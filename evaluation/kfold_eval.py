"""
evaluation/kfold_eval.py

10-Fold Cross-Validation Evaluation & Optical Degradation Sweep.
Evaluates verification accuracy on:
1. Clean images
2. 3 Degradation axes x 5 severity levels (15 degraded conditions)
3. Three-class verification outcomes: Same-person, Twin/Sibling imposter, General imposter.
4. Relative Attention Shift (Regional vs Global reliance under degradation)
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import LOGS_DIR, PROCESSED_CTTS80_DIR, PROCESSED_DIGIFACE_DIR, PROCESSED_FIW_DIR, resolve_fiw_path
from degradation.simulator import DegradationSimulator
from evaluation.metrics import (
    calibrate_three_class_thresholds,
    compute_roc_eer,
    compute_tar_at_far,
    evaluate_three_class_predictions,
    predict_three_class,
)
from models.dual_path_net import DualPathFaceNet

MANIFESTS_DIR = PROJECT_DIR / "splits" / "manifests"


def load_and_preprocess_img(path_str: str) -> np.ndarray:
    """Loads BGR image and ensures 112x112 uint8 format."""
    p = Path(path_str)
    if not p.exists():
        return np.zeros((112, 112, 3), dtype=np.uint8)
    bgr = cv2.imread(str(p))
    if bgr is None:
        return np.zeros((112, 112, 3), dtype=np.uint8)
    if bgr.shape[:2] != (112, 112):
        bgr = cv2.resize(bgr, (112, 112))
    return bgr


def to_tensor(bgr_img: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    norm = (rgb.astype(np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


class ModelEvaluator:
    def __init__(
        self,
        checkpoint_path: Path,
        backbone_name: str = "resnet50",
        use_regional: bool = True,
        device_str: str = "auto",
    ):
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model = DualPathFaceNet(
            backbone_name=backbone_name,
            pretrained_backbone=False,
            use_regional=use_regional,
        ).to(self.device)

        if Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint from: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Using base weights.")

        self.model.eval()
        self.simulator = DegradationSimulator(split="eval")

    @torch.no_grad()
    def get_embedding(self, bgr_img: np.ndarray, return_dict: bool = False):
        t = to_tensor(bgr_img, self.device)
        return self.model(t, return_dict=return_dict)

    def compute_similarity(self, bgr1: np.ndarray, bgr2: np.ndarray) -> Tuple[float, Optional[Dict]]:
        out1 = self.get_embedding(bgr1, return_dict=True)
        out2 = self.get_embedding(bgr2, return_dict=True)
        e1 = out1["embedding"]
        e2 = out2["embedding"]
        sim = F.cosine_similarity(e1, e2).item()

        # Average reliance metrics across the pair
        g_norm = (out1["reliance_metrics"]["global_norm"] + out2["reliance_metrics"]["global_norm"]) / 2.0
        r_norm = (out1["reliance_metrics"]["regional_norm"] + out2["reliance_metrics"]["regional_norm"]) / 2.0
        r_ratio = (out1["reliance_metrics"]["regional_ratio"] + out2["reliance_metrics"]["regional_ratio"]) / 2.0
        gate_val = (out1["reliance_metrics"]["gate_value"] + out2["reliance_metrics"]["gate_value"]) / 2.0

        metrics = {
            "global_norm": g_norm,
            "regional_norm": r_norm,
            "regional_ratio": r_ratio,
            "gate_value": gate_val,
        }
        return sim, metrics


def build_evaluation_pairs(
    fold_idx: int = 0,
    kfold_manifest: Path = MANIFESTS_DIR / "ctts80_kfold_splits.json",
    fiw_manifest: Path = MANIFESTS_DIR / "fiw_disjoint_splits.json",
    digiface_manifest: Path = MANIFESTS_DIR / "digiface_disjoint_splits.json",
    max_eval_pairs: int = 100,
) -> Dict[str, List[Tuple[str, str]]]:
    """Builds evaluation pairs: same-person, twin-imposter, sibling-imposter, unrelated-imposter."""
    pairs = {
        "same": [],
        "twin": [],
        "sibling": [],
        "unrelated": [],
    }

    # 1. CTTS-80 Held-out Test Twin Pairs (Fold fold_idx)
    if kfold_manifest.exists():
        with open(kfold_manifest, "r", encoding="utf-8") as f:
            kf = json.load(f)
        test_twins = kf["folds"][fold_idx]["test_twin_pairs"]
        for p_name in test_twins:
            p_dir = PROCESSED_CTTS80_DIR / p_name
            if not p_dir.exists() and (PROCESSED_CTTS80_DIR / "ctts80_112x112" / p_name).exists():
                p_dir = PROCESSED_CTTS80_DIR / "ctts80_112x112" / p_name
            if p_dir.exists():
                twins = sorted([d for d in p_dir.iterdir() if d.is_dir()])
                if len(twins) >= 2:
                    imgs_a = sorted(list(twins[0].glob("*.*")))
                    imgs_b = sorted(list(twins[1].glob("*.*")))
                    # Same person pairs from Twin A
                    if len(imgs_a) >= 2:
                        pairs["same"].append((str(imgs_a[0]), str(imgs_a[1])))
                    # Same person pairs from Twin B
                    if len(imgs_b) >= 2:
                        pairs["same"].append((str(imgs_b[0]), str(imgs_b[1])))
                    # Twin imposter pairs
                    if len(imgs_a) >= 1 and len(imgs_b) >= 1:
                        pairs["twin"].append((str(imgs_a[0]), str(imgs_b[0])))

    # 2. FIW Sibling Imposter Pairs (Test Split)
    if fiw_manifest.exists():
        with open(fiw_manifest, "r", encoding="utf-8") as f:
            fiw_data = json.load(f)
        for p in fiw_data.get("test_pairs", []):
            if p.get("pair_type") == "same_family_hard_negative":
                r1 = resolve_fiw_path(p["image_1"])
                r2 = resolve_fiw_path(p["image_2"])
                if r1 and r2 and r1.exists() and r2.exists():
                    pairs["sibling"].append((str(r1), str(r2)))

    # 3. DigiFace-1M Unrelated Imposter Pairs (Val Split)
    if digiface_manifest.exists():
        with open(digiface_manifest, "r", encoding="utf-8") as f:
            digi_data = json.load(f)
        val_subs = digi_data.get("val_subjects", [])
        if len(val_subs) >= 2:
            for i in range(min(max_eval_pairs, len(val_subs) - 1)):
                d1 = PROCESSED_DIGIFACE_DIR / val_subs[i]
                d2 = PROCESSED_DIGIFACE_DIR / val_subs[i + 1]
                if d1.exists() and d2.exists():
                    im1 = list(d1.glob("*.png")) + list(d1.glob("*.jpg"))
                    im2 = list(d2.glob("*.png")) + list(d2.glob("*.jpg"))
                    if im1 and im2:
                        pairs["unrelated"].append((str(im1[0]), str(im2[0])))

    # Limit to max_eval_pairs per group
    for k in pairs:
        if max_eval_pairs:
            pairs[k] = pairs[k][:max_eval_pairs]
        print(f"Evaluation group [{k:9s}]: {len(pairs[k])} test pairs.")

    return pairs


def run_evaluation_sweep(
    evaluator: ModelEvaluator,
    eval_pairs: Dict[str, List[Tuple[str, str]]],
    output_json: Path,
) -> Dict:
    """Runs verification sweep over Clean and all 15 Degradation conditions."""
    conditions = [("clean", 0)]
    for axis in ["blur", "jpeg", "illumination"]:
        for sev in range(1, 6):
            conditions.append((axis, sev))

    results = {}

    for axis, sev in conditions:
        cond_name = f"{axis}_sev{sev}" if sev > 0 else "clean"
        print(f"\n--- Evaluating Condition: {cond_name.upper()} ---")

        all_scores = {}
        all_metrics = {}

        for group in ["same", "twin", "sibling", "unrelated"]:
            scores = []
            rel_ratios = []
            gate_vals = []

            for p1, p2 in eval_pairs[group]:
                im1 = load_and_preprocess_img(p1)
                im2 = load_and_preprocess_img(p2)

                # Apply optical degradation to image 2 if degraded condition
                if sev > 0:
                    im2, _ = evaluator.simulator.apply(im2, axis=axis, severity=sev)

                sim, reliance = evaluator.compute_similarity(im1, im2)
                scores.append(sim)
                rel_ratios.append(reliance["regional_ratio"])
                gate_vals.append(reliance["gate_value"])

            all_scores[group] = np.array(scores)
            all_metrics[group] = {
                "mean_sim": float(np.mean(scores)) if scores else 0.0,
                "std_sim": float(np.std(scores)) if scores else 0.0,
                "mean_regional_ratio": float(np.mean(rel_ratios)) if rel_ratios else 0.0,
                "mean_gate_value": float(np.mean(gate_vals)) if gate_vals else 0.0,
            }

        # 1. Binary Twin-Verification EER (Same vs Twin Imposter)
        y_twin = np.concatenate([np.ones(len(all_scores["same"])), np.zeros(len(all_scores["twin"]))])
        s_twin = np.concatenate([all_scores["same"], all_scores["twin"]])
        eer_twin, opt_th_twin, fpr_tw, tpr_tw, _ = compute_roc_eer(s_twin, y_twin)
        tar_at_far1 = compute_tar_at_far(fpr_tw, tpr_tw, target_far=0.01)

        # 2. General-Verification EER (Same vs Unrelated)
        y_unrel = np.concatenate([np.ones(len(all_scores["same"])), np.zeros(len(all_scores["unrelated"]))])
        s_unrel = np.concatenate([all_scores["same"], all_scores["unrelated"]])
        eer_unrel, opt_th_unrel, _, _, _ = compute_roc_eer(s_unrel, y_unrel)

        # 3. 3-Class Decision Rule (Same [0], Twin/Sibling [1], Unrelated [2])
        combined_rel = np.concatenate([all_scores["twin"], all_scores["sibling"]])
        thresh_dict = calibrate_three_class_thresholds(all_scores["same"], combined_rel, all_scores["unrelated"])

        # Ground truth: 0 for same, 1 for twin/sibling, 2 for unrelated
        y_true_3 = np.concatenate([
            np.zeros(len(all_scores["same"]), dtype=np.int64),
            np.ones(len(combined_rel), dtype=np.int64),
            np.full(len(all_scores["unrelated"]), 2, dtype=np.int64),
        ])
        all_sims_3 = np.concatenate([all_scores["same"], combined_rel, all_scores["unrelated"]])
        y_pred_3 = predict_three_class(all_sims_3, thresh_dict["tau_twin_same"], thresh_dict["tau_twin_unrelated"])
        three_class_eval = evaluate_three_class_predictions(y_true_3, y_pred_3)

        results[cond_name] = {
            "axis": axis,
            "severity": sev,
            "twin_eer": eer_twin,
            "twin_optimal_threshold": opt_th_twin,
            "twin_tar_at_far001": tar_at_far1,
            "unrelated_eer": eer_unrel,
            "group_metrics": all_metrics,
            "three_class_metrics": {
                "accuracy": three_class_eval["accuracy"],
                "thresholds": thresh_dict,
                "confusion_matrix": three_class_eval["confusion_matrix"],
            },
            "attention_shift": {
                "regional_ratio_same": all_metrics["same"]["mean_regional_ratio"],
                "regional_ratio_twin": all_metrics["twin"]["mean_regional_ratio"],
                "gate_value": all_metrics["same"]["mean_gate_value"],
            },
        }

        print(f"[{cond_name}] Twin EER: {eer_twin * 100:.2f}% | 3-Class Accuracy: {three_class_eval['accuracy'] * 100:.2f}% | Reg Ratio: {all_metrics['same']['mean_regional_ratio']:.3f}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation Complete! Results saved to: {output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate face verification model under optical degradation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--fold", type=int, default=0, help="CTTS-80 fold index (0-9)")
    parser.add_argument("--max-pairs", type=int, default=80, help="Max test pairs per evaluation group")
    parser.add_argument("--no-regional", action="store_true", help="Disable regional branch")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        checkpoint_path=Path(args.checkpoint),
        use_regional=not args.no_regional,
    )

    eval_pairs = build_evaluation_pairs(
        fold_idx=args.fold,
        max_eval_pairs=args.max_pairs,
    )

    out_p = Path(args.output) if args.output else (LOGS_DIR / f"eval_results_fold{args.fold}.json")
    run_evaluation_sweep(evaluator, eval_pairs, out_p)

