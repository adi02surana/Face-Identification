"""
training/phase2_triplet.py

Phase 2 — Twin/Sibling-Aware Fine-Tuning with Mandatory Hard-Negative Mining.
References:
- Schroff et al., "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015.
- Paone et al. (2014) & Ahmad et al. (2020) on twin face discrimination limitations.

CRITICAL DESIGN REQUIREMENT (README §8):
For any given anchor, negative samples MUST be mined from that subject's own twin (CTTS-80)
or own sibling / family member (FIW) — NEVER random strangers.
Random-negative triplets never force the network to resolve genuinely similar faces.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import LOGS_DIR, PROCESSED_CTTS80_DIR, PROCESSED_FIW_DIR, resolve_fiw_path
from degradation.simulator import DegradationSimulator
from models.dual_path_net import DualPathFaceNet

CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR = PROJECT_DIR / "splits" / "manifests"


# -----------------------------------------------------------------------------
# 1. Twin-Aware Triplet Loss
# -----------------------------------------------------------------------------

class TwinAwareTripletLoss(nn.Module):
    """
    Triplet loss operating on cosine distance between L2-normalized face embeddings:
    D(u, v) = 1.0 - cos_sim(u, v)
    Loss = max(0, D(anchor, positive) - D(anchor, negative) + margin)
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Cosine similarity for L2-normalized embeddings
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
        neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=1)

        d_pos = 1.0 - pos_sim
        d_neg = 1.0 - neg_sim

        losses = F.relu(d_pos - d_neg + self.margin)
        loss = losses.mean()

        # Metrics for monitoring
        violation_rate = (losses > 0).float().mean().item()
        stats = {
            "d_pos_mean": d_pos.mean().item(),
            "d_neg_mean": d_neg.mean().item(),
            "margin_violation_pct": violation_rate * 100.0,
            "raw_loss": loss.item(),
        }
        return loss, stats


# -----------------------------------------------------------------------------
# 2. Hard-Negative Twin & Sibling Triplet Dataset
# -----------------------------------------------------------------------------

class TwinSiblingTripletDataset(Dataset):
    """
    Constructs hard triplets strictly from subject's own twin (CTTS-80)
    or sibling / family member (FIW).
    
    Anchor (a): Image of subject A
    Positive (p): Different image of subject A
    Hard Negative (n): Image of subject A's twin (CTTS-80) or sibling (FIW)
    """

    def __init__(
        self,
        ctts80_dir: Path = PROCESSED_CTTS80_DIR,
        ctts80_twin_pairs: Optional[List[str]] = None,
        fiw_dir: Path = PROCESSED_FIW_DIR,
        fiw_pairs_manifest: Optional[Path] = MANIFESTS_DIR / "fiw_disjoint_splits.json",
        max_triplets: Optional[int] = None,
        is_train: bool = True,
        apply_degradation: bool = False,
        degradation_prob: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        self.is_train = is_train
        self.apply_degradation = apply_degradation
        self.degradation_prob = degradation_prob
        self.rng = random.Random(seed)
        self.triplets: List[Tuple[str, str, str, str]] = []  # (anchor, pos, neg, source)
        self.simulator = DegradationSimulator(split="train" if is_train else "eval", seed=seed) if apply_degradation else None

        # 1. Build CTTS-80 Twin Triplets
        if ctts80_dir.exists():
            all_pair_dirs = sorted([d for d in ctts80_dir.iterdir() if d.is_dir()])
            if ctts80_twin_pairs:
                active_pairs = [d for d in all_pair_dirs if d.name in ctts80_twin_pairs]
            else:
                active_pairs = all_pair_dirs

            for pair_dir in active_pairs:
                twins = sorted([d for d in pair_dir.iterdir() if d.is_dir()])
                if len(twins) < 2:
                    continue

                imgs_a = sorted(list(twins[0].glob("*.*")))
                imgs_b = sorted(list(twins[1].glob("*.*")))

                if len(imgs_a) >= 2 and len(imgs_b) >= 1:
                    # Anchor from Twin A, Positive from Twin A, Negative from Twin B
                    for i in range(len(imgs_a)):
                        for j in range(len(imgs_a)):
                            if i == j:
                                continue
                            for k in range(min(3, len(imgs_b))):
                                self.triplets.append((
                                    str(imgs_a[i]),
                                    str(imgs_a[j]),
                                    str(imgs_b[k]),
                                    "ctts80_twin",
                                ))

                if len(imgs_b) >= 2 and len(imgs_a) >= 1:
                    # Anchor from Twin B, Positive from Twin B, Negative from Twin A
                    for i in range(len(imgs_b)):
                        for j in range(len(imgs_b)):
                            if i == j:
                                continue
                            for k in range(min(3, len(imgs_a))):
                                self.triplets.append((
                                    str(imgs_b[i]),
                                    str(imgs_b[j]),
                                    str(imgs_a[k]),
                                    "ctts80_twin",
                                ))

        # 2. Build FIW Sibling / Family Hard-Negative Triplets
        if fiw_pairs_manifest and Path(fiw_pairs_manifest).exists():
            try:
                with open(fiw_pairs_manifest, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                fiw_split_pairs = manifest.get("train_pairs", []) if is_train else manifest.get("val_pairs", [])

                # Map positive pairs (same person) and hard negative pairs (same family)
                # Group by (FID, MID)
                member_imgs = {}
                family_members = {}

                for p in fiw_split_pairs:
                    im1 = p["image_1"]
                    im2 = p["image_2"]
                    ptype = p["pair_type"]
                    f1 = p.get("fid_1")
                    m1 = p.get("mid_1")

                    # We can use the hard_negative pairs directly: (im1, im2) are same family, different member
                    if ptype == "same_family_hard_negative":
                        # We need a positive for im1
                        resolved_1 = resolve_fiw_path(im1)
                        resolved_2 = resolve_fiw_path(im2)
                        if resolved_1 and resolved_2 and resolved_1.exists() and resolved_2.exists():
                            # Find another image from same member folder
                            sibling_imgs = list(resolved_1.parent.glob("*.jpg"))
                            if len(sibling_imgs) >= 2:
                                pos_img = sibling_imgs[0] if sibling_imgs[0] != resolved_1 else sibling_imgs[1]
                                self.triplets.append((
                                    str(resolved_1),
                                    str(pos_img),
                                    str(resolved_2),
                                    "fiw_sibling",
                                ))
            except Exception as e:
                print(f"Notice reading FIW manifest: {e}")

        self.rng.shuffle(self.triplets)
        if max_triplets:
            self.triplets = self.triplets[:max_triplets]

        print(f"TwinSiblingTripletDataset ({'train' if is_train else 'val'}): loaded {len(self.triplets)} hard-negative triplets.")

    def __len__(self) -> int:
        return len(self.triplets)

    def _load_and_preprocess(self, path_str: str) -> torch.Tensor:
        bgr = cv2.imread(path_str)
        if bgr is None:
            return torch.zeros(3, 112, 112, dtype=torch.float32)

        # Ensure 112x112
        if bgr.shape[:2] != (112, 112):
            bgr = cv2.resize(bgr, (112, 112))

        # Optical Degradation channel: apply online blur, compression, or illumination shift
        if self.apply_degradation and self.is_train and self.simulator is not None:
            if random.random() < self.degradation_prob:
                axis = random.choice(["blur", "jpeg", "illumination"])
                sev = random.randint(1, 5)
                bgr, _ = self.simulator.apply(bgr, axis=axis, severity=sev)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.is_train and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)

        norm = (rgb.astype(np.float32) / 127.5) - 1.0
        return torch.from_numpy(norm).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        a_path, p_path, n_path, source = self.triplets[idx]
        t_a = self._load_and_preprocess(a_path)
        t_p = self._load_and_preprocess(p_path)
        t_n = self._load_and_preprocess(n_path)
        return t_a, t_p, t_n, source


# -----------------------------------------------------------------------------
# 3. Fine-Tuning Pipeline
# -----------------------------------------------------------------------------

def train_phase2(
    fold_idx: int = 0,
    kfold_manifest: Path = MANIFESTS_DIR / "ctts80_kfold_splits.json",
    fiw_manifest: Path = MANIFESTS_DIR / "fiw_disjoint_splits.json",
    phase1_checkpoint: Optional[Path] = CHECKPOINTS_DIR / "phase1_arcface_best.pth",
    backbone_name: str = "resnet50",
    use_regional: bool = True,  # Full Dual-Path network with regional jawline path
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    margin: float = 0.3,
    max_triplets: Optional[int] = 2000,
    apply_degradation: bool = False,
    degradation_prob: float = 0.5,
    num_workers: int = 4,
    device_str: str = "auto",
    save_prefix: Optional[str] = None,
) -> Dict:
    """Runs Phase 2 fine-tuning with hard-negative twin/sibling triplet loss."""
    if save_prefix is None:
        save_prefix = "phase2_triplet_degraded" if apply_degradation else "phase2_triplet_clean"

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 70)
    mode_str = "DEGRADATION-AWARE" if apply_degradation else "CLEAN"
    print(f"PHASE 2: {mode_str} TWIN/SIBLING TRIPLET FINE-TUNING (Fold {fold_idx})")
    print(f"Device: {device} | Use Regional: {use_regional} | Margin: {margin} | Degrade Prob: {degradation_prob if apply_degradation else 0.0}")
    print("=" * 70)

    # 1. Load CTTS-80 fold training twin pairs
    train_twin_pairs = []
    if kfold_manifest and kfold_manifest.exists():
        with open(kfold_manifest, "r", encoding="utf-8") as f:
            kf_data = json.load(f)
            folds = kf_data.get("folds", [])
            if fold_idx < len(folds):
                train_twin_pairs = folds[fold_idx]["train_twin_pairs"]
                print(f"Fold {fold_idx}: Loaded {len(train_twin_pairs)} training twin pairs (held-out {len(folds[fold_idx]['test_twin_pairs'])} test pairs).")

    # 2. Dataset and Loader
    dataset = TwinSiblingTripletDataset(
        ctts80_twin_pairs=train_twin_pairs,
        fiw_pairs_manifest=fiw_manifest,
        max_triplets=max_triplets,
        is_train=True,
        apply_degradation=apply_degradation,
        degradation_prob=degradation_prob,
    )
    if len(dataset) == 0:
        print("No triplets generated. Check processed data paths.")
        return {}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if str(device) != "cpu" else 0,
        drop_last=True,
    )

    # 3. Model Initialization
    model = DualPathFaceNet(
        backbone_name=backbone_name,
        pretrained_backbone=True,
        use_cbam=True,
        use_regional=use_regional,
        embedding_dim=512,
    ).to(device)

    # Initialize from Phase 1 weights if available
    if phase1_checkpoint and Path(phase1_checkpoint).exists():
        print(f"Loading Phase 1 pretrained weights from {phase1_checkpoint}...")
        try:
            state_dict = torch.load(phase1_checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded Phase 1 checkpoint!")
        except Exception as e:
            print(f"Could not load Phase 1 checkpoint ({e}), proceeding with backbone weights.")

    criterion = TwinAwareTripletLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "loss": [],
        "d_pos_mean": [],
        "d_neg_mean": [],
        "margin_violation_pct": [],
    }
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_d_pos = 0.0
        total_d_neg = 0.0
        total_violations = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"Fine-Tuning Epoch {epoch}/{epochs}")
        for anchors, positives, negatives, _ in pbar:
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()

            emb_a = model(anchors)
            emb_p = model(positives)
            emb_n = model(negatives)

            loss, stats = criterion(emb_a, emb_p, emb_n)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += stats["raw_loss"]
            total_d_pos += stats["d_pos_mean"]
            total_d_neg += stats["d_neg_mean"]
            total_violations += stats["margin_violation_pct"]
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss / n_batches:.4f}",
                "d_pos": f"{total_d_pos / n_batches:.3f}",
                "d_neg": f"{total_d_neg / n_batches:.3f}",
                "viol%": f"{total_violations / n_batches:.1f}%",
            })

        scheduler.step()
        ep_loss = total_loss / max(1, n_batches)
        ep_d_pos = total_d_pos / max(1, n_batches)
        ep_d_neg = total_d_neg / max(1, n_batches)
        ep_viol = total_violations / max(1, n_batches)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) - "
            f"Loss: {ep_loss:.4f} | D_pos: {ep_d_pos:.3f} | D_neg: {ep_d_neg:.3f} | "
            f"Margin Violations: {ep_viol:.1f}%"
        )

        history["loss"].append(ep_loss)
        history["d_pos_mean"].append(ep_d_pos)
        history["d_neg_mean"].append(ep_d_neg)
        history["margin_violation_pct"].append(ep_viol)

        # Checkpoint saving
        ckpt_path = CHECKPOINTS_DIR / f"{save_prefix}_fold{fold_idx}_latest.pth"
        torch.save(
            {
                "epoch": epoch,
                "fold_idx": fold_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": ep_loss,
                "use_regional": use_regional,
            },
            ckpt_path,
        )

        if ep_loss < best_loss:
            best_loss = ep_loss
            best_ckpt = CHECKPOINTS_DIR / f"{save_prefix}_fold{fold_idx}_best.pth"
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> Saved best checkpoint to {best_ckpt}")

    # Save log
    log_file = LOGS_DIR / f"{save_prefix}_fold{fold_idx}_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nPhase 2 Fine-Tuning Complete! Log saved to {log_file}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Twin/Sibling Hard-Negative Fine-Tuning")
    parser.add_argument("--fold", type=int, default=0, help="CTTS-80 fold index (0-9)")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--max-triplets", type=int, default=2000, help="Max triplets to train on")
    parser.add_argument("--apply-degradation", action="store_true", help="Enable optical degradation channel during fine-tuning")
    parser.add_argument("--degradation-prob", type=float, default=0.5, help="Probability of applying optical degradation per image")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--no-regional", action="store_true", help="Disable regional branch")
    parser.add_argument("--save-prefix", type=str, default=None, help="Checkpoint name prefix")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 1-epoch sanity test")

    args = parser.parse_args()

    if args.smoke_test:
        train_phase2(
            fold_idx=args.fold,
            epochs=1,
            batch_size=4,
            max_triplets=20,
            use_regional=not args.no_regional,
            apply_degradation=args.apply_degradation,
            degradation_prob=args.degradation_prob,
            num_workers=0,
            save_prefix=args.save_prefix or ("phase2_triplet_degraded_smoketest" if args.apply_degradation else "phase2_triplet_clean_smoketest"),
        )
    else:
        train_phase2(
            fold_idx=args.fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            margin=args.margin,
            max_triplets=args.max_triplets,
            use_regional=not args.no_regional,
            apply_degradation=args.apply_degradation,
            degradation_prob=args.degradation_prob,
            num_workers=args.num_workers,
            save_prefix=args.save_prefix,
        )

