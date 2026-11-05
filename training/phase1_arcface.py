"""
training/phase1_arcface.py

Phase 1 — General Pretraining on DigiFace-1M with ArcFace Angular Margin Loss.
Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019.

Optimizes angular margin between ground-truth identity class and imposter classes:
L = -log( exp(s * cos(theta_yi + m)) / (exp(s * cos(theta_yi + m)) + sum_{j != yi} exp(s * cos(theta_j))) )
where s = 64.0 (scale factor), m = 0.50 (angular margin).
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

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

from config import LOGS_DIR, PROCESSED_DIGIFACE_DIR
from models.dual_path_net import DualPathFaceNet

CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. ArcFace Angular Margin Head (Deng et al., CVPR 2019)
# -----------------------------------------------------------------------------

class ArcMarginProduct(nn.Module):
    """
    ArcFace Additive Angular Margin Head.
    
    Applies additive angular margin m to the target class angle theta_yi,
    scaled by hyperequatorial scale factor s:
        cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
    """

    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 1000,
        s: float = 64.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (B, in_features) L2-normalized embedding
            label: (B,) ground truth identity indices
        Returns:
            output: (B, out_features) scaled ArcFace logits
        """
        # Ensure weight is L2-normalized
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity between feature embeddings and class weights: cos(theta)
        cosine = F.linear(input_features, normalized_weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for ground-truth identity classes
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# -----------------------------------------------------------------------------
# 2. DigiFace-1M Pretraining Dataset
# -----------------------------------------------------------------------------

class DigiFaceDataset(Dataset):
    """
    Dataset loader for preprocessed 112x112 DigiFace-1M faces.
    """

    def __init__(
        self,
        root_dir: Path = PROCESSED_DIGIFACE_DIR,
        subject_list: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
        max_imgs_per_subject: Optional[int] = None,
        is_train: bool = True,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.is_train = is_train

        # Collect subjects
        if subject_list is not None:
            valid_dirs = [self.root_dir / s for s in subject_list if (self.root_dir / s).exists()]
        else:
            valid_dirs = []
            if self.root_dir.exists():
                for subset in sorted(self.root_dir.iterdir()):
                    if subset.is_dir():
                        for s in sorted(subset.iterdir()):
                            if s.is_dir():
                                valid_dirs.append(s)

        if max_subjects:
            valid_dirs = valid_dirs[:max_subjects]

        self.samples = []  # List of (img_path, class_idx)
        self.class_to_idx = {d.name: idx for idx, d in enumerate(valid_dirs)}
        self.num_classes = len(self.class_to_idx)

        for d in valid_dirs:
            cls_idx = self.class_to_idx[d.name]
            imgs = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
            if max_imgs_per_subject:
                imgs = imgs[:max_imgs_per_subject]
            for p in imgs:
                self.samples.append((str(p), cls_idx))

        print(f"DigiFaceDataset ({'train' if is_train else 'val'}): loaded {len(self.samples)} images across {self.num_classes} identities.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            # Fallback zero tensor if image corrupted
            tensor = torch.zeros(3, 112, 112, dtype=torch.float32)
            return tensor, label

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Training augmentation: random horizontal flip
        if self.is_train and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)

        # Normalize pixel values to [-1, 1] standard for ArcFace
        img_np = (rgb.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        return tensor, label


# -----------------------------------------------------------------------------
# 3. Training Loop
# -----------------------------------------------------------------------------

def train_phase1(
    data_dir: Path = PROCESSED_DIGIFACE_DIR,
    manifest_path: Optional[Path] = PROJECT_DIR / "splits" / "manifests" / "digiface_disjoint_splits.json",
    backbone_name: str = "resnet50",
    use_regional: bool = False,  # Baseline global-CBAM model first (README §10.3)
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_subjects: Optional[int] = None,
    num_workers: int = 4,
    device_str: str = "auto",
    save_prefix: str = "phase1_arcface",
) -> Dict:
    """Runs Phase 1 general pretraining with ArcFace angular margin loss."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 70)
    print("PHASE 1: ARCFACE GENERAL PRETRAINING (DIGIFACE-1M)")
    print(f"Device: {device} | Backbone: {backbone_name} | Use Regional: {use_regional}")
    print("=" * 70)

    # Load split manifest if available
    train_subs, val_subs = None, None
    if manifest_path and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            all_train = meta.get("train_subjects", [])
            all_val = meta.get("val_subjects", [])
            if max_subjects and max_subjects > 0:
                train_subs = all_train[:max_subjects]
                val_subs = all_val[:max(10, max_subjects // 10)]
            else:
                train_subs = all_train
                val_subs = all_val

    # Datasets and Loaders
    effective_max_sub = max_subjects if (max_subjects and max_subjects > 0) else None
    train_dataset = DigiFaceDataset(
        root_dir=data_dir,
        subject_list=train_subs,
        max_subjects=effective_max_sub,
        max_imgs_per_subject=20,
        is_train=True,
    )
    if len(train_dataset) == 0:
        print("No training images found in dataset. Check data directory.")
        return {}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if str(device) != "cpu" else 0,
        drop_last=True,
    )

    num_classes = train_dataset.num_classes
    print(f"Initializing DualPathFaceNet ({backbone_name}) and ArcMarginProduct (num_classes={num_classes})...")

    model = DualPathFaceNet(
        backbone_name=backbone_name,
        pretrained_backbone=True,
        use_cbam=True,
        use_regional=use_regional,
        embedding_dim=512,
    ).to(device)

    arcface_head = ArcMarginProduct(
        in_features=512,
        out_features=num_classes,
        s=64.0,
        m=0.50,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface_head.parameters()),
        lr=lr,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": []}
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        arcface_head.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(imgs)  # 512-d L2-normalized embeddings
            logits = arcface_head(embeddings, labels)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Stats
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

            curr_loss = total_loss / total_samples
            curr_acc = (correct / total_samples) * 100.0
            pbar.set_postfix({"loss": f"{curr_loss:.4f}", "acc": f"{curr_acc:.2f}%"})

        scheduler.step()
        epoch_loss = total_loss / total_samples
        epoch_acc = (correct / total_samples) * 100.0
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} Complete ({elapsed:.1f}s) - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        # Save checkpoint
        ckpt_path = CHECKPOINTS_DIR / f"{save_prefix}_latest.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "head_state_dict": arcface_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "num_classes": num_classes,
                "use_regional": use_regional,
            },
            ckpt_path,
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_ckpt_path = CHECKPOINTS_DIR / f"{save_prefix}_best.pth"
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> Saved best model checkpoint to {best_ckpt_path}")

    # Save training log
    log_file = LOGS_DIR / f"{save_prefix}_training_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nPhase 1 Pretraining Complete! History saved to {log_file}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 ArcFace General Pretraining")
    parser.add_argument("--epochs", type=int, default=5, help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-subjects", type=int, default=0, help="Max subjects to train on (0 = all 72,833 subjects)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--use-regional", action="store_true", help="Activate regional branch")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 1-epoch sanity test")

    args = parser.parse_args()

    if args.smoke_test:
        train_phase1(
            epochs=1,
            batch_size=4,
            max_subjects=10,
            num_workers=0,
            save_prefix="phase1_arcface_smoketest",
        )
    else:
        train_phase1(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_subjects=args.max_subjects,
            num_workers=args.num_workers,
            use_regional=args.use_regional,
        )

