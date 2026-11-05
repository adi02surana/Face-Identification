"""
training package

Provides:
- ArcMarginProduct, DigiFaceDataset, train_phase1 (Phase 1 pretraining)
- TwinAwareTripletLoss, TwinSiblingTripletDataset, train_phase2 (Phase 2 fine-tuning)
"""

from training.phase1_arcface import ArcMarginProduct, DigiFaceDataset, train_phase1
from training.phase2_triplet import (
    TwinAwareTripletLoss,
    TwinSiblingTripletDataset,
    train_phase2,
)

__all__ = [
    "ArcMarginProduct",
    "DigiFaceDataset",
    "train_phase1",
    "TwinAwareTripletLoss",
    "TwinSiblingTripletDataset",
    "train_phase2",
]

