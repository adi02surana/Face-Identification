"""
evaluation/metrics.py

Evaluation metrics for face verification under optical degradation:
1. EER (Equal Error Rate) and optimal threshold calculation
2. TAR@FAR (True Acceptance Rate at fixed False Acceptance Rate)
3. 3-Class Decision Rule (Same-person, Twin/Sibling-imposter, General-imposter)
4. Confusion matrix and multi-class classification metrics
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report


def compute_roc_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes ROC curve, AUC, and EER (Equal Error Rate).

    Args:
        scores: 1D array of cosine similarities in [-1, 1]
        labels: 1D array of binary labels (1 = positive/same, 0 = negative/imposter)
    Returns:
        eer: float (Equal Error Rate in [0, 1])
        optimal_thresh: float (Threshold where FAR == FRR)
        fpr: False Positive Rate array
        tpr: True Positive Rate array
        thresholds: Thresholds array
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # EER is point where FAR (fpr) == FRR (fnr)
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    optimal_thresh = float(thresholds[idx])

    return float(eer), optimal_thresh, fpr, tpr, thresholds


def compute_tar_at_far(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_far: float = 0.001,
) -> float:
    """Computes True Acceptance Rate (TAR) at a specified False Acceptance Rate (FAR)."""
    idx = np.where(fpr <= target_far)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def calibrate_three_class_thresholds(
    same_scores: np.ndarray,
    twin_sibling_scores: np.ndarray,
    unrelated_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Calibrates decision boundaries for the 3-class verification task:
      - Class 0: Same Person
      - Class 1: Twin / Sibling Imposter
      - Class 2: General / Unrelated Imposter

    Decision Rule:
      - sim >= tau_twin_same: Same Person (0)
      - tau_twin_unrelated <= sim < tau_twin_same: Twin/Sibling Imposter (1)
      - sim < tau_twin_unrelated: General Imposter (2)
    """
    # 1. Threshold separating same-identity vs twin/sibling imposter
    y_twin_same = np.concatenate([np.ones_like(same_scores), np.zeros_like(twin_sibling_scores)])
    scores_twin_same = np.concatenate([same_scores, twin_sibling_scores])
    _, tau_twin_same, _, _, _ = compute_roc_eer(scores_twin_same, y_twin_same)

    # 2. Threshold separating twin/sibling imposter vs general unrelated imposter
    y_rel_unrel = np.concatenate([np.ones_like(twin_sibling_scores), np.zeros_like(unrelated_scores)])
    scores_rel_unrel = np.concatenate([twin_sibling_scores, unrelated_scores])
    _, tau_twin_unrelated, _, _, _ = compute_roc_eer(scores_rel_unrel, y_rel_unrel)

    return {
        "tau_twin_same": float(tau_twin_same),
        "tau_twin_unrelated": float(tau_twin_unrelated),
    }


def predict_three_class(
    similarities: np.ndarray,
    tau_twin_same: float,
    tau_twin_unrelated: float,
) -> np.ndarray:
    """
    Predicts 3-class verification outcome based on calibrated thresholds:
      0: Same Person
      1: Twin / Sibling Imposter
      2: General / Unrelated Imposter
    """
    preds = np.full(similarities.shape, 2, dtype=np.int64)  # default: general imposter
    preds[similarities >= tau_twin_same] = 0
    preds[(similarities >= tau_twin_unrelated) & (similarities < tau_twin_same)] = 1
    return preds


def evaluate_three_class_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Evaluates multi-class accuracy and confusion matrix for 3-class outcomes."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    target_names = ["same_person", "twin_sibling_imposter", "general_imposter"]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

