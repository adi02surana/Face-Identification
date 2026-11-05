"""
degradation/simulator.py

Deterministic optical degradation simulator built strictly from OpenCV, NumPy, and Pillow.
Supports 3 degradation axes with 5 calibrated severity levels each:
1. Blur (Isotropic + Anisotropic Gaussian + Linear Motion Blur; disjoint train/eval kernels)
2. JPEG compression (Calibrated quality factors)
3. Illumination shift (Gamma correction + linear gain/offset over- and under-exposure)

Follows specifications in README §6 and Yasarla & Patel (2020).
"""

import sys
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from degradation.kernels import get_disjoint_blur_kernel

# JPEG Quality Bands across 5 Severity Levels
JPEG_QUALITY_CONFIGS = {
    1: {"q_range_train": (75, 80), "q_eval": 78},  # Mild
    2: {"q_range_train": (55, 65), "q_eval": 60},  # Low-Medium
    3: {"q_range_train": (35, 45), "q_eval": 40},  # Medium
    4: {"q_range_train": (18, 25), "q_eval": 20},  # Heavy
    5: {"q_range_train": (5, 10), "q_eval": 7},    # Severe
}

# Illumination Shift Parameters across 5 Severity Levels
ILLUMINATION_CONFIGS = {
    # Level 1: Mild
    1: {
        "under": {"gamma": 0.80, "gain": 0.85, "offset": -10},
        "over": {"gamma": 1.25, "gain": 1.15, "offset": 10},
    },
    # Level 2: Low-Medium
    2: {
        "under": {"gamma": 0.65, "gain": 0.70, "offset": -20},
        "over": {"gamma": 1.50, "gain": 1.30, "offset": 20},
    },
    # Level 3: Medium
    3: {
        "under": {"gamma": 0.50, "gain": 0.55, "offset": -30},
        "over": {"gamma": 1.80, "gain": 1.45, "offset": 30},
    },
    # Level 4: Heavy
    4: {
        "under": {"gamma": 0.35, "gain": 0.40, "offset": -45},
        "over": {"gamma": 2.20, "gain": 1.65, "offset": 45},
    },
    # Level 5: Severe
    5: {
        "under": {"gamma": 0.20, "gain": 0.25, "offset": -60},
        "over": {"gamma": 2.80, "gain": 1.90, "offset": 60},
    },
}


def _ensure_bgr_uint8(img_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Helper to ensure input is a uint8 BGR numpy array."""
    if isinstance(img_input, (str, Path)):
        img = cv2.imread(str(img_input))
        if img is None:
            raise FileNotFoundError(f"Could not load image from {img_input}")
        return img
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    elif isinstance(img_input, Image.Image):
        # PIL RGB to OpenCV BGR
        rgb = np.array(img_input)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported image type: {type(img_input)}")


def apply_blur(
    img_input: Union[str, Path, np.ndarray],
    severity: int = 1,
    kernel_type: Literal["isotropic", "anisotropic", "motion", "random"] = "random",
    split: Literal["train", "eval"] = "eval",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Applies blur degradation using disjoint train/eval kernels.

    Args:
        img_input: image path or array
        severity: 1 (mild) to 5 (severe)
        kernel_type: 'isotropic', 'anisotropic', 'motion', or 'random'
        split: 'train' or 'eval'
        seed: optional random seed

    Returns:
        degraded_bgr: uint8 BGR array
        metadata: dict containing degradation parameters
    """
    img = _ensure_bgr_uint8(img_input)
    kernel, meta = get_disjoint_blur_kernel(
        severity=severity, kernel_type=kernel_type, split=split, seed=seed
    )
    # Apply convolution
    degraded = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    return degraded, meta


def apply_jpeg_compression(
    img_input: Union[str, Path, np.ndarray],
    severity: int = 1,
    split: Literal["train", "eval"] = "eval",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Applies JPEG compression artifact degradation at calibrated quality levels.

    Args:
        img_input: image path or array
        severity: 1 (mild) to 5 (severe)
        split: 'train' or 'eval'
        seed: optional random seed

    Returns:
        degraded_bgr: uint8 BGR array
        metadata: dict containing degradation parameters
    """
    img = _ensure_bgr_uint8(img_input)
    if severity not in JPEG_QUALITY_CONFIGS:
        raise ValueError(f"Severity must be 1..5, got {severity}")

    cfg = JPEG_QUALITY_CONFIGS[severity]
    rng = np.random.RandomState(seed)

    if split == "train":
        q_low, q_high = cfg["q_range_train"]
        quality = int(rng.randint(q_low, q_high + 1))
    else:
        quality = int(cfg["q_eval"])

    # Re-encode in memory with OpenCV
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encimg = cv2.imencode(".jpg", img, encode_param)
    if not success:
        raise RuntimeError("Failed to encode image to JPEG format.")
    degraded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    meta = {
        "axis": "jpeg",
        "severity": severity,
        "split": split,
        "jpeg_quality": quality,
    }
    return degraded, meta


def apply_illumination_shift(
    img_input: Union[str, Path, np.ndarray],
    severity: int = 1,
    mode: Literal["under", "over", "random"] = "random",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Applies illumination degradation via non-linear gamma correction and linear gain/offset shift.

    Args:
        img_input: image path or array
        severity: 1 (mild) to 5 (severe)
        mode: 'under' (underexposure), 'over' (overexposure), or 'random'
        seed: optional random seed

    Returns:
        degraded_bgr: uint8 BGR array
        metadata: dict containing degradation parameters
    """
    img = _ensure_bgr_uint8(img_input)
    if severity not in ILLUMINATION_CONFIGS:
        raise ValueError(f"Severity must be 1..5, got {severity}")

    cfg = ILLUMINATION_CONFIGS[severity]
    rng = np.random.RandomState(seed)

    if mode == "random":
        chosen_mode = rng.choice(["under", "over"])
    else:
        chosen_mode = mode

    params = cfg[chosen_mode]
    gamma = params["gamma"]
    gain = params["gain"]
    offset = params["offset"]

    # 1. Non-linear Gamma adjustment via lookup table
    inv_gamma = 1.0 / max(1e-5, gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(img, table)

    # 2. Linear gain & offset shift
    shifted = gamma_corrected.astype(np.float32) * gain + offset
    degraded = np.clip(shifted, 0, 255).astype(np.uint8)

    meta = {
        "axis": "illumination",
        "severity": severity,
        "illumination_mode": chosen_mode,
        "gamma": gamma,
        "gain": gain,
        "offset": offset,
    }
    return degraded, meta


def apply_degradation(
    img_input: Union[str, Path, np.ndarray],
    axis: Literal["blur", "jpeg", "illumination"] = "blur",
    severity: int = 1,
    split: Literal["train", "eval"] = "eval",
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    """
    Unified function to apply any degradation axis at specified severity level (1..5).
    """
    if axis == "blur":
        return apply_blur(img_input, severity=severity, split=split, seed=seed, **kwargs)
    elif axis == "jpeg":
        return apply_jpeg_compression(img_input, severity=severity, split=split, seed=seed)
    elif axis == "illumination":
        return apply_illumination_shift(img_input, severity=severity, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown degradation axis: '{axis}'. Must be blur, jpeg, or illumination.")


class DegradationSimulator:
    """
    Class interface to manage dataset-level and pair-level optical degradation.
    """

    def __init__(self, split: Literal["train", "eval"] = "eval", seed: Optional[int] = None):
        self.split = split
        self.seed = seed

    def apply(
        self,
        img_input: Union[str, Path, np.ndarray],
        axis: Literal["blur", "jpeg", "illumination"],
        severity: int,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """Applies a single degradation condition."""
        return apply_degradation(
            img_input, axis=axis, severity=severity, split=self.split, seed=self.seed, **kwargs
        )

    def apply_all_axes_and_levels(
        self, img_input: Union[str, Path, np.ndarray]
    ) -> Dict[Tuple[str, int], Tuple[np.ndarray, Dict]]:
        """
        Generates all 15 degraded variants (3 axes x 5 severity levels) for evaluation sweeps.

        Returns:
            dict mapping (axis, severity) -> (degraded_img, metadata)
        """
        results = {}
        for axis in ["blur", "jpeg", "illumination"]:
            for sev in range(1, 6):
                degraded, meta = self.apply(img_input, axis=axis, severity=sev)
                results[(axis, sev)] = (degraded, meta)
        return results
