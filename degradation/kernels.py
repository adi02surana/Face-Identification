"""
degradation/kernels.py

Blur kernel generation following Yasarla & Patel's kernel-convolution recipe.
Enforces disjoint kernel parameter sets between training and evaluation splits
to prevent optimistic evaluation leakage (as specified in README §6).
"""

import math
from typing import Dict, Literal, Tuple

import cv2
import numpy as np


def generate_isotropic_gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """Generates a 2D isotropic Gaussian blur kernel normalized to sum to 1.0."""
    if ksize % 2 == 0:
        ksize += 1
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d.astype(np.float32)


def generate_anisotropic_gaussian_kernel(
    ksize: int, sigma_x: float, sigma_y: float, theta_deg: float = 0.0
) -> np.ndarray:
    """
    Generates a 2D anisotropic (directional) Gaussian kernel rotated by theta_deg.
    """
    if ksize % 2 == 0:
        ksize += 1

    center = (ksize - 1) / 2.0
    y, x = np.mgrid[-center : center + 1, -center : center + 1]

    # Rotate coordinates by theta
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    x_rot = cos_t * x + sin_t * y
    y_rot = -sin_t * x + cos_t * y

    # Compute Gaussian
    kernel = np.exp(-0.5 * ((x_rot**2) / (sigma_x**2 + 1e-8) + (y_rot**2) / (sigma_y**2 + 1e-8)))
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel.astype(np.float32)


def generate_motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    """
    Generates a linear motion blur kernel of specified length and angle in degrees.
    """
    length = max(3, int(length))
    ksize = length if length % 2 != 0 else length + 1
    center = (ksize - 1) // 2

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    for i in range(-center, center + 1):
        x = int(round(center + i * dx))
        y = int(round(center + i * dy))
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0

    ksum = np.sum(kernel)
    if ksum > 0:
        kernel /= ksum
    else:
        kernel[center, center] = 1.0
    return kernel.astype(np.float32)


# 5-Level Severity Calibrations with Disjoint Train/Eval Parameter Sets
# Train and Eval use disjoint angles, aspect ratios, and sigma ranges.
BLUR_LEVEL_CONFIGS = {
    # Level 1: Mild
    1: {
        "ksize": 5,
        "iso_sigma_range_train": (0.8, 1.2),
        "iso_sigma_range_eval": (1.21, 1.5),
        "aniso_sigma_train": ((1.2, 0.6), (0.6, 1.2)),
        "aniso_sigma_eval": ((1.4, 0.7), (0.7, 1.4)),
        "motion_length_train": 3,
        "motion_length_eval": 4,
        "train_angles": [0.0, 90.0],
        "eval_angles": [45.0, 135.0],
    },
    # Level 2: Low-Medium
    2: {
        "ksize": 7,
        "iso_sigma_range_train": (1.6, 2.2),
        "iso_sigma_range_eval": (2.21, 2.8),
        "aniso_sigma_train": ((2.2, 0.9), (0.9, 2.2)),
        "aniso_sigma_eval": ((2.6, 1.1), (1.1, 2.6)),
        "motion_length_train": 5,
        "motion_length_eval": 6,
        "train_angles": [15.0, 105.0],
        "eval_angles": [60.0, 150.0],
    },
    # Level 3: Medium
    3: {
        "ksize": 11,
        "iso_sigma_range_train": (2.9, 3.8),
        "iso_sigma_range_eval": (3.81, 4.7),
        "aniso_sigma_train": ((3.8, 1.4), (1.4, 3.8)),
        "aniso_sigma_eval": ((4.4, 1.6), (1.6, 4.4)),
        "motion_length_train": 7,
        "motion_length_eval": 9,
        "train_angles": [30.0, 120.0],
        "eval_angles": [75.0, 165.0],
    },
    # Level 4: Heavy
    4: {
        "ksize": 15,
        "iso_sigma_range_train": (4.8, 6.0),
        "iso_sigma_range_eval": (6.01, 7.2),
        "aniso_sigma_train": ((5.8, 1.9), (1.9, 5.8)),
        "aniso_sigma_eval": ((6.8, 2.2), (2.2, 6.8)),
        "motion_length_train": 11,
        "motion_length_eval": 13,
        "train_angles": [45.0, 135.0],
        "eval_angles": [15.0, 105.0],
    },
    # Level 5: Severe
    5: {
        "ksize": 21,
        "iso_sigma_range_train": (7.3, 9.0),
        "iso_sigma_range_eval": (9.01, 11.0),
        "aniso_sigma_train": ((8.5, 2.5), (2.5, 8.5)),
        "aniso_sigma_eval": ((10.0, 2.8), (2.8, 10.0)),
        "motion_length_train": 15,
        "motion_length_eval": 19,
        "train_angles": [0.0, 60.0, 120.0],
        "eval_angles": [30.0, 90.0, 150.0],
    },
}


def get_disjoint_blur_kernel(
    severity: int = 1,
    kernel_type: Literal["isotropic", "anisotropic", "motion", "random"] = "random",
    split: Literal["train", "eval"] = "eval",
    seed: int = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Returns a normalized blur kernel adhering to the disjoint train/eval kernel protocol.

    Args:
        severity: int from 1 (mild) to 5 (severe)
        kernel_type: 'isotropic', 'anisotropic', 'motion', or 'random'
        split: 'train' or 'eval' (uses mutually disjoint kernel parameters)
        seed: optional random seed for reproducibility

    Returns:
        kernel: 2D np.ndarray float32 normalized kernel
        params: dict of parameters used
    """
    if severity not in BLUR_LEVEL_CONFIGS:
        raise ValueError(f"Severity level must be 1..5, got {severity}")

    cfg = BLUR_LEVEL_CONFIGS[severity]
    rng = np.random.RandomState(seed)

    if kernel_type == "random":
        chosen_type = rng.choice(["isotropic", "anisotropic", "motion"])
    else:
        chosen_type = kernel_type

    ksize = cfg["ksize"]
    meta = {
        "axis": "blur",
        "severity": severity,
        "split": split,
        "kernel_type": chosen_type,
        "ksize": ksize,
    }

    if chosen_type == "isotropic":
        sigma_range = cfg[f"iso_sigma_range_{split}"]
        sigma = float(rng.uniform(sigma_range[0], sigma_range[1]))
        kernel = generate_isotropic_gaussian_kernel(ksize, sigma)
        meta["sigma"] = round(sigma, 3)

    elif chosen_type == "anisotropic":
        sigmas = cfg[f"aniso_sigma_{split}"]
        idx = rng.randint(0, len(sigmas))
        sx, sy = sigmas[idx]
        angles = cfg[f"{split}_angles"]
        theta = float(rng.choice(angles))
        kernel = generate_anisotropic_gaussian_kernel(ksize, sx, sy, theta)
        meta["sigma_x"] = sx
        meta["sigma_y"] = sy
        meta["theta_deg"] = theta

    elif chosen_type == "motion":
        length = cfg[f"motion_length_{split}"]
        angles = cfg[f"{split}_angles"]
        theta = float(rng.choice(angles))
        kernel = generate_motion_blur_kernel(length, theta)
        meta["motion_length"] = length
        meta["theta_deg"] = theta

    else:
        raise ValueError(f"Unknown kernel type: {chosen_type}")

    return kernel, meta
