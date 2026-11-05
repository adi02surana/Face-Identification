"""
degradation package

Controlled optical degradation simulator across 3 axes:
- Blur (Isotropic/Anisotropic Gaussian, Motion Blur with train/eval disjoint kernels)
- JPEG Compression (5 calibrated quality bands)
- Illumination Shift (Gamma + linear gain/offset over-/under-exposure)
"""

from .kernels import (
    generate_isotropic_gaussian_kernel,
    generate_anisotropic_gaussian_kernel,
    generate_motion_blur_kernel,
    get_disjoint_blur_kernel,
)
from .simulator import (
    apply_blur,
    apply_jpeg_compression,
    apply_illumination_shift,
    apply_degradation,
    DegradationSimulator,
)

__all__ = [
    "generate_isotropic_gaussian_kernel",
    "generate_anisotropic_gaussian_kernel",
    "generate_motion_blur_kernel",
    "get_disjoint_blur_kernel",
    "apply_blur",
    "apply_jpeg_compression",
    "apply_illumination_shift",
    "apply_degradation",
    "DegradationSimulator",
]
