"""
models/regional_branch.py

Landmark-guided face-curve / lower-jaw region extraction and small dedicated conv branch.
Reference: Mousavi et al., "Facial feature analysis for identical twin identification", 
           Multimedia Tools Appl. 2021.

Mousavi et al. identified the lower-face / mandibular jaw-curve region as the most
diagnostic biometrics for discriminating identical twins (~35% of distinguishing variance).
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical coordinates for aligned 112x112 ArcFace faces:
# - Nose tip: y ~ 71.7, x ~ 56.0
# - Mouth corners: y ~ 92.3, x ~ 41.5 & 70.7
# Lower face region begins around y=56 (just above nose tip) to y=112 (chin)
CANONICAL_LOWER_FACE_Y_RANGE = (56, 112)
CANONICAL_LOWER_FACE_X_RANGE = (12, 100)


def extract_lower_face_crop(
    images: torch.Tensor,
    landmarks: Optional[torch.Tensor] = None,
    target_size: Tuple[int, int] = (56, 56),
) -> torch.Tensor:
    """
    Extracts the lower-face/jaw-curve region from aligned 112x112 face images.
    
    Args:
        images: Batch of aligned face images of shape (B, 3, 112, 112).
        landmarks: Optional tensor of 5-point landmarks (B, 5, 2) in image coordinates.
                   Points order: [left_eye, right_eye, nose, left_mouth, right_mouth].
        target_size: (H, W) to resize the cropped regional patch into.
        
    Returns:
        crops: Tensor of shape (B, 3, target_size[0], target_size[1])
    """
    B, C, H, W = images.shape

    # If explicit landmarks are provided per sample, perform landmark-guided crop
    if landmarks is not None and landmarks.numel() > 0:
        batch_crops = []
        for i in range(B):
            lm = landmarks[i]  # (5, 2)
            # Nose tip is index 2, left mouth is 3, right mouth is 4
            nose_y = lm[2, 1].item()
            lm_x = lm[3, 0].item()
            rm_x = lm[4, 0].item()

            # Dynamic bounding box with margins
            top = max(0, int(nose_y - 8))
            bottom = H
            left = max(0, int(min(lm_x, rm_x) - 16))
            right = min(W, int(max(lm_x, rm_x) + 16))

            if right - left < 16 or bottom - top < 16:
                # Fallback to canonical crop if landmark bounding box is degenerate
                crop = images[i : i + 1, :, 56:112, 12:100]
            else:
                crop = images[i : i + 1, :, top:bottom, left:right]

            crop_resized = F.interpolate(
                crop, size=target_size, mode="bilinear", align_corners=False
            )
            batch_crops.append(crop_resized)
        return torch.cat(batch_crops, dim=0)

    # Fast vectorised canonical lower-face crop from aligned 112x112 image
    y1, y2 = CANONICAL_LOWER_FACE_Y_RANGE
    x1, x2 = CANONICAL_LOWER_FACE_X_RANGE
    crops = images[:, :, y1:y2, x1:x2]
    if (crops.shape[2], crops.shape[3]) != target_size:
        crops = F.interpolate(crops, size=target_size, mode="bilinear", align_corners=False)
    return crops


class ConvBNReLU(nn.Module):
    """Basic Conv2d + BatchNorm2d + ReLU block"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RegionalConvBranch(nn.Module):
    """
    Dedicated small convolutional branch focusing specifically on the lower-face /
    jawline / mouth geometry (Mousavi et al., 2021).
    
    Accepts 56x56 regional face crops and outputs a compact regional feature vector
    (e.g., 256-dimensional).
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim

        self.features = nn.Sequential(
            # Stage 1: 56x56 -> 28x28
            ConvBNReLU(in_channels, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 2: 28x28 -> 14x14
            ConvBNReLU(32, 64, kernel_size=3, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 3: 14x14 -> 7x7
            ConvBNReLU(64, 128, kernel_size=3, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 4: 7x7 -> 7x7 (deep features)
            ConvBNReLU(128, 256, kernel_size=3, padding=1),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.PReLU(out_dim),
        )

    def forward(
        self,
        images_or_crops: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None,
        is_already_cropped: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            images_or_crops: Either full 112x112 face images or pre-cropped 56x56 regional patches.
            landmarks: Optional 5-point landmarks for guided extraction.
            is_already_cropped: If True, skips extraction and feeds directly to conv layers.
        Returns:
            regional_emb: Regional feature embedding of shape (B, out_dim).
        """
        if is_already_cropped:
            x = images_or_crops
        else:
            x = extract_lower_face_crop(images_or_crops, landmarks=landmarks, target_size=(56, 56))

        feats = self.features(x)
        pooled = self.gap(feats).flatten(1)
        out = self.fc(pooled)
        return out

