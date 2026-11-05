"""
models/cbam.py

Convolutional Block Attention Module (CBAM) implementation.
Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

Implements sequential Channel Attention followed by Spatial Attention
to enhance feature representations across face verification representations.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).
    Compresses spatial dimensions via Adaptive Average and Max Pooling,
    passing through a shared 2-layer MLP with reduction ratio r.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = max(1, reduction_ratio)
        hidden_dim = max(8, in_channels // self.reduction_ratio)

        # Shared MLP via 1x1 convolutions for seamless 4D tensor operations
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map of shape (B, C, H, W)
        Returns:
            out: Channel-attended feature map of shape (B, C, H, W)
            channel_att: Attention weights of shape (B, C, 1, 1)
        """
        # Average pooling along spatial dimensions -> (B, C, 1, 1)
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        # Max pooling along spatial dimensions -> (B, C, 1, 1)
        max_pool = F.adaptive_max_pool2d(x, (1, 1))

        # Shared MLP forward
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att, channel_att


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).
    Compresses channel dimension via Average and Max pooling along channels,
    followed by a standard large-kernel (7x7) convolution and Sigmoid.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError(f"Kernel size must be 3 or 7, got {kernel_size}")
        padding = 3 if kernel_size == 7 else 1

        # Input has 2 channels: channel-avg and channel-max
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map of shape (B, C, H, W)
        Returns:
            out: Spatially-attended feature map of shape (B, C, H, W)
            spatial_att: Attention map of shape (B, 1, H, W)
        """
        # Average along channel axis -> (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max along channel axis -> (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        spatial_att = self.sigmoid(self.bn(self.conv(concat)))
        return x * spatial_att, spatial_att


class CBAM(nn.Module):
    """
    Complete Convolutional Block Attention Module (Woo et al., 2018).
    Applies Channel Attention followed sequentially by Spatial Attention.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_gate = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_gate = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(
        self, x: torch.Tensor, return_maps: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x: Input feature map of shape (B, C, H, W)
            return_maps: If True, returns dict with channel and spatial attention weights.
        Returns:
            out: Refined feature map of shape (B, C, H, W)
            att_maps: Optional dictionary {"channel": (B, C, 1, 1), "spatial": (B, 1, H, W)}
        """
        x_ca, ca_weights = self.channel_gate(x)
        x_sa, sa_weights = self.spatial_gate(x_ca)

        if return_maps:
            return x_sa, {"channel": ca_weights, "spatial": sa_weights}
        return x_sa, None

