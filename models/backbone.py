"""
models/backbone.py

Backbone implementations for face verification:
1. ResNet50Backbone: Standard ResNet-50 adapted for 112x112 face inputs,
   initialized from pretrained weights (He et al., CVPR 2016).
2. IResNet50Backbone: Canonical ArcFace modified ResNet-50 (Deng et al., CVPR 2019)
   with PReLU activations and improved residual units for 112x112 aligned faces.

Both extract intermediate 2D feature maps for downstream CBAM attention processing.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "resnet50_pretrained.pth"


# -----------------------------------------------------------------------------
# 1. ResNet-50 Backbone (Torchvision-based with 112x112 face adaptation)
# -----------------------------------------------------------------------------

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone adapted for 112x112 facial images.
    
    When preserve_spatial_resolution=True:
      - MaxPool2d is bypassed/identity, so for 112x112 inputs:
        conv1 -> 56x56
        layer1 -> 56x56 (256 ch)
        layer2 -> 28x28 (512 ch)
        layer3 -> 14x14 (1024 ch)
        layer4 -> 7x7 (2048 ch)
      This yields the standard 7x7 spatial feature map for CBAM attention.
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: Optional[Union[str, Path]] = DEFAULT_WEIGHTS_PATH,
        preserve_spatial_resolution: bool = True,
    ):
        super().__init__()
        self.out_channels = 2048
        self.preserve_spatial_resolution = preserve_spatial_resolution

        # Instantiate standard ResNet-50 architecture
        base = models.resnet50(weights=None)

        if pretrained:
            path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
            if path.exists():
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                # Filter out any mismatched head keys if present
                base.load_state_dict(state_dict, strict=False)
            else:
                # Fallback to downloading or uninitialized
                try:
                    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                except Exception:
                    pass

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = nn.Identity() if preserve_spatial_resolution else base.maxpool

        self.layer1 = base.layer1  # 256 ch
        self.layer2 = base.layer2  # 512 ch
        self.layer3 = base.layer3  # 1024 ch
        self.layer4 = base.layer4  # 2048 ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Aligned face tensor of shape (B, 3, 112, 112)
        Returns:
            feat_map: 2D feature map of shape (B, 2048, 7, 7)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)  # (B, 2048, 7, 7)
        return feat_map


# -----------------------------------------------------------------------------
# 2. Canonical ArcFace IResNet-50 (Deng et al., CVPR 2019 / InsightFace)
# -----------------------------------------------------------------------------

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class IResNet50Backbone(nn.Module):
    """
    ArcFace IResNet-50 backbone (Deng et al., CVPR 2019).
    Input: (B, 3, 112, 112) -> Output: (B, 512, 7, 7) feature map.
    """

    def __init__(self, layers=(3, 4, 14, 3), in_channels: int = 3, out_channels: int = 512):
        super().__init__()
        self.inplanes = 64
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(IBasicBlock, 64, layers[0], stride=2)    # 56x56
        self.layer2 = self._make_layer(IBasicBlock, 128, layers[1], stride=2)   # 28x28
        self.layer3 = self._make_layer(IBasicBlock, 256, layers[2], stride=2)   # 14x14
        self.layer4 = self._make_layer(IBasicBlock, 512, layers[3], stride=2)   # 7x7

        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112)
        Returns:
            feat_map: (B, 512, 7, 7)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        return x


# -----------------------------------------------------------------------------
# Factory function
# -----------------------------------------------------------------------------

def get_backbone(
    name: str = "resnet50",
    pretrained: bool = True,
    weights_path: Optional[Union[str, Path]] = None,
) -> nn.Module:
    """
    Factory function returning the specified backbone.
    
    Args:
        name: 'resnet50' (torchvision-adapted) or 'iresnet50' (ArcFace canonical)
        pretrained: whether to load pretrained weights
        weights_path: custom weights path
    """
    name_clean = name.lower().replace("-", "").replace("_", "")
    if name_clean in ("resnet50", "r50"):
        return ResNet50Backbone(
            pretrained=pretrained,
            weights_path=weights_path or DEFAULT_WEIGHTS_PATH,
            preserve_spatial_resolution=True,
        )
    elif name_clean in ("iresnet50", "arcfaceresnet50", "ir50"):
        model = IResNet50Backbone()
        if pretrained and weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        return model
    else:
        raise ValueError(f"Unknown backbone name '{name}'. Choose 'resnet50' or 'iresnet50'.")

