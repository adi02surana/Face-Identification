"""
models package

Provides:
- ResNet50Backbone, IResNet50Backbone, get_backbone
- ChannelAttention, SpatialAttention, CBAM
- RegionalConvBranch, extract_lower_face_crop
- DualPathFaceNet
"""

from models.backbone import IResNet50Backbone, ResNet50Backbone, get_backbone
from models.cbam import CBAM, ChannelAttention, SpatialAttention
from models.dual_path_net import DualPathFaceNet
from models.regional_branch import RegionalConvBranch, extract_lower_face_crop

__all__ = [
    "ResNet50Backbone",
    "IResNet50Backbone",
    "get_backbone",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "RegionalConvBranch",
    "extract_lower_face_crop",
    "DualPathFaceNet",
]

