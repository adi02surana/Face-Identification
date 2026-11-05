"""
models/dual_path_net.py

Dual-Path Attention Embedding Network for Face Verification.
Fuses:
1. Global Path: ResNet-50 backbone + CBAM (Channel & Spatial Attention)
2. Regional Path: Dedicated CNN branch on lower-face / jaw-curve crop (Mousavi et al., 2021)
3. Fusion Projection: Concatenation -> Linear -> BatchNorm -> 512-d L2-normalized embedding.

Supports:
- Mode "dual_path": Full dual-path architecture with fused global and regional representations.
- Mode "global_only": Baseline with Global CBAM path only (README §7 & §10.3 fallback).
- Mode "regional_only": Ablation study baseline using regional features only.
- Reliance Tracking: Logs relative activation magnitude and contribution of regional vs global paths
  under varying degradation levels (README §9 novel metric).
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import get_backbone
from models.cbam import CBAM
from models.regional_branch import RegionalConvBranch, extract_lower_face_crop


class DualPathFaceNet(nn.Module):
    """
    Dual-Path Attention Face Verification Network.
    
    Produces 512-dimensional L2-normalized embeddings for 112x112 face images.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        weights_path: Optional[str] = None,
        use_cbam: bool = True,
        use_regional: bool = True,
        cbam_reduction: int = 16,
        global_dim: int = 512,
        regional_dim: int = 256,
        embedding_dim: int = 512,
        dropout_p: float = 0.1,
    ):
        """
        Args:
            backbone_name: 'resnet50' or 'iresnet50'
            pretrained_backbone: whether to load pretrained backbone weights
            weights_path: optional path to backbone weights file
            use_cbam: whether to apply CBAM on the global feature map
            use_regional: whether to activate the regional lower-face branch
            cbam_reduction: reduction ratio for CBAM channel attention
            global_dim: dimension of pooled global feature
            regional_dim: dimension of regional conv branch feature
            embedding_dim: final L2-normalized embedding dimension (512)
            dropout_p: dropout probability before final projection
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.use_cbam = use_cbam
        self.use_regional = use_regional
        self.embedding_dim = embedding_dim
        self.global_dim = global_dim
        self.regional_dim = regional_dim

        # 1. Global Backbone
        self.backbone = get_backbone(
            name=backbone_name,
            pretrained=pretrained_backbone,
            weights_path=weights_path,
        )
        backbone_channels = getattr(self.backbone, "out_channels", 2048)

        # 2. CBAM Attention Module
        if self.use_cbam:
            self.cbam = CBAM(
                in_channels=backbone_channels,
                reduction_ratio=cbam_reduction,
                spatial_kernel_size=7,
            )
        else:
            self.cbam = None

        # Global Pooling and Projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Sequential(
            nn.Linear(backbone_channels, global_dim, bias=False),
            nn.BatchNorm1d(global_dim),
            nn.PReLU(global_dim),
        )

        # 3. Regional Path (Mousavi et al., 2021 jaw-curve branch)
        if self.use_regional:
            self.regional_branch = RegionalConvBranch(in_channels=3, out_dim=regional_dim)
            fusion_in_dim = global_dim + regional_dim
        else:
            self.regional_branch = None
            fusion_in_dim = global_dim

        # 4. Fusion and Final Projection to 512-d
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.fusion_fc = nn.Linear(fusion_in_dim, embedding_dim, bias=False)
        self.fusion_bn = nn.BatchNorm1d(embedding_dim, eps=1e-5)

        # Learnable gating parameter to quantify relative branch weighting (README §9)
        if self.use_regional:
            self.gate_param = nn.Parameter(torch.tensor([0.5]))
        else:
            self.gate_param = None

    def forward_features(
        self,
        x: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through feature extractors.
        
        Returns:
            global_feat: (B, global_dim)
            regional_feat: (B, regional_dim) or None
            att_maps: Dict of CBAM attention maps if requested
        """
        # --- Global Path ---
        feat_map = self.backbone(x)  # (B, C, 7, 7)

        att_maps = None
        if self.cbam is not None:
            feat_map, att_maps = self.cbam(feat_map, return_maps=return_attention)

        pooled_global = self.global_pool(feat_map).flatten(1)  # (B, C)
        global_feat = self.global_fc(pooled_global)             # (B, global_dim)

        # --- Regional Path ---
        regional_feat = None
        if self.use_regional and self.regional_branch is not None:
            regional_feat = self.regional_branch(x, landmarks=landmarks)  # (B, regional_dim)

        return global_feat, regional_feat, att_maps

    def forward(
        self,
        x: torch.Tensor,
        landmarks: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict]:
        """
        Computes 512-d L2-normalized face embedding.
        
        Args:
            x: Aligned face batch (B, 3, 112, 112)
            landmarks: Optional (B, 5, 2) facial landmark coordinates
            return_dict: If True, returns detailed metrics including attention maps
                         and relative reliance ratios for evaluation (README §9).
        Returns:
            embedding: (B, 512) L2-normalized embedding tensor, or dict if return_dict=True.
        """
        global_feat, regional_feat, att_maps = self.forward_features(
            x, landmarks=landmarks, return_attention=return_dict
        )

        # Fuse representations
        if self.use_regional and regional_feat is not None:
            # Gated concatenation for observable path weighting
            gate = torch.sigmoid(self.gate_param)
            gated_global = gate * global_feat
            gated_regional = (1.0 - gate) * regional_feat
            fused = torch.cat([gated_global, gated_regional], dim=1)
        else:
            fused = global_feat

        fused = self.dropout(fused)
        proj = self.fusion_bn(self.fusion_fc(fused))
        embedding = F.normalize(proj, p=2, dim=1)  # 512-d L2-normalized embedding

        if not return_dict:
            return embedding

        # Calculate relative path activation norms for degradation analysis
        g_norm = torch.norm(global_feat, p=2, dim=1).mean().item()
        r_norm = torch.norm(regional_feat, p=2, dim=1).mean().item() if regional_feat is not None else 0.0
        total_norm = g_norm + r_norm + 1e-8
        regional_ratio = r_norm / total_norm

        return {
            "embedding": embedding,
            "global_feat": global_feat,
            "regional_feat": regional_feat,
            "cbam_attention": att_maps,
            "reliance_metrics": {
                "global_norm": g_norm,
                "regional_norm": r_norm,
                "regional_ratio": regional_ratio,
                "gate_value": torch.sigmoid(self.gate_param).item() if self.gate_param is not None else 1.0,
            },
        }

    def forward_pair(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        landmarks1: Optional[torch.Tensor] = None,
        landmarks2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Siamese verification forward pass using tied weights.
        
        Args:
            img1, img2: Batches of aligned face images (B, 3, 112, 112)
        Returns:
            emb1: (B, 512) L2-normalized embeddings for img1
            emb2: (B, 512) L2-normalized embeddings for img2
            similarity: (B,) Cosine similarities in [-1, 1]
        """
        emb1 = self.forward(img1, landmarks=landmarks1)
        emb2 = self.forward(img2, landmarks=landmarks2)
        similarity = F.cosine_similarity(emb1, emb2, dim=1)
        return emb1, emb2, similarity

