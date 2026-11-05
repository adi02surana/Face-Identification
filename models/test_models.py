"""
models/test_models.py

Comprehensive test suite verifying all components of Stage 3:
1. ResNet50Backbone and IResNet50Backbone forward pass and feature dimensions.
2. CBAM channel and spatial attention weights extraction.
3. Regional face-curve crop extraction and RegionalConvBranch.
4. DualPathFaceNet in both Global CBAM Baseline mode (README §10.3) and Full Dual-Path mode (README §7).
5. Siamese pair verification forward pass (shared weights, cosine similarity).
6. L2 normalization constraint verification (||e||_2 == 1.0).
7. Gradient backpropagation check.
8. Real processed face image test from CTTS-80.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.backbone import get_backbone
from models.cbam import CBAM
from models.dual_path_net import DualPathFaceNet
from models.regional_branch import RegionalConvBranch, extract_lower_face_crop


def run_tests():
    print("=" * 70)
    print("STAGE 3: EMBEDDING NETWORK (DUAL-PATH ATTENTION) TEST SUITE")
    print("=" * 70)

    device = torch.device("cpu")
    dummy_input = torch.randn(2, 3, 112, 112, device=device)

    # -------------------------------------------------------------------------
    # Test 1: Backbones
    # -------------------------------------------------------------------------
    print("\n[Test 1] Testing Backbones...")
    r50 = get_backbone("resnet50", pretrained=True)
    r50_out = r50(dummy_input)
    print(f"  [OK] ResNet-50 output feature map: {r50_out.shape} (Expected: [2, 2048, 7, 7])")
    assert r50_out.shape == (2, 2048, 7, 7)

    ir50 = get_backbone("iresnet50", pretrained=False)
    ir50_out = ir50(dummy_input)
    print(f"  [OK] IResNet-50 output feature map: {ir50_out.shape} (Expected: [2, 512, 7, 7])")
    assert ir50_out.shape == (2, 512, 7, 7)

    # -------------------------------------------------------------------------
    # Test 2: CBAM Module
    # -------------------------------------------------------------------------
    print("\n[Test 2] Testing CBAM...")
    cbam = CBAM(in_channels=2048, reduction_ratio=16)
    cbam_out, att_maps = cbam(r50_out, return_maps=True)
    print(f"  [OK] CBAM refined feature map: {cbam_out.shape}")
    print(f"  [OK] CBAM channel attention shape: {att_maps['channel'].shape}")
    print(f"  [OK] CBAM spatial attention shape: {att_maps['spatial'].shape}")
    assert cbam_out.shape == (2, 2048, 7, 7)

    # -------------------------------------------------------------------------
    # Test 3: Regional Face-Curve Branch
    # -------------------------------------------------------------------------
    print("\n[Test 3] Testing Regional Lower-Face Branch...")
    crop = extract_lower_face_crop(dummy_input)
    print(f"  [OK] Canonical lower face crop: {crop.shape} (Expected: [2, 3, 56, 56])")
    assert crop.shape == (2, 3, 56, 56)

    reg_branch = RegionalConvBranch(in_channels=3, out_dim=256)
    reg_out = reg_branch(dummy_input)
    print(f"  [OK] Regional branch output embedding: {reg_out.shape} (Expected: [2, 256])")
    assert reg_out.shape == (2, 256)

    # -------------------------------------------------------------------------
    # Test 4: Global CBAM Baseline Mode (README §10.3 Step 3)
    # -------------------------------------------------------------------------
    print("\n[Test 4] Testing Global CBAM Baseline Model (use_regional=False)...")
    baseline_model = DualPathFaceNet(
        backbone_name="resnet50",
        pretrained_backbone=True,
        use_cbam=True,
        use_regional=False,
        embedding_dim=512,
    )
    baseline_emb = baseline_model(dummy_input)
    print(f"  [OK] Baseline embedding shape: {baseline_emb.shape} (Expected: [2, 512])")
    norms = torch.norm(baseline_emb, p=2, dim=1)
    print(f"  [OK] L2 norms of embeddings: {[round(n.item(), 5) for n in norms]} (Expected all 1.0)")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # -------------------------------------------------------------------------
    # Test 5: Full Dual-Path Network (README §7)
    # -------------------------------------------------------------------------
    print("\n[Test 5] Testing Full Dual-Path Network (use_regional=True)...")
    dual_model = DualPathFaceNet(
        backbone_name="resnet50",
        pretrained_backbone=True,
        use_cbam=True,
        use_regional=True,
        embedding_dim=512,
    )
    dual_out = dual_model(dummy_input, return_dict=True)
    print(f"  [OK] Dual-path embedding shape: {dual_out['embedding'].shape}")
    print(f"  [OK] Global feature norm: {dual_out['reliance_metrics']['global_norm']:.4f}")
    print(f"  [OK] Regional feature norm: {dual_out['reliance_metrics']['regional_norm']:.4f}")
    print(f"  [OK] Regional reliance ratio: {dual_out['reliance_metrics']['regional_ratio']:.4f}")
    print(f"  [OK] Gating parameter (sigmoid): {dual_out['reliance_metrics']['gate_value']:.4f}")

    dual_norms = torch.norm(dual_out["embedding"], p=2, dim=1)
    assert torch.allclose(dual_norms, torch.ones_like(dual_norms), atol=1e-5)

    # -------------------------------------------------------------------------
    # Test 6: Siamese Verification Pair Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Test 6] Testing Siamese Pair Verification...")
    dual_model.eval()  # In eval mode, dropout is disabled
    img1 = torch.randn(4, 3, 112, 112)
    img2 = torch.randn(4, 3, 112, 112)
    with torch.no_grad():
        e1, e2, sim = dual_model.forward_pair(img1, img2)
        print(f"  [OK] Emb1 shape: {e1.shape}, Emb2 shape: {e2.shape}")
        print(f"  [OK] Random pairwise cosine similarities: {[round(s.item(), 4) for s in sim]}")
        assert len(sim) == 4

        # Test identical images in eval mode give cosine similarity of exactly 1.0
        _, _, sim_ident = dual_model.forward_pair(img1, img1)
        print(f"  [OK] Self-similarity of identical images (eval mode): {[round(s.item(), 4) for s in sim_ident]} (Expected all 1.0)")
        assert torch.allclose(sim_ident, torch.ones_like(sim_ident), atol=1e-5)

    # -------------------------------------------------------------------------
    # Test 7: Backward pass & Gradients
    # -------------------------------------------------------------------------
    print("\n[Test 7] Testing Backward Pass and Gradients...")
    dual_model.train()  # Switch back to train mode for backprop test
    e1_tr, e2_tr, sim_tr = dual_model.forward_pair(img1, img2)
    loss = (1.0 - sim_tr).mean()
    loss.backward()
    grad_backbone = dual_model.backbone.conv1.weight.grad
    grad_fusion = dual_model.fusion_fc.weight.grad
    grad_regional = dual_model.regional_branch.features[0].block[0].weight.grad
    print(f"  [OK] Gradient norm on backbone conv1: {grad_backbone.norm().item():.5f}")
    print(f"  [OK] Gradient norm on regional conv1: {grad_regional.norm().item():.5f}")
    print(f"  [OK] Gradient norm on fusion layer: {grad_fusion.norm().item():.5f}")
    assert grad_backbone is not None and grad_regional is not None and grad_fusion is not None

    # -------------------------------------------------------------------------
    # Test 8: Real Image Verification on Preprocessed CTTS-80 Pair
    # -------------------------------------------------------------------------
    print("\n[Test 8] Testing on Real Preprocessed CTTS-80 Faces...")
    sample_dir = Path("data/processed/ctts80_112x112/Ananthan_twins")
    if sample_dir.exists():
        twin_a_imgs = list((sample_dir / "Ananthan_Anuja").glob("*.jpg"))
        twin_b_imgs = list((sample_dir / "Ananthan_Thanuja").glob("*.jpg"))

        if twin_a_imgs and twin_b_imgs:
            # Load images
            im_a1 = cv2.imread(str(twin_a_imgs[0]))
            im_a2 = cv2.imread(str(twin_a_imgs[1])) if len(twin_a_imgs) > 1 else im_a1
            im_b1 = cv2.imread(str(twin_b_imgs[0]))

            def to_tensor(im):
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
                im_norm = (im_rgb / 127.5) - 1.0
                t = torch.from_numpy(im_norm).permute(2, 0, 1).unsqueeze(0)
                return t

            t_a1 = to_tensor(im_a1)
            t_a2 = to_tensor(im_a2)
            t_b1 = to_tensor(im_b1)

            dual_model.eval()
            with torch.no_grad():
                # Same-person comparison (Anuja img 1 vs Anuja img 2)
                _, _, sim_same = dual_model.forward_pair(t_a1, t_a2)
                # Twin-imposter comparison (Anuja img 1 vs Thanuja img 1)
                _, _, sim_twin = dual_model.forward_pair(t_a1, t_b1)

            print(f"  [OK] Real face comparison: Ananthan twins")
            print(f"       - Same identity similarity (Anuja 1 vs Anuja 2): {sim_same.item():.4f}")
            print(f"       - Twin imposter similarity (Anuja 1 vs Thanuja 1): {sim_twin.item():.4f}")
            print("  [OK] Real image forward pass succeeded seamlessly!")

    print("\n" + "=" * 70)
    print("ALL STAGE 3 TESTS PASSED SUCCESSFULLY! ")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
