"""
test_dgx_setup.py

Quick sanity verification script to run on the DGX server after setup:
1. Checks CUDA GPU device detection and memory
2. Validates all processed dataset paths and manifests
3. Runs a forward pass through DualPathFaceNet on the GPU
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROCESSED_CTTS80_DIR,
    PROCESSED_FIW_DIR,
    PROCESSED_DIGIFACE_DIR,
    resolve_fiw_path,
)

print("=" * 60)
print("DGX SERVER ENVIRONMENT & PATH SANITY CHECK")
print("=" * 60)

# 1. GPU Check
print(f"Python version : {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available : {cuda_avail}")

if cuda_avail:
    device_cnt = torch.cuda.device_count()
    print(f"GPU Count      : {device_cnt}")
    for i in range(device_cnt):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU [{i}]: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GB")
    device = torch.device("cuda:0")
else:
    print("WARNING: CUDA is not available! Training will run on CPU.")
    device = torch.device("cpu")

# 2. Dataset Paths Check
print("\nChecking Processed Dataset Paths:")
for name, p in [
    ("CTTS-80", PROCESSED_CTTS80_DIR),
    ("FIW", PROCESSED_FIW_DIR),
    ("DigiFace-1M", PROCESSED_DIGIFACE_DIR),
]:
    if p.exists():
        sub_items = list(p.iterdir())
        print(f"  [OK] {name:12s} found at {p} ({len(sub_items)} items)")
    else:
        print(f"  [MISSING] {name:12s} NOT found at {p}")

# 3. Splits Manifests Check
print("\nChecking Split Manifests:")
manifests_dir = PROJECT_ROOT / "splits" / "manifests"
for m in ["ctts80_kfold_splits.json", "fiw_disjoint_splits.json", "digiface_disjoint_splits.json"]:
    mp = manifests_dir / m
    if mp.exists():
        print(f"  [OK] {m} ({mp.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  [MISSING] {m} NOT found")

# 4. FIW Path Resolver Check
print("\nChecking FIW Path Resolution:")
sample_fiw_manifest_path = "D:/public\\FIDs\\F0010\\MID1\\P00103_face2.jpg"
resolved = resolve_fiw_path(sample_fiw_manifest_path)
print(f"  Sample manifest path: {sample_fiw_manifest_path}")
print(f"  Resolved on server  : {resolved}")
print(f"  File exists on disk : {resolved.exists() if resolved else False}")

# 5. DualPathFaceNet GPU Forward Pass Test
print(f"\nTesting DualPathFaceNet forward pass on {device}...")
try:
    from models.dual_path_net import DualPathFaceNet
    model = DualPathFaceNet(backbone_name="resnet50", pretrained_backbone=False, use_regional=True).to(device)
    dummy_x = torch.randn(2, 3, 112, 112, device=device)
    with torch.no_grad():
        emb = model(dummy_x)
    print(f"  [OK] Forward pass successful! Output embedding shape: {emb.shape}")
    print(f"  [OK] L2 Norm check: {torch.norm(emb, p=2, dim=1).cpu().tolist()}")
except Exception as e:
    print(f"  [FAIL] Model forward pass error: {e}")

print("=" * 60)
print("SANITY CHECK COMPLETE!")
print("=" * 60)

