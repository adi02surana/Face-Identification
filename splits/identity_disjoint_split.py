"""
splits/identity_disjoint_split.py

Identity-Disjoint Train/Val/Test Split and K-Fold Cross-Validation Protocol.
Strictly enforces zero identity leakage across splits as mandated in README §2 & §9:

1. CTTS-80 K-Fold Split:
   - Partitions 80 twin pairs into K folds (default K=10, 8 pairs/fold, or K=5).
   - If a twin pair is held out in test, neither twin (nor any of their pairs)
     appears in training folds.
   - Generates same-person, twin-imposter, and general-imposter verification pairs.

2. FIW Family-Level Split:
   - Partitions by Family ID (FID). All members and pairs of a family belong
     strictly to one split (train, val, or test).

3. DigiFace-1M Split:
   - Partitions synthetic subjects into disjoint train/val subsets for Phase 1.
"""

from collections import defaultdict
import json
from pathlib import Path
import random
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    CTTS80_ROOT,
    DATA_DIR,
    FIW_PAIRS_DIR,
    LOGS_DIR,
    PROCESSED_CTTS80_DIR,
    PROCESSED_DIGIFACE_DIR,
    PROCESSED_FIW_DIR,
)

MANIFESTS_DIR = PROJECT_DIR / "splits" / "manifests"
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. CTTS-80 K-Fold Cross-Validation Split Generator
# -----------------------------------------------------------------------------

def generate_ctts80_kfold_splits(
    data_dir: Path = PROCESSED_CTTS80_DIR,
    n_splits: int = 10,
    seed: int = 42,
    output_path: Path = MANIFESTS_DIR / "ctts80_kfold_splits.json",
) -> Dict:
    """
    Partitions CTTS-80 twin pairs into K disjoint folds.
    
    Returns a dict with:
      - n_splits: K
      - twin_pairs: List of all 80 twin pair folder names
      - folds: List of K fold dicts, each containing:
          - fold_idx: int
          - train_pairs: List of twin pair names
          - test_pairs: List of held-out twin pair names
          - test_verification_pairs: Dict with 'same_person', 'twin_imposter', 'general_imposter'
    """
    if not data_dir.exists():
        data_dir = CTTS80_ROOT

    # Each directory in data_dir is a twin pair (e.g. 'Ananthan_twins')
    pair_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    total_pairs = len(pair_dirs)
    if total_pairs == 0:
        raise FileNotFoundError(f"No twin pair directories found in {data_dir}")

    rng = random.Random(seed)
    shuffled_pairs = pair_dirs.copy()
    rng.shuffle(shuffled_pairs)

    # Distribute pairs into K folds
    fold_size = total_pairs // n_splits
    remainder = total_pairs % n_splits

    fold_pair_allocations = []
    idx = 0
    for f in range(n_splits):
        curr_size = fold_size + (1 if f < remainder else 0)
        fold_pair_allocations.append(shuffled_pairs[idx : idx + curr_size])
        idx += curr_size

    # Build detailed metadata per fold
    folds = []
    for f_idx in range(n_splits):
        test_pairs = fold_pair_allocations[f_idx]
        train_pairs = [p for p in pair_dirs if p not in test_pairs]

        # Verify disjointness
        assert set(train_pairs).isdisjoint(set(test_pairs)), "Identity leakage detected in CTTS-80 fold!"

        # Build verification pairs for this test fold
        test_verif = _generate_ctts80_test_pairs(data_dir, test_pairs, rng)

        folds.append({
            "fold_idx": f_idx,
            "train_twin_pairs": train_pairs,
            "test_twin_pairs": test_pairs,
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
            "verification_counts": {
                "same_person": len(test_verif["same_person"]),
                "twin_imposter": len(test_verif["twin_imposter"]),
                "general_imposter": len(test_verif["general_imposter"]),
            },
            "verification_pairs": test_verif,
        })

    manifest = {
        "dataset": "CTTS-80",
        "total_twin_pairs": total_pairs,
        "n_splits": n_splits,
        "seed": seed,
        "data_dir": str(data_dir.resolve()),
        "folds": folds,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {n_splits}-fold identity-disjoint splits for CTTS-80 ({total_pairs} pairs).")
    print(f"  -> Saved manifest to {output_path}")
    return manifest


def _generate_ctts80_test_pairs(
    data_dir: Path, test_pairs: List[str], rng: random.Random
) -> Dict[str, List[Tuple[str, str]]]:
    """Generates same-person, twin-imposter, and general-imposter pairs for held-out twin pairs."""
    same_person = []
    twin_imposter = []
    general_imposter = []

    # Map: pair_name -> {twin_A: [img_paths], twin_B: [img_paths]}
    pair_img_map = {}
    for p_name in test_pairs:
        p_dir = data_dir / p_name
        sub_dirs = sorted([d for d in p_dir.iterdir() if d.is_dir()])
        if len(sub_dirs) < 2:
            continue
        twin_a_imgs = sorted([str(p.relative_to(data_dir).as_posix()) for p in sub_dirs[0].glob("*.*") if p.is_file()])
        twin_b_imgs = sorted([str(p.relative_to(data_dir).as_posix()) for p in sub_dirs[1].glob("*.*") if p.is_file()])
        pair_img_map[p_name] = (twin_a_imgs, twin_b_imgs)

        # 1. Same-person pairs
        for imgs in [twin_a_imgs, twin_b_imgs]:
            for i in range(len(imgs)):
                for j in range(i + 1, min(i + 4, len(imgs))):  # sample pairs per person
                    same_person.append((imgs[i], imgs[j]))

        # 2. Twin-imposter pairs (Twin A vs Twin B of the same twin set)
        for i in range(min(len(twin_a_imgs), len(twin_b_imgs))):
            twin_imposter.append((twin_a_imgs[i], twin_b_imgs[i]))

    # 3. General-imposter pairs (different twin sets in the test fold)
    all_test_pair_names = list(pair_img_map.keys())
    for i, p1 in enumerate(all_test_pair_names):
        p2 = all_test_pair_names[(i + 1) % len(all_test_pair_names)]
        imgs1 = pair_img_map[p1][0]
        imgs2 = pair_img_map[p2][0]
        if imgs1 and imgs2:
            general_imposter.append((imgs1[0], imgs2[0]))

    return {
        "same_person": same_person,
        "twin_imposter": twin_imposter,
        "general_imposter": general_imposter,
    }


# -----------------------------------------------------------------------------
# 2. FIW Family-Disjoint Split Generator
# -----------------------------------------------------------------------------

def extract_fiw_fid_mid(path_str: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts (FID, MID) e.g. ('F0841', 'MID2') from path."""
    clean = str(path_str).replace("\\", "/")
    fid_match = re.search(r"F(\d{4})", clean)
    mid_match = re.search(r"MID(\d+)", clean)
    fid = f"F{fid_match.group(1)}" if fid_match else None
    mid = f"MID{mid_match.group(1)}" if mid_match else None
    return fid, mid


def generate_fiw_disjoint_splits(
    csv_dir: Path = FIW_PAIRS_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_path: Path = MANIFESTS_DIR / "fiw_disjoint_splits.json",
) -> Dict:
    """
    Partitions FIW data by Family ID (FID), ensuring no family or member
    spans across train, val, or test sets.
    """
    import pandas as pd

    all_pairs = []
    fids_set = set()

    for csv_file in csv_dir.glob("*.csv"):
        if csv_file.name == "all_pairs_combined.csv":
            continue
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                im1 = str(row["image_1"])
                im2 = str(row["image_2"])
                label = int(row.get("label", 0))
                ptype = str(row.get("pair_type", "unknown"))

                fid1, mid1 = extract_fiw_fid_mid(im1)
                fid2, mid2 = extract_fiw_fid_mid(im2)

                if fid1:
                    fids_set.add(fid1)
                if fid2:
                    fids_set.add(fid2)

                all_pairs.append({
                    "image_1": im1,
                    "image_2": im2,
                    "label": label,
                    "pair_type": ptype,
                    "fid_1": fid1,
                    "fid_2": fid2,
                })
        except Exception as e:
            print(f"Warning reading {csv_file}: {e}")

    fids = sorted(list(fids_set))
    rng = random.Random(seed)
    shuffled_fids = fids.copy()
    rng.shuffle(shuffled_fids)

    n_total = len(shuffled_fids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_fids = set(shuffled_fids[:n_train])
    val_fids = set(shuffled_fids[n_train : n_train + n_val])
    test_fids = set(shuffled_fids[n_train + n_val :])

    # Strict identity disjointness validation
    assert train_fids.isdisjoint(val_fids), "Leakage between train and val FIDs!"
    assert train_fids.isdisjoint(test_fids), "Leakage between train and test FIDs!"
    assert val_fids.isdisjoint(test_fids), "Leakage between val and test FIDs!"

    # Split pairs strictly according to FID assignment
    train_pairs, val_pairs, test_pairs = [], [], []

    for p in all_pairs:
        f1, f2 = p["fid_1"], p["fid_2"]
        # Same-person or hard negative (same family): both belong to same fid
        if f1 == f2 and f1 is not None:
            if f1 in train_fids:
                train_pairs.append(p)
            elif f1 in val_fids:
                val_pairs.append(p)
            elif f1 in test_fids:
                test_pairs.append(p)
        else:
            # Cross-family negative pair: assign to a split only if BOTH families belong to that split
            if f1 in train_fids and f2 in train_fids:
                train_pairs.append(p)
            elif f1 in val_fids and f2 in val_fids:
                val_pairs.append(p)
            elif f1 in test_fids and f2 in test_fids:
                test_pairs.append(p)

    manifest = {
        "dataset": "FIW",
        "total_fids": n_total,
        "split_counts": {
            "train_fids": len(train_fids),
            "val_fids": len(val_fids),
            "test_fids": len(test_fids),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
        },
        "train_fids": sorted(list(train_fids)),
        "val_fids": sorted(list(val_fids)),
        "test_fids": sorted(list(test_fids)),
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated identity-disjoint splits for FIW ({n_total} families):")
    print(f"  -> Train: {len(train_fids)} families ({len(train_pairs)} pairs)")
    print(f"  -> Val:   {len(val_fids)} families ({len(val_pairs)} pairs)")
    print(f"  -> Test:  {len(test_fids)} families ({len(test_pairs)} pairs)")
    print(f"  -> Saved manifest to {output_path}")
    return manifest


# -----------------------------------------------------------------------------
# 3. DigiFace-1M Disjoint Train/Val Split Generator
# -----------------------------------------------------------------------------

def generate_digiface_disjoint_splits(
    data_dir: Path = PROCESSED_DIGIFACE_DIR,
    val_ratio: float = 0.05,
    seed: int = 42,
    output_path: Path = MANIFESTS_DIR / "digiface_disjoint_splits.json",
    max_subjects: Optional[int] = None,
) -> Dict:
    """Partitions processed DigiFace-1M subjects into disjoint train/val sets."""
    subject_dirs = []
    if data_dir.exists():
        for subset in data_dir.iterdir():
            if subset.is_dir():
                for s in subset.iterdir():
                    if s.is_dir():
                        subject_dirs.append(s.relative_to(data_dir).as_posix())

    if not subject_dirs:
        # Fallback: scan raw if processed not fully populated yet
        from config import DIGIFACE_ROOT
        if DIGIFACE_ROOT.exists():
            for subset in DIGIFACE_ROOT.iterdir():
                if subset.is_dir():
                    for s in subset.iterdir():
                        if s.is_dir():
                            subject_dirs.append(s.relative_to(DIGIFACE_ROOT).as_posix())

    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]

    total_subs = len(subject_dirs)
    rng = random.Random(seed)
    shuffled = subject_dirs.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(total_subs * val_ratio))
    val_subs = set(shuffled[:n_val])
    train_subs = set(shuffled[n_val:])

    assert train_subs.isdisjoint(val_subs), "Leakage in DigiFace-1M subjects!"

    manifest = {
        "dataset": "DigiFace-1M",
        "total_subjects": total_subs,
        "n_train": len(train_subs),
        "n_val": len(val_subs),
        "train_subjects": sorted(list(train_subs)),
        "val_subjects": sorted(list(val_subs)),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated disjoint splits for DigiFace-1M: {len(train_subs)} train, {len(val_subs)} val subjects.")
    print(f"  -> Saved manifest to {output_path}")
    return manifest


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING IDENTITY-DISJOINT SPLITS (NO-LEAKAGE ENFORCEMENT)")
    print("=" * 70)

    # 1. CTTS-80 10-fold cross-validation split
    generate_ctts80_kfold_splits(n_splits=10, seed=42)

    # 2. FIW Family-disjoint split
    generate_fiw_disjoint_splits(seed=42)

    # 3. DigiFace-1M Subject-disjoint split (all subjects)
    generate_digiface_disjoint_splits(max_subjects=None, seed=42)

    print("\nAll identity-disjoint splits generated and verified with 0 leakage! [OK]")

