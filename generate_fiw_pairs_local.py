"""
generate_fiw_pairs_local.py

Parses your local FIW download directly from disk -- no FiftyOne needed --
matching the structure you actually have:

    FIW_ROOT/
      FIDs/
        F0010/
          mid.csv
          MID1/
            P00103_face2.jpg
            ...
          MID2/
            ...
          unrelated_and_nonfaces/     (excluded automatically -- not a
                                        verified single identity)
      FIW_FIDs.csv
      FIW_PIDs_v2.csv
      FIW_RIDs.csv

A few members (like F0018/MID1 in your listing) have a nested subfolder
inside them (e.g. "n005175/") holding extra images from what looks like an
external dataset merged in. This script does NOT recurse into those --
only image files sitting directly inside a MID folder are used -- and it
prints a list of any it skipped so you can inspect them yourself later.

Produces the same CSV format as before, so apply_degradation.py from
before works on this output unchanged:
  - positive_pairs.csv        same person (same F####/MID#), label 1
  - negative_pairs.csv        different families, label 0
  - hard_negative_pairs.csv   different member, SAME family
                               (siblings/parent-child -- your closest
                               available proxy for 'highly similar
                               identity' pairs while ND-TWINS is pending)
  - all_pairs_combined.csv    everything, shuffled
"""

import os
import csv
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---------------- CONFIG ----------------
FIW_ROOT = "D:/public"
FIDS_DIR_NAME = "FIDs"
OUTPUT_DIR = "fiw_pairs_output"
NUM_POSITIVE_PAIRS = 5000
NUM_NEGATIVE_PAIRS = 5000
NUM_HARD_NEGATIVE_PAIRS = 3000
RANDOM_SEED = 42

EXCLUDE_FOLDER_NAMES = {"unrelated_and_nonfaces"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# -----------------------------------------

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_image(filename):
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def scan_fiw(fiw_root):
    """
    Walks FIDs/F####/MID#/ and builds:
      identity_to_images: {"F0010_MID1": [filepaths...]}
      family_to_members:  {"F0010": ["MID1", "MID2", ...]}
    """
    fids_dir = os.path.join(fiw_root, FIDS_DIR_NAME)
    identity_to_images = defaultdict(list)
    family_to_members = defaultdict(list)
    skipped_nested_folders = []

    for family_name in sorted(os.listdir(fids_dir)):
        family_path = os.path.join(fids_dir, family_name)
        if not os.path.isdir(family_path) or not family_name.startswith("F"):
            continue

        for member_name in sorted(os.listdir(family_path)):
            member_path = os.path.join(family_path, member_name)
            if not os.path.isdir(member_path):
                continue
            if member_name in EXCLUDE_FOLDER_NAMES or not member_name.startswith("MID"):
                continue

            images = []
            for entry in sorted(os.listdir(member_path)):
                entry_path = os.path.join(member_path, entry)
                if os.path.isdir(entry_path):
                    skipped_nested_folders.append(entry_path)
                    continue
                if is_image(entry):
                    images.append(entry_path)

            if images:
                identity_key = f"{family_name}_{member_name}"
                identity_to_images[identity_key] = images
                family_to_members[family_name].append(member_name)

    if skipped_nested_folders:
        print(f"\nNote: skipped {len(skipped_nested_folders)} nested subfolders found "
              f"inside MID directories (e.g. extra external-dataset folders like "
              f"'n005175'). Not used for pair generation. First few:")
        for p in skipped_nested_folders[:5]:
            print(f"   {p}")

    return identity_to_images, family_to_members


def try_load_relationship_csv(fiw_root):
    """
    Attempts to load FIW_RIDs.csv, which should hold the authoritative
    relationship-type pairs (e.g. specifically siblings vs parent-child).
    Column names have varied across FIW releases, so this just prints what
    it finds -- use the printed columns to add sibling-only filtering to
    build_hard_negative_pairs() below if you want that extra precision.
    """
    rid_path = os.path.join(fiw_root, "FIW_RIDs.csv")
    if not os.path.exists(rid_path):
        print(f"\nNo FIW_RIDs.csv found at {rid_path} -- hard negatives will use "
              f"'any two members of the same family' rather than siblings only.")
        return None

    df = pd.read_csv(rid_path)
    print(f"\nLoaded {rid_path} -- columns found: {list(df.columns)}")
    print(df.head(3))
    return df


def build_positive_pairs(identity_to_images, num_pairs):
    pairs = []
    identities = [k for k, v in identity_to_images.items() if len(v) >= 2]
    attempts = 0
    while len(pairs) < num_pairs and attempts < num_pairs * 20 and identities:
        identity = random.choice(identities)
        img1, img2 = random.sample(identity_to_images[identity], 2)
        pairs.append((img1, img2, 1, "same_person"))
        attempts += 1
    return pairs


def build_negative_pairs(identity_to_images, family_to_members, num_pairs):
    """Cross-family pairs -- guaranteed unrelated."""
    pairs = []
    families = list(family_to_members.keys())
    attempts = 0
    while len(pairs) < num_pairs and attempts < num_pairs * 20 and len(families) >= 2:
        fam1, fam2 = random.sample(families, 2)
        mid1 = random.choice(family_to_members[fam1])
        mid2 = random.choice(family_to_members[fam2])
        img1 = random.choice(identity_to_images[f"{fam1}_{mid1}"])
        img2 = random.choice(identity_to_images[f"{fam2}_{mid2}"])
        pairs.append((img1, img2, 0, "different_family"))
        attempts += 1
    return pairs


def build_hard_negative_pairs(identity_to_images, family_to_members, num_pairs):
    """
    Same-family, different-member pairs -- covers siblings and
    parent-child together. Your closest available proxy for 'highly
    similar identity' hard negatives while ND-TWINS access is pending.
    """
    pairs = []
    families_with_multiple = {f: m for f, m in family_to_members.items() if len(m) >= 2}
    families = list(families_with_multiple.keys())
    attempts = 0
    while len(pairs) < num_pairs and attempts < num_pairs * 20 and families:
        fam = random.choice(families)
        members = families_with_multiple[fam]
        mid1, mid2 = random.sample(members, 2)
        img1 = random.choice(identity_to_images[f"{fam}_{mid1}"])
        img2 = random.choice(identity_to_images[f"{fam}_{mid2}"])
        pairs.append((img1, img2, 0, "same_family_hard_negative"))
        attempts += 1
    return pairs


def export_pairs_csv(pairs, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_1", "image_2", "label", "pair_type"])
        writer.writerows(pairs)
    print(f"Saved {len(pairs)} pairs to {path}")


def main():
    identity_to_images, family_to_members = scan_fiw(FIW_ROOT)
    print(f"\nFound {len(identity_to_images)} identities across "
          f"{len(family_to_members)} families")

    if not identity_to_images:
        print("No identities found. Check FIW_ROOT points to the folder that "
              "directly contains 'FIDs/' (same level as FIW_RIDs.csv).")
        return

    try_load_relationship_csv(FIW_ROOT)  # informational -- see printed columns

    positive_pairs = build_positive_pairs(identity_to_images, NUM_POSITIVE_PAIRS)
    negative_pairs = build_negative_pairs(identity_to_images, family_to_members, NUM_NEGATIVE_PAIRS)
    hard_negative_pairs = build_hard_negative_pairs(identity_to_images, family_to_members, NUM_HARD_NEGATIVE_PAIRS)

    all_pairs = positive_pairs + negative_pairs + hard_negative_pairs
    random.shuffle(all_pairs)

    export_pairs_csv(positive_pairs, "positive_pairs.csv")
    export_pairs_csv(negative_pairs, "negative_pairs.csv")
    export_pairs_csv(hard_negative_pairs, "hard_negative_pairs.csv")
    export_pairs_csv(all_pairs, "all_pairs_combined.csv")

    print("\nSummary:")
    print(f"  Positive (same person) pairs:               {len(positive_pairs)}")
    print(f"  Negative (different family) pairs:          {len(negative_pairs)}")
    print(f"  Hard negative (same-family, diff member):    {len(hard_negative_pairs)}")


if __name__ == "__main__":
    main()
