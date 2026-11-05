"""
config.py

Centralized configuration and path management for:
"Robust Face Verification for Highly Similar Identities under Optical Degradation"
LNMIIT Jaipur, Dept. of Communication and Computer Engineering
"""

import os
from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset raw roots
DATA_DIR = PROJECT_ROOT / "data"
CTTS80_ROOT = DATA_DIR / "ctts80"
FIW_PAIRS_DIR = DATA_DIR / "fiw_pairs_output"
DIGIFACE_ROOT = DATA_DIR / "digiface1m"

# FIW Raw Image Root (external disk location)
# Supports 'D:/Public', 'D:/public', or custom override via environment variable
FIW_RAW_ROOT_DEFAULT = Path("D:/Public") if Path("D:/Public").exists() else Path("D:/public")
FIW_RAW_ROOT = Path(os.getenv("FIW_RAW_ROOT", str(FIW_RAW_ROOT_DEFAULT)))

# Processed & Preprocessed output directories
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_CTTS80_DIR = PROCESSED_DATA_DIR / "ctts80_112x112"
PROCESSED_FIW_DIR = PROCESSED_DATA_DIR / "fiw_112x112"
PROCESSED_DIGIFACE_DIR = PROCESSED_DATA_DIR / "digiface_112x112"

# Degraded output directories
DEGRADED_DATA_DIR = DATA_DIR / "degraded"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure essential output directories exist
for p in [PROCESSED_DATA_DIR, DEGRADED_DATA_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Reference 5-point face landmarks for standard 112x112 ArcFace alignment
# (Deng et al., ArcFace / RetinaFace standard template)
ARCFACE_REFERENCE_LANDMARKS_112x112 = [
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
]


def resolve_fiw_path(path_str: str) -> Path:
    """
    Resolves an image path from FIW metadata/CSV files to the actual file location.
    
    Handles:
      1. Processed 112x112 directory (highest priority for training & evaluation)
      2. Paths starting with 'D:/public', 'D:/Public', or 'D:\\public'
      3. Relative paths like 'FIDs/F0841/MID2/P08881_face2.jpg'
      4. Windows vs Linux POSIX slash differences
    """
    if not path_str:
        return None
    
    cleaned = str(path_str).strip().replace("\\", "/")
    
    # Check direct path first
    direct_p = Path(cleaned)
    if direct_p.exists():
        return direct_p
    
    # If path contains 'FIDs/', extract suffix starting from FIDs
    idx = cleaned.find("FIDs/")
    if idx != -1:
        rel_subpath = cleaned[idx:]
        
        # Priority 1: Check in PROCESSED_FIW_DIR (for training/evaluation on aligned faces)
        proc_candidate = PROCESSED_FIW_DIR / rel_subpath
        if proc_candidate.exists():
            return proc_candidate

        # Priority 1b: Check nested processed directory if present
        proc_nested = PROCESSED_FIW_DIR / "fiw_112x112" / rel_subpath
        if proc_nested.exists():
            return proc_nested

        # Priority 2: Check relative to FIW_RAW_ROOT (raw images)
        raw_candidate = FIW_RAW_ROOT / rel_subpath
        if raw_candidate.exists():
            return raw_candidate

        # Priority 3: Alternative roots (local data/fiw or data/processed)
        for root_candidate in [DATA_DIR / "fiw", PROCESSED_DATA_DIR / "fiw"]:
            candidate = root_candidate / rel_subpath
            if candidate.exists():
                return candidate
    
    # Fallback to direct candidate or default
    return direct_p
