"""
preprocessing/align.py

Face detection, 5-point landmark localization, and affine similarity alignment to 112x112.
Follows RetinaFace (Deng et al., CVPR 2020) and ArcFace standard 112x112 alignment protocol.

Logs detection and alignment failure rates explicitly, as specified in README §5.
"""

import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    ARCFACE_REFERENCE_LANDMARKS_112x112,
    CTTS80_ROOT,
    DIGIFACE_ROOT,
    FIW_PAIRS_DIR,
    LOGS_DIR,
    PROCESSED_CTTS80_DIR,
    PROCESSED_DIGIFACE_DIR,
    PROCESSED_FIW_DIR,
    resolve_fiw_path,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "preprocessing.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("preprocessing.align")

# Reference landmarks for 112x112 ArcFace alignment
# (viewer left eye, viewer right eye, nose tip, viewer left mouth, viewer right mouth)
REF_LANDMARKS = np.array(ARCFACE_REFERENCE_LANDMARKS_112x112, dtype=np.float32)

# Path to cached YuNet ONNX model
MODELS_DIR = PROJECT_DIR / "preprocessing" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def ensure_yunet_model() -> Path:
    """Ensures YuNet ONNX model is available locally."""
    if not YUNET_MODEL_PATH.exists():
        logger.info(f"Downloading lightweight YuNet face detector model to {YUNET_MODEL_PATH}...")
        try:
            urllib.request.urlretrieve(YUNET_URL, str(YUNET_MODEL_PATH))
            logger.info("YuNet model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download YuNet model: {e}")
    return YUNET_MODEL_PATH


def estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Estimates 2D similarity transform matrix (scale, rotation, translation)
    mapping src_pts to dst_pts.
    """
    src_pts = np.asarray(src_pts, dtype=np.float32)
    dst_pts = np.asarray(dst_pts, dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return M


class FaceAligner:
    """
    Face detector and aligner using RetinaFace / YuNet to produce
    112x112 aligned face crops matching ArcFace input standards.
    """

    def __init__(self, target_size: Tuple[int, int] = (112, 112), detector_backend: str = "auto"):
        self.target_size = target_size
        self.detector_backend = detector_backend
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Initializes RetinaFace if available, else OpenCV YuNet / DNN detector."""
        if self.detector_backend in ["auto", "retinaface", "insightface"]:
            try:
                import insightface
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                self.detector = app
                self.detector_backend = "insightface"
                logger.info("Initialized InsightFace/RetinaFace detector backend successfully.")
                return
            except Exception as e:
                logger.debug(
                    f"InsightFace detector not available ({e}). Using OpenCV YuNet face detector."
                )

        # Fallback to OpenCV YuNet (FaceDetectorYN)
        try:
            model_path = ensure_yunet_model()
            if model_path.exists() and hasattr(cv2, "FaceDetectorYN"):
                self.detector = cv2.FaceDetectorYN.create(
                    str(model_path), "", (320, 320), score_threshold=0.5, nms_threshold=0.3
                )
                self.detector_backend = "yunet"
                logger.info("Initialized OpenCV YuNet 5-point landmark detector.")
                return
        except Exception as e:
            logger.warning(f"Could not initialize YuNet detector: {e}")

        self.detector_backend = "center_crop"
        logger.warning("No deep detector initialized. Using center-crop fallback.")

    def detect_and_align(
        self, image_input: Union[str, Path, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, str]:
        """
        Detects face and aligns to 112x112.

        Returns:
            aligned_img: 112x112 uint8 BGR image
            landmarks: (5, 2) array of detected 5-point landmarks or None
            success: bool indicating if detection/alignment succeeded
            msg: status string
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                return None, None, False, f"File does not exist: {img_path}"
            img = cv2.imread(str(img_path))
            if img is None:
                return None, None, False, f"Failed to decode image: {img_path}"
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            return None, None, False, "Invalid image input type."

        h, w = img.shape[:2]

        # 1. InsightFace / RetinaFace Backend
        if self.detector_backend == "insightface" and self.detector is not None:
            try:
                faces = self.detector.get(img)
                if len(faces) > 0:
                    faces = sorted(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                        reverse=True,
                    )
                    landmarks = faces[0].kps  # 5 landmarks
                    M = estimate_similarity_transform(landmarks, REF_LANDMARKS)
                    if M is not None:
                        aligned = cv2.warpAffine(
                            img, M, self.target_size, borderValue=0.0, flags=cv2.INTER_LINEAR
                        )
                        return aligned, landmarks, True, "RetinaFace detection + alignment succeeded"
            except Exception as e:
                logger.debug(f"RetinaFace error: {e}")

        # 2. YuNet Backend (OpenCV FaceDetectorYN)
        if self.detector_backend == "yunet" and self.detector is not None:
            try:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(img)
                if faces is not None and len(faces) > 0:
                    # Pick largest face by area
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    f = faces[0]
                    # YuNet outputs: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
                    # Order in image coordinates (viewer left to right):
                    # Point 1 (viewer left eye), Point 2 (viewer right eye), Point 3 (nose), Point 4 (viewer left mouth), Point 5 (viewer right mouth)
                    eye1 = np.array([f[4], f[5]], dtype=np.float32)
                    eye2 = np.array([f[6], f[7]], dtype=np.float32)
                    if eye1[0] > eye2[0]:
                        eye1, eye2 = eye2, eye1

                    nose = np.array([f[8], f[9]], dtype=np.float32)
                    m1 = np.array([f[10], f[11]], dtype=np.float32)
                    m2 = np.array([f[12], f[13]], dtype=np.float32)
                    if m1[0] > m2[0]:
                        m1, m2 = m2, m1

                    landmarks = np.array([eye1, eye2, nose, m1, m2], dtype=np.float32)
                    M = estimate_similarity_transform(landmarks, REF_LANDMARKS)
                    if M is not None:
                        aligned = cv2.warpAffine(
                            img, M, self.target_size, borderValue=0.0, flags=cv2.INTER_LINEAR
                        )
                        return aligned, landmarks, True, "YuNet detection + 5-pt alignment succeeded"
            except Exception as e:
                logger.debug(f"YuNet detection error: {e}")

        # 3. Center Crop Fallback (Logged as alignment failure)
        min_dim = min(h, w)
        sy = (h - min_dim) // 2
        sx = (w - min_dim) // 2
        crop = img[sy : sy + min_dim, sx : sx + min_dim]
        aligned = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        return aligned, None, False, "Face detection failed: used center crop fallback"


def align_face(
    image_input: Union[str, Path, np.ndarray], target_size: Tuple[int, int] = (112, 112)
) -> Tuple[np.ndarray, bool]:
    """Convenience functional wrapper for face alignment."""
    aligner = FaceAligner(target_size=target_size)
    aligned_img, _, success, _ = aligner.detect_and_align(image_input)
    return aligned_img, success


def process_ctts80_dataset(
    ctts80_root: Path = CTTS80_ROOT,
    output_dir: Path = PROCESSED_CTTS80_DIR,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Processes all CTTS-80 images, aligns to 112x112, and explicitly logs
    the detection/alignment failure rate as required in README §5 & §10.
    """
    logger.info(f"Starting CTTS-80 alignment from {ctts80_root} to {output_dir}...")
    aligner = FaceAligner()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        all_imgs.extend(list(ctts80_root.rglob(ext)))

    all_imgs = sorted(all_imgs)
    if max_samples:
        all_imgs = all_imgs[:max_samples]

    total = len(all_imgs)
    if total == 0:
        logger.warning(f"No images found in {ctts80_root}")
        return {"total": 0, "success": 0, "failed": 0, "failure_rate_pct": 0.0}

    success_count = 0
    failure_count = 0
    failed_files = []

    for img_p in tqdm(all_imgs, desc="Aligning CTTS-80"):
        rel_path = img_p.relative_to(ctts80_root)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip already-processed files (resume-friendly)
        if out_path.exists():
            success_count += 1
            continue

        aligned, _, success, msg = aligner.detect_and_align(img_p)
        if aligned is not None:
            cv2.imwrite(str(out_path), aligned)

        if success:
            success_count += 1
        else:
            failure_count += 1
            failed_files.append({"path": str(img_p), "reason": msg})

    failure_rate = (failure_count / total) * 100.0 if total > 0 else 0.0
    stats = {
        "dataset": "CTTS-80",
        "total_images": total,
        "aligned_success": success_count,
        "detection_failures": failure_count,
        "failure_rate_pct": round(failure_rate, 2),
        "failed_samples": failed_files[:100],
    }

    report_path = LOGS_DIR / "ctts80_alignment_stats.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"CTTS-80 Alignment Completed: {success_count}/{total} succeeded "
        f"({failure_count} failures, failure rate: {failure_rate:.2f}%). Stats saved to {report_path}"
    )
    return stats


def process_fiw_dataset(
    csv_dir: Path = FIW_PAIRS_DIR,
    output_dir: Path = PROCESSED_FIW_DIR,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Processes FIW images referenced in the pairs CSVs, resolves raw paths
    from D:/Public or local paths, aligns to 112x112, and logs failure rates.
    """
    logger.info(f"Starting FIW alignment from CSVs in {csv_dir} to {output_dir}...")
    aligner = FaceAligner()
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_paths = set()
    for csv_file in csv_dir.glob("*.csv"):
        try:
            import pandas as pd

            df = pd.read_csv(csv_file)
            for col in ["image_1", "image_2"]:
                if col in df.columns:
                    for val in df[col].dropna():
                        unique_paths.add(str(val))
        except Exception as e:
            logger.warning(f"Error reading CSV {csv_file}: {e}")

    path_list = sorted(list(unique_paths))
    if max_samples:
        path_list = path_list[:max_samples]

    total = len(path_list)
    if total == 0:
        logger.warning(f"No image references found in {csv_dir}")
        return {"total": 0, "success": 0, "failed": 0, "failure_rate_pct": 0.0}

    success_count = 0
    failure_count = 0
    missing_count = 0
    failed_files = []

    for raw_path_str in tqdm(path_list, desc="Aligning FIW"):
        resolved_path = resolve_fiw_path(raw_path_str)
        if not resolved_path or not resolved_path.exists():
            missing_count += 1
            failure_count += 1
            failed_files.append({"path": raw_path_str, "reason": "File not found at resolved path"})
            continue

        str_p = str(resolved_path).replace("\\", "/")
        idx = str_p.find("FIDs/")
        rel_subpath = str_p[idx:] if idx != -1 else resolved_path.name
        out_path = output_dir / rel_subpath
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip already-processed files (resume-friendly)
        if out_path.exists():
            success_count += 1
            continue

        aligned, _, success, msg = aligner.detect_and_align(resolved_path)
        if aligned is not None:
            cv2.imwrite(str(out_path), aligned)

        if success:
            success_count += 1
        else:
            failure_count += 1
            failed_files.append({"path": str(resolved_path), "reason": msg})

    failure_rate = (failure_count / total) * 100.0 if total > 0 else 0.0
    stats = {
        "dataset": "FIW",
        "total_images": total,
        "aligned_success": success_count,
        "detection_failures": failure_count,
        "missing_files": missing_count,
        "failure_rate_pct": round(failure_rate, 2),
        "failed_samples": failed_files[:100],
    }

    report_path = LOGS_DIR / "fiw_alignment_stats.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"FIW Alignment Completed: {success_count}/{total} succeeded "
        f"({failure_count} failures, {missing_count} missing, failure rate: {failure_rate:.2f}%). Stats saved to {report_path}"
    )
    return stats


def process_digiface_dataset(
    digiface_root: Path = DIGIFACE_ROOT,
    output_dir: Path = PROCESSED_DIGIFACE_DIR,
    max_samples: Optional[int] = None,
) -> Dict:
    """Processes DigiFace-1M images and aligns to 112x112.

    Uses structured directory iteration instead of rglob to avoid
    multi-minute file scanning overhead on 1M+ image datasets.
    Structure: digiface_root / subset_dir / subject_id / {0..N}.png
    """
    logger.info(f"Starting DigiFace-1M alignment from {digiface_root} to {output_dir}...")
    aligner = FaceAligner()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images by iterating the known 2-level structure (subset/subject/images)
    # instead of rglob, which is extremely slow on 1M+ files.
    logger.info("Scanning DigiFace-1M directory structure...")
    all_imgs = []
    for subset_dir in sorted(digiface_root.iterdir()):
        if not subset_dir.is_dir():
            continue
        for subject_dir in sorted(subset_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            for img_file in sorted(subject_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    all_imgs.append(img_file)
    logger.info(f"Found {len(all_imgs)} DigiFace-1M images to process.")

    if max_samples:
        all_imgs = all_imgs[:max_samples]

    total = len(all_imgs)
    if total == 0:
        logger.warning(f"No images found in {digiface_root}")
        return {"total": 0, "success": 0, "failed": 0, "failure_rate_pct": 0.0}

    success_count = 0
    failure_count = 0

    for img_p in tqdm(all_imgs, desc="Aligning DigiFace-1M"):
        rel_path = img_p.relative_to(digiface_root)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip already-processed files (resume-friendly)
        if out_path.exists():
            success_count += 1
            continue

        aligned, _, success, _ = aligner.detect_and_align(img_p)
        if aligned is not None:
            cv2.imwrite(str(out_path), aligned)

        if success:
            success_count += 1
        else:
            failure_count += 1

    failure_rate = (failure_count / total) * 100.0 if total > 0 else 0.0
    stats = {
        "dataset": "DigiFace-1M",
        "total_images": total,
        "aligned_success": success_count,
        "detection_failures": failure_count,
        "failure_rate_pct": round(failure_rate, 2),
    }

    report_path = LOGS_DIR / "digiface_alignment_stats.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"DigiFace-1M Alignment Completed: {success_count}/{total} succeeded "
        f"({failure_count} failures, failure rate: {failure_rate:.2f}%). Stats saved to {report_path}"
    )
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="5-point landmark face alignment to 112x112")
    parser.add_argument(
        "--dataset",
        choices=["ctts80", "fiw", "digiface", "all"],
        default="ctts80",
        help="Dataset to align",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to process (useful for testing)",
    )
    args = parser.parse_args()

    if args.dataset in ["ctts80", "all"]:
        process_ctts80_dataset(max_samples=args.limit)
    if args.dataset in ["fiw", "all"]:
        process_fiw_dataset(max_samples=args.limit)
    if args.dataset in ["digiface", "all"]:
        process_digiface_dataset(max_samples=args.limit)
