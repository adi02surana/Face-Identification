import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import CTTS80_ROOT, FIW_RAW_ROOT, LOGS_DIR
from degradation.simulator import DegradationSimulator
from preprocessing.align import align_face


def run_degradation_demo(
    sample_img_path: Path = None,
    dataset: str = "ctts80",
    random_sample: bool = False,
    seed: int = None,
    output_plot_path: Path = LOGS_DIR / "degradation_grid_demo.png",
):
    """Generates a 3x6 grid visualization showing Original + 5 severity levels per axis."""
    if seed is not None:
        random.seed(seed)

    if sample_img_path is None:
        target_root = FIW_RAW_ROOT if dataset.lower() == "fiw" else CTTS80_ROOT
        sample_imgs = list(target_root.rglob("*.jpg")) + list(target_root.rglob("*.png"))
        if not sample_imgs:
            raise FileNotFoundError(f"No sample images found in {target_root}")
        
        if random_sample:
            sample_img_path = random.choice(sample_imgs)
        else:
            sample_img_path = sample_imgs[0]

    sample_img_path = Path(sample_img_path)
    print(f"Loading sample image from: {sample_img_path}")

    # First align face to 112x112
    aligned_img, success = align_face(sample_img_path, target_size=(112, 112))
    if aligned_img is None:
        raise RuntimeError(f"Failed to detect/align face in sample image: {sample_img_path}")

    simulator = DegradationSimulator(split="eval", seed=seed or 42)
    all_degraded = simulator.apply_all_axes_and_levels(aligned_img)

    # Prepare 3x6 grid: Col 0 is Clean/Aligned, Cols 1..5 are Severities 1..5
    axes_names = ["blur", "jpeg", "illumination"]
    titles = ["Blur", "JPEG Compression", "Illumination Shift"]

    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    person_name = sample_img_path.parent.name
    fig.suptitle(
        f"Degradation Simulator Verification Grid ({person_name} / {sample_img_path.name})\n"
        f"3 Axes × 5 Calibrated Severity Levels",
        fontsize=15,
    )

    for row_idx, axis_name in enumerate(axes_names):
        # Column 0: Clean Aligned Image
        rgb_clean = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
        axes[row_idx, 0].imshow(rgb_clean)
        axes[row_idx, 0].set_title(f"{titles[row_idx]}\n(Clean 112×112)", fontsize=10)
        axes[row_idx, 0].axis("off")

        # Columns 1..5: Severities 1..5
        for sev in range(1, 6):
            degraded_bgr, meta = all_degraded[(axis_name, sev)]
            rgb_deg = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)
            col_idx = sev
            ax = axes[row_idx, col_idx]
            ax.imshow(rgb_deg)

            if axis_name == "blur":
                sub_label = f"L{sev}: {meta['kernel_type']}\n(k={meta['ksize']})"
            elif axis_name == "jpeg":
                sub_label = f"L{sev}: Q={meta['jpeg_quality']}"
            elif axis_name == "illumination":
                sub_label = f"L{sev}: γ={meta['gamma']:.2f}\n({meta['illumination_mode']})"
            else:
                sub_label = f"Level {sev}"

            ax.set_title(sub_label, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved degradation demo grid plot to: {output_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Degradation Simulator Visual Demo Grid Generator")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a specific image file to test.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ctts80",
        choices=["ctts80", "fiw"],
        help="Dataset to pick from if --image is not specified.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        default=True,
        help="Randomly pick an image from the dataset instead of always picking the first.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for selection and degradation reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(LOGS_DIR / "degradation_grid_demo.png"),
        help="Path to save the generated output grid image.",
    )
    args = parser.parse_args()

    run_degradation_demo(
        sample_img_path=args.image,
        dataset=args.dataset,
        random_sample=args.random,
        seed=args.seed,
        output_plot_path=Path(args.output),
    )
