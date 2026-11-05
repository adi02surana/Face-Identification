# Robust Face Verification for Highly Similar Identities under Optical Degradation

**Mini-project — LNMIIT Jaipur, Dept. of Communication and Computer Engineering**
Team: Ananya Surana (24UCC090), Samarth Gupta (24UCC134) · Guide: Dr. Gaurav Verma

This README specifies the full architecture, data pipeline, training procedure, and evaluation protocol as finalized in the project's progress report, so implementation can start directly from this document. Every design choice below is literature-grounded — the "why" is included so implementation doesn't drift from the justified design if questions come up.

---


## 1. Problem Statement

Build a face verification system that classifies a face pair into one of three outcomes — **same-identity**, **twin-imposter**, or **general-imposter** — and remains as accurate as possible as image quality is deliberately degraded (blur, JPEG compression, illumination shift). No prior work evaluates twin discrimination and *controlled, engineered* optical degradation jointly — that is this project's contribution. We additionally test across a **genetic-similarity spectrum** (twins → full siblings → unrelated individuals) rather than a binary twin/non-twin split.

---

## 2. Datasets (already acquired — do not block on further access requests)

| Dataset | Role | Status | Notes |
|---|---|---|---|
| **CTTS-80** (Celeb Twins Test Set) | Primary — twin pairs | ✅ In hand | 80 celebrity twin pairs, 12 images/person. Verification protocol: 132 same-person + 144 different-person pairs per twin set (same structure as LFW/CALFW/CPLFW/CFP-FP/AgeDB-30). Web-scraped → **uncontrolled pose/lighting/quality**, unlike studio-shot ND-TWINS. Includes metadata: skin-mark distinguishability (43/80 pairs) and mirror-asymmetry (7/80 pairs) — worth using if time permits. Reference: Zang et al., arXiv:2609.01141, 2026. |
| **FIW** (Families in the Wild) | Secondary — sibling pairs | ✅ In hand | ~12,000 photos, ~1,000 families, 11 kinship relationship types. Use only full-sibling pairs for our purposes. |
| **DigiFace-1M** | Baseline — unrelated pairs + Phase 1 pretraining data | ✅ In hand | 1M synthetic (computer-generated) faces, Microsoft 2023. Used specifically because MS-Celeb-1M and VGGFace2 were retracted over consent issues. |
| ND-TWINS-2009-2010 / WVU-FBI Twins Days | Stretch-goal validation set | ⏳ Access pending | Do not block current pipeline on this. Integrate later if/when access clears. |

**Critical constraint:** keep twin/sibling pairs **fully identity-disjoint** across train/val/test splits — no subject's face may appear in more than one split, even under a different pair. Identity leakage here silently inflates every reported number.

---

## 3. Environment & Dependencies (accessibility constraint: no proprietary/inaccessible tools)

Everything below is free, pip-installable, and requires no special access or paid API:

```
torch, torchvision          # model + training
insightface                 # RetinaFace detector + pretrained ArcFace/InsightFace checkpoints
opencv-python                # image I/O, blur kernels, JPEG re-encoding
numpy
pillow
scikit-learn                 # EER, ROC, TAR@FAR, k-fold splitting
scikit-image                 # optional: classical restoration fallback (Wiener/blind deconvolution)
matplotlib, seaborn           # plots, t-SNE visualization
tqdm
```

Do **not** introduce: GAN-based degradation models (e.g. MPGNet), wave-optics/phase-screen turbulence simulators, or any dataset/checkpoint requiring a restrictive access agreement. These were reviewed in the literature (Ahn et al. 2025; Leonard et al. 2012) but explicitly **not adopted** — they need GPU-heavy training or specialized simulation code beyond this project's scope. If a "why not X" question comes up during implementation, the answer is this constraint.

---

## 4. Directory Structure (suggested)

```
project/
├── data/
│   ├── ctts80/                # raw CTTS-80 images + pair-list metadata
│   ├── fiw/                   # raw FIW images, organized by family/relationship
│   └── digiface1m/            # raw DigiFace-1M synthetic images
├── preprocessing/
│   └── align.py                # RetinaFace detect + align + crop to 112x112
├── degradation/
│   └── simulator.py            # blur / compression / illumination, 5 severity levels each
├── models/
│   ├── backbone.py             # ResNet-50 (ArcFace-pretrained) wrapper
│   ├── cbam.py                 # global channel+spatial attention
│   ├── regional_branch.py       # landmark-guided face-curve crop + small conv branch
│   └── dual_path_net.py         # fuses backbone + CBAM + regional branch → 512-d embedding
├── training/
│   ├── phase1_arcface.py        # general pretraining on DigiFace-1M
│   └── phase2_triplet.py        # twin/sibling-aware hard-negative triplet fine-tuning
├── evaluation/
│   ├── metrics.py               # EER, TAR@FAR, ROC
│   ├── kfold_eval.py            # k-fold protocol for CTTS-80 (only 80 pairs)
│   └── visualize.py             # t-SNE, attention-weight (global vs regional) shift plots
└── splits/
    └── identity_disjoint_split.py  # enforce no-identity-leakage train/val/test split
```

---

## 5. Pipeline Stage 1 — Preprocessing

- **Detector/aligner:** RetinaFace (via `insightface`), using its 5-point landmarks (eyes, nose, mouth corners).
- **Output:** every face cropped and aligned to **112×112**, matching ArcFace-family backbone input expectations.
- **CTTS-80-specific note:** its web-scraped images will have a higher detection/alignment failure rate than FIW/DigiFace-1M. Log and report this failure rate explicitly — it's a documented dataset limitation, not a bug.

---

## 6. Pipeline Stage 2 — Degradation Simulator

Built **entirely** from deterministic OpenCV/NumPy/Pillow operations — no learned or paired-data-dependent degradation model. Three axes, **5 fixed and logged severity levels each** (not random/unlogged), so accuracy can be plotted against severity:

| Axis | Method | Justification |
|---|---|---|
| **Blur** (widest severity range) | Isotropic + anisotropic Gaussian kernels, plus linear motion-blur kernels, randomly sampled strength within each severity band — following Yasarla & Patel's kernel-convolution recipe | Karahan et al. (2016) and Grm et al. (2018) both found blur causes the sharpest CNN face-recognition accuracy drop among common degradations |
| **JPEG compression** (narrower range) | Re-encode at randomly sampled quality factor per severity level | Same two studies found compression/illumination comparatively well tolerated until pushed to extremes — hence the narrower range |
| **Illumination** (narrower range) | Gamma correction + linear gain/offset shift (under- and over-exposure) | Same as above |

**Train/test kernel discipline:** following Yasarla & Patel, use a **disjoint set of blur kernels between training and evaluation** — don't test on the exact kernel distribution trained on, or robustness numbers will be optimistic.

**Do not implement:** GAN-learned degradation (Ahn et al.) or phase-screen/PSF turbulence simulation (Leonard et al.) as primary methods — reviewed but out of scope (see §3).

---

## 7. Pipeline Stage 3 — Embedding Network (Dual-Path Attention)

**Backbone:** ResNet-50, initialized from a pretrained ArcFace/InsightFace checkpoint (not trained from scratch). *(See flagged inconsistency at the top of this doc — confirm ResNet-50 vs ResNet-101 with the team before locking this in; ResNet-50 is used throughout Figures 1/2 and is the recommended default.)*

**Two parallel paths after the backbone:**

1. **Global path:** CBAM (Woo et al., 2018) — channel attention then spatial attention — applied to the full feature map. Cheap, well-tested, keeps a safe baseline.
2. **Regional path:** using RetinaFace's landmarks, crop the **lower-face/jaw-curve region** specifically, and pass just that crop through a small dedicated conv branch. This targets the exact region Mousavi et al. (2021) found most diagnostic for twins (~35% of pairs).

**Fusion:** concatenate global-path and regional-path outputs → linear projection → **512-dimensional, L2-normalized embedding**.

**Implementation fallback:** if the regional path proves too time-consuming, drop it and use the global CBAM path alone — this is a fully literature-justified standalone baseline, not a broken partial implementation. Build and validate this simpler version **first**, then add the regional path.

Two identical copies of this network (shared/tied weights) process the anchor and comparison image respectively — same-network, two forward passes, not two separate models.

---

## 8. Pipeline Stage 4 — Training (Two Phases)

**Phase 1 — General pretraining:**
- Data: DigiFace-1M
- Loss: ArcFace angular margin loss
- Goal: sharpen general discrimination before touching twin-specific data

**Phase 2 — Twin/sibling-aware fine-tuning:**
- Data: CTTS-80 (twins) + FIW (siblings)
- Loss: triplet loss
- **Hard-negative mining is mandatory and specific:** for a given anchor, mine negatives from that subject's own twin (CTTS-80) or sibling (FIW) — not random strangers. This is the single most load-bearing design decision in the whole project: random-negative triplet training never forces the network to resolve genuinely similar faces (this is why Paone et al. and Ahmad et al. found persistently high error rates even in twin-aware deep models trained the naive way).

---

## 9. Pipeline Stage 5 — Similarity Scoring & Evaluation

- **Scoring:** cosine distance between L2-normalized embeddings, thresholded into a decision. **Calibrate the threshold separately per condition** (twin-pair score distributions sit closer together than general-imposter distributions — one global threshold will misbehave).
- **Evaluation groups:** same-person / twin / sibling / unrelated, evaluated **separately at each of the 5 severity levels on each of the 3 degradation axes**.
- **Metrics:** EER, TAR@FAR, ROC curves, t-SNE visualization of the embedding space.
- **Novel metric enabled by the dual-path design:** track whether the network's relative reliance shifts from the regional (face-curve) path toward the global path as degradation increases — this would indicate the fine-grained twin-distinguishing signal is being lost first, which is the project's hypothesis.
- **CTTS-80-specific protocol:** only 80 twin pairs exist, so use **k-fold cross-validation** (rotate which pairs are held out across folds), reporting mean ± variance — not a single fixed train/test split, which would be too sensitive to which pairs land where.

---

## 10. Implementation Order (matches the report's "Next Steps")

1. RetinaFace preprocessing across all three datasets; measure and log CTTS-80's detection/alignment failure rate specifically.
2. Degradation simulator (§6) — build and sanity-check on sample images from all three datasets.
3. **Global CBAM path only** (no regional path yet) — get this working end-to-end first.
4. Phase 1 training (ArcFace on DigiFace-1M).
5. Phase 2 training (triplet + hard-negative mining on CTTS-80/FIW), with identity-disjoint splits and k-fold for CTTS-80.
6. Baseline evaluation sweep (clean vs. degraded, global-CBAM-only model).
7. Add the regional face-curve path; re-run the same evaluation sweep; compare.
8. t-SNE and attention-shift visualizations; write up results.

---

## 11. Key References (for inline citation in code comments/docstrings)

- ResNet-50 backbone: He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
- ArcFace loss/checkpoints: Deng et al., CVPR 2019
- RetinaFace: Deng et al., CVPR 2020
- Triplet loss: Schroff et al. (FaceNet), CVPR 2015
- CBAM: Woo et al., ECCV 2018
- Regional face-curve justification: Mousavi et al., Multimedia Tools Appl. 2021
- Degradation ranking (blur worst): Karahan et al., BIOSIG 2016; Grm et al., IET Biometrics 2018
- Blur kernel-convolution recipe: Yasarla & Patel, arXiv:2007.08404, 2020
- CTTS-80 dataset: Zang et al., arXiv:2609.01141, 2026
- FIW dataset: Robinson et al., ACM Multimedia 2016
- DigiFace-1M: Microsoft Research, WACV 2023
- AHAN (state of the art comparison target): Nguyen, AAAI 2026

Full citations with page numbers are in the project's progress report (`Progress_Report_Literature_and_Methodology_updated.docx`).
