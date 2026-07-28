# Grounding Data Prep + QC — 2026-07-23

Prepared new grounding supervision on the remote (AutoDL) and validated it with the
new `dataset-qc` skill.

## Skill

`.cursor/skills/dataset-qc/` — standardized, tiered data QC (BLOCKER / WARN / INFO)
for LLaVA-conversation grounding JSONL. Encodes TFDV (schema+anomaly),
Great Expectations (expectation suite), Deequ (constraints), COCO/FiftyOne (bbox
validity), Datasheets-for-Datasets (documentation) and train/eval leakage checks.

- `SKILL.md` — gates, workflow, run command.
- `reference.md` — check catalog, thresholds rationale, norm mapping.
- `scripts/dataset_qc.py` — runnable; emits `report.md` + `report.json`; exit 1 iff any BLOCKER.

## Data prepared: DIOR-RSVG

Source: `LittleCollections/DIOR-RSVG` on HF (pulled via `hf-mirror.com`; Google Drive
and huggingface.co are unreachable from the remote). Canonical VOC layout
(`Annotations.zip` + `JPEGImages.zip` 5.4GB + split txt).

Converter: `data/build_dior_rsvg.py` (VOC XML → L0 grounding JSONL, boxes → 0..1000).

- Remote images: `/root/autodl-tmp/datasets/l0_cpt/images/dior_rsvg/` (17,402 jpg, 800×800).
- Train JSONL: `/root/autodl-tmp/datasets/dior_rsvg/dior_rsvg_train.jsonl` (30,820 = train+val).
- Test JSONL:  `/root/autodl-tmp/datasets/dior_rsvg/dior_rsvg_test.jsonl` (7,500).
- Split reconciliation exact: total objects 38,320; built train=26,991 / val=3,829 /
  test=7,500 == official txt counts (enumeration order matches the official split).

## QC results

`dior_rsvg_train.jsonl` → **PASS** (0 BLOCKER, 3 WARN):

- B1–B5 all PASS: JSON parse, schema+unique id, 0 unresolved/failed images
  (400 sampled opened), 30,820/30,820 valid boxes (0 out-of-range / degenerate / NaN),
  every grounding row carries a box.
- W1 center bias PASS: center mean (0.504, 0.503), std (0.222, 0.213) — well spread.
- W2 WARN: 11.7% tiny boxes (<0.1% area) — DIOR has many small objects; hard at 384px.
- W4 WARN: expression diversity 0.30 — templated captions ("The X in the left/right").
- W6 WARN: single-source file (100% dior_rsvg) — expected pre-merge into `l0_mixed`.
- Box geometry: area p10/50/90 = 0.0009 / 0.020 / 0.22; aspect p50 = 0.99.

### Leakage gate demonstration (B6)

Running QC on `dior_rsvg_test.jsonl` with `--eval-jsonl dior_rsvg_train.jsonl` →
**FAIL**: 4,446 overlapping image basenames. DIOR-RSVG official splits are
**expression-level**, so images are reused across train/val/test. Implication: a clean
held-out **grounding** eval must be split by image, not by DIOR's official split. This
does not affect our use — DIOR is auxiliary L0 grounding; S2/S3 evaluation is on
UAV-Flow / airspatial / open3d imagery (disjoint from DIOR).

## Data prepared: Open3DVQA-REC (mask → 2D box)

Source: `EmbodiedCity/Open3DVQA-v2` on HF (`O3DVQA.zip`, 2.1GB, via hf-mirror).
Only 4 scenes ship mask-bearing `chunk_*.pkl` (RealworldUAV / UrbanScene / WildUAV);
`EmbodiedCity/Wuhan` chunks carry captions only. Each mask-bearing row has aligned
`caption[i]` / `masks[i]` (uint8 0/255 HxW) / `valid_idx[i]`.

Extractor: `data/build_open3dvqa_rec.py` — box = tight rect of each mask's nonzero
pixels, normalized to 0..1000; drops empty / degenerate / near-full (>0.92) / within-image
duplicate boxes; RGB saved from the embedded PIL image.

- Remote images: `/root/autodl-tmp/datasets/l0_cpt/images/o3dvqa_rec/` (506 jpg, 640×480).
- JSONL: `/root/autodl-tmp/datasets/o3dvqa_full/o3dvqa_rec_grounding.jsonl` (1,396 rows).
- Dropped: full-image=48, dup-box=12, degenerate/empty=0.

QC → **PASS** (0 BLOCKER, 1 WARN = single-source pre-merge). Notable geometry:
area p10/50/90 = 0.064 / 0.235 / 0.569 (median box 23.5% of image — CLIPSeg+SAM
pseudo-masks are **coarse/large**, unlike DIOR's precise small boxes); expression
diversity 0.86 (rich GPT-4o captions). Caveat: pseudo-label boxes teach coarse
region attention rather than tight localization; treat as auxiliary / consider
down-weighting relative to DIOR human boxes.

## Source comparison

| | DIOR-RSVG | Open3DVQA-REC |
|---|---|---|
| boxes | human GT, precise | CLIPSeg+SAM pseudo, coarse |
| rows / images | 30,820 / 17,402 | 1,396 / 506 |
| median box area | 0.020 | 0.235 |
| expr diversity | 0.30 (templated) | 0.86 (rich) |
| domain | RS overhead | real UAV / urban oblique |

## Merge into S2 grounding mix + re-QC

Existing `l0_mixed.jsonl` had only **495 grounding rows** (215,552 total) — the P5
scarcity root cause. Merged `l0_mixed + dior_rsvg_train + o3dvqa_rec` (O3D-REC
regenerated with `--exclude-eval-json` to drop **79 eval-leak images**, e.g.
UrbanScene/Campus frames that appear in `eval_subset.json`).

Two artifacts on remote (`/root/autodl-tmp/datasets/l0_cpt/`):

| file | rows | grounding | B6 leakage gate |
|---|---|---|---|
| `l0_mixed_grd.jsonl` | 247,545 | 32,488 | **FAIL** |
| `l0_mixed_grd_clean.jsonl` | 187,605 | 32,488 | **PASS** |

Grounding boosted ~66× (495 → 32,488). QC B1–B5 pass on both; W6 source balance now
PASS (top source `general` 32.6%). W2 tiny 11.4% (DIOR small boxes), W4 diversity 0.35.

### Pre-existing eval leakage (important)

B6 on the full merge FAILs with **437 overlapping basenames** — attribution:
`eval ∩ DIOR = 0`, `eval ∩ O3D-REC = 0`, **`eval ∩ original l0_mixed = 437` (all of
eval)**. So `eval_subset.json` was drawn from the same images used to train
`l0_mixed`; this predates and is independent of the grounding additions, which are
fully eval-disjoint. `l0_mixed_grd_clean.jsonl` removes all rows on the 437 eval
images (dropped 59,940 rows, mostly repeated coco/uav/cognitive VQA turns) → B6 PASS.

Recommendation: for a rigorous evaluation, rebuild `eval_subset` from images **held
out** of training rather than shrinking the training set; then S0/S1/S2 metrics on it
are no longer optimistic. `l0_mixed_grd_clean.jsonl` is the eval-disjoint training
input for S2 retraining in the meantime.

- Consider down-weighting `open3d_vqa_rec` (coarse pseudo-labels) vs `dior_rsvg`
  (precise human boxes) in the grounding sampler.

## Image-level held-out eval (rebuilt)

The original `eval_subset.json` was an item-level `random_split` val slice of
`stage2_mixed_data_v2.json`, so eval images also appear in training (image leakage).
Rebuilt as a true **image-level** hold-out via `data/build_heldout_eval.py`: reserves
whole images per source (seed 42), removes every row on those images from training,
asserts path-level disjointness. Includes a grounding split for clean P5 IoU.

On remote (`/root/autodl-tmp/datasets/l0_cpt/`):

| file | content |
|---|---|
| `eval_subset_v2.json` / `.jsonl` | 1,350 held-out items over 1,240 images; grounding split = 520 boxes (dior 250 / o3d 150 / airspatial 120) + capability items (general/aerial_spatial/cognitive/hrvqa/uav_motion/airspatial 150–200 each) |
| `l0_mixed_grd_heldout.jsonl` | 232,602 training rows (31,559 grounding); **image-disjoint** from eval_v2 |

QC of `l0_mixed_grd_heldout.jsonl` vs `eval_subset_v2.jsonl` → **PASS**: B6 exact-path
leakage = 0. One benign cross-source basename collision is reported as B6b WARN (paths
differ; recycled frame names / dup filename, not a real leak).

QC B6 was refined: exact relative-path overlap is the BLOCKER (definite same image);
basename-only overlap is B6b WARN (handles `uav_motion` recycled per-episode frame
numbering, which previously over-flagged).

### Recommended training/eval pair

- Train S2 on `l0_mixed_grd_heldout.jsonl` (grounding boosted 495 → 31,559; eval-clean).
- Evaluate on `eval_subset_v2.json`; measure P5 IoU on its 520 held-out grounding boxes.
