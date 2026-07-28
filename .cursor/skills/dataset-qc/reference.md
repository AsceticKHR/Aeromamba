# Dataset QC — Check Catalog & Rationale

Detailed reference for `dataset-qc`. Read this when adding checks, tuning thresholds,
or justifying gate decisions.

## Norm mapping

| Industry practice | What we adopt |
|---|---|
| **TFX Data Validation (TFDV)** | Schema inference + anomaly detection → B1/B2 schema gates, distribution drift as INFO. |
| **Great Expectations** | Expectation suite with pass/fail per expectation → tiered CHECKS table + report.json. |
| **Amazon Deequ** | Constraint & metric checks (completeness, uniqueness, range) → B2 uniqueness, B4 range. |
| **COCO / FiftyOne bbox rules** | Valid box: inside image, positive area, correct ordering → B4. |
| **Datasheets for Datasets (Gebru et al.)** | Composition, collection, distribution documentation → report.md datasheet section. |
| **Data leakage guidance (ML eval integrity)** | No train/eval overlap → B6. |

## BLOCKER thresholds & rationale

- **B4 bbox validity.** Coordinates must satisfy `0 <= x1 < x2 <= scale` and
  `0 <= y1 < y2 <= scale`. The repo's `parse_bbox_0_1000` does **not** enforce
  ordering, so inverted/zero-area boxes would silently train the grounding head on
  garbage targets and cause the IoU collapse seen in S2 pilots. Area fraction must be
  in `(0, 1]`; exactly-zero area is a BLOCKER, near-zero (<0.001) is W2.
- **B6 leakage.** Primary check compares exact relative `image` paths: any shared
  path ⇒ the identical image is in both train and eval ⇒ contaminated evaluation ⇒
  BLOCKER. Basename-only overlap (same filename, different path) is reported as W
  (B6b): it can mean a renamed duplicate (real leak) OR merely recycled per-episode
  frame numbering (e.g. `uav_motion` frames), so it is flagged for review rather than
  hard-failed. When ingesting a renamed source (e.g. mask-derived crops), exclude
  eval images explicitly by their original (scene, filename) key, since renaming
  breaks both path and basename matching.

## WARN thresholds & rationale

- **W1 center bias.** If box centers cluster at (0.5, 0.5) with std < 0.1, labels are
  likely mode-collapsed (a known failure signature: the head predicts a constant
  center box). Report mean/std of centers.
- **W2 size outliers.** Tiny boxes (<0.1% area) are near-unlearnable at 384px / 27×27
  patch grid; full-image boxes (>90%) provide no localization signal.
- **W3 duplication.** Exact `(image, expression, box)` triples repeated > 1% inflate
  effective epoch count on a few samples.
- **W4 expression diversity.** unique(expression)/count < 0.5 means heavy templating;
  acceptable for CPT breadth but flagged so it is not mistaken for rich supervision.

## Coordinate conventions

- Internal canonical: `[x1, y1, x2, y2]` top-left origin, `x` horizontal.
- Repo scale: integers in `[0, 1000)` (Qwen-style), normalized by `/1000` at load.
- When ingesting a new source, convert to this scale **before** QC so B4 ranges apply.
- Mask-derived boxes (Open3DVQA): `x1=min col, y1=min row, x2=max col, y2=max row`
  of the binary mask, then normalize by image `(W, H)`.

## Report structure (report.md)

```
# Dataset QC Report — <jsonl name>
## Verdict: PASS | FAIL (n BLOCKER)
## Gates
<table: id | tier | check | result | detail>
## Datasheet
- Composition: rows, per-source counts, per-task counts
- Grounding: #boxes, box geometry stats (w,h,area,center,aspect)
- Text: instruction/answer length percentiles, unique-expression ratio
- Integrity: images resolved / missing / sampled-opened
- Leakage: eval overlap count
## WARN details
## Recommendations
```

## Adding a check

1. Implement a function returning `(tier, check_id, name, passed: bool, detail: str)`.
2. Append to the `CHECKS` list execution in `run_all`.
3. Add threshold constants at the top of `scripts/dataset_qc.py`.
4. BLOCKER checks must set the process exit code via the aggregate in `main`.
