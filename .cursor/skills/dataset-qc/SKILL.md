---
name: dataset-qc
description: Standardized quality-control gates for VLA/VLM grounding & instruction datasets in LLaVA-conversation JSONL. Runs schema, referential integrity, bbox label validity, distribution/bias, duplication & train-eval leakage, and text-quality checks; emits a pass/fail report with tiered gates. Use when preparing, merging, or accepting new training data (e.g. L0/S2 grounding JSONL, DIOR-RSVG, Open3DVQA), or when the user asks to QC / validate / 质检 a dataset.
---

# Dataset QC (grounding & instruction JSONL)

Industry-standard data validation adapted for this repo's LLaVA-conversation JSONL
(`{id, source, task, image, image_root, conversations:[{from,value}]}`, boxes as
`[x1,y1,x2,y2]` in 0–1000 int coords inside `gpt` answers).

Norms this encodes: schema+anomaly (TFDV), expectation suites (Great Expectations),
constraint/metric checks (Deequ), bbox validity (COCO/FiftyOne), and dataset
documentation (Datasheets for Datasets).

## When to run

Run **before** any new data enters the training mix, and again after merging into
`l0_mixed.jsonl`. Treat it as a merge gate: BLOCKER failures must be fixed or the
data is rejected.

## Gate tiers

| Tier | Meaning | Action |
|------|---------|--------|
| **BLOCKER** | Corrupts training or evaluation validity | Reject / fix before use. Script exits non-zero. |
| **WARN** | Quality/limitation risk | Allowed, but record in datasheet; consider filtering. |
| **INFO** | Descriptive statistics | Report only. |

### BLOCKER checks
- **B1 Parse**: every line is valid JSON.
- **B2 Schema**: required keys present (`id, source, task, image, conversations`);
  `conversations` is a non-empty list of `{from,value}`; `id` unique.
- **B3 Image integrity**: image path resolves via `image_roots`; file exists;
  (sampled) opens and has positive dimensions.
- **B4 BBox validity** (grounding rows): coords parse; all in `[0, scale]`;
  `x1<x2` and `y1<y2` (no degenerate/inverted); area fraction in `(0, 1]`; no NaN/inf.
- **B5 Non-empty target**: `gpt` answer non-empty; grounding rows actually contain a box.
- **B6 Leakage**: when an eval/held-out split is provided, **zero** exact
  relative-path overlap between new data and the eval split (definite same image).
  Basename-only overlap (paths differ, e.g. recycled per-episode frame names) is a
  separate **W** advisory (B6b), not a blocker.

### WARN checks
- **W1 Center bias**: box-center distribution concentrated (mean center within
  0.45–0.55 on both axes with low variance ⇒ likely degenerate/mode-collapsed labels).
- **W2 Size outliers**: >5% boxes tiny (area frac < 0.001) or >5% near-full-image (>0.9).
- **W3 Duplication**: exact-duplicate (image+expression+box) ratio > 1%.
- **W4 Expression diversity**: unique-referring-expression ratio < 0.5 (heavy templating).
- **W5 Text bounds**: instruction/answer length outliers (empty, or > max_text_len tokens proxy).
- **W6 Source/task balance**: any single source > 80% (over-domination) — report mix.

## Workflow

```
- [ ] 1. Locate the JSONL and its image_roots
- [ ] 2. Run scripts/dataset_qc.py (add --eval-jsonl for leakage check)
- [ ] 3. Read report.md; if BLOCKER present, STOP and fix
- [ ] 4. Record WARN items in the datasheet section of report.md
- [ ] 5. Only merge/accept when zero BLOCKER
```

**Run:**

```bash
python scripts/dataset_qc.py \
  --jsonl /path/to/new_grounding.jsonl \
  --image-roots '{"l0":"/root/autodl-tmp/datasets/l0_cpt","stage2":"/root/autodl-tmp/Aeromamba/data"}' \
  --grounding-tasks grounding \
  --coord-scale 1000 \
  --sample-images 300 \
  --eval-jsonl /path/to/held_out_eval.jsonl \
  --out-dir ./qc_out
```

Exit code `0` = no BLOCKER; `1` = at least one BLOCKER (do not use the data).

Dependencies: Python 3.8+, stdlib only for logic; `Pillow` (optional) for B3 image
open/dimension check and `numpy` (optional) for faster stats. The script degrades
gracefully (skips B3 open-check with a WARN if Pillow is missing) — never silently passes.

## Output

- `qc_out/report.md` — human-readable gates + datasheet-style statistics.
- `qc_out/report.json` — machine-readable results (for CI / merge gating).

## Extending

Add new checks in `scripts/dataset_qc.py` under the matching tier and register them
in the `CHECKS` table so they appear in the report. Keep thresholds at the top of the
file. For the full check catalog, thresholds rationale, and norm mapping, see
[reference.md](reference.md).
