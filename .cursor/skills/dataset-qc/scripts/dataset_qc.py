#!/usr/bin/env python3
"""Standardized QC for VLA/VLM grounding & instruction JSONL (dataset-qc skill).

Runs tiered gates (BLOCKER / WARN / INFO) over a LLaVA-conversation JSONL:
  {id, source, task, image, image_root, conversations:[{from,value}]}
with boxes as [x1,y1,x2,y2] in 0..scale int coords inside `gpt` answers.

Emits report.md + report.json. Exit code 1 iff any BLOCKER fails.

Deps: stdlib only for logic; Pillow (optional) for image-open check; numpy (optional)
for faster stats. Missing optional deps degrade to a WARN, never a silent pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image  # optional
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# ------------------------------------------------------------- thresholds
TINY_AREA_FRAC = 0.001
FULL_AREA_FRAC = 0.90
CENTER_STD_MIN = 0.10           # below this on both axes => W1 center bias
CENTER_MEAN_LO, CENTER_MEAN_HI = 0.45, 0.55
SIZE_OUTLIER_MAX_RATIO = 0.05   # >5% tiny or full => W2
DUP_MAX_RATIO = 0.01            # >1% exact dup triples => W3
EXPR_DIVERSITY_MIN = 0.50       # unique/total below => W4
SOURCE_DOMINATION = 0.80        # any source > 80% => W6

_BBOX_RE = re.compile(
    r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]"
)


def parse_box(text, scale):
    m = _BBOX_RE.search(text or "")
    if not m:
        return None
    return [float(m.group(i)) for i in range(1, 5)]


# ------------------------------------------------------------- load
def load_rows(path):
    rows, parse_errors = [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                parse_errors.append((ln, str(e)))
    return rows, parse_errors


def resolve_image(row, image_roots):
    rel = row.get("image", "")
    root = Path(image_roots.get(row.get("image_root", "l0"),
                                image_roots.get("l0", ".")))
    for cand in (root / rel, root / "images" / rel, Path(rel)):
        if cand.exists():
            return cand
    return None


def iter_turns(row):
    """Yield (human, gpt) pairs from conversations."""
    turns = row.get("conversations") or []
    i = 0
    while i < len(turns):
        if turns[i].get("from") != "human":
            i += 1
            continue
        human = turns[i].get("value", "")
        if i + 1 < len(turns) and turns[i + 1].get("from") == "gpt":
            yield human, turns[i + 1].get("value", "")
            i += 2
        else:
            i += 1


# ------------------------------------------------------------- checks
class Result:
    def __init__(self, tier, cid, name, passed, detail):
        self.tier, self.cid, self.name, self.passed, self.detail = \
            tier, cid, name, passed, detail

    def as_dict(self):
        return {"tier": self.tier, "id": self.cid, "name": self.name,
                "passed": self.passed, "detail": self.detail}


def run_all(rows, parse_errors, image_roots, grounding_tasks, scale,
            sample_images, eval_jsonl):
    results = []
    n = len(rows)

    # B1 parse
    results.append(Result("BLOCKER", "B1", "JSON parse",
                          len(parse_errors) == 0,
                          f"{len(parse_errors)} unparseable lines"
                          + (f" (e.g. line {parse_errors[0][0]})" if parse_errors else "")))

    # B2 schema + uniqueness
    req = ("id", "source", "task", "image", "conversations")
    bad_schema, ids = 0, []
    for r in rows:
        ok = all(k in r for k in req) and isinstance(r.get("conversations"), list) \
            and len(r["conversations"]) > 0
        if not ok:
            bad_schema += 1
        ids.append(r.get("id"))
    dup_ids = n - len(set(ids))
    results.append(Result("BLOCKER", "B2", "Schema + unique id",
                          bad_schema == 0 and dup_ids == 0,
                          f"{bad_schema} rows missing keys/empty conv; {dup_ids} duplicate ids"))

    # collect grounding turns + all turns
    grd = []       # dicts: image, image_root, expr, box, row_id
    txt_lens = []  # (human_len, gpt_len) char proxy
    empty_ans = 0
    for r in rows:
        is_grd = r.get("task") in grounding_tasks
        for human, gpt in iter_turns(r):
            txt_lens.append((len(human.strip()), len(gpt.strip())))
            if not gpt.strip():
                empty_ans += 1
            if is_grd:
                box = parse_box(gpt, scale)
                grd.append({"image": r.get("image"), "image_root": r.get("image_root", "l0"),
                            "expr": human.replace("<image>", "").strip(),
                            "box": box, "id": r.get("id")})

    # B3 image integrity (existence for all; open for a sample)
    missing, checked_paths = 0, {}
    for r in rows:
        key = (r.get("image"), r.get("image_root"))
        if key not in checked_paths:
            checked_paths[key] = resolve_image(r, image_roots)
        if checked_paths[key] is None:
            missing += 1
    open_fail, opened, dims_bad = 0, 0, 0
    if _HAS_PIL and sample_images > 0:
        uniq = [p for p in {v for v in checked_paths.values() if v is not None}]
        step = max(1, len(uniq) // sample_images)
        for p in uniq[::step][:sample_images]:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    opened += 1
                    if w <= 0 or h <= 0:
                        dims_bad += 1
            except Exception:
                open_fail += 1
    results.append(Result("BLOCKER", "B3", "Image resolves & opens",
                          missing == 0 and open_fail == 0 and dims_bad == 0,
                          f"{missing} unresolved paths; sampled_open={opened}, "
                          f"open_fail={open_fail}, bad_dims={dims_bad}"
                          + ("" if _HAS_PIL else " [PIL missing: open-check skipped]")))
    if not _HAS_PIL:
        results.append(Result("WARN", "B3b", "PIL available", False,
                              "Pillow not installed; image open/dimension check skipped"))

    # B4 bbox validity + B5 grounding-has-box
    n_grd = len(grd)
    boxes_ok, oob, degen, nan_bad = 0, 0, 0, 0
    geoms = []  # (w,h,area,cx,cy,aspect)
    no_box = 0
    for g in grd:
        b = g["box"]
        if b is None:
            no_box += 1
            continue
        if any(map(lambda v: (math.isnan(v) or math.isinf(v)), b)):
            nan_bad += 1
            continue
        x1, y1, x2, y2 = b
        if not (0 <= x1 <= scale and 0 <= y1 <= scale and 0 <= x2 <= scale and 0 <= y2 <= scale):
            oob += 1
            continue
        if not (x1 < x2 and y1 < y2):
            degen += 1
            continue
        boxes_ok += 1
        w, h = (x2 - x1) / scale, (y2 - y1) / scale
        geoms.append((w, h, w * h, (x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale,
                      (w / h if h > 0 else 0)))
    results.append(Result("BLOCKER", "B4", "BBox validity",
                          (oob == 0 and degen == 0 and nan_bad == 0) and n_grd > 0,
                          f"grd_boxes={n_grd}, valid={boxes_ok}, out_of_range={oob}, "
                          f"degenerate={degen}, nan/inf={nan_bad}"))
    results.append(Result("BLOCKER", "B5", "Grounding rows carry a box & non-empty answer",
                          no_box == 0 and empty_ans == 0,
                          f"grounding turns w/o box={no_box}; empty answers={empty_ans}"))

    # B6 leakage. Exact relative-path overlap = definite same image (BLOCKER).
    # Basename-only overlap (paths differ) = possible dup or recycled frame names
    # (e.g. per-episode frame numbering) => WARN, investigate but not proof of leak.
    if eval_jsonl:
        eval_rows, _ = load_rows(eval_jsonl)
        eval_paths = {r.get("image", "") for r in eval_rows}
        new_paths = {r.get("image", "") for r in rows}
        path_overlap = eval_paths & new_paths
        results.append(Result("BLOCKER", "B6", "No train/eval image leakage (exact path)",
                              len(path_overlap) == 0,
                              f"{len(path_overlap)} identical image paths shared with eval"))
        eval_bn = {os.path.basename(p) for p in eval_paths}
        new_bn = {os.path.basename(p) for p in new_paths}
        bn_only = len(eval_bn & new_bn) - len({os.path.basename(p) for p in path_overlap})
        results.append(Result("WARN", "B6b", "Shared basenames with eval (paths differ)",
                              bn_only <= 0,
                              f"{bn_only} basenames shared but paths differ "
                              f"(dup images or recycled frame names — verify)"))

    # ---- WARN
    if geoms:
        cxs = [g[3] for g in geoms]
        cys = [g[4] for g in geoms]
        cx_m, cy_m = statistics.mean(cxs), statistics.mean(cys)
        cx_s = statistics.pstdev(cxs) if len(cxs) > 1 else 0
        cy_s = statistics.pstdev(cys) if len(cys) > 1 else 0
        center_bias = (CENTER_MEAN_LO <= cx_m <= CENTER_MEAN_HI and
                       CENTER_MEAN_LO <= cy_m <= CENTER_MEAN_HI and
                       cx_s < CENTER_STD_MIN and cy_s < CENTER_STD_MIN)
        results.append(Result("WARN", "W1", "Center bias / mode collapse",
                              not center_bias,
                              f"center mean=({cx_m:.3f},{cy_m:.3f}) std=({cx_s:.3f},{cy_s:.3f})"))
        areas = [g[2] for g in geoms]
        tiny = sum(a < TINY_AREA_FRAC for a in areas) / len(areas)
        full = sum(a > FULL_AREA_FRAC for a in areas) / len(areas)
        results.append(Result("WARN", "W2", "Box size outliers",
                              tiny <= SIZE_OUTLIER_MAX_RATIO and full <= SIZE_OUTLIER_MAX_RATIO,
                              f"tiny(<{TINY_AREA_FRAC})={tiny:.1%}, full(>{FULL_AREA_FRAC})={full:.1%}"))

    if grd:
        triples = [(g["image"], g["expr"], tuple(g["box"]) if g["box"] else None) for g in grd]
        dup_ratio = 1 - len(set(triples)) / len(triples)
        results.append(Result("WARN", "W3", "Exact duplicate (image,expr,box)",
                              dup_ratio <= DUP_MAX_RATIO, f"dup ratio={dup_ratio:.2%}"))
        exprs = [g["expr"] for g in grd if g["expr"]]
        div = len(set(exprs)) / len(exprs) if exprs else 0
        results.append(Result("WARN", "W4", "Expression diversity",
                              div >= EXPR_DIVERSITY_MIN, f"unique/total={div:.2f}"))

    # W6 source balance
    src_counts = Counter(r.get("source", "?") for r in rows)
    top_src, top_n = (src_counts.most_common(1)[0] if src_counts else ("-", 0))
    dom = top_n / n if n else 0
    results.append(Result("WARN", "W6", "Source balance",
                          dom <= SOURCE_DOMINATION,
                          f"top source '{top_src}'={dom:.1%} of rows"))

    # ---- stats for datasheet
    stats = {
        "rows": n,
        "per_source": dict(src_counts),
        "per_task": dict(Counter(r.get("task", "?") for r in rows)),
        "grounding_turns": n_grd,
        "valid_boxes": boxes_ok,
        "images_missing": missing,
        "images_sampled_opened": opened,
    }
    if geoms:
        def pct(xs, p):
            xs = sorted(xs)
            return round(xs[min(len(xs) - 1, int(p * len(xs)))], 4)
        ws = [g[0] for g in geoms]
        hs = [g[1] for g in geoms]
        ars = [g[5] for g in geoms]
        stats["box_geometry"] = {
            "w_p10_50_90": [pct(ws, .1), pct(ws, .5), pct(ws, .9)],
            "h_p10_50_90": [pct(hs, .1), pct(hs, .5), pct(hs, .9)],
            "area_p10_50_90": [pct([g[2] for g in geoms], .1),
                               pct([g[2] for g in geoms], .5),
                               pct([g[2] for g in geoms], .9)],
            "aspect_p10_50_90": [pct(ars, .1), pct(ars, .5), pct(ars, .9)],
            "center_mean": [round(statistics.mean([g[3] for g in geoms]), 3),
                            round(statistics.mean([g[4] for g in geoms]), 3)],
        }
    if txt_lens:
        h_lens = sorted(t[0] for t in txt_lens)
        g_lens = sorted(t[1] for t in txt_lens)
        stats["text_len_chars"] = {
            "human_p50_p95": [h_lens[len(h_lens) // 2], h_lens[int(0.95 * (len(h_lens) - 1))]],
            "gpt_p50_p95": [g_lens[len(g_lens) // 2], g_lens[int(0.95 * (len(g_lens) - 1))]],
        }
    return results, stats


# ------------------------------------------------------------- report
def write_reports(results, stats, jsonl, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blockers = [r for r in results if r.tier == "BLOCKER" and not r.passed]
    warns = [r for r in results if r.tier == "WARN" and not r.passed]
    verdict = "FAIL" if blockers else "PASS"

    lines = [f"# Dataset QC Report — {Path(jsonl).name}", ""]
    lines.append(f"## Verdict: {verdict}"
                 + (f" ({len(blockers)} BLOCKER)" if blockers else "")
                 + (f" · {len(warns)} WARN" if warns else ""))
    lines += ["", "## Gates", "", "| tier | id | check | result | detail |",
              "|---|---|---|---|---|"]
    for r in results:
        mark = "PASS" if r.passed else ("**FAIL**" if r.tier == "BLOCKER" else "WARN")
        lines.append(f"| {r.tier} | {r.cid} | {r.name} | {mark} | {r.detail} |")
    lines += ["", "## Datasheet", "", "```json",
              json.dumps(stats, ensure_ascii=False, indent=2), "```"]
    if warns:
        lines += ["", "## WARN details"]
        lines += [f"- **{r.cid} {r.name}**: {r.detail}" for r in warns]
    lines += ["", "## Recommendations"]
    if blockers:
        lines += [f"- Fix BLOCKER {r.cid} ({r.name}) before using this data." for r in blockers]
    else:
        lines.append("- No BLOCKER. Safe to merge; record WARN items in the datasheet.")

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(
        {"verdict": verdict, "results": [r.as_dict() for r in results], "stats": stats},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return verdict, blockers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--image-roots", default="{}",
                    help='JSON map, e.g. {"l0":"/path","stage2":"/path"}')
    ap.add_argument("--grounding-tasks", default="grounding",
                    help="comma list of task names treated as grounding")
    ap.add_argument("--coord-scale", type=float, default=1000.0)
    ap.add_argument("--sample-images", type=int, default=300)
    ap.add_argument("--eval-jsonl", default=None)
    ap.add_argument("--out-dir", default="./qc_out")
    args = ap.parse_args()

    rows, parse_errors = load_rows(args.jsonl)
    image_roots = json.loads(args.image_roots)
    gtasks = {t.strip() for t in args.grounding_tasks.split(",") if t.strip()}
    results, stats = run_all(rows, parse_errors, image_roots, gtasks,
                             args.coord_scale, args.sample_images, args.eval_jsonl)
    verdict, blockers = write_reports(results, stats, args.jsonl, args.out_dir)
    print(f"[dataset-qc] verdict={verdict} rows={len(rows)} "
          f"blockers={len(blockers)} -> {args.out_dir}/report.md")
    for r in results:
        if not r.passed:
            print(f"  [{r.tier}] {r.cid} {r.name}: {r.detail}")
    raise SystemExit(1 if blockers else 0)


if __name__ == "__main__":
    main()
