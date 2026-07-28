"""L0 aerial CPT dataset — streaming JSONL in LLaVA conversation format.

Resolves image paths via build_report image_roots:
  stage2 -> Aeromamba/data
  l0     -> datasets/l0_cpt
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class L0CPTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Any,
        transform: Any,
        image_roots: Optional[Dict[str, str]] = None,
        max_text_len: int = 192,
        sources: Optional[List[str]] = None,
    ):
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_len = max_text_len
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        default_roots = {
            "stage2": str(Path("/root/autodl-tmp/Aeromamba/data")),
            "l0": str(self.jsonl_path.parent),
        }
        self.image_roots = {**default_roots, **(image_roots or {})}

        self.rows: List[dict] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if sources and row.get("source") not in sources:
                    continue
                self.rows.append(row)

        src_counts: Dict[str, int] = {}
        for r in self.rows:
            s = r.get("source", "?")
            src_counts[s] = src_counts.get(s, 0) + 1
        print(f"[L0CPTDataset] {len(self.rows)} rows from {self.jsonl_path}")
        print(f"[L0CPTDataset] sources={src_counts}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image(self, row: dict) -> Path:
        rel = row["image"]
        root_key = row.get("image_root", "l0")
        root = Path(self.image_roots.get(root_key, self.image_roots["l0"]))
        p = root / rel
        if p.exists():
            return p
        # common fallbacks
        for cand in (root / "images" / rel, Path(rel)):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"L0 image not found: {p} (root_key={root_key})")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        img = Image.open(self._resolve_image(row)).convert("RGB")
        pixel_values = self.transform(img)

        # flatten multi-turn into last Q/A supervised turn (CPT style)
        human_text, gpt_text = "", ""
        for turn in row["conversations"]:
            if turn["from"] == "human":
                human_text = turn["value"].replace("<image>", "").replace("\n", " ").strip()
            elif turn["from"] == "gpt":
                gpt_text = turn["value"].strip()

        prompt = f"User: {human_text}\nAssistant: "
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(gpt_text, add_special_tokens=False)["input_ids"]
        response_ids.append(self.tokenizer.eos_token_id)

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        if len(input_ids) < self.max_text_len:
            pad = self.max_text_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad
            labels += [-100] * pad
        else:
            input_ids = input_ids[: self.max_text_len]
            labels = labels[: self.max_text_len]

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


_BBOX_RE = __import__("re").compile(
    r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]"
)


def parse_bbox_0_1000(text: str):
    """Parse first [x1,y1,x2,y2] in 0-1000 coords → float tensor [4] in [0,1]."""
    m = _BBOX_RE.search(text or "")
    if not m:
        return None
    vals = [float(m.group(i)) for i in range(1, 5)]
    if max(vals) > 1000.0 + 1e-3:
        return None
    return torch.tensor([v / 1000.0 for v in vals], dtype=torch.float32)


class L0S2Dataset(Dataset):
    """S2 dataset: explode multi-turn Q/A and attach bbox labels when present.

    Designed to force vision use via grounding supervision (airspatial boxes)
    instead of short motion-phrase CE memorisation.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Any,
        transform: Any,
        image_roots: Optional[Dict[str, str]] = None,
        max_text_len: int = 192,
        sources: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        require_bbox: bool = False,
        exclude_sources: Optional[List[str]] = None,
    ):
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_len = max_text_len
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        default_roots = {
            "stage2": str(Path("/root/autodl-tmp/Aeromamba/data")),
            "l0": str(self.jsonl_path.parent),
        }
        self.image_roots = {**default_roots, **(image_roots or {})}
        exclude = set(exclude_sources or [])

        self.samples: List[dict] = []
        src_counts: Dict[str, int] = {}
        n_bbox = 0
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                src = row.get("source", "?")
                task = row.get("task", "?")
                if sources and src not in sources:
                    continue
                if exclude and src in exclude:
                    continue
                if tasks and task not in tasks:
                    continue

                turns = row.get("conversations") or []
                # pair consecutive human/gpt turns
                i = 0
                while i < len(turns):
                    if turns[i].get("from") != "human":
                        i += 1
                        continue
                    human = turns[i]["value"]
                    gpt = ""
                    if i + 1 < len(turns) and turns[i + 1].get("from") == "gpt":
                        gpt = turns[i + 1]["value"]
                        i += 2
                    else:
                        i += 1
                        continue
                    bbox = parse_bbox_0_1000(gpt)
                    if require_bbox and bbox is None:
                        continue
                    # also accept bbox asked in question for metric tasks? no — only answer boxes
                    self.samples.append({
                        "image": row["image"],
                        "image_root": row.get("image_root", "l0"),
                        "source": src,
                        "task": task,
                        "human": human.replace("<image>", "").replace("\n", " ").strip(),
                        "gpt": gpt.strip(),
                        "bbox": bbox,
                    })
                    src_counts[src] = src_counts.get(src, 0) + 1
                    if bbox is not None:
                        n_bbox += 1

        print(f"[L0S2Dataset] {len(self.samples)} turns from {self.jsonl_path} "
              f"(with_bbox={n_bbox})")
        print(f"[L0S2Dataset] sources={src_counts}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image(self, sample: dict) -> Path:
        rel = sample["image"]
        root_key = sample.get("image_root", "l0")
        root = Path(self.image_roots.get(root_key, self.image_roots["l0"]))
        p = root / rel
        if p.exists():
            return p
        for cand in (root / "images" / rel, Path(rel)):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"L0 image not found: {p}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img = Image.open(self._resolve_image(s)).convert("RGB")
        pixel_values = self.transform(img)

        prompt = f"User: {s['human']}\nAssistant: "
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(s["gpt"], add_special_tokens=False)["input_ids"]
        response_ids.append(self.tokenizer.eos_token_id)
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        if len(input_ids) < self.max_text_len:
            pad = self.max_text_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad
            labels += [-100] * pad
        else:
            input_ids = input_ids[: self.max_text_len]
            labels = labels[: self.max_text_len]

        has_bbox = s["bbox"] is not None
        bbox = s["bbox"] if has_bbox else torch.zeros(4, dtype=torch.float32)
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "bbox": bbox,
            "has_bbox": torch.tensor(1 if has_bbox else 0, dtype=torch.float32),
            "source_id": hash(s["source"]) % 10_000,
        }


def l0_s2_collate_fn(batch: List[Dict]) -> Dict:
    from torch.utils.data.dataloader import default_collate

    sample_pv = batch[0]["pixel_values"]
    if isinstance(sample_pv, dict):
        collated_pv = {
            k: torch.stack([s["pixel_values"][k] for s in batch], dim=0)
            for k in sample_pv.keys()
        }
    else:
        collated_pv = torch.stack([s["pixel_values"] for s in batch], dim=0)
    rest = default_collate(
        [{k: v for k, v in s.items() if k != "pixel_values"} for s in batch]
    )
    rest["pixel_values"] = collated_pv
    return rest
