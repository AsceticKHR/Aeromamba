import json
from pathlib import Path
from typing import Dict, List, Any
import torch
from torch.utils.data import Dataset
from PIL import Image

class LLaVADataset(Dataset):
    """
    Dataset loader for LLaVA-Pretrain formatted JSON data.
    
    Each sample contains:
      - pixel_values: image tensor(s) (processed by vision transform)
      - input_ids: tokenised text [L] (prompt + response)
      - labels: tokenised labels [L] (prompt is masked with -100, response is active)
    """
    def __init__(
        self,
        data_root: str,
        tokenizer: Any,
        transform: Any,
        max_text_len: int = 128,
        json_name: str = "llava_subset.json"
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_len = max_text_len
        
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        json_path = self.data_root / json_name
        if not json_path.exists():
            raise FileNotFoundError(f"LLaVA JSON file not found at: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Per-sample data source tag ("general" when absent). Consumed by
        # WeightedRandomSampler in Stage2Trainer so the mixing ratio between
        # e.g. general instruction data and aerial spatial QA is an explicit
        # training knob instead of whatever the JSON happened to contain.
        self.sources: List[str] = [
            item.get("source", "general") for item in self.data
        ]
        source_counts: Dict[str, int] = {}
        for s in self.sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        print(f"[LLaVADataset] Loaded {len(self.data)} samples from {json_path}")
        print(f"[LLaVADataset] Source distribution: {source_counts}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 1. Load and transform image
        img_path = self.data_root / item["image"]
        if not img_path.exists():
            # Check if it exists inside a nested 'images' directory
            nested_path = self.data_root / "images" / item["image"]
            if nested_path.exists():
                img_path = nested_path
            else:
                # Check if it exists inside a train2017 subdirectory
                fallback_path = self.data_root / "train2017" / item["image"]
                if fallback_path.exists():
                    img_path = fallback_path
                else:
                    raise FileNotFoundError(
                        f"Image not found: {img_path} (nor nested {nested_path} nor fallback {fallback_path})"
                    )
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img)
        
        # 2. Extract conversation
        # Pretraining is always a 2-turn conversation: human (caption request) and gpt (caption response)
        convs = item["conversations"]
        human_text = ""
        gpt_text = ""
        
        for turn in convs:
            if turn["from"] == "human":
                # Clean out the <image> token and newlines
                human_text = turn["value"].replace("<image>", "").replace("\n", "").strip()
            elif turn["from"] == "gpt":
                gpt_text = turn["value"].strip()
                
        # Format Cobra / LLaVA pretrain prompt style:
        # "User: {human_text}\nAssistant: {gpt_text}"
        prompt = f"User: {human_text}\nAssistant: "
        response = gpt_text
        
        # Tokenize separately to mask prompt loss
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        # Add EOS token to the end of the response
        response_ids.append(self.tokenizer.eos_token_id)
        
        input_ids = prompt_ids + response_ids
        # Mask out prompt in labels with -100 (which PyTorch CrossEntropyLoss ignores)
        labels = [-100] * len(prompt_ids) + response_ids
        
        # Pad or truncate to max_text_len
        if len(input_ids) < self.max_text_len:
            padding_len = self.max_text_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            labels += [-100] * padding_len
        else:
            input_ids = input_ids[:self.max_text_len]
            labels = labels[:self.max_text_len]
            
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def llava_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function that handles both single and dual vision encoders."""
    from torch.utils.data.dataloader import default_collate

    sample_pv = batch[0]["pixel_values"]
    if isinstance(sample_pv, dict):
        keys = sample_pv.keys()
        collated_pv = {
            k: torch.stack([s["pixel_values"][k] for s in batch], dim=0)
            for k in keys
        }
    else:
        collated_pv = torch.stack([s["pixel_values"] for s in batch], dim=0)

    rest = default_collate(
        [{k: v for k, v in s.items() if k != "pixel_values"} for s in batch]
    )
    rest["pixel_values"] = collated_pv
    return rest
