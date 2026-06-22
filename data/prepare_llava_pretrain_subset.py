import json
import sys
from pathlib import Path

# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def main():
    root = Path(__file__).resolve().parent / "llava_pretrain"
    json_path = root / "blip_laion_cc_sbu_558k.json"
    out_path = root / "llava_pretrain_subset.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} does not exist!")
        sys.exit(1)
        
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("Filtering samples with valid images...")
    subset = []
    for item in data:
        img_path = root / item["image"]
        if img_path.exists():
            subset.append(item)
            if len(subset) == 200:
                break
                
    if len(subset) < 200:
        print(f"Warning: Only found {len(subset)} valid samples out of 200 requested.")
    else:
        print(f"Successfully found 200 valid samples.")
        
    print(f"Saving to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
