import json
import sys
from pathlib import Path

# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    root = Path(__file__).resolve().parent / "llava_instruct"
    json_path = root / "llava_instruct_150k.json"
    out_path = root / "llava_instruct_subset.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} does not exist!")
        return
        
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Extract first 100 samples
    subset = data[:100]
    
    print(f"Saving {len(subset)} samples to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
