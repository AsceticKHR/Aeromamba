import argparse
import sys
import time
from pathlib import Path
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model.uav_mamba_vla import AeroMambaVLA

def generate_text(model, pixels, prompt_text, device, max_new_tokens=50):
    prompt = f"User: {prompt_text}\nAssistant: "
    input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    
    with torch.no_grad():
        vis_patches = model._encode_vision(pixels)      # [B, N_pooled, D_v]
        vis_tokens = model.projector(vis_patches)       # [B, N_pooled, D_m]
        text_embs = model._embed_text(input_ids)        # [B, L, D_m]
        inputs_embeds = torch.cat([vis_tokens, text_embs], dim=1)  # [B, N_pooled + L, D_m]
        
        curr_embeds = inputs_embeds
        generated_ids = []
        for _ in range(max_new_tokens):
            out = model.mamba(inputs_embeds=curr_embeds)
            next_token_logits = out.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            token_id = next_token.item()
            
            if token_id == model.tokenizer.eos_token_id:
                break
                
            generated_ids.append(token_id)
            
            new_emb = model._embed_text(next_token.unsqueeze(0))
            curr_embeds = torch.cat([curr_embeds, new_emb], dim=1)
            
        return model.tokenizer.decode(generated_ids, skip_special_tokens=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Systematic evaluation running on device: {device}")

    # 1. Test images list
    images = [
        "000000001072.jpg",
        "000000001393.jpg",
        "000000001737.jpg",
        "000000001999.jpg",
        "000000003244.jpg"
    ]

    # 2. Prompts list
    prompts = [
        "Describe this image.",
        "What is in the image?",
        "What do you see?"
    ]

    # 3. Model Initialization
    print("Initializing model...")
    model = AeroMambaVLA(
        mamba_type="mamba-130m",
        vision_type="dinosiglip_so_384",
        use_token_pooling=True,
        pool_size=8
    )
    print("Applying LoRA...")
    model.configure_stage2(lora_r=16, lora_alpha=32)
    model.to(device)
    model.eval()

    # 4. Load checkpoint
    ckpt_path = ROOT / "checkpoints" / "remote_backup" / "stage2" / "best.pth"
    print(f"Loading weights from {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    results_dir = ROOT / "checkpoints" / "remote_backup" / "inference"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_markdown = results_dir / "stage2_systematic_results.md"

    # Start writing report
    report_lines = [
        "# AeroMamba-VLA Stage 2 Systematic Inference Test Results\n",
        f"**Device**: {device}",
        f"**Weights File**: `checkpoints/remote_backup/stage2/best.pth`",
        f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Missing Keys**: {len(missing)}",
        f"**Unexpected Keys**: {len(unexpected)}\n",
        "## 🔍 VQA Systematic Tests\n"
    ]

    for img_name in images:
        img_path = ROOT / "data" / img_name
        if not img_path.exists():
            print(f"Image {img_path} not found, skipping...")
            continue
            
        print(f"Processing {img_name}...")
        img = Image.open(img_path).convert("RGB")
        transformed = model.vision_encoder.transform(img)
        if isinstance(transformed, dict):
            pixels = {k: v.unsqueeze(0).to(device) for k, v in transformed.items()}
        else:
            pixels = transformed.unsqueeze(0).to(device)

        report_lines.append(f"### 🖼️ Image: `{img_name}`")
        report_lines.append("| Prompt | Generated VQA Response |")
        report_lines.append("| :--- | :--- |")

        for prompt in prompts:
            t0 = time.perf_counter()
            out_text = generate_text(model, pixels, prompt, device)
            latency = (time.perf_counter() - t0) * 1000.0
            print(f"Prompt: '{prompt}' -> {out_text} ({latency:.1f}ms)")
            report_lines.append(f"| {prompt} | {out_text} |")
        report_lines.append("\n" + "---" + "\n")

    with open(out_markdown, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nSystematic evaluation report written to {out_markdown}")

if __name__ == "__main__":
    main()
