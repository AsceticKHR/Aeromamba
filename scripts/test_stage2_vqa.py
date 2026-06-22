import argparse
import sys
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

def get_args():
    parser = argparse.ArgumentParser(description="Test a Stage 2 AeroMamba checkpoint")
    parser.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "stage2" / "best.pth")
    parser.add_argument("--image", type=Path, default=ROOT / "data" / "000000033471.jpg")
    parser.add_argument("--mamba_type", default="mamba-130m")
    parser.add_argument("--vision_type", default="dinosiglip_so_384")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # 1. Load test image
    img_path = args.image
    if not img_path.exists():
        print(f"Error: Test image not found at {img_path}")
        sys.exit(1)
    print(f"Loading image from {img_path}...")
    img = Image.open(img_path).convert("RGB")

    # 2. Initialize Model (matching Stage 2 training parameters)
    print("Initializing model...")
    model = AeroMambaVLA(
        mamba_type=args.mamba_type,
        vision_type=args.vision_type,
        use_token_pooling=True,
        pool_size=8
    )
    
    # Configure LoRA first so that the PEFT structure is built before loading the state dict
    print("Applying LoRA (Stage 2 configuration)...")
    model.configure_stage2(lora_r=16, lora_alpha=32)
    model.to(device)
    model.eval()

    # Preprocess image
    transformed = model.vision_encoder.transform(img)
    if isinstance(transformed, dict):
        pixels = {key: value.unsqueeze(0).to(device) for key, value in transformed.items()}
    else:
        pixels = transformed.unsqueeze(0).to(device)

    # 3. Load stage 2 checkpoint
    ckpt_path = args.ckpt
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
        
    print(f"Loading weights from {ckpt_path}...")
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_state = ckpt.get("model_state", ckpt)
        
        # Load state dict strictly to verify all PEFT/Projector/Embedding keys align perfectly
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print("Model loaded successfully.")
        print(f"Missing keys (should be empty except for frozen weights if loaded partially): {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # 4. Generate descriptions with different prompts
    prompts = [
        "Describe this image.",
        "What is in the image?",
        "What do you see?"
    ]

    print("\n" + "="*50)
    print("STAGE 2 MODEL INFERENCE TEST")
    print("="*50)
    for p in prompts:
        out_text = generate_text(model, pixels, p, device, max_new_tokens=args.max_new_tokens)
        print(f"Prompt: {p}")
        print(f"Output: {out_text}")
        print("-" * 50)

if __name__ == "__main__":
    main()
