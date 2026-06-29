"""
Common Trainer base class for all AeroMamba-VLA training stages.

Each stage only needs to override three methods:
    configure_model(model)            — which parameters to train
    compute_loss(model, batch, device)— stage-specific loss function
    load_pretrained(model)            — (optional) load prior stage checkpoint

Everything else (DataLoader, optimizer, scheduler, AMP, epoch loop,
checkpoint saving, logging) is handled here.

Usage:
    class Stage1Trainer(BaseTrainer):
        def configure_model(self, model): model.configure_stage1()
        def compute_loss(self, model, batch, device): ...

    trainer = Stage1Trainer(args)
    trainer.run()
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Reconfigure stdout/stderr to UTF-8 on Windows (default is CP1252 in PowerShell)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.uav_mamba_vla import AeroMambaVLA
from data.dataset import UAVFlowDataset, UAVFlowHFDataset, DummyUAVDataset, aero_collate_fn


class BaseTrainer(ABC):
    """
    Abstract base trainer shared by all three training stages.

    Subclasses must implement:
        configure_model(model)
        compute_loss(model, batch, device) → (loss, detail_dict)
    """

    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")

    # ─────────────────────────────────────────────────────────────────────────
    # Abstract interface
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def configure_model(self, model: AeroMambaVLA) -> None:
        """Set which parameters are trainable for this stage."""
        ...

    @abstractmethod
    def compute_loss(
        self,
        model:  AeroMambaVLA,
        batch:  dict,
        device: torch.device,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute and return (loss_tensor, detail_dict)."""
        ...

    def load_pretrained(self, model: AeroMambaVLA) -> None:
        """Override in subclasses to load a checkpoint from a prior stage."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset  (override for custom data pipelines)
    # ─────────────────────────────────────────────────────────────────────────

    def get_collate_fn(self):
        """Return the collate function to use for DataLoaders.
        
        Override in subclasses to use a different collate function.
        Default: aero_collate_fn (for UAVFlowDataset / DummyUAVDataset).
        """
        return aero_collate_fn

    def get_dataset(self, model: AeroMambaVLA):
        """
        Build train/val datasets.

        Falls back to DummyUAVDataset if --dummy is set or data_root is empty.
        Returns (train_dataset, val_dataset).
        """
        args = self.args
        hf_dataset = getattr(args, "hf_dataset", "")
        if hf_dataset:
            print(f"[Trainer] Using HuggingFace UAV-Flow dataset: {hf_dataset}")
            ds = UAVFlowHFDataset(
                dataset_name=hf_dataset,
                split=getattr(args, "hf_split", "train"),
                data_files=getattr(args, "hf_data_files", None),
                cache_dir=getattr(args, "hf_cache_dir", None),
                tokenizer=model.tokenizer,
                transform=model.vision_encoder.transform,
                chunk_size=getattr(args, "chunk_size", 5),
                max_text_len=getattr(args, "max_text_len", 64),
                instruction=getattr(
                    args,
                    "instruction",
                    "Navigate the UAV along the planned trajectory.",
                ),
                pos_scale=getattr(args, "pos_scale", 100.0),
                aug_flip=getattr(args, "aug_flip", False),
            )
        elif getattr(args, "dummy", False) or not getattr(args, "data_root", ""):
            print("[Trainer] Using DummyUAVDataset (no real data)")
            vision_type = getattr(args, "vision_type", "dinosiglip_so_384")
            dual = vision_type.startswith("dino") and ("siglip" in vision_type)
            ds = DummyUAVDataset(
                size=getattr(args, "dummy_size", 2000),
                chunk_size=getattr(args, "chunk_size", 5),
                max_text_len=getattr(args, "max_text_len", 64),
                dual_vision=dual,
            )
        else:
            ds = UAVFlowDataset(
                data_root=args.data_root,
                tokenizer=model.tokenizer,
                transform=model.vision_encoder.transform,
                chunk_size=getattr(args, "chunk_size", 5),
                max_text_len=getattr(args, "max_text_len", 64),
            )

        val_frac = getattr(args, "val_frac", 0.1)
        n_val    = max(1, int(len(ds) * val_frac))
        n_train  = len(ds) - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        if getattr(ds, "sequential_loading_preferred", False):
            train_ds.indices.sort()
            val_ds.indices.sort()
        print(f"[Trainer] Dataset: {n_train} train  |  {n_val} val")
        return train_ds, val_ds

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop internals
    # ─────────────────────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        model:     AeroMambaVLA,
        loader:    DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler:    torch.amp.GradScaler,
        epoch:     int,
    ) -> float:
        model.train()
        total_loss = 0.0
        t0         = time.time()
        log_every  = getattr(self.args, "log_every", 20)
        max_steps  = getattr(self.args, "max_steps", None)

        for bidx, batch in enumerate(loader):
            if max_steps and bidx >= max_steps:
                print(f"  [Ep{epoch:02d}] Reached max_steps={max_steps}. Stopping epoch early.")
                break
            optimizer.zero_grad(set_to_none=True)

            amp_enabled = (self.device.type == "cuda") and not getattr(self.args, "no_amp", False)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, detail = self.compute_loss(model, batch, self.device)

            if not torch.isfinite(loss):
                print(f"  [Ep{epoch:02d} | {bidx+1:05d}/{len(loader)}] WARNING non-finite loss; skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if (bidx + 1) % log_every == 0:
                detail_str = "  ".join(f"{k}={v:.4f}" for k, v in detail.items())
                elapsed    = time.time() - t0
                print(
                    f"  [Ep{epoch:02d} | {bidx+1:05d}/{len(loader)}] "
                    f"loss={loss.item():.4f}  {detail_str}  ({elapsed:.1f}s)"
                )
                t0 = time.time()

            save_every_steps = getattr(self.args, "save_every_steps", None)
            if save_every_steps and (bidx + 1) % save_every_steps == 0:
                save_dir = Path(getattr(self.args, "save_dir", "./checkpoints"))
                self._save_checkpoint(
                    str(save_dir / "latest.pth"),
                    model,
                    optimizer,
                    epoch,
                    float("inf"),
                )

        steps_run = min(max_steps, len(loader)) if (max_steps and len(loader) > max_steps) else len(loader)
        return total_loss / max(steps_run, 1)

    @torch.no_grad()
    def _validate(
        self,
        model:  AeroMambaVLA,
        loader: DataLoader,
    ) -> dict:
        model.eval()
        total_loss = 0.0
        metrics_sum = {}
        max_val_steps = getattr(self.args, "max_val_steps", 100)
        steps_run = 0
        for batch in loader:
            if max_val_steps and steps_run >= max_val_steps:
                break
            amp_enabled = (self.device.type == "cuda") and not getattr(self.args, "no_amp", False)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, detail = self.compute_loss(model, batch, self.device)
            if not torch.isfinite(loss):
                print("  [Val] WARNING non-finite loss; skipping batch")
                continue
            total_loss += loss.item()
            for k, v in detail.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            steps_run += 1
                
        n = max(steps_run, 1)
        res = {"loss": total_loss / n}
        for k, v in metrics_sum.items():
            res[k] = v / n
        return res

    @staticmethod
    def _save_checkpoint(
        path:     str,
        model:    AeroMambaVLA,
        optimizer: torch.optim.Optimizer,
        epoch:    int,
        best_val: float,
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch":       epoch,
            "best_val":    best_val,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        }, path)
        print(f"  ✓ Saved → {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> AeroMambaVLA:
        args = self.args

        # ── Build model ───────────────────────────────────────────────────────
        model = AeroMambaVLA(
            mamba_type=getattr(args, "mamba_type",   "mamba-370m"),
            vision_type=getattr(args, "vision_type",  "dinosiglip_so_384"),
            chunk_size=getattr(args, "chunk_size",   5),
            use_token_pooling=getattr(args, "use_token_pooling", False),
            pool_size=getattr(args, "pool_size", 8),
            token_resampler=getattr(args, "token_resampler", "none"),
            num_visual_queries=getattr(args, "num_visual_queries", 32),
            resampler_layers=getattr(args, "resampler_layers", 2),
            resampler_heads=getattr(args, "resampler_heads", 8),
            action_head_type=getattr(args, "action_head_type", "mlp"),
            action_bound=getattr(args, "action_bound", 1.0),
        )

        # ── Load prior-stage weights (optional) ───────────────────────────────
        self.load_pretrained(model)

        # ── Configure trainable parameters ────────────────────────────────────
        self.configure_model(model)
        model.print_trainable_params()
        model.to(self.device)

        # ── Datasets & loaders ─────────────────────────────────────────────────
        train_ds, val_ds = self.get_dataset(model)
        workers = getattr(args, "workers", 4)
        batch   = getattr(args, "batch",   16)
        collate_fn = self.get_collate_fn()
        pin = (self.device.type == "cuda")
        base_train_ds = getattr(train_ds, "dataset", train_ds)
        shuffle_train = not getattr(base_train_ds, "sequential_loading_preferred", False)
        train_loader = DataLoader(
            train_ds, batch_size=batch, shuffle=shuffle_train,
            num_workers=workers, pin_memory=pin, drop_last=True,
            collate_fn=collate_fn, persistent_workers=(workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch, shuffle=False,
            num_workers=workers, pin_memory=pin,
            collate_fn=collate_fn, persistent_workers=(workers > 0),
        )

        # ── Optimizer & scheduler ──────────────────────────────────────────────
        lr = getattr(args, "lr", 1e-4)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=1e-4,
        )
        epochs = getattr(args, "epochs", 10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.1,
        )
        try:
            scaler = torch.amp.GradScaler(
                enabled=(self.device.type == "cuda") and not getattr(args, "no_amp", False)
            )
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler(
                enabled=(self.device.type == "cuda") and not getattr(args, "no_amp", False)
            )

        save_dir    = Path(getattr(args, "save_dir", "./checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)
        start_epoch = 1
        best_val    = float("inf")

        # ── Optional: resume from checkpoint ──────────────────────────────────
        resume = getattr(args, "resume", None)
        if resume and Path(resume).exists():
            ckpt = torch.load(resume, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val    = ckpt["best_val"]
            print(f"[Trainer] Resumed from epoch {ckpt['epoch']}  "
                  f"(best_val={best_val:.4f})")

        # ── Training loop ──────────────────────────────────────────────────────
        for epoch in range(start_epoch, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            train_loss = self._train_epoch(model, train_loader, optimizer, scaler, epoch)
            val_metrics = self._validate(model, val_loader)
            val_loss = val_metrics.pop("loss")
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items())
            print(
                f"  >> train={train_loss:.4f}  val_loss={val_loss:.4f}  {val_str}  lr={lr_now:.2e}"
            )

            # Save best checkpoint
            if val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(
                    str(save_dir / "best.pth"), model, optimizer, epoch, best_val
                )

            # Periodic checkpoint every 5 epochs
            if epoch % 5 == 0:
                self._save_checkpoint(
                    str(save_dir / f"epoch_{epoch:03d}.pth"),
                    model, optimizer, epoch, best_val,
                )

        print(f"\n[Trainer] Training complete. Best val loss: {best_val:.4f}")
        print(f"[Trainer] Best checkpoint: {save_dir}/best.pth")
        return model
