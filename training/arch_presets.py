"""Shared architecture presets for Stage 1/2/3 training entry points."""

from __future__ import annotations

import argparse
from typing import Literal

StageName = Literal["stage1", "stage2", "stage3"]


def add_arch_preset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arch_preset",
        default="none",
        choices=[
            "none",
            "uav_lite_compatible",
            "uav_lite_siglip",
            "aeromamba_opt",
            "aeromamba_opt_fast",
        ],
        help=(
            "Architecture preset. Use 'aeromamba_opt' for the full-stage SigLIP2 + "
            "Perceiver(64) + Mamba-2-370M recipe."
        ),
    )


def apply_arch_preset(args: argparse.Namespace, stage: StageName) -> argparse.Namespace:
    if args.arch_preset == "none":
        return args

    if args.arch_preset == "uav_lite_compatible":
        args.vision_type = "dinosiglip_so_384"
        args.token_resampler = "perceiver"
        args.num_visual_queries = 32
        args.resampler_layers = 2
        args.resampler_heads = 8
        if stage == "stage3":
            args.action_head_type = "dynamics"
            args.stage3_train_lora = True
            args.no_amp = True
            if args.lr == 5e-4:
                args.lr = 5e-5
    elif args.arch_preset == "uav_lite_siglip":
        args.vision_type = "siglip2_base_384"
        if args.mamba_type == "mamba-130m":
            args.mamba_type = "mamba-2-370m"
        args.token_resampler = "perceiver"
        args.num_visual_queries = 32
        args.resampler_layers = 2
        args.resampler_heads = 8
        if stage == "stage3":
            args.action_head_type = "dynamics"
            args.stage3_train_lora = True
            args.no_amp = True
            if args.lr == 5e-4:
                args.lr = 5e-5
    elif args.arch_preset == "aeromamba_opt":
        args.vision_type = "siglip2_base_384"
        if args.mamba_type == "mamba-130m":
            args.mamba_type = "mamba-2-370m"
        args.token_resampler = "perceiver"
        args.num_visual_queries = 64
        args.resampler_layers = 2
        args.resampler_heads = 8
        args.proprio_dim = 8
        if getattr(args, "chunk_size", 5) == 5:
            args.chunk_size = 8
        if stage == "stage3":
            args.action_head_type = "mlp"
            args.no_amp = True
            if args.lr == 5e-4:
                args.lr = 5e-5
    elif args.arch_preset == "aeromamba_opt_fast":
        args.vision_type = "siglip2_base_p32_256"
        if args.mamba_type == "mamba-130m":
            args.mamba_type = "mamba-2-370m"
        args.token_resampler = "perceiver"
        args.num_visual_queries = 32
        args.resampler_layers = 2
        args.resampler_heads = 8
        args.proprio_dim = 8
        if getattr(args, "chunk_size", 5) == 5:
            args.chunk_size = 8
        if stage == "stage3":
            args.action_head_type = "mlp"
            args.no_amp = True
            if args.lr == 5e-4:
                args.lr = 5e-5

    return args
