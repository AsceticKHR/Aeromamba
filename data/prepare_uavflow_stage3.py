"""
Build AeroMamba Stage-3 training folders from official UAV-Flow parquet files.

The official UAV-Flow script groups rows by trajectory id and writes:
    <output>/<trajectory_id>/000000.jpg
    <output>/<trajectory_id>/log.json

This script keeps that layout, adds validation metadata, and preserves the
fields needed by AeroMamba's Stage-3 loader:
    raw_logs, preprocessed_logs, instruction, instruction_unified, length
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from PIL import Image


@dataclass
class TrajectoryBuffer:
    trajectory_id: str
    raw_logs: List[Any]
    preprocessed_logs: List[Any]
    instruction: str
    instruction_unified: str
    length: int
    images: Dict[int, Any] = field(default_factory=dict)
    source_files: Set[str] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UAV-Flow Stage-3 folders.")
    parser.add_argument(
        "--parquet_glob",
        required=True,
        help="Local parquet glob, e.g. /root/autodl-tmp/datasets/uav-flow/train-*.parquet",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output folder for official UAV-Flow trajectory directories.",
    )
    parser.add_argument("--split", default="train", help="Split name stored in metadata.")
    parser.add_argument("--chunk_size", type=int, default=5, help="Minimum usable trajectory length.")
    parser.add_argument("--max_trajectories", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output_dir before writing.")
    parser.add_argument("--verify_images", action="store_true", help="Re-open saved JPEGs while writing.")
    parser.add_argument("--hf_cache_dir", default=None, help="HuggingFace datasets cache directory.")
    parser.add_argument(
        "--delete_parquet_after_success",
        action="store_true",
        help=(
            "Delete a parquet shard only after every trajectory seen in that shard "
            "has been written or safely skipped. Use only after testing."
        ),
    )
    parser.add_argument(
        "--manifest_name",
        default="manifest.jsonl",
        help="Manifest filename under output_dir/metadata.",
    )
    return parser.parse_args()


def expand_parquet_files(parquet_glob: str) -> List[Path]:
    files = [Path(path) for path in sorted(glob.glob(parquet_glob))]
    if not files:
        raise FileNotFoundError(f"No parquet files matched {parquet_glob!r}")
    return files


def iter_parquet_rows(parquet_file: Path, cache_dir: Optional[str]) -> Iterator[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets with `pip install datasets pyarrow`.") from exc

    dataset = load_dataset(
        "parquet",
        data_files=[str(parquet_file)],
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    for row in dataset:
        yield row


def parse_log(log_value: Any) -> Dict[str, Any]:
    if isinstance(log_value, dict):
        return log_value
    if isinstance(log_value, str):
        try:
            payload = json.loads(log_value)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def trajectory_length(payload: Dict[str, Any]) -> int:
    raw_logs = payload.get("raw_logs") or []
    preprocessed_logs = payload.get("preprocessed_logs") or []
    return int(payload.get("length") or len(preprocessed_logs) or len(raw_logs))


def coerce_image(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = image_value.get("path")
        if image_path:
            return Image.open(image_path).convert("RGB")
    raise ValueError("Row does not contain a valid image payload.")


def save_image(image_value: Any, path: Path, verify: bool) -> None:
    image = coerce_image(image_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)
    if verify:
        with Image.open(path) as saved:
            saved.verify()


def build_buffer(row: Dict[str, Any]) -> Optional[TrajectoryBuffer]:
    payload = parse_log(row.get("log", ""))
    length = trajectory_length(payload)
    if length <= 0:
        return None
    trajectory_id = str(row["id"])
    return TrajectoryBuffer(
        trajectory_id=trajectory_id,
        raw_logs=payload.get("raw_logs") or [],
        preprocessed_logs=payload.get("preprocessed_logs") or [],
        instruction=payload.get("instruction") or "",
        instruction_unified=payload.get("instruction_unified") or payload.get("instruction") or "",
        length=length,
    )


def write_trajectory(
    buffer: TrajectoryBuffer,
    output_dir: Path,
    manifest_file,
    split: str,
    chunk_size: int,
    verify_images: bool,
) -> bool:
    if buffer.length < chunk_size:
        return False
    if len(buffer.images) < buffer.length:
        return False

    traj_dir = output_dir / buffer.trajectory_id
    traj_dir.mkdir(parents=True, exist_ok=True)
    for output_idx, frame_idx in enumerate(sorted(buffer.images)):
        save_image(buffer.images[frame_idx], traj_dir / f"{output_idx:06d}.jpg", verify_images)

    log_payload = {
        "id": buffer.trajectory_id,
        "raw_logs": buffer.raw_logs,
        "preprocessed_logs": buffer.preprocessed_logs,
        "instruction": buffer.instruction,
        "instruction_unified": buffer.instruction_unified,
        "length": buffer.length,
        "split": split,
    }
    log_path = traj_dir / "log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, ensure_ascii=False)

    manifest = {
        "id": buffer.trajectory_id,
        "split": split,
        "length": buffer.length,
        "num_images": len(buffer.images),
        "instruction": buffer.instruction,
        "instruction_unified": buffer.instruction_unified,
        "log_path": str(log_path.relative_to(output_dir)).replace("\\", "/"),
    }
    manifest_file.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    return True


def prepare_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    buffers: Dict[str, TrajectoryBuffer] = {}
    stats = {
        "rows": 0,
        "written_trajectories": 0,
        "skipped_short": 0,
        "bad_rows": 0,
        "incomplete_buffers": 0,
        "parquet_files": 0,
        "deleted_parquet_files": 0,
    }
    processed_files: Set[str] = set()
    deleted_files: Set[str] = set()
    parquet_files = expand_parquet_files(args.parquet_glob)
    stats["parquet_files"] = len(parquet_files)

    manifest_path = metadata_dir / args.manifest_name
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for parquet_file in parquet_files:
            source_name = str(parquet_file)
            processed_files.add(source_name)
            for row in iter_parquet_rows(parquet_file, args.hf_cache_dir):
                stats["rows"] += 1
                try:
                    trajectory_id = str(row["id"])
                    frame_idx = int(row["frame_idx"])
                    if trajectory_id not in buffers:
                        buffer = build_buffer(row)
                        if buffer is None:
                            stats["bad_rows"] += 1
                            continue
                        buffers[trajectory_id] = buffer
                    buffers[trajectory_id].source_files.add(source_name)
                    buffers[trajectory_id].images[frame_idx] = row["image"]
                    buffer = buffers[trajectory_id]
                    if len(buffer.images) >= buffer.length:
                        if buffer.length < args.chunk_size:
                            stats["skipped_short"] += 1
                        elif write_trajectory(
                            buffer,
                            output_dir,
                            manifest_file,
                            args.split,
                            args.chunk_size,
                            args.verify_images,
                        ):
                            stats["written_trajectories"] += 1
                        buffers.pop(trajectory_id, None)
                        if (
                            args.max_trajectories is not None
                            and stats["written_trajectories"] >= args.max_trajectories
                        ):
                            break
                except Exception:
                    stats["bad_rows"] += 1

            if args.delete_parquet_after_success:
                blocked_files = {
                    source_file
                    for buffer in buffers.values()
                    for source_file in buffer.source_files
                }
                for candidate in sorted(processed_files - deleted_files - blocked_files):
                    path = Path(candidate)
                    if path.exists():
                        path.unlink()
                        stats["deleted_parquet_files"] += 1
                    deleted_files.add(candidate)

            if (
                args.max_trajectories is not None
                and stats["written_trajectories"] >= args.max_trajectories
            ):
                break

    stats["incomplete_buffers"] = len(buffers)
    summary_path = metadata_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def main() -> None:
    args = parse_args()
    stats = prepare_dataset(args)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
