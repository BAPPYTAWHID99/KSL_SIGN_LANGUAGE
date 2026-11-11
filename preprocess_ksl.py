"""Convert MediaPipe Holistic KSL arrays to 75-landmark position/velocity features."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

import numpy as np


NUM_LANDMARKS = 75
NUM_COORDINATES = 3
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDINATES * 2
DEFAULT_ARCHIVE = Path("keypoint_dataset") / "keypoint_cache.zip"
DEFAULT_OUTPUT_DIR = Path("C:/KSL_Processed_450")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _iter_zip_files(
    archive: zipfile.ZipFile, subset: str, output_dir: Path
) -> Iterator[tuple[str, str]]:
    for info in archive.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".npy"):
            continue
        member = PurePosixPath(info.filename)
        if subset not in member.parts:
            continue
        subset_index = member.parts.index(subset)
        relative_path = Path(subset, *member.parts[subset_index + 1 :])
        target_path = (output_dir / relative_path).resolve()
        if not _is_relative_to(target_path, output_dir):
            raise ValueError(f"Archive member escapes the output directory: {info.filename}")
        yield info.filename, relative_path.as_posix()


def _iter_directory_files(
    source_dir: Path, subset: str, output_dir: Path
) -> Iterator[tuple[Path, str]]:
    if source_dir.name == subset:
        subset_root = source_dir
    else:
        candidates = [path for path in source_dir.rglob(subset) if path.is_dir()]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one {subset!r} directory under {source_dir}, "
                f"found {len(candidates)}."
            )
        subset_root = candidates[0]

    if subset_root.resolve() == output_dir:
        raise ValueError("The output directory must not be the input training directory.")
    for path in sorted(subset_root.rglob("*.npy")):
        target_path = (output_dir / subset / path.relative_to(subset_root)).resolve()
        if _is_relative_to(target_path, output_dir):
            yield path, (Path(subset) / path.relative_to(subset_root)).as_posix()


def _convert_array(source: BinaryIO, source_name: str, max_frames: int) -> np.ndarray:
    try:
        sequence = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load NumPy array {source_name}: {exc}") from exc

    if sequence.ndim != 3 or sequence.shape[0] == 0:
        raise ValueError(
            f"Expected a non-empty (frames, landmarks, channels) array in "
            f"{source_name}, got {sequence.shape}."
        )
    if sequence.shape[1] < NUM_LANDMARKS or sequence.shape[2] < NUM_COORDINATES:
        raise ValueError(
            f"Expected at least {NUM_LANDMARKS} landmarks and {NUM_COORDINATES} "
            f"coordinate channels in {source_name}, got {sequence.shape}."
        )

    positions = np.asarray(
        sequence[:max_frames, :NUM_LANDMARKS, :NUM_COORDINATES], dtype=np.float32
    ).reshape(min(sequence.shape[0], max_frames), NUM_LANDMARKS * NUM_COORDINATES)
    if not np.isfinite(positions).all():
        raise ValueError(f"Keypoint array contains NaN or infinite values: {source_name}")

    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]
    features = np.concatenate((positions, velocities), axis=1)
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError(
            f"Internal feature dimension error for {source_name}: {features.shape}"
        )
    return features


def _write_array(output_path: Path, features: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as file:
            np.save(file, features, allow_pickle=False)
        temporary_path.replace(output_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def _process(
    source_path: Path, output_dir: Path, subset: str, max_frames: int, limit: int | None
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    processed_count = 0

    if source_path.is_file() and zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as archive:
            file_entries = _iter_zip_files(archive, subset, output_dir)
            for member, relative_name in file_entries:
                output_path = output_dir / relative_name
                with archive.open(member) as source:
                    features = _convert_array(source, member, max_frames)
                _write_array(output_path, features)
                processed_count += 1
                if processed_count % 10000 == 0:
                    print(f"Processed {processed_count} files...", flush=True)
                if limit is not None and processed_count >= limit:
                    break
    elif source_path.is_dir():
        file_entries = _iter_directory_files(source_path, subset, output_dir)
        for source_file, relative_name in file_entries:
            with source_file.open("rb") as source:
                features = _convert_array(source, str(source_file), max_frames)
            _write_array(output_dir / relative_name, features)
            processed_count += 1
            if processed_count % 10000 == 0:
                print(f"Processed {processed_count} files...", flush=True)
            if limit is not None and processed_count >= limit:
                break
    else:
        raise FileNotFoundError(
            f"Input must be an existing ZIP archive or extracted directory: {source_path}"
        )

    if processed_count == 0:
        raise ValueError(f"No .npy files found in the requested {subset!r} subset.")
    print(
        f"Processed {processed_count} files from {source_path} "
        f"into {output_dir}; each frame has {FEATURE_DIM} features.",
        flush=True,
    )
    return processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep pose/left-hand/right-hand landmarks (indices 0:75), drop "
            "channel 4, append temporal deltas, and write 450-feature arrays."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset", default="1.Training")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many files (useful for a smoke test).",
    )
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    return args


def main() -> int:
    args = parse_args()
    try:
        _process(args.source, args.output_dir, args.subset, args.max_frames, args.limit)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
