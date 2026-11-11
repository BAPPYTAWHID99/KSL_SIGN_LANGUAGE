#!/usr/bin/env python3
"""Create deterministic signer-independent train/validation/test manifests."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path


_WORD_PATTERN = re.compile(r"(?:^|_)WORD(?P<label>[A-Za-z0-9-]+)(?=_|$)", re.IGNORECASE)
_REAL_SIGNER_PATTERN = re.compile(r"(?:^|_)REAL(?P<signer>\d+)(?=_|$)", re.IGNORECASE)
_PREFIX_SIGNER_PATTERN = re.compile(r"^(?P<signer>\d{2,})(?:_|-)")


@dataclass(frozen=True)
class Sample:
    path: Path
    signer_id: str
    label: str


def parse_filename(path: Path) -> tuple[str, str]:
    """Return (signer_id, label) from supported KSL keypoint filenames.

    The AI-Hub-derived files encode the class as ``WORD####`` and the signer
    as ``REAL##``. Also accept ``01_<label>_...npy`` style filenames.
    """
    stem = path.stem
    word_match = _WORD_PATTERN.search(stem)
    real_signer_match = _REAL_SIGNER_PATTERN.search(stem)
    prefix_signer_match = _PREFIX_SIGNER_PATTERN.match(stem)

    if word_match:
        label = f"WORD{word_match.group('label')}"
    elif prefix_signer_match:
        remaining_parts = stem[len(prefix_signer_match.group(0)) :].split("_")
        if not remaining_parts or not remaining_parts[0]:
            raise ValueError(f"Cannot determine word label from filename: {path.name}")
        label = remaining_parts[0]
    else:
        raise ValueError(f"Cannot determine word label from filename: {path.name}")

    if real_signer_match:
        signer_id = real_signer_match.group("signer")
    elif prefix_signer_match:
        signer_id = prefix_signer_match.group("signer")
    else:
        raise ValueError(f"Cannot determine signer ID from filename: {path.name}")

    return signer_id, label


def split_signers(
    signer_ids: list[str], ratios: tuple[float, float, float], seed: int
) -> tuple[set[str], set[str], set[str]]:
    """Shuffle and apportion whole signers across train, validation, and test."""
    if len(signer_ids) < 3:
        raise ValueError(
            "At least 3 unique signers are required to create non-empty "
            "train, validation, and test splits."
        )

    counts = [ratio * len(signer_ids) for ratio in ratios]
    sizes = [math.floor(count) for count in counts]
    for index in sorted(
        range(len(ratios)), key=lambda i: counts[i] - sizes[i], reverse=True
    )[: len(signer_ids) - sum(sizes)]:
        sizes[index] += 1

    # Ensure each requested split is represented when there are enough signers.
    for index, ratio in enumerate(ratios):
        if ratio > 0 and sizes[index] == 0:
            donor = max(range(len(sizes)), key=lambda i: sizes[i])
            sizes[donor] -= 1
            sizes[index] = 1

    shuffled = sorted(signer_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = sizes[0]
    val_end = train_end + sizes[1]
    return (
        set(shuffled[:train_end]),
        set(shuffled[train_end:val_end]),
        set(shuffled[val_end:]),
    )


def build_splits(
    data_dir: Path, ratios: tuple[float, float, float], seed: int
) -> dict[str, list[Sample]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Training data directory does not exist: {data_dir}")

    files = sorted(path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".npy")
    if not files:
        raise ValueError(f"No .npy files found under: {data_dir}")

    samples: list[Sample] = []
    failures: list[str] = []
    for path in files:
        try:
            signer_id, label = parse_filename(path)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        samples.append(Sample(path.resolve(), signer_id, label))

    if failures:
        preview = "\n".join(f"  - {failure}" for failure in failures[:10])
        more = f"\n  ... and {len(failures) - 10} more" if len(failures) > 10 else ""
        raise ValueError(
            f"Could not parse signer/label from {len(failures)} of {len(files)} "
            f"filenames. No split files were written.\n{preview}{more}"
        )

    signers = sorted({sample.signer_id for sample in samples})
    train_signers, val_signers, test_signers = split_signers(signers, ratios, seed)

    splits = {
        "train": [sample for sample in samples if sample.signer_id in train_signers],
        "val": [sample for sample in samples if sample.signer_id in val_signers],
        "test": [sample for sample in samples if sample.signer_id in test_signers],
    }

    # Keep this explicit: signer leakage invalidates the intended evaluation.
    signer_sets = [
        {sample.signer_id for sample in splits[name]}
        for name in ("train", "val", "test")
    ]
    if any(
        signer_sets[left] & signer_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("Internal error: signer IDs overlap between splits.")

    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic, signer-independent KSL data splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("keypoint_cache") / "1.Training",
        help="Directory containing training .npy files (default: keypoint_cache/1.Training).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for train_split.json, val_split.json, and test_split.json.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.70, 0.15, 0.15),
        help="Train/validation/test signer proportions (default: 0.70 0.15 0.15).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    if any(ratio < 0 for ratio in args.ratios) or not math.isclose(
        sum(args.ratios), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        parser.error("--ratios must be non-negative and sum to 1.")
    if sum(ratio > 0 for ratio in args.ratios) < 3:
        parser.error("All three split ratios must be greater than zero.")
    return args


def main() -> int:
    args = parse_args()
    try:
        splits = build_splits(args.data_dir, tuple(args.ratios), args.seed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    all_samples = [sample for split in splits.values() for sample in split]
    signers = {sample.signer_id for sample in all_samples}
    classes = {sample.label for sample in all_samples}
    print(f"Total files: {len(all_samples)}")
    print(f"Unique signers: {len(signers)}")
    print(f"Unique classes (filename word IDs): {len(classes)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        output_path = args.output_dir / f"{split_name}_split.json"
        output_path.write_text(
            json.dumps([str(sample.path) for sample in samples], indent=2) + "\n",
            encoding="utf-8",
        )
        split_signer_count = len({sample.signer_id for sample in samples})
        print(
            f"{split_name.capitalize()}: {len(samples)} files "
            f"from {split_signer_count} signers -> {output_path.resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
