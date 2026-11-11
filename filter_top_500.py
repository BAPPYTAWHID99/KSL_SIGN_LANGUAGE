"""Filter the existing signer-independent KSL splits to their top 500 classes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from create_splits import parse_filename


SPLIT_NAMES = ("train", "val", "test")


def read_split(project_dir: Path, split_name: str) -> list[tuple[Path, str, str]]:
    manifest_path = project_dir / f"{split_name}_split.json"
    try:
        contents = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read split manifest {manifest_path}: {exc}") from exc

    if not isinstance(contents, list) or not contents:
        raise ValueError(f"Split manifest must contain a non-empty JSON list: {manifest_path}")

    parsed: list[tuple[Path, str, str]] = []
    for item in contents:
        if not isinstance(item, str) or not item:
            raise ValueError(f"Invalid path entry in {manifest_path}: {item!r}")
        source_path = Path(item)
        if source_path.suffix.lower() != ".npy":
            raise ValueError(f"Expected a .npy sample path in {manifest_path}: {item}")
        try:
            signer_id, label = parse_filename(source_path)
        except ValueError as exc:
            raise ValueError(f"Could not parse {item} from {manifest_path}: {exc}") from exc
        parsed.append((source_path, signer_id, label))
    return parsed


def verify_signer_independence(
    splits: dict[str, list[tuple[Path, str, str]]],
) -> dict[str, set[str]]:
    signer_sets = {
        name: {signer_id for _, signer_id, _ in samples}
        for name, samples in splits.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = signer_sets[left] & signer_sets[right]
        if overlap:
            raise ValueError(
                f"Signer leakage between {left} and {right} splits: {sorted(overlap)}"
            )
    return signer_sets


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Copy the top-frequency KSL classes while preserving existing signer splits."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=project_dir,
        help="Directory containing train_split.json, val_split.json, and test_split.json.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(r"C:\KSL_Processed_450"),
        help="Root directory containing the source .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "KSL_Processed_500",
        help="Destination directory (must not already contain .npy files).",
    )
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive.")
    return args


def main() -> int:
    args = parse_args()
    try:
        source_root = args.source_dir.resolve(strict=True)
        if not source_root.is_dir():
            raise ValueError(f"Source root is not a directory: {source_root}")

        splits = {name: read_split(args.project_dir, name) for name in SPLIT_NAMES}
        original_signers = verify_signer_independence(splits)

        train_counts = Counter(label for _, _, label in splits["train"])
        if len(train_counts) < args.top_k:
            raise ValueError(
                f"Requested top {args.top_k} classes, but train_split.json contains "
                f"only {len(train_counts)} classes."
            )
        top_classes = {
            label
            for label, _ in sorted(
                train_counts.items(), key=lambda item: (-item[1], item[0])
            )[: args.top_k]
        }

        filtered_splits = {
            name: [
                sample
                for sample in samples
                if sample[2] in top_classes
            ]
            for name, samples in splits.items()
        }
        if any(not filtered_splits[name] for name in SPLIT_NAMES):
            empty_splits = [name for name in SPLIT_NAMES if not filtered_splits[name]]
            raise ValueError(f"Top-class filtering produced empty split(s): {empty_splits}")
        filtered_signers = verify_signer_independence(filtered_splits)

        output_dir = args.output_dir.resolve()
        copy_plan: dict[Path, Path] = {}
        output_manifests: dict[str, list[str]] = {}
        for split_name, samples in filtered_splits.items():
            manifest_paths: list[str] = []
            for source_path, _, _ in samples:
                try:
                    resolved_source = source_path.resolve(strict=True)
                    relative_path = resolved_source.relative_to(source_root)
                except (OSError, ValueError) as exc:
                    raise ValueError(
                        f"Sample path must exist under {source_root}: {source_path}"
                    ) from exc
                destination_path = output_dir / relative_path
                copy_plan[resolved_source] = destination_path
                manifest_paths.append(str(destination_path))
            output_manifests[split_name] = manifest_paths

        if output_dir.exists():
            planned_destinations = set(copy_plan.values())
            unexpected_files = [
                path
                for path in output_dir.rglob("*.npy")
                if path.resolve() not in planned_destinations
            ]
            if unexpected_files:
                raise FileExistsError(
                    f"Output directory contains .npy files outside the selected "
                    f"top-{args.top_k} splits: {unexpected_files[:5]}"
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        for source_path, destination_path in copy_plan.items():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

        splits_dir = output_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        for split_name, manifest_paths in output_manifests.items():
            contents = json.dumps(manifest_paths, ensure_ascii=False, indent=2) + "\n"
            (output_dir / f"{split_name}_split_500.json").write_text(
                contents, encoding="utf-8"
            )
            # train_ksl.py and dataset.build_label_mapping expect these names.
            (splits_dir / f"{split_name}_split.json").write_text(
                contents, encoding="utf-8"
            )

        selected_counts = {
            name: len(samples) for name, samples in filtered_splits.items()
        }
        top_20 = sorted(
            train_counts.items(), key=lambda item: (-item[1], item[0])
        )[:20]
        original_signer_ids = {
            name: sorted(ids) for name, ids in original_signers.items()
        }
        filtered_signer_ids = {
            name: sorted(ids) for name, ids in filtered_signers.items()
        }
        print(f"Selected {len(top_classes)} classes from the training split.")
        print(f"Top 20 by training frequency: {top_20}")
        print(f"Copied {len(copy_plan)} unique .npy files to {output_dir}")
        print(f"Filtered sample counts: {selected_counts}")
        print(f"Original signer IDs: {original_signer_ids}")
        print(f"Filtered signer IDs: {filtered_signer_ids}")
        print("Signer independence verified for train, validation, and test.")
        print(f"Named 500-class manifests: {output_dir}\\*_split_500.json")
        print(f"Training manifests: {splits_dir}")
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"Filtering failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
