"""PyTorch dataset for the signer-independent KSL split manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from create_splits import parse_filename


SPLIT_FILENAMES = ("train_split.json", "val_split.json", "test_split.json")


def _read_split_paths(split_file: Path) -> list[Path]:
    try:
        contents = json.loads(split_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read split manifest {split_file}: {exc}") from exc

    if not isinstance(contents, list) or not contents:
        raise ValueError(f"Split manifest must contain a non-empty JSON list: {split_file}")
    if any(not isinstance(item, str) or not item for item in contents):
        raise ValueError(f"Every split entry must be a non-empty path string: {split_file}")
    return [Path(item) for item in contents]


def build_label_mapping(split_dir: str | Path = ".") -> dict[str, int]:
    """Build one contiguous label map from the union of all three split manifests."""
    directory = Path(split_dir)
    labels: set[str] = set()
    for filename in SPLIT_FILENAMES:
        manifest_path = directory / filename
        paths = _read_split_paths(manifest_path)
        for path in paths:
            try:
                _, label = parse_filename(path)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot determine class label for {path} in {manifest_path}: {exc}"
                ) from exc
            labels.add(label)

    if not labels:
        raise ValueError(f"No class labels found in split manifests under {directory}")
    return {label: index for index, label in enumerate(sorted(labels))}


class KSLKeypointDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load, flatten, optionally augment, and pad/truncate KSL keypoint sequences.

    The cached archive arrays use ``(frames, landmarks, channels)`` layout.
    All channels are preserved when flattening. With ``add_velocity=True``,
    per-frame temporal differences are appended to the original features;
    the first frame's delta is zero.
    """

    def __init__(
        self,
        split_file: str | Path,
        *,
        max_frames: int = 200,
        add_velocity: bool = True,
        label_to_idx: Mapping[str, int] | None = None,
    ) -> None:
        self.split_file = Path(split_file)
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        self.max_frames = max_frames
        self.add_velocity = add_velocity
        self.paths = _read_split_paths(self.split_file)

        if label_to_idx is None:
            self.label_to_idx = build_label_mapping(self.split_file.parent)
        else:
            self.label_to_idx = dict(label_to_idx)
            if not self.label_to_idx:
                raise ValueError("label_to_idx must not be empty.")
            if sorted(self.label_to_idx.values()) != list(range(len(self.label_to_idx))):
                raise ValueError("label_to_idx values must be unique contiguous integers starting at 0.")

        self.labels: list[int] = []
        for path in self.paths:
            try:
                _, label = parse_filename(path)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot determine class label for {path} in {self.split_file}: {exc}"
                ) from exc
            if label not in self.label_to_idx:
                raise ValueError(
                    f"Class {label!r} from {path} is missing from label_to_idx. "
                    "Build the mapping from all three split manifests."
                )
            self.labels.append(self.label_to_idx[label])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.paths[index]
        try:
            sequence = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load keypoint array {path}: {exc}") from exc

        if sequence.ndim < 2 or sequence.shape[0] == 0:
            raise ValueError(
                f"Expected a non-empty sequence with shape (frames, ...), "
                f"got {sequence.shape} from {path}"
            )
        if not np.issubdtype(sequence.dtype, np.number):
            raise ValueError(f"Expected numeric keypoint data in {path}, got {sequence.dtype}")

        features = np.asarray(sequence, dtype=np.float32).reshape(sequence.shape[0], -1)
        if not np.isfinite(features).all():
            raise ValueError(f"Keypoint sequence contains NaN or infinite values: {path}")

        if self.add_velocity:
            delta = np.empty_like(features)
            delta[0] = 0.0
            delta[1:] = features[1:] - features[:-1]
            features = np.concatenate((features, delta), axis=1)

        frame_count = min(features.shape[0], self.max_frames)
        padded = np.zeros((self.max_frames, features.shape[1]), dtype=np.float32)
        padded[:frame_count] = features[:frame_count]

        return torch.from_numpy(padded), self.labels[index]
