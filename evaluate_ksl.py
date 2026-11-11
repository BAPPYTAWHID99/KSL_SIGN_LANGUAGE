"""Evaluate saved KSL checkpoints once on the unseen-signer test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import KSLKeypointDataset
from models import BiGRUAttention, SmallTransformer


MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "bigru": BiGRUAttention,
    "transformer": SmallTransformer,
}
CHECKPOINT_NAMES = {
    "bigru": "best_bigru_ksl.pt",
    "transformer": "best_transformer_ksl.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained KSL models on the held-out test signers."
    )
    parser.add_argument("--split-dir", type=Path, default=Path("."))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--model",
        choices=("both", *MODEL_CLASSES),
        default="both",
        help="Which checkpoint(s) to evaluate (default: both).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device: auto, cpu, cuda, or another PyTorch device string.",
    )
    parser.add_argument(
        "--velocity",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Delta feature setting used during training. Auto recognizes the "
            "preprocessed 450-feature arrays and the raw-cache widths of 460/920."
        ),
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")
    return args


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device {device_name!r}: {exc}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this PyTorch environment.")
    return device


def load_class_mapping(mapping_path: Path) -> dict[str, int]:
    try:
        raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read class mapping {mapping_path}: {exc}") from exc

    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError(f"Class mapping must be a non-empty JSON object: {mapping_path}")
    if any(not isinstance(label, str) or not isinstance(index, int) for label, index in raw_mapping.items()):
        raise ValueError(f"Class mapping must map string labels to integer indices: {mapping_path}")
    mapping = dict(raw_mapping)
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Class mapping indices must be unique and contiguous starting at 0.")
    return mapping


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not load model checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(state, Mapping) or not state:
        raise ValueError(
            f"Checkpoint must contain a non-empty model state_dict: {checkpoint_path}"
        )
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError(f"Checkpoint contains unexpected non-tensor state: {checkpoint_path}")
    return dict(state)


def checkpoint_dimensions(
    model_name: str, state: Mapping[str, torch.Tensor]
) -> tuple[int, int, int]:
    try:
        input_dim = state["input_projection.weight"].shape[1]
        num_classes = state["classifier.weight"].shape[0]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Checkpoint is missing expected {model_name} model weights."
        ) from exc

    if model_name == "transformer":
        try:
            max_frames = state["positional_embedding"].shape[1] - 1
        except (KeyError, IndexError) as exc:
            raise ValueError("Transformer checkpoint is missing positional embeddings.") from exc
    else:
        max_frames = 200
    if input_dim <= 0 or num_classes <= 0 or max_frames <= 0:
        raise ValueError(f"Checkpoint has invalid model dimensions: {model_name}")
    return input_dim, num_classes, max_frames


def macro_f1_from_counts(
    true_positive: torch.Tensor,
    actual_count: torch.Tensor,
    predicted_count: torch.Tensor,
) -> float:
    denominator = actual_count + predicted_count
    per_class_f1 = torch.where(
        denominator > 0,
        2.0 * true_positive / denominator.clamp_min(1),
        torch.zeros_like(denominator, dtype=torch.float64),
    )
    return per_class_f1.mean().item()


def main() -> int:
    args = parse_args()
    try:
        device = resolve_device(args.device)
        label_to_idx = load_class_mapping(args.model_dir / "class_mapping.json")
        selected_models = (
            tuple(MODEL_CLASSES)
            if args.model == "both"
            else (args.model,)
        )

        model_states: dict[str, dict[str, torch.Tensor]] = {}
        model_dimensions: dict[str, tuple[int, int, int]] = {}
        for model_name in selected_models:
            checkpoint_path = args.model_dir / CHECKPOINT_NAMES[model_name]
            state = load_state_dict(checkpoint_path, device)
            dimensions = checkpoint_dimensions(model_name, state)
            if dimensions[1] != len(label_to_idx):
                raise ValueError(
                    f"{model_name} checkpoint has {dimensions[1]} classes, but "
                    f"class_mapping.json has {len(label_to_idx)}."
                )
            model_states[model_name] = state
            model_dimensions[model_name] = dimensions

        dimensions = set(model_dimensions.values())
        if len(dimensions) != 1:
            raise ValueError(
                "Selected checkpoints do not share the same input dimensions and frame length."
            )
        input_dim, _, max_frames = next(iter(dimensions))

        if args.velocity == "auto":
            if input_dim == 450 or input_dim == 460:
                add_velocity = False
            elif input_dim == 920:
                add_velocity = True
            else:
                raise ValueError(
                    f"Cannot infer velocity setting for {input_dim} input features. "
                    "Specify --velocity on or --velocity off to match training."
                )
        else:
            add_velocity = args.velocity == "on"

        dataset = KSLKeypointDataset(
            args.split_dir / "test_split.json",
            max_frames=max_frames,
            add_velocity=add_velocity,
            label_to_idx=label_to_idx,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        models: dict[str, nn.Module] = {}
        for model_name in selected_models:
            model = MODEL_CLASSES[model_name](
                num_classes=len(label_to_idx),
                input_dim=input_dim,
                max_frames=max_frames,
            ).to(device)
            model.load_state_dict(model_states[model_name], strict=True)
            model.eval()
            models[model_name] = model

        top_k = min(5, len(label_to_idx))
        counts = {
            model_name: {
                "top1": 0,
                "top5": 0,
                "true_positive": torch.zeros(len(label_to_idx), dtype=torch.int64),
                "actual": torch.zeros(len(label_to_idx), dtype=torch.int64),
                "predicted": torch.zeros(len(label_to_idx), dtype=torch.int64),
            }
            for model_name in selected_models
        }
        total = 0

        # Iterate over the held-out test files once and reuse each batch for all models.
        with torch.inference_mode():
            for features, labels in loader:
                labels = labels.to(device, non_blocking=True)
                features = features.to(device, non_blocking=True)
                batch_count = labels.shape[0]
                total += batch_count

                for model_name, model in models.items():
                    logits = model(features)
                    predictions = logits.argmax(dim=1)
                    top_predictions = logits.topk(top_k, dim=1).indices
                    model_counts = counts[model_name]
                    model_counts["top1"] += (predictions == labels).sum().item()
                    model_counts["top5"] += (
                        top_predictions == labels.unsqueeze(1)
                    ).any(dim=1).sum().item()

                    labels_cpu = labels.cpu()
                    predictions_cpu = predictions.cpu()
                    model_counts["actual"] += torch.bincount(
                        labels_cpu, minlength=len(label_to_idx)
                    )
                    model_counts["predicted"] += torch.bincount(
                        predictions_cpu, minlength=len(label_to_idx)
                    )
                    correct = predictions_cpu == labels_cpu
                    model_counts["true_positive"] += torch.bincount(
                        labels_cpu[correct], minlength=len(label_to_idx)
                    )

        if total == 0:
            raise ValueError("Test split contains no samples.")

        print(f"Test split: {args.split_dir / 'test_split.json'}")
        print(f"Unseen-signer test samples: {total}")
        print(f"Classes in mapping: {len(label_to_idx)}")
        print(f"Evaluation device: {device}")
        for model_name in selected_models:
            checkpoint_path = args.model_dir / CHECKPOINT_NAMES[model_name]
            model_counts = counts[model_name]
            file_size_bytes = checkpoint_path.stat().st_size
            macro_f1 = macro_f1_from_counts(
                model_counts["true_positive"],
                model_counts["actual"],
                model_counts["predicted"],
            )
            metrics = (
                ("Total Test Samples", f"{total:,}"),
                ("Top-1 Accuracy", f"{model_counts['top1'] / total:.6f}"),
                (f"Top-{top_k} Accuracy", f"{model_counts['top5'] / total:.6f}"),
                ("Macro F1-Score", f"{macro_f1:.6f}"),
                ("Model File Size (MB)", f"{file_size_bytes / 1_000_000:.3f}"),
            )
            label_width = max(len(label) for label, _ in metrics)
            value_width = max(len(value) for _, value in metrics)
            border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"
            print(f"\n{model_name.upper()} evaluation summary")
            print(border)
            print(f"| {'Metric':<{label_width}} | {'Value':>{value_width}} |")
            print(border)
            for label, value in metrics:
                print(f"| {label:<{label_width}} | {value:>{value_width}} |")
            print(border)
            print(f"Checkpoint: {checkpoint_path.resolve()}")
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
