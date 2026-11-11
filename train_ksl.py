"""Train the KSL temporal models with validation-loss early stopping."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Type

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import KSLKeypointDataset, build_label_mapping
from models import BiGRUAttention, SmallTransformer


MODEL_CONFIGS: dict[str, tuple[Type[nn.Module], float, str]] = {
    "bigru": (BiGRUAttention, 3e-4, "best_bigru_ksl.pt"),
    "transformer": (SmallTransformer, 2e-4, "best_transformer_ksl.pt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KSL temporal classifiers using train/validation manifests."
    )
    parser.add_argument("--split-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--model",
        choices=("both", *MODEL_CONFIGS),
        default="both",
        help="Which architecture(s) to train (default: both).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume a single model from a checkpoint_epoch_*.pt checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, or another PyTorch device string.",
    )
    parser.add_argument("--max-frames", type=int, default=200)
    velocity_group = parser.add_mutually_exclusive_group()
    velocity_group.add_argument(
        "--add-velocity",
        dest="add_velocity",
        action="store_true",
        help="Append temporal deltas to source position features.",
    )
    velocity_group.add_argument(
        "--no-velocity",
        dest="add_velocity",
        action="store_false",
        help="Do not append deltas (default; use for 450-feature preprocessed arrays).",
    )
    parser.set_defaults(add_velocity=False)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.patience <= 0:
        parser.error("--patience must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive.")
    if args.resume_from is not None and args.model == "both":
        parser.error("--resume-from requires --model bigru or --model transformer.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def save_checkpoint(checkpoint: object, output_path: Path) -> None:
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_state_dict(model: nn.Module, output_path: Path) -> None:
    save_checkpoint(model.state_dict(), output_path)


def train_one_model(
    model_name: str,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    patience: int,
    resume_from: Path | None = None,
) -> Path:
    model_class, learning_rate, checkpoint_name = MODEL_CONFIGS[model_name]
    model = model_class(
        num_classes=num_classes,
        input_dim=input_dim,
        max_frames=train_loader.dataset.max_frames,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    checkpoint_path = output_dir / checkpoint_name

    if resume_from is not None:
        if not resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_from}")
        try:
            checkpoint = torch.load(
                resume_from, map_location=device, weights_only=True
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Could not load resume checkpoint {resume_from}: {exc}") from exc
        expected_metadata = {
            "model_name": model_name,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "max_frames": train_loader.dataset.max_frames,
        }
        for key, expected_value in expected_metadata.items():
            if checkpoint.get(key) != expected_value:
                raise ValueError(
                    f"Resume checkpoint {resume_from} has incompatible {key}: "
                    f"{checkpoint.get(key)!r} (expected {expected_value!r})."
                )
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            epochs_without_improvement = checkpoint["epochs_without_improvement"]
            torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())
            loader_generator = train_loader.generator
            if loader_generator is not None and checkpoint["loader_generator_state"] is not None:
                loader_generator.set_state(checkpoint["loader_generator_state"].cpu())
            if device.type == "cuda" and "cuda_rng_states" in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        except (KeyError, RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid resume checkpoint {resume_from}: {exc}") from exc
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Best-model checkpoint is missing while resuming: {checkpoint_path}"
            )
        print(
            f"Resuming {model_name} from epoch {start_epoch - 1}: {resume_from.resolve()}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            print(
                f"Checkpoint already reached early-stopping patience ({patience}); "
                "no further epochs will run.",
                flush=True,
            )
            return checkpoint_path

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += batch_size

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.inference_mode():
            for features, labels in val_loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(features)
                loss = criterion(logits, labels)

                batch_size = labels.shape[0]
                val_loss_sum += loss.item() * batch_size
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_count += batch_size

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        print(
            f"{model_name} epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_accuracy={train_correct / train_count:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_correct / val_count:.4f}",
            flush=True,
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_state_dict(model, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path.resolve()}", flush=True)
        else:
            epochs_without_improvement += 1

        epoch_checkpoint_name = (
            f"checkpoint_epoch_{epoch}.pt"
            if model_name == "bigru"
            else f"checkpoint_epoch_{epoch}_{model_name}.pt"
        )
        epoch_checkpoint_path = output_dir / epoch_checkpoint_name
        epoch_checkpoint: dict[str, object] = {
            "model_name": model_name,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "max_frames": train_loader.dataset.max_frames,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "cpu_rng_state": torch.get_rng_state(),
            "loader_generator_state": (
                train_loader.generator.get_state()
                if train_loader.generator is not None
                else None
            ),
        }
        if device.type == "cuda":
            epoch_checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        save_checkpoint(epoch_checkpoint, epoch_checkpoint_path)
        print(f"Saved epoch checkpoint: {epoch_checkpoint_path.resolve()}", flush=True)

        if not improved and epochs_without_improvement >= patience:
            print(
                f"Early stopping {model_name}: validation loss did not improve "
                f"for {patience} epochs.",
                flush=True,
            )
            break

    return checkpoint_path


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    try:
        device = resolve_device(args.device)
        label_to_idx = build_label_mapping(args.split_dir)
        train_dataset = KSLKeypointDataset(
            args.split_dir / "train_split.json",
            max_frames=args.max_frames,
            add_velocity=args.add_velocity,
            label_to_idx=label_to_idx,
        )
        val_dataset = KSLKeypointDataset(
            args.split_dir / "val_split.json",
            max_frames=args.max_frames,
            add_velocity=args.add_velocity,
            label_to_idx=label_to_idx,
        )

        input_dim = train_dataset[0][0].shape[1]
        if val_dataset[0][0].shape[1] != input_dim:
            raise ValueError("Train and validation datasets have different feature dimensions.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = args.output_dir / "class_mapping.json"
        mapping_path.write_text(
            json.dumps(label_to_idx, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=loader_generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        print(
            f"Device: {device}; input features: {input_dim}; classes: {len(label_to_idx)}; "
            f"train samples: {len(train_dataset)}; validation samples: {len(val_dataset)}",
            flush=True,
        )
        print(f"Saved class mapping: {mapping_path.resolve()}", flush=True)

        selected_models = (
            tuple(MODEL_CONFIGS)
            if args.model == "both"
            else (args.model,)
        )
        for model_name in selected_models:
            checkpoint_path = train_one_model(
                model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                num_classes=len(label_to_idx),
                device=device,
                output_dir=args.output_dir,
                epochs=args.epochs,
                patience=args.patience,
                resume_from=args.resume_from,
            )
            print(f"Finished {model_name}: {checkpoint_path.resolve()}", flush=True)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
