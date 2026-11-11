"""Web views and inference API for the 500-class KSL BiGRU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from models import BiGRUAttention


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "KSL_Processed_500" / "models"
MODEL_PATH = MODEL_DIR / "best_bigru_ksl.pt"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"
GLOSS_MAPPING_PATH = MODEL_DIR / "korean_gloss_map.json"
MAX_FRAMES = 200
INPUT_DIM = 450


def _read_class_mapping(path: Path) -> dict[str, int]:
    mapping: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(
            not isinstance(class_id, str)
            or not isinstance(index, int)
            or isinstance(index, bool)
            for class_id, index in mapping.items()
        )
        or sorted(mapping.values()) != list(range(len(mapping)))
    ):
        raise ValueError(f"Invalid contiguous class mapping: {path}")
    return mapping


def _read_gloss_mapping(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    glosses: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(glosses, dict) or any(
        not isinstance(class_id, str) or not isinstance(gloss, str)
        for class_id, gloss in glosses.items()
    ):
        raise ValueError(f"Invalid Korean gloss mapping: {path}")
    return {class_id: gloss for class_id, gloss in glosses.items() if gloss.strip()}


def _load_model() -> tuple[BiGRUAttention, list[str], dict[str, str]]:
    class_to_index = _read_class_mapping(CLASS_MAPPING_PATH)
    class_ids = [
        class_id
        for class_id, _ in sorted(class_to_index.items(), key=lambda item: item[1])
    ]
    model = BiGRUAttention(
        num_classes=len(class_ids),
        input_dim=INPUT_DIM,
        max_frames=MAX_FRAMES,
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, class_ids, _read_gloss_mapping(GLOSS_MAPPING_PATH)


# Load during Django URL configuration so missing or incompatible assets fail at startup.
MODEL, CLASS_IDS, KOREAN_GLOSSES = _load_model()


@ensure_csrf_cookie
def index(request: HttpRequest):
    return render(request, "index.html")


def _parse_input(payload: Any) -> torch.Tensor:
    if not isinstance(payload, dict) or "sequence" not in payload:
        raise ValueError('JSON body must contain a "sequence" field.')
    try:
        sequence = np.asarray(payload["sequence"], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sequence must contain numeric values.") from exc

    expected_shape = (1, MAX_FRAMES, INPUT_DIM)
    if sequence.shape != expected_shape:
        raise ValueError(
            f"Expected sequence shape {expected_shape}, got {sequence.shape}."
        )
    if not np.isfinite(sequence).all():
        raise ValueError("Sequence must contain only finite values.")
    return torch.from_numpy(np.ascontiguousarray(sequence))


@require_POST
def predict(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body)
        inputs = _parse_input(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    with torch.inference_mode():
        probabilities = torch.softmax(MODEL(inputs), dim=1)[0]
        class_index = int(torch.argmax(probabilities).item())

    class_id = CLASS_IDS[class_index]
    korean_word = KOREAN_GLOSSES.get(class_id)
    response = {
        "class_id": class_id,
        "class_index": class_index,
        "word": korean_word,
        "confidence": float(probabilities[class_index].item()),
    }
    if korean_word:
        response["korean_word"] = korean_word
    return JsonResponse(response)
