"""Generate a WORD#### to Korean gloss map from AI-Hub label metadata."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


WORD_ID_PATTERN = re.compile(r"\bWORD[\s_-]*(\d+)\b", re.IGNORECASE)
NUMERIC_ID_FIELDS = {
    "wordid",
    "wordcode",
    "classid",
    "labelid",
    "signid",
    "어휘코드",
    "단어코드",
    "수어단어코드",
    "단어번호",
}
ID_FIELDS = NUMERIC_ID_FIELDS | {"id"}
GLOSS_FIELDS = {
    "gloss",
    "korean",
    "koreanword",
    "word",
    "label",
    "classname",
    "signword",
    "signname",
    "어휘",
    "어휘명",
    "단어",
    "단어명",
    "한국어",
    "한국어단어",
    "수어",
    "수어단어",
    "표제어",
}


def normalize_field_name(value: object) -> str:
    return re.sub(r"[\W_]+", "", str(value).casefold(), flags=re.UNICODE)


def normalize_word_id(value: object, *, numeric_allowed: bool = False) -> str | None:
    text = str(value).strip()
    match = WORD_ID_PATTERN.search(text)
    if match:
        return f"WORD{int(match.group(1)):04d}"
    if numeric_allowed and text.isdigit():
        return f"WORD{int(text):04d}"
    return None


def clean_gloss(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    gloss = value.strip()
    if not gloss or normalize_word_id(gloss) is not None:
        return None
    return gloss


def gloss_from_record(record: dict[Any, Any]) -> tuple[str, str] | None:
    identifiers: set[str] = set()
    fields = {normalize_field_name(key): value for key, value in record.items()}

    for raw_key, value in record.items():
        key = normalize_field_name(raw_key)
        key_word_id = normalize_word_id(raw_key)
        if key_word_id is not None:
            identifiers.add(key_word_id)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            word_id = normalize_word_id(value, numeric_allowed=key in ID_FIELDS)
            if word_id is not None:
                identifiers.add(word_id)

    glosses = [
        gloss
        for key, value in fields.items()
        if key in GLOSS_FIELDS and (gloss := clean_gloss(value)) is not None
    ]
    if len(identifiers) != 1 or not glosses:
        return None
    return next(iter(identifiers)), glosses[0]


def iter_json_records(
    value: Any, contextual_word_id: str | None = None
) -> Iterator[tuple[dict[Any, Any], str | None]]:
    if isinstance(value, dict):
        local_word_id = contextual_word_id
        for key in value:
            local_word_id = normalize_word_id(key) or local_word_id
        yield value, local_word_id
        for key, child in value.items():
            yield from iter_json_records(
                child, normalize_word_id(key) or local_word_id
            )
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_records(child, contextual_word_id)


def _read_csv_text(text: str) -> Iterator[dict[str, Any]]:
    yield from csv.DictReader(io.StringIO(text, newline=""))


def _decode_metadata(data: bytes, source: str) -> str:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode metadata {source}")


def read_csv_records(path: Path) -> Iterator[dict[str, Any]]:
    last_error: UnicodeError | None = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                yield from csv.DictReader(file)
            return
        except UnicodeError as exc:
            last_error = exc
    if last_error is not None:
        raise ValueError(f"Could not decode CSV metadata {path}: {last_error}") from last_error


def metadata_files(metadata_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in metadata_dir.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in {".csv", ".json", ".zip"}
        and not any(".cache" in part.casefold() for part in path.parts)
        and any(
            "label" in part.casefold() or "라벨" in part
            for part in path.parts
        )
    )


def _iter_metadata_file(path: Path) -> Iterator[tuple[dict[Any, Any], str | None]]:
    if path.suffix.casefold() == ".csv":
        contextual_word_id = normalize_word_id(str(path))
        yield from (
            (record, contextual_word_id) for record in read_csv_records(path)
        )
        return
    if path.suffix.casefold() == ".json":
        content = json.loads(_decode_metadata(path.read_bytes(), str(path)))
        yield from iter_json_records(content, normalize_word_id(str(path)))
        return
    if path.suffix.casefold() != ".zip":
        return

    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or member.suffix.casefold() not in {".csv", ".json"}:
                continue
            source_name = f"{path}!{info.filename}"
            data = archive.read(info)
            contextual_word_id = normalize_word_id(info.filename)
            if member.suffix.casefold() == ".csv":
                text = _decode_metadata(data, source_name)
                for record in _read_csv_text(text):
                    yield record, contextual_word_id
            else:
                content = json.loads(_decode_metadata(data, source_name))
                yield from iter_json_records(content, contextual_word_id)


def scan_metadata(metadata_dir: Path) -> tuple[dict[str, str], list[str]]:
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory does not exist: {metadata_dir}")

    glosses: dict[str, str] = {}
    failures: list[str] = []
    files = metadata_files(metadata_dir)
    if not files:
        failures.append(
            f"No CSV/JSON files or label archives found under {metadata_dir}"
        )
    for path in files:
        try:
            for record, contextual_word_id in _iter_metadata_file(path):
                parsed = gloss_from_record(record)
                if parsed is None and contextual_word_id is not None:
                    gloss_fields = {
                        normalize_field_name(key): value
                        for key, value in record.items()
                    }
                    for field_name, value in gloss_fields.items():
                        gloss = clean_gloss(value) if field_name in GLOSS_FIELDS else None
                        if gloss is not None:
                            parsed = contextual_word_id, gloss
                            break
                if parsed is None:
                    continue
                word_id, gloss = parsed
                existing = glosses.get(word_id)
                if existing is not None and existing != gloss:
                    raise ValueError(
                        f"Conflicting glosses for {word_id}: {existing!r} and {gloss!r} "
                        f"(in {path})"
                    )
                glosses[word_id] = gloss
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            csv.Error,
            zipfile.BadZipFile,
            ValueError,
        ) as exc:
            failures.append(f"{path}: {exc}")
    return glosses, failures


def load_class_ids(project_dir: Path, model_dir: Path) -> list[str]:
    mapping_path = model_dir / "class_mapping.json"
    if mapping_path.is_file():
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read class mapping {mapping_path}: {exc}") from exc
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(
                not isinstance(label, str)
                or not isinstance(index, int)
                or isinstance(index, bool)
                for label, index in mapping.items()
            )
            or sorted(mapping.values()) != list(range(len(mapping)))
        ):
            raise ValueError(f"Invalid contiguous label-to-index mapping: {mapping_path}")
        return [
            label
            for label, _ in sorted(mapping.items(), key=lambda item: item[1])
        ]

    manifest_path = project_dir / "KSL_Processed_500" / "train_split_500.json"
    try:
        paths = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot determine model classes; missing/invalid {mapping_path} "
            f"and {manifest_path}: {exc}"
        ) from exc
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"Expected a non-empty path list in {manifest_path}")
    return sorted(
        {
            word_id
            for item in paths
            if isinstance(item, str)
            and (word_id := normalize_word_id(Path(item).stem)) is not None
        }
    )


def load_existing_glosses(output_path: Path) -> dict[str, str]:
    if not output_path.is_file():
        return {}
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read existing gloss map {output_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Existing gloss map must be a JSON object: {output_path}")
    return {
        word_id: gloss
        for key, value in data.items()
        if (word_id := normalize_word_id(key)) is not None
        and (gloss := clean_gloss(value)) is not None
    }


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a Korean gloss map from AI-Hub CSV/JSON label metadata."
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(r"C:\KSL_Processed_450"),
        help="Root folder to search recursively for CSV/JSON label metadata.",
    )
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_dir / "KSL_Processed_500" / "models",
        help="Folder containing class_mapping.json and the output gloss map.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path (default: MODEL_DIR/korean_gloss_map.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or args.model_dir / "korean_gloss_map.json"
    try:
        class_ids = load_class_ids(args.project_dir, args.model_dir)
        discovered, failures = scan_metadata(args.metadata_dir)
        existing = load_existing_glosses(output_path)
        result = {
            word_id: discovered.get(word_id, existing.get(word_id, ""))
            for word_id in class_ids
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        matched_count = sum(bool(discovered.get(word_id)) for word_id in class_ids)
        preserved_count = sum(
            bool(existing.get(word_id)) and word_id not in discovered
            for word_id in class_ids
        )
        print(f"Classes in mapping: {len(class_ids)}")
        print(f"Glosses discovered from metadata: {matched_count}")
        print(f"Existing manual glosses preserved: {preserved_count}")
        missing_count = len(class_ids) - sum(bool(value) for value in result.values())
        print(f"Classes needing manual glosses: {missing_count}")
        print(f"Gloss map written to: {output_path.resolve()}")
        if not discovered:
            print(
                "No recognizable WORD#### gloss metadata was found. "
                "Edit the empty string values in the JSON file manually."
            )
        for failure in failures:
            print(f"Warning: {failure}", file=sys.stderr)
        if missing_count:
            print(
                "Gloss map is incomplete; supply the original AI-Hub label metadata "
                "or add the missing Korean glosses.",
                file=sys.stderr,
            )
            return 2
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Gloss-map generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
