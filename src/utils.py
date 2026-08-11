"""Small file and serialization utilities used across pipeline scripts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd


TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory of a file path and return the file Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def now_timestamp(fmt: str = TIMESTAMP_FORMAT) -> str:
    return datetime.now().strftime(fmt)


def sanitize_tag(text: str) -> str:
    """Convert free-form text into a file/folder friendly tag."""
    safe = str(text).strip().replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in safe)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "untitled"


def build_experiment_tag(*parts: Any, timestamp: bool = False) -> str:
    """Join meaningful run descriptors into one stable experiment tag."""
    cleaned = [sanitize_tag(str(part)) for part in parts if part is not None and str(part).strip()]
    if timestamp:
        cleaned.append(now_timestamp())
    return "_".join(cleaned)


def save_json(data: Mapping[str, Any], path: str | Path, indent: int = 2) -> Path:
    out = ensure_parent_dir(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    return out


def save_text(text: str, path: str | Path) -> Path:
    out = ensure_parent_dir(path)
    out.write_text(text, encoding="utf-8")
    return out


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    out = ensure_parent_dir(path)
    df.to_csv(out, index=index)
    return out


def save_snapshot_bundle(
    *,
    df: Optional[pd.DataFrame],
    csv_path: str | Path,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str | Path] = None,
    notes: Optional[str] = None,
    notes_path: Optional[str | Path] = None,
    index: bool = False,
) -> None:
    """Save a dataframe snapshot with optional JSON config and text notes."""
    if df is not None:
        save_dataframe(df, csv_path, index=index)

    if config is not None and config_path is not None:
        save_json(config, config_path)

    if notes is not None and notes_path is not None:
        save_text(notes, notes_path)


def describe_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Small dataframe summary that is handy for config/result metadata."""
    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
    }


def detector_dataset_stem(detector: str, dataset: str) -> str:
    return f"{detector}_{dataset}"


def lodo_stem(detector: str, dataset: str) -> str:
    return f"{detector}_LODO_{dataset}"
