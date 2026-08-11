"""
Benchmark preprocessing.

Purpose
-------
Convert raw benchmark result CSVs into completed detector-configuration rows
with raw and transformed objectives.

Inputs
------
- data/raw_data/benchmarking_results/<detector>/<detector>_<dataset>.csv

Outputs
-------
- data/processed_benchmark_data/<detector>/<detector>_<dataset>.csv

Important behavior
------------------
Raw ACCURACY is higher-is-better and raw RUNTIME is lower-is-better.
The exported transformed_accuracy and transformed_runtime are both
higher-is-better. Runtime clipping is controlled by `src.config`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import (
    ALL_DATASETS,
    ALL_DETECTORS,
    PREPROCESSING_ACCURACY_MODE,
    PREPROCESSING_ADD_INDEX_IF_MISSING,
    PREPROCESSING_EXPORT_TRANSFORMED_OBJECTIVES,
    PREPROCESSING_RUNTIME_CLIP_IQR_MULTIPLIER,
    PREPROCESSING_RUNTIME_CLIP_METHOD,
    PREPROCESSING_RUNTIME_CLIP_PERCENTILE,
    PREPROCESSING_RUNTIME_MODE,
    PREPROCESSING_RUNTIME_UPPER_CLIPPING,
    PREPROCESSING_STATUS_TO_KEEP,
)
from src.paths import get_paths, get_paths_from_script
from src.target_utils import pareto_layer_rank
from src.transforms import transform_accuracy, transform_runtime
from src.utils import ensure_dir


STATUS_TO_KEEP = PREPROCESSING_STATUS_TO_KEEP
ACCURACY_MODE = PREPROCESSING_ACCURACY_MODE
RUNTIME_MODE = PREPROCESSING_RUNTIME_MODE
RUNTIME_UPPER_CLIPPING = PREPROCESSING_RUNTIME_UPPER_CLIPPING
RUNTIME_CLIP_METHOD = PREPROCESSING_RUNTIME_CLIP_METHOD
RUNTIME_CLIP_IQR_MULTIPLIER = PREPROCESSING_RUNTIME_CLIP_IQR_MULTIPLIER
RUNTIME_CLIP_PERCENTILE = PREPROCESSING_RUNTIME_CLIP_PERCENTILE
EXPORT_TRANSFORMED_OBJECTIVES = PREPROCESSING_EXPORT_TRANSFORMED_OBJECTIVES
ADD_INDEX_IF_MISSING = PREPROCESSING_ADD_INDEX_IF_MISSING
DETECTORS = ALL_DETECTORS
DATASETS = ALL_DATASETS
FILENAME_RE = re.compile(r"^(?P<detector>[^_]+)_(?P<dataset>[^_]+)\.csv$", re.IGNORECASE)
_BOOL_SET = {"TRUE", "FALSE"}


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _is_bool_token_series(series: pd.Series) -> bool:
    values = series.astype(str).str.strip().str.upper()
    values = values[values != ""]
    return (len(values) > 0) and set(values.unique()).issubset(_BOOL_SET)


def _coerce_numeric_strict(series: pd.Series) -> tuple[pd.Series | None, str | None]:
    values = series.astype(str).str.strip()
    values_nonempty = values[values != ""]
    if values_nonempty.empty:
        return None, None
    converted = pd.to_numeric(values_nonempty, errors="coerce")
    if converted.isna().any():
        return None, None
    if (converted % 1 == 0).all():
        return pd.to_numeric(values, errors="raise").astype(int), "int"
    return pd.to_numeric(values, errors="coerce").astype(float), "float"


def _enforce_representation(df: pd.DataFrame, config_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in config_cols:
        series = out[col]
        if _is_bool_token_series(series):
            out[col] = series.astype(str).str.strip().str.upper()
            continue
        numeric_series, _ = _coerce_numeric_strict(series)
        if numeric_series is not None:
            out[col] = numeric_series
        else:
            out[col] = series.astype(str).str.strip()
    return out


def parse_detector_dataset_from_filename(filename: str) -> Optional[tuple[str, str]]:
    match = FILENAME_RE.match(filename)
    if not match:
        return None
    return match.group("detector"), match.group("dataset")


def _process_valid_rows(df: pd.DataFrame, *, input_name: str) -> pd.DataFrame:
    status_col = _find_column(df, ["Status"])
    accuracy_col = _find_column(df, ["ACCURACY"])
    runtime_col = _find_column(df, ["RUNTIME"])

    if status_col is None:
        raise ValueError(f"{input_name}: missing Status column.")
    if accuracy_col is None or runtime_col is None:
        raise ValueError(f"{input_name}: missing ACCURACY and/or RUNTIME columns.")

    out = df[df[status_col].astype(str).str.strip().str.casefold() == STATUS_TO_KEEP.casefold()].copy()
    if out.empty:
        raise ValueError(f"{input_name}: no rows with Status == '{STATUS_TO_KEEP}'.")

    out[accuracy_col] = pd.to_numeric(out[accuracy_col], errors="coerce")
    out[runtime_col] = pd.to_numeric(out[runtime_col], errors="coerce")
    out = out[np.isfinite(out[accuracy_col]) & np.isfinite(out[runtime_col])].copy()
    if out.empty:
        raise ValueError(f"{input_name}: no valid completed rows after numeric cleanup.")

    return out


def process_file(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    df = _process_valid_rows(df, input_name=input_csv.name)

    accuracy_col = _find_column(df, ["ACCURACY"])
    runtime_col = _find_column(df, ["RUNTIME"])
    assert accuracy_col is not None and runtime_col is not None

    if ADD_INDEX_IF_MISSING and "index" not in df.columns and "Index" not in df.columns:
        df.insert(0, "index", np.arange(len(df), dtype=int))
    index_col = "index" if "index" in df.columns else ("Index" if "Index" in df.columns else None)

    transformed_accuracy = transform_accuracy(df[accuracy_col], mode=ACCURACY_MODE)
    transformed_runtime = transform_runtime(
        df[runtime_col],
        mode=RUNTIME_MODE,
        use_upper_clipping=RUNTIME_UPPER_CLIPPING,
        clip_method=RUNTIME_CLIP_METHOD,
        clip_iqr_multiplier=RUNTIME_CLIP_IQR_MULTIPLIER,
        clip_percentile=RUNTIME_CLIP_PERCENTILE,
    )
    points = np.column_stack([transformed_accuracy, transformed_runtime])
    pareto_layer = pareto_layer_rank(points).astype(int)
    df["is_pareto"] = (pareto_layer == 1).astype(int)
    df["pareto_layer"] = pareto_layer

    if EXPORT_TRANSFORMED_OBJECTIVES:
        df["transformed_accuracy"] = transformed_accuracy
        df["transformed_runtime"] = transformed_runtime

    preferred_first: list[str] = []
    if index_col is not None:
        preferred_first.append(index_col)
    preferred_first.extend([accuracy_col, runtime_col])
    if EXPORT_TRANSFORMED_OBJECTIVES:
        preferred_first.extend(["transformed_accuracy", "transformed_runtime"])
    preferred_first.extend(["is_pareto", "pareto_layer"])

    excluded = set(preferred_first) | {"Status"}
    config_cols = [c for c in df.columns if c not in excluded]
    df = _enforce_representation(df, config_cols)
    df_out = df[preferred_first + config_cols].copy()

    ensure_dir(output_csv.parent)
    df_out.to_csv(output_csv, index=False, na_rep="")


def run_batch(
    *,
    project_root: Optional[Path],
    detector_arg: str,
    dataset_arg: str,
    verbose: bool = True,
) -> None:
    paths = get_paths(project_root)

    detector_arg = detector_arg.strip()
    dataset_arg = dataset_arg.strip()

    if detector_arg.upper() == "ALL":
        detectors = DETECTORS[:]
    else:
        detectors = [part.strip() for part in detector_arg.split(",") if part.strip()]
        unknown = [detector for detector in detectors if detector not in DETECTORS]
        if unknown:
            raise ValueError(f"Unknown detector(s) {unknown}. Expected exact names from: {DETECTORS}, or ALL.")

    dataset_filter: Optional[str] = None
    dataset_filter_set: Optional[set[str]] = None
    if dataset_arg.upper() != "ALL":
        requested_datasets = [part.strip() for part in dataset_arg.split(",") if part.strip()]
        unknown = [dataset for dataset in requested_datasets if dataset not in DATASETS]
        if unknown:
            raise ValueError(f"Unknown dataset(s) {unknown}. Expected exact names from: {DATASETS}, or ALL.")
        dataset_filter_set = set(requested_datasets)

    total_found = 0
    total_done = 0
    errors: list[str] = []

    for detector in detectors:
        in_dir = paths.raw_detector_dir(detector)
        out_dir = paths.processed_detector_dir(detector)
        if not in_dir.exists():
            errors.append(f"[{detector}] input directory not found: {in_dir}")
            continue

        ensure_dir(out_dir)
        for file_path in sorted(in_dir.glob("*.csv")):
            parsed = parse_detector_dataset_from_filename(file_path.name)
            if parsed is None:
                continue
            detector_in_name, dataset_in_name = parsed
            if detector_in_name != detector:
                continue
            if dataset_filter is not None and dataset_in_name != dataset_filter:
                continue
            if dataset_filter_set is not None and dataset_in_name not in dataset_filter_set:
                continue

            total_found += 1
            output_csv = out_dir / f"{detector}_{dataset_in_name}_processed.csv"
            try:
                process_file(file_path, output_csv)
                total_done += 1
                if verbose:
                    print(f"[OK] {file_path} -> {output_csv}")
            except Exception as exc:
                errors.append(f"[FAIL] {file_path}: {exc}")

    print(f"\nSummary: matched={total_found}, processed={total_done}, failed={len(errors)}")
    if errors:
        print("\nErrors:")
        for message in errors:
            print(" -", message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw benchmark CSVs into transformed benchmark data.")
    parser.add_argument("--detector", type=str, required=True, help="Detector name, ALL, or comma-separated exact names.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, ALL, or comma-separated exact names.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root directory.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    args = parser.parse_args()

    if args.project_root is None:
        project_root = get_paths_from_script(__file__).project_root
    else:
        project_root = Path(args.project_root).resolve()

    run_batch(
        project_root=project_root,
        detector_arg=args.detector,
        dataset_arg=args.dataset,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
