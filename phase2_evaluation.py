"""
Phase 2 evaluation.

Purpose
-------
Compute final evaluation metrics from externally measured Rec, Par, and Def
result files.

Inputs
------
- measured recommendation rows (Rec)
- measured benchmark-reference rows (Par)
- measured default rows (Def)

Outputs
-------
- pair-level, detector-level, dataset-level, and global metric tables

Important behavior
------------------
The script does not generate, export, or execute configurations. Transformed
objective bounds for distance/utility metrics come from completed Rec union Par;
Def does not define those bounds. The reported R2 gap is r2_rec - r2_ref, and
global quality aggregation is the mean of detector means.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    project_root_for_bootstrap = Path(__file__).resolve().parent
    venv_python = (
        project_root_for_bootstrap / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else project_root_for_bootstrap / ".venv" / "bin" / "python"
    )
    if (
        venv_python.exists()
        and Path(sys.executable).resolve() != venv_python.resolve()
        and os.environ.get("PHASE2_EVALUATION_BOOTSTRAPPED") != "1"
    ):
        env = os.environ.copy()
        env["PHASE2_EVALUATION_BOOTSTRAPPED"] = "1"
        completed = subprocess.run([str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
        sys.exit(completed.returncode)
    raise

from src.metrics import (
    exact_r2_from_transformed_objectives,
    gd_plus,
    igd_plus,
    raw_accuracy_runtime_dominance_rates,
    safe_spearman_correlation,
)


INPUT_RELATIVE_PATH = Path("results") / "phase_2" / "final_thesis_results"
OUTPUT_RELATIVE_PATH = Path("results") / "phase_2" / "computed_metrics"

BASE_REQUIRED_COLUMNS = (
    "source",
    "Status",
    "ACCURACY",
    "RUNTIME",
)
PREDICTION_COLUMNS = (
    "predicted_transformed_accuracy",
    "predicted_transformed_runtime",
)

SOURCE_ALIASES = {
    "rec": "rec",
    "recommendation": "rec",
    "recommendation_system": "rec",
    "par": "par",
    "benchmark_pareto": "par",
    "benchmark_reference": "par",
    "reference": "par",
    "def": "def",
    "default": "def",
    "paper_default": "def",
}
IGNORED_SOURCES = {"benchmark_max_rt", "benchmark_min_acc"}

PAIR_COLUMNS = [
    "dataset",
    "rec_attempted_count",
    "rec_completed_count",
    "completion_rate",
    "par_completed_count",
    "def_total_count",
    "def_completed_count",
    "gd_plus",
    "igd_plus",
    "r2_rec",
    "r2_ref",
    "r2_gap",
    "rho_acc",
    "rho_rt",
    "do_rec_par",
    "db_par_rec",
    "do_rec_def",
    "db_def_rec",
]

COUNT_COLUMNS = [
    "rec_attempted_count",
    "rec_completed_count",
    "par_completed_count",
    "def_total_count",
    "def_completed_count",
]

QUALITY_METRICS = [
    "gd_plus",
    "igd_plus",
    "r2_rec",
    "r2_ref",
    "r2_gap",
    "rho_acc",
    "rho_rt",
    "do_rec_par",
    "db_par_rec",
    "do_rec_def",
    "db_def_rec",
]


def _sort_key(path_or_text: Path | str) -> str:
    return str(path_or_text).casefold()


def _empty_quality_metrics() -> dict[str, float]:
    return {metric: float("nan") for metric in QUALITY_METRICS}


def _finite_mean(values: pd.Series) -> tuple[float, int]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return float("nan"), 0
    return float(np.mean(finite)), int(finite.size)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _source_kind(value: Any) -> str | None:
    token = str(value).strip().casefold()
    return SOURCE_ALIASES.get(token)


def discover_result_files(input_root: Path) -> dict[str, list[tuple[str, Path]]]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_root}")

    discovered: dict[str, list[tuple[str, Path]]] = {}
    suffix = "_final_results.csv"
    detector_dirs = sorted((p for p in input_root.iterdir() if p.is_dir()), key=_sort_key)
    if len(detector_dirs) != 20:
        logging.warning("Discovered %d detector directories; expected 20.", len(detector_dirs))

    for detector_dir in detector_dirs:
        detector = detector_dir.name
        prefix = f"{detector}_"
        files: list[tuple[str, Path]] = []
        for csv_path in sorted(detector_dir.glob(f"*{suffix}"), key=_sort_key):
            name = csv_path.name
            if not name.startswith(prefix):
                logging.warning("Ignoring result file with unexpected detector prefix: %s", csv_path)
                continue
            dataset = name.removeprefix(prefix).removesuffix(suffix)
            if not dataset:
                logging.warning("Ignoring result file with empty dataset name: %s", csv_path)
                continue
            files.append((dataset, csv_path))
        if not files:
            logging.warning("Detector directory contains no valid result files: %s", detector_dir)
        discovered[detector] = sorted(files, key=lambda item: item[0].casefold())
    return discovered


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read CSV file {csv_path}: {exc}") from exc

    missing = [column for column in BASE_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV file {csv_path} is missing required column(s): {missing}")

    out = df.copy()
    out["__source_kind"] = out["source"].map(_source_kind)
    out["__source_norm"] = out["source"].astype("string").str.strip().str.casefold()
    ignored_count = int(out["__source_norm"].isin(IGNORED_SOURCES).sum())
    if ignored_count:
        logging.info("%s: ignored %d non-Rec/Par/Def row(s).", csv_path, ignored_count)
    unexpected = sorted(
        str(value)
        for value in out.loc[
            out["__source_kind"].isna() & ~out["__source_norm"].isin(IGNORED_SOURCES),
            "__source_norm",
        ].dropna().unique()
    )
    if unexpected:
        logging.warning("%s has unexpected source value(s): %s", csv_path, unexpected)

    out["__status_norm"] = out["Status"].astype("string").str.strip().str.casefold()
    out["__accuracy"] = pd.to_numeric(out["ACCURACY"], errors="coerce")
    out["__runtime"] = pd.to_numeric(out["RUNTIME"], errors="coerce")

    for column in PREDICTION_COLUMNS:
        if column in out.columns:
            out[f"__{column}"] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[f"__{column}"] = np.nan

    marked_completed = out["__status_norm"].eq("completed")
    finite_objectives = np.isfinite(out["__accuracy"].to_numpy(dtype=float)) & np.isfinite(
        out["__runtime"].to_numpy(dtype=float)
    )
    invalid_completed_count = int((marked_completed.to_numpy(dtype=bool) & ~finite_objectives).sum())
    if invalid_completed_count:
        logging.warning(
            "%s has %d row(s) marked completed with missing or non-finite real objective(s).",
            csv_path,
            invalid_completed_count,
        )
    out["__completed"] = marked_completed & finite_objectives
    return out


def _objective_points(df: pd.DataFrame, accuracy_col: str, runtime_col: str) -> np.ndarray:
    return df[[accuracy_col, runtime_col]].to_numpy(dtype=float)


def _transform_completed_rec_par(
    recommendations: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform completed Rec and Par using bounds from Rec union Par only."""

    fit_df = pd.concat([recommendations, reference], ignore_index=True)
    accuracy = fit_df["__accuracy"].to_numpy(dtype=float)
    runtime = fit_df["__runtime"].to_numpy(dtype=float)
    log_runtime = np.log1p(runtime)
    if not np.isfinite(log_runtime).all():
        raise ValueError(f"{label}: log1p runtime transformation produced non-finite values.")

    accuracy_min = float(np.min(accuracy))
    accuracy_max = float(np.max(accuracy))
    log_runtime_min = float(np.min(log_runtime))
    log_runtime_max = float(np.max(log_runtime))

    accuracy_range = accuracy_max - accuracy_min
    log_runtime_range = log_runtime_max - log_runtime_min

    if np.isclose(accuracy_range, 0.0, rtol=0.0, atol=1e-12):
        logging.warning("%s has zero transformed accuracy range; assigning accuracy objective 1.0.", label)
    if np.isclose(log_runtime_range, 0.0, rtol=0.0, atol=1e-12):
        logging.warning("%s has zero transformed runtime range; assigning runtime objective 1.0.", label)

    def transform(df: pd.DataFrame) -> np.ndarray:
        acc_values = df["__accuracy"].to_numpy(dtype=float)
        rt_log_values = np.log1p(df["__runtime"].to_numpy(dtype=float))
        if np.isclose(accuracy_range, 0.0, rtol=0.0, atol=1e-12):
            transformed_accuracy = np.ones(len(df), dtype=float)
        else:
            transformed_accuracy = (acc_values - accuracy_min) / accuracy_range
        if np.isclose(log_runtime_range, 0.0, rtol=0.0, atol=1e-12):
            transformed_runtime = np.ones(len(df), dtype=float)
        else:
            transformed_runtime = 1.0 - ((rt_log_values - log_runtime_min) / log_runtime_range)
        return np.clip(np.column_stack([transformed_accuracy, transformed_runtime]), 0.0, 1.0)

    return transform(recommendations), transform(reference)


def _reference_dominance_metrics(recommendations: pd.DataFrame, references: pd.DataFrame) -> tuple[float, float]:
    # Thesis convention: if the comparison reference is empty, both rates are 1.
    if references.empty:
        return 1.0, 1.0
    return raw_accuracy_runtime_dominance_rates(
        _objective_points(recommendations, "__accuracy", "__runtime"),
        _objective_points(references, "__accuracy", "__runtime"),
    )


def compute_pair_metrics(detector: str, dataset: str, csv_path: Path) -> dict[str, Any]:
    """
    Compute Phase 2 metrics for one measured detector-dataset result file.

    Rec and Par completed rows define transformed objective bounds for
    distance/R2 metrics. Def rows are used only for dominance comparisons.
    """
    df = load_and_validate_csv(csv_path)

    rec_mask = df["__source_kind"].eq("rec")
    par_mask = df["__source_kind"].eq("par")
    def_mask = df["__source_kind"].eq("def")

    rec_all = df.loc[rec_mask].copy()
    rec = df.loc[rec_mask & df["__completed"]].copy()
    par = df.loc[par_mask & df["__completed"]].copy()
    def_all = df.loc[def_mask].copy()
    default = df.loc[def_mask & df["__completed"]].copy()

    attempted_count = int(len(rec_all))
    completed_count = int(len(rec))

    if attempted_count == 0:
        logging.warning("%s has no Rec/recommendation rows.", csv_path)

    row: dict[str, Any] = {
        "dataset": dataset,
        "rec_attempted_count": attempted_count,
        "rec_completed_count": completed_count,
        "completion_rate": _rate(completed_count, attempted_count),
        "par_completed_count": int(len(par)),
        "def_total_count": int(len(def_all)),
        "def_completed_count": int(len(default)),
    }

    if completed_count == 0:
        row.update(_empty_quality_metrics())
        return row

    label = f"{detector}/{dataset}"
    rec_points, par_points = _transform_completed_rec_par(rec, par, label=label)

    # Empty Par convention: GD+/IGD+ are zero and r2_ref is fixed to 0.75.
    r2_rec = exact_r2_from_transformed_objectives(rec_points)
    if par.empty:
        gd_value = 0.0
        igd_value = 0.0
        r2_ref = 0.75
    else:
        gd_value = gd_plus(rec_points, par_points)
        igd_value = igd_plus(rec_points, par_points)
        r2_ref = exact_r2_from_transformed_objectives(par_points)

    do_rec_par, db_par_rec = _reference_dominance_metrics(rec, par)
    do_rec_def, db_def_rec = _reference_dominance_metrics(rec, default)

    pred_acc = rec["__predicted_transformed_accuracy"]
    pred_rt = rec["__predicted_transformed_runtime"]
    if pred_acc.notna().sum() == 0 or pred_rt.notna().sum() == 0:
        logging.warning("%s: standardized prediction columns are missing or empty for Rec rows.", label)
    rho_acc = safe_spearman_correlation(pred_acc, rec_points[:, 0])
    rho_rt = safe_spearman_correlation(pred_rt, rec_points[:, 1])
    if not np.isfinite(rho_acc):
        logging.warning("%s: rho_acc is undefined.", label)
    if not np.isfinite(rho_rt):
        logging.warning("%s: rho_rt is undefined.", label)

    row.update(
        {
            "gd_plus": gd_value,
            "igd_plus": igd_value,
            "r2_rec": r2_rec,
            "r2_ref": r2_ref,
            "r2_gap": r2_rec - r2_ref,
            "rho_acc": rho_acc,
            "rho_rt": rho_rt,
            "do_rec_par": do_rec_par,
            "db_par_rec": db_par_rec,
            "do_rec_def": do_rec_def,
            "db_def_rec": db_def_rec,
        }
    )
    return row


def aggregate_detector(detector: str, pair_df: pd.DataFrame) -> dict[str, Any]:
    attempted = int(pair_df["rec_attempted_count"].sum()) if not pair_df.empty else 0
    completed = int(pair_df["rec_completed_count"].sum()) if not pair_df.empty else 0
    row: dict[str, Any] = {
        "detector": detector,
        "dataset_file_count": int(len(pair_df)),
        "completed_pair_count": int((pair_df["rec_completed_count"] > 0).sum()) if not pair_df.empty else 0,
        "zero_completion_pair_count": int((pair_df["rec_completed_count"] == 0).sum()) if not pair_df.empty else 0,
        "rec_attempted_count": attempted,
        "rec_completed_count": completed,
        "completion_rate": _rate(completed, attempted),
        "par_completed_count": int(pair_df["par_completed_count"].sum()) if not pair_df.empty else 0,
        "def_total_count": int(pair_df["def_total_count"].sum()) if not pair_df.empty else 0,
        "def_completed_count": int(pair_df["def_completed_count"].sum()) if not pair_df.empty else 0,
    }
    for metric in QUALITY_METRICS:
        mean_value, valid_count = _finite_mean(pair_df[metric]) if not pair_df.empty else (float("nan"), 0)
        row[metric] = mean_value
        row[f"valid_pair_count__{metric}"] = valid_count
    return row


def aggregate_dataset(dataset: str, pair_df: pd.DataFrame) -> dict[str, Any]:
    attempted = int(pair_df["rec_attempted_count"].sum()) if not pair_df.empty else 0
    completed = int(pair_df["rec_completed_count"].sum()) if not pair_df.empty else 0
    row: dict[str, Any] = {
        "dataset": dataset,
        "detector_file_count": int(len(pair_df)),
        "completed_pair_count": int((pair_df["rec_completed_count"] > 0).sum()) if not pair_df.empty else 0,
        "zero_completion_pair_count": int((pair_df["rec_completed_count"] == 0).sum()) if not pair_df.empty else 0,
        "rec_attempted_count": attempted,
        "rec_completed_count": completed,
        "completion_rate": _rate(completed, attempted),
        "par_completed_count": int(pair_df["par_completed_count"].sum()) if not pair_df.empty else 0,
        "def_total_count": int(pair_df["def_total_count"].sum()) if not pair_df.empty else 0,
        "def_completed_count": int(pair_df["def_completed_count"].sum()) if not pair_df.empty else 0,
    }
    for metric in QUALITY_METRICS:
        mean_value, valid_count = _finite_mean(pair_df[metric]) if not pair_df.empty else (float("nan"), 0)
        row[metric] = mean_value
        row[f"valid_pair_count__{metric}"] = valid_count
    return row


def aggregate_global(detector_df: pd.DataFrame) -> dict[str, Any]:
    attempted = int(detector_df["rec_attempted_count"].sum()) if not detector_df.empty else 0
    completed = int(detector_df["rec_completed_count"].sum()) if not detector_df.empty else 0
    row: dict[str, Any] = {
        "detector_count": int(len(detector_df)),
        "dataset_file_count": int(detector_df["dataset_file_count"].sum()) if not detector_df.empty else 0,
        "completed_pair_count": int(detector_df["completed_pair_count"].sum()) if not detector_df.empty else 0,
        "zero_completion_pair_count": int(detector_df["zero_completion_pair_count"].sum()) if not detector_df.empty else 0,
        "rec_attempted_count": attempted,
        "rec_completed_count": completed,
        "completion_rate": _rate(completed, attempted),
        "par_completed_count": int(detector_df["par_completed_count"].sum()) if not detector_df.empty else 0,
        "def_total_count": int(detector_df["def_total_count"].sum()) if not detector_df.empty else 0,
        "def_completed_count": int(detector_df["def_completed_count"].sum()) if not detector_df.empty else 0,
    }
    for metric in QUALITY_METRICS:
        mean_value, valid_detector_count = _finite_mean(detector_df[metric]) if not detector_df.empty else (float("nan"), 0)
        row[metric] = mean_value
        row[f"valid_detector_count__{metric}"] = valid_detector_count
        valid_pair_column = f"valid_pair_count__{metric}"
        row[f"valid_pair_count_total__{metric}"] = (
            int(detector_df[valid_pair_column].sum()) if valid_pair_column in detector_df.columns else 0
        )
    return row


def write_all_final_metric_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    preferred_columns = [
        "detector",
        "dataset",
        "detector_count",
        "dataset_file_count",
        "detector_file_count",
        "completed_pair_count",
        "zero_completion_pair_count",
        *COUNT_COLUMNS,
        "completion_rate",
    ]
    preferred_columns.extend(QUALITY_METRICS)
    for metric in QUALITY_METRICS:
        preferred_columns.extend(
            [
                f"valid_pair_count__{metric}",
                f"valid_detector_count__{metric}",
                f"valid_pair_count_total__{metric}",
            ]
        )
    ordered_columns = [column for column in preferred_columns if column in df.columns]
    ordered_columns.extend(column for column in df.columns if column not in ordered_columns)
    write_csv_if_changed(df[ordered_columns], output_path)


def _normalize_csv_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def write_csv_if_changed(df: pd.DataFrame, output_path: Path) -> None:
    csv_text = df.to_csv(index=False)
    if output_path.exists():
        existing_text = output_path.read_text(encoding="utf-8")
        if _normalize_csv_text(existing_text) == _normalize_csv_text(csv_text):
            logging.info("Skipping unchanged CSV: %s", output_path)
            return
    output_path.write_text(csv_text, encoding="utf-8", newline="")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = Path(__file__).resolve().parent
    input_root = project_root / INPUT_RELATIVE_PATH
    output_root = project_root / OUTPUT_RELATIVE_PATH
    detector_output_root = output_root / "detector_level"
    dataset_output_root = output_root / "dataset_level"
    output_root.mkdir(parents=True, exist_ok=True)
    detector_output_root.mkdir(parents=True, exist_ok=True)
    dataset_output_root.mkdir(parents=True, exist_ok=True)

    discovered = discover_result_files(input_root)
    generated_files: list[Path] = []
    detector_rows: list[dict[str, Any]] = []
    dataset_pair_rows: dict[str, list[dict[str, Any]]] = {}
    dataset_rows: list[dict[str, Any]] = []
    all_final_rows: list[dict[str, Any]] = []
    processed_file_count = 0

    for detector in sorted(discovered, key=str.casefold):
        pair_rows: list[dict[str, Any]] = []
        for dataset, csv_path in discovered[detector]:
            logging.info("Processing %s/%s", detector, dataset)
            pair_row = compute_pair_metrics(detector, dataset, csv_path)
            pair_rows.append(pair_row)
            dataset_pair_rows.setdefault(dataset, []).append({"detector": detector, **pair_row})
            processed_file_count += 1

        pair_df = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)
        if not pair_df.empty:
            pair_df = pair_df.sort_values("dataset", key=lambda values: values.str.casefold()).reset_index(drop=True)
        detector_output = detector_output_root / f"{detector}_phase2_metrics.csv"
        write_csv_if_changed(pair_df, detector_output)
        generated_files.append(detector_output)

        detector_row = aggregate_detector(detector, pair_df)
        detector_rows.append(detector_row)
        for pair_row in pair_df.to_dict(orient="records"):
            all_final_rows.append({"detector": detector, **pair_row})
        all_final_rows.append({**detector_row, "dataset": "ALL"})

    detector_df = pd.DataFrame(detector_rows)
    detector_df = detector_df.sort_values("detector", key=lambda values: values.str.casefold()).reset_index(drop=True)
    detector_output = output_root / "detector_aggregated_metrics.csv"
    write_csv_if_changed(detector_df, detector_output)
    generated_files.append(detector_output)

    for dataset in sorted(dataset_pair_rows, key=str.casefold):
        dataset_pair_df = pd.DataFrame(dataset_pair_rows[dataset], columns=["detector", *PAIR_COLUMNS])
        dataset_pair_df = dataset_pair_df.sort_values("detector", key=lambda values: values.str.casefold()).reset_index(drop=True)
        dataset_output = dataset_output_root / f"{dataset}_phase2_metrics.csv"
        write_csv_if_changed(dataset_pair_df, dataset_output)
        generated_files.append(dataset_output)
        dataset_rows.append(aggregate_dataset(dataset, dataset_pair_df))

    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df = dataset_df.sort_values("dataset", key=lambda values: values.str.casefold()).reset_index(drop=True)
    dataset_aggregate_output = output_root / "dataset_aggregated_metrics.csv"
    write_csv_if_changed(dataset_df, dataset_aggregate_output)
    generated_files.append(dataset_aggregate_output)

    global_row = aggregate_global(detector_df)
    global_df = pd.DataFrame([global_row])
    global_output = output_root / "global_metrics.csv"
    write_csv_if_changed(global_df, global_output)
    generated_files.append(global_output)

    for dataset_row in dataset_df.to_dict(orient="records"):
        all_final_rows.append({"detector": "ALL", **dataset_row})
    all_final_rows.append({**global_row, "detector": "ALL", "dataset": "ALL"})
    all_final_output = output_root / "all_final_metric_results.csv"
    write_all_final_metric_results(all_final_rows, all_final_output)
    generated_files.append(all_final_output)

    print()
    print("Phase 2 evaluation summary")
    print(f"Discovered detector count: {len(discovered)}")
    print(f"Processed file count: {processed_file_count}")
    print(f"Completed detector-dataset pair count: {global_row['completed_pair_count']}")
    print(f"Zero-completion pair count: {global_row['zero_completion_pair_count']}")
    print(f"Total attempted recommendations: {global_row['rec_attempted_count']}")
    print(f"Total completed recommendations: {global_row['rec_completed_count']}")
    print(f"Global completion rate: {global_row['completion_rate']:.6f}")
    print("Generated output files:")
    for path in generated_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
