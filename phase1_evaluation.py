"""
Phase 1 evaluation.

Purpose
-------
Evaluate saved LODO models on each held-out detector-dataset pair.

Inputs
------
- setup-specific Phase 1 model artifacts
- processed held-out benchmark CSVs
- configured metadata representation

Outputs
-------
- held-out predictions, selected recommendation details, metric summaries, plots

Important behavior
------------------
Selection uses predictions only. Observed held-out transformed objectives and
real Pareto layers are used only after the selected set is frozen. This script
does not train or retrain models.
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any, Iterable

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from src.config import (
    COMPUTE_PREDICTION_UNCERTAINTY,
    DEFAULT_ACCURACY_WEIGHT,
    FINAL_SINGLE_TARGET_FORMULATIONS,
    LEGACY_SINGLE_TARGET_FORMULATIONS,
    PHASE1_MIN_COMPLETED_CONFIGS,
    PHASE1_MIN_COMPLETED_ENABLED,
    PHASE1_RECOMMENDATION_BUDGET,
    PCA_VARIANCES,
    PREPROCESSING_ACCURACY_MODE,
    PREPROCESSING_RUNTIME_CLIP_IQR_MULTIPLIER,
    PREPROCESSING_RUNTIME_CLIP_METHOD,
    PREPROCESSING_RUNTIME_CLIP_PERCENTILE,
    PREPROCESSING_RUNTIME_MODE,
    PREPROCESSING_RUNTIME_UPPER_CLIPPING,
    TRAIN_APD_ALPHA,
    TRAIN_APD_EVAL_RATIO,
    TRAIN_DATASETS,
    TRAIN_DETECTORS,
    SINGLE_TARGET_FORMULATION,
    TRAIN_METADATA_SCALE_METHOD,
    TRAIN_METADATA_VARIANT,
    TRAIN_MODEL_FAMILY,
    TRAIN_MODEL_PARAMS,
    TRAIN_NPD_THRESHOLDS,
    TRAIN_OVERWRITE_ARTIFACTS,
    TRAIN_PLOT_PARETO,
    TRAIN_PLOT_PARETO_LOG,
    TRAIN_PREFERENCE_REGION_ACCURACY_WEIGHTS,
    TRAIN_PREFERENCE_REGION_NAMES,
    TRAIN_PBI_THETA,
    TRAIN_RANDOM_STATE,
    TRAIN_SCALE_METADATA,
    TRAIN_SCALARIZATION_IDEAL_POINT,
    TRAIN_TARGET_MODE,
    TRAIN_TOP_K_VALUES,
    TRAIN_USE_METADATA,
    TRAIN_USE_PREFERENCE_REGIONS,
)
from src.evaluation import (
    _stable_config_keys,
    evaluate_separate_pareto,
    evaluate_regional_preference_regression,
    evaluate_single_regression,
)
from src.metrics import clean_global_metric_columns
from src.model_factory import merge_model_params, model_family_tag
from src.paths import ProjectPaths, get_paths_from_script
from src.sweeper_setup import add_pipeline_setup_args, resolve_pipeline_setup
from src.plotting import (
    plot_details_csv,
    plot_details_csv_log,
)
from src.target_utils import (
    REGIONAL_TARGETS,
    ordered_preference_regions,
    validate_distance_method,
)
from src.training_data import (
    build_lodo_data_bundle,
    get_pca_lodo_variance,
    get_pca_ranked_number_of_features,
    get_pca_ranked_selected_positions,
    metadata_variant_tag,
)
from src.utils import build_experiment_tag, ensure_dir, save_dataframe, save_json


def _validate_positive_int_values(name: str, values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"{name} values must be positive integers. Got {value!r}.")
        out.append(int(value))
    return sorted(set(out))


def _validate_unit_interval_values(name: str, values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{name} values must be finite and in [0, 1]. Got {value!r}.")
        out.append(numeric)
    return sorted(set(out))


def validate_supported_single_target(target_method: str) -> str:
    value = str(target_method).strip()
    if value in FINAL_SINGLE_TARGET_FORMULATIONS or value in LEGACY_SINGLE_TARGET_FORMULATIONS:
        return value
    raise ValueError(
        f"Unknown single-target formulation '{value}'. "
        f"Supported formulations: {list(FINAL_SINGLE_TARGET_FORMULATIONS + LEGACY_SINGLE_TARGET_FORMULATIONS)}."
    )


MODEL_FAMILY = TRAIN_MODEL_FAMILY
MODEL_PARAMS = TRAIN_MODEL_PARAMS
RANDOM_STATE = TRAIN_RANDOM_STATE
TARGET_MODE = TRAIN_TARGET_MODE
SINGLE_TARGET_METHOD = SINGLE_TARGET_FORMULATION
USE_PREFERENCE_REGIONS = TRAIN_USE_PREFERENCE_REGIONS
PREFERENCE_REGION_NAMES = list(TRAIN_PREFERENCE_REGION_NAMES)
USE_METADATA = TRAIN_USE_METADATA
METADATA_VARIANT = TRAIN_METADATA_VARIANT
SCALE_METADATA = TRAIN_SCALE_METADATA
METADATA_SCALE_METHOD = TRAIN_METADATA_SCALE_METHOD
NEEDS_UNCERTAINTY = COMPUTE_PREDICTION_UNCERTAINTY
OVERWRITE_ARTIFACTS = TRAIN_OVERWRITE_ARTIFACTS
PLOT_PARETO = TRAIN_PLOT_PARETO
PLOT_PARETO_LOG = TRAIN_PLOT_PARETO_LOG
TOP_K_VALUES = _validate_positive_int_values("TRAIN_TOP_K_VALUES", TRAIN_TOP_K_VALUES)
NPD_THRESHOLDS = _validate_unit_interval_values("TRAIN_NPD_THRESHOLDS", TRAIN_NPD_THRESHOLDS)
DETECTORS = TRAIN_DETECTORS
DATASETS = TRAIN_DATASETS


class Phase1EligibilitySkip(Exception):
    """Raised when a held-out pair is intentionally excluded from Phase 1 evaluation."""


def _regional_active() -> bool:
    return bool(
        USE_PREFERENCE_REGIONS
        and TARGET_MODE == "single"
        and SINGLE_TARGET_METHOD in REGIONAL_TARGETS
    )


def _ordered_preference_regions() -> list[tuple[str, float, float]]:
    return ordered_preference_regions(
        tuple(PREFERENCE_REGION_NAMES),
        TRAIN_PREFERENCE_REGION_ACCURACY_WEIGHTS,
    )


def _validate_phase1_regional_budget() -> None:
    region_count = len(PREFERENCE_REGION_NAMES)
    if int(PHASE1_RECOMMENDATION_BUDGET) % region_count != 0:
        raise ValueError(
            "PHASE1_RECOMMENDATION_BUDGET must be divisible by the number of "
            f"preference regions ({region_count}) for regional Phase 1 evaluation."
        )


def validate_evaluation_configuration() -> None:
    if TARGET_MODE == "single":
        validate_distance_method(SINGLE_TARGET_METHOD)
        validate_supported_single_target(SINGLE_TARGET_METHOD)
        if _regional_active():
            _ordered_preference_regions()
            _validate_phase1_regional_budget()
    elif TARGET_MODE == "separate":
        return
    else:
        raise ValueError("TRAIN_TARGET_MODE must be 'single' or 'separate'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved Phase 1 LODO models.")
    parser.add_argument("--detector", type=str, default="ALL", help="Detector name, ALL, or comma-separated exact names.")
    parser.add_argument("--dataset", type=str, default="ALL", help="Dataset name, ALL, or comma-separated exact names.")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate Phase 1 plots from existing held-out details CSVs without loading models.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Compute Phase 1 predictions and metrics without generating plot files.",
    )
    add_pipeline_setup_args(parser)
    return parser.parse_args()


def _apply_pipeline_setup(args: argparse.Namespace) -> None:
    """Apply sweep setup overrides when supplied; otherwise keep config defaults."""
    setup = resolve_pipeline_setup(
        args,
        default_target_mode=TRAIN_TARGET_MODE,
        default_single_target_formulation=SINGLE_TARGET_FORMULATION,
        default_use_metadata=TRAIN_USE_METADATA,
        default_metadata_variant=TRAIN_METADATA_VARIANT,
    )
    global TARGET_MODE, SINGLE_TARGET_METHOD, USE_METADATA, METADATA_VARIANT
    TARGET_MODE = setup.target_mode
    SINGLE_TARGET_METHOD = setup.single_target_formulation
    USE_METADATA = setup.use_metadata
    METADATA_VARIANT = setup.metadata_variant


def _resolve_cli_selection(selected: str, available: list[str], what: str) -> list[str]:
    selected = str(selected).strip()
    if selected.upper() == "ALL":
        return list(available)
    requested = [part.strip() for part in selected.split(",") if part.strip()]
    if not requested:
        raise ValueError(f"Empty {what} selection is not allowed.")
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise ValueError(f"Unknown {what}(s) {unknown}. Available: {available}")
    seen: set[str] = set()
    ordered: list[str] = []
    for item in requested:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _metadata_artifact_tag() -> str:
    return metadata_variant_tag(METADATA_VARIANT, use_metadata=USE_METADATA)


def _experiment_subfolders() -> tuple[str, str]:
    mode_tag = "SINGLE" if TARGET_MODE == "single" else "SEPARATE"
    family_tag = model_family_tag(MODEL_FAMILY)
    ranking_tag_source = "pareto_layer" if TARGET_MODE == "separate" else SINGLE_TARGET_METHOD
    if ranking_tag_source == "euc_dist":
        dist_tag = "EUC"
    elif ranking_tag_source == "mod_dist":
        dist_tag = "Dist"
    elif ranking_tag_source == "pareto_layer":
        dist_tag = "PL"
    elif ranking_tag_source == "pareto_loss":
        dist_tag = "PLOSS"
    elif ranking_tag_source == "pareto_rank":
        dist_tag = "PRANK"
    elif ranking_tag_source == "tchebycheff":
        dist_tag = "TCH"
    elif ranking_tag_source == "pbi":
        dist_tag = "PBI" + str(TRAIN_PBI_THETA).replace(".", "p")
    elif ranking_tag_source == "apd":
        dist_tag = (
            "APD"
            + "A" + str(TRAIN_APD_ALPHA).replace(".", "p")
            + "E" + str(TRAIN_APD_EVAL_RATIO).replace(".", "p")
        )
    else:
        dist_tag = "PS"
    if USE_METADATA:
        variant = str(METADATA_VARIANT).strip().lower()
        if variant == "lodo_pca":
            metadata_tag = "LPCAN" + f"{get_pca_lodo_variance():.2f}".replace(".", "p")
        elif variant == "lodo_pca_ranked":
            selected = get_pca_ranked_selected_positions()
            if selected:
                metadata_tag = "LPCAS" + "".join(str(v) for v in selected)
            else:
                metadata_tag = f"LPCAR{get_pca_ranked_number_of_features()}"
        else:
            metadata_tag = str(metadata_variant_tag(METADATA_VARIANT, use_metadata=True)).replace("META_", "M").replace("_", "")
    else:
        metadata_tag = "CFG"
    acc_tag = "AID" if PREPROCESSING_ACCURACY_MODE == "identity" else "AMM"
    rt_tag = "RLMM" if PREPROCESSING_RUNTIME_MODE == "log1p_minmax" else "RMM"
    if not PREPROCESSING_RUNTIME_UPPER_CLIPPING:
        clip_tag = "Cnone"
    elif PREPROCESSING_RUNTIME_CLIP_METHOD == "iqr_upper":
        clip_tag = f"Ci{PREPROCESSING_RUNTIME_CLIP_IQR_MULTIPLIER}"
    else:
        clip_tag = f"Cp{PREPROCESSING_RUNTIME_CLIP_PERCENTILE}"
    pref_region_tag = "PREG5" if _regional_active() else "PREG0"
    tag_parts = [
        family_tag,
        dist_tag,
        metadata_tag,
        acc_tag,
        rt_tag,
        clip_tag,
    ]
    if TARGET_MODE == "single":
        tag_parts.append("L" + str(DEFAULT_ACCURACY_WEIGHT).replace("0.", "").replace(".", ""))
    tag_parts.append(pref_region_tag)
    return mode_tag, build_experiment_tag(*tag_parts)


def _phase1_model_path(
    paths: ProjectPaths,
    detector: str,
    lodo_dataset: str,
    *,
    target_method: str,
    preference_region: str | None = None,
    objective: str | None = None,
) -> Path:
    """Resolve the unique Phase 1 model path created by train_models.py."""
    return paths.phase1_model_file(
        detector,
        lodo_dataset,
        target_mode=TARGET_MODE,
        target_method=target_method,
        metadata_tag=_metadata_artifact_tag(),
        preference_region=preference_region,
        objective=objective,
    )


def _prepare_loaded_model(model: Pipeline) -> Pipeline:
    estimator = model.named_steps.get("estimator") if hasattr(model, "named_steps") else None
    if estimator is not None and hasattr(estimator, "n_jobs"):
        estimator.n_jobs = MODEL_PARAMS.get("n_jobs", -1)
    return model


def _load_model_or_fail(path: Path, *, label: str) -> Pipeline:
    """Load the exact setup-specific fitted model without retraining."""
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}. "
            "Run train_models.py first with matching configuration settings."
        )
    return _prepare_loaded_model(load(path))


def _load_single_model(paths: ProjectPaths, detector: str, lodo_dataset: str, *, target_method: str) -> Pipeline:
    return _load_model_or_fail(
        _phase1_model_path(paths, detector, lodo_dataset, target_method=target_method),
        label="Single Phase 1 LODO model",
    )


def _load_separate_models(paths: ProjectPaths, detector: str, lodo_dataset: str) -> tuple[Pipeline, Pipeline]:
    acc_model = _load_model_or_fail(
        _phase1_model_path(paths, detector, lodo_dataset, target_method="separate", objective="accuracy"),
        label="Separate accuracy Phase 1 LODO model",
    )
    runtime_model = _load_model_or_fail(
        _phase1_model_path(paths, detector, lodo_dataset, target_method="separate", objective="runtime"),
        label="Separate runtime Phase 1 LODO model",
    )
    return acc_model, runtime_model


def _load_region_model(paths: ProjectPaths, detector: str, lodo_dataset: str, *, region_name: str) -> Pipeline:
    return _load_model_or_fail(
        _phase1_model_path(
            paths,
            detector,
            lodo_dataset,
            target_method=SINGLE_TARGET_METHOD,
            preference_region=region_name,
        ),
        label=f"Preference-region Phase 1 model ({region_name})",
    )


def _predict_with_uncertainty(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = np.asarray(model.predict(X), dtype=float)
    estimator = model.named_steps["estimator"]
    preprocessor = model.named_steps["preprocessor"]
    if not hasattr(estimator, "estimators_"):
        nan_arr = np.full(len(preds), np.nan, dtype=float)
        return preds, nan_arr, nan_arr
    Xt = preprocessor.transform(X)
    member_preds = np.column_stack([tree.predict(Xt) for tree in estimator.estimators_])
    tree_std = np.nanstd(member_preds, axis=1)
    tree_iqr = np.nanpercentile(member_preds, 75, axis=1) - np.nanpercentile(member_preds, 25, axis=1)
    return preds, tree_std, tree_iqr


def _predict_for_evaluation(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if NEEDS_UNCERTAINTY:
        return _predict_with_uncertainty(model, X)
    preds = np.asarray(model.predict(X), dtype=float)
    nan_arr = np.full(len(preds), np.nan, dtype=float)
    return preds, nan_arr, nan_arr


def _pca_lodo_component_counts(paths: ProjectPaths, lodo_datasets: Iterable[str]) -> list[int]:
    variance = get_pca_lodo_variance()
    counts: list[int] = []
    for dataset_name in sorted({str(name) for name in lodo_datasets if pd.notna(name)}):
        path = paths.metadata_pca_lodo_file(dataset_name, variance)
        if not path.exists():
            continue
        columns = pd.read_csv(path, nrows=0).columns
        counts.append(int(sum(str(col) != "dataset_name" for col in columns)))
    return counts


def _metadata_global_columns(paths: ProjectPaths, *, lodo_datasets: Iterable[str] | None = None) -> dict[str, Any]:
    if not USE_METADATA:
        return {"metadata": "CFGONLY", "number_of_features": 0}

    variant = str(METADATA_VARIANT).strip()
    lower = variant.lower()
    if lower == "lodo_pca":
        variance = get_pca_lodo_variance()
        datasets_for_count = [] if lodo_datasets is None else list(lodo_datasets)
        counts = _pca_lodo_component_counts(paths, datasets_for_count)
        n_features = float(np.mean(counts)) if counts else float("nan")
        n_features_min = int(np.min(counts)) if counts else float("nan")
        n_features_max = int(np.max(counts)) if counts else float("nan")
        return {
            "metadata": f"{variant}_{variance:.2f}",
            "number_of_features": n_features,
            "number_of_features_min": n_features_min,
            "number_of_features_max": n_features_max,
        }
    if lower == "lodo_pca_ranked":
        return {"metadata": variant, "number_of_features": int(get_pca_ranked_number_of_features())}

    metadata_path_by_variant = {
        "all": paths.all_meta_features_file,
        "cleaned": paths.cleaned_meta_features_file,
        "pruned": paths.pruned_meta_features_file,
    }
    metadata_path = metadata_path_by_variant.get(lower)
    if metadata_path is None or not metadata_path.exists():
        return {"metadata": variant, "number_of_features": float("nan")}

    columns = pd.read_csv(metadata_path, nrows=0).columns
    return {
        "metadata": variant,
        "number_of_features": int(sum(str(col) != "dataset_name" for col in columns)),
    }


def _effective_run_config(
    *,
    paths: ProjectPaths,
    run_dir: Path,
    experiment_parent: str,
    experiment_tag: str,
    selected_detectors: list[str],
    selected_datasets: list[str],
) -> dict[str, Any]:
    return {
        "SCRIPT": "phase1_evaluation.py",
        "RESPONSIBILITY": "phase1_evaluation_only",
        "TARGET_MODE": TARGET_MODE,
        "SINGLE_TARGET_FORMULATION": SINGLE_TARGET_METHOD if TARGET_MODE == "single" else "not_applicable",
        "SEPARATE_SELECTION_METHOD": "pareto_layer" if TARGET_MODE == "separate" else "not_applicable",
        "EXPERIMENT_PARENT": experiment_parent,
        "EXPERIMENT_TAG": experiment_tag,
        "RESULT_DIRECTORY": str(run_dir),
        "FINAL_SINGLE_TARGET_FORMULATIONS": list(FINAL_SINGLE_TARGET_FORMULATIONS),
        "LEGACY_SINGLE_TARGET_FORMULATIONS": list(LEGACY_SINGLE_TARGET_FORMULATIONS),
        "TRAIN_MODEL_FAMILY": MODEL_FAMILY,
        "EFFECTIVE_MODEL_PARAMS": merge_model_params(
            MODEL_FAMILY,
            random_state=RANDOM_STATE,
            override_params=MODEL_PARAMS,
        ),
        "TRAIN_RANDOM_STATE": RANDOM_STATE,
        "TRAIN_USE_METADATA": USE_METADATA,
        "TRAIN_METADATA_VARIANT": METADATA_VARIANT,
        "TRAIN_SCALE_METADATA": SCALE_METADATA,
        "TRAIN_METADATA_SCALE_METHOD": METADATA_SCALE_METHOD,
        "PCA_VARIANCES": list(PCA_VARIANCES),
        "EFFECTIVE_PCA_LODO_VARIANCE": float(get_pca_lodo_variance()),
        "EFFECTIVE_PCA_RANKED_NUMBER_OF_FEATURES": int(get_pca_ranked_number_of_features()),
        "EFFECTIVE_PCA_RANKED_SELECTED_POSITIONS": get_pca_ranked_selected_positions(),
        "SELECTED_DETECTORS": list(selected_detectors),
        "SELECTED_DATASETS": list(selected_datasets),
        "DEFAULT_ACCURACY_WEIGHT": DEFAULT_ACCURACY_WEIGHT,
        "TRAIN_SCALARIZATION_IDEAL_POINT": TRAIN_SCALARIZATION_IDEAL_POINT,
        "TRAIN_PBI_THETA": TRAIN_PBI_THETA,
        "TRAIN_APD_ALPHA": TRAIN_APD_ALPHA,
        "TRAIN_APD_EVAL_RATIO": TRAIN_APD_EVAL_RATIO,
        "TRAIN_TOP_K_VALUES": TOP_K_VALUES,
        "TRAIN_NPD_THRESHOLDS": NPD_THRESHOLDS,
        "PHASE1_MIN_COMPLETED_ENABLED": PHASE1_MIN_COMPLETED_ENABLED,
        "PHASE1_MIN_COMPLETED_CONFIGS": PHASE1_MIN_COMPLETED_CONFIGS,
        "TRAIN_USE_PREFERENCE_REGIONS": TRAIN_USE_PREFERENCE_REGIONS,
        "REGIONAL_ACTIVE": _regional_active(),
        "TRAIN_PREFERENCE_REGION_NAMES": TRAIN_PREFERENCE_REGION_NAMES,
        "TRAIN_PREFERENCE_REGION_ACCURACY_WEIGHTS": TRAIN_PREFERENCE_REGION_ACCURACY_WEIGHTS,
        "COMPUTE_PREDICTION_UNCERTAINTY": COMPUTE_PREDICTION_UNCERTAINTY,
        "PROJECT_ROOT": str(paths.project_root),
    }


def _save_summary_bundle(target_dir: Path, aggregate_rows: list[dict[str, Any]], *, paths: ProjectPaths) -> None:
    aggregate_df = pd.DataFrame(aggregate_rows)
    save_dataframe(aggregate_df, target_dir / "aggregate_metrics.csv", index=False)
    registered_metric_cols = clean_global_metric_columns()
    metric_cols = [
        c
        for c in registered_metric_cols
        if c in aggregate_df.columns and pd.api.types.is_numeric_dtype(pd.to_numeric(aggregate_df[c], errors="coerce"))
    ] if not aggregate_df.empty else registered_metric_cols
    if aggregate_df.empty:
        detector_columns = ["detector"]
        for col in metric_cols:
            detector_columns.extend([col, f"n_valid_datasets__{col}"])
        save_dataframe(pd.DataFrame(columns=detector_columns), target_dir / "detector_means.csv", index=False)
        global_empty = _metadata_global_columns(paths, lodo_datasets=[])
        global_empty["n_completed_pairs"] = 0
        global_empty["n_detectors_with_results"] = 0
        for col in metric_cols:
            global_empty[col] = float("nan")
            global_empty[f"n_valid_detectors__{col}"] = 0
            global_empty[f"n_valid_pairs__{col}"] = 0
        save_dataframe(pd.DataFrame([global_empty]), target_dir / "global_means.csv", index=False)
        return

    detector_rows: list[dict[str, Any]] = []
    for detector, group in aggregate_df.groupby("detector", dropna=False):
        row: dict[str, Any] = {"detector": detector}
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[col] = float(values.mean())
            row[f"n_valid_datasets__{col}"] = int(values.notna().sum())
        detector_rows.append(row)
    detector_means_df = pd.DataFrame(detector_rows)
    save_dataframe(detector_means_df, target_dir / "detector_means.csv", index=False)

    lodo_datasets = aggregate_df["lodo_dataset"].dropna().astype(str).unique() if "lodo_dataset" in aggregate_df.columns else []
    global_row: dict[str, Any] = _metadata_global_columns(paths, lodo_datasets=lodo_datasets)
    global_row["n_completed_pairs"] = int(len(aggregate_df))
    global_row["n_detectors_with_results"] = int(aggregate_df["detector"].nunique(dropna=True)) if "detector" in aggregate_df.columns else 0
    for col in metric_cols:
        detector_values = pd.to_numeric(detector_means_df[col], errors="coerce")
        pair_values = pd.to_numeric(aggregate_df[col], errors="coerce")
        global_row[col] = float(detector_values.mean())
        global_row[f"n_valid_detectors__{col}"] = int(detector_values.notna().sum())
        global_row[f"n_valid_pairs__{col}"] = int(pair_values.notna().sum())
    save_dataframe(pd.DataFrame([global_row]), target_dir / "global_means.csv", index=False)


SKIPPED_PAIR_COLUMNS = [
    "detector",
    "lodo_dataset",
    "target_mode",
    "single_target_formulation",
    "separate_selection_method",
    "use_metadata",
    "metadata_variant",
    "stage",
    "exception_type",
    "error_message",
]

SKIPPED_PLOT_COLUMNS = [
    "detector",
    "lodo_dataset",
    "target_mode",
    "single_target_formulation",
    "metadata_variant",
    "plot_stage",
    "exception_type",
    "error_message",
    "reason",
    "n_heldout_configs",
]


def _save_failure_reports(target_dir: Path, skipped_pairs: list[dict[str, Any]], skipped_plots: list[dict[str, Any]]) -> None:
    save_dataframe(pd.DataFrame(skipped_pairs, columns=SKIPPED_PAIR_COLUMNS), target_dir / "skipped_pairs.csv", index=False)
    save_dataframe(pd.DataFrame(skipped_plots, columns=SKIPPED_PLOT_COLUMNS), target_dir / "skipped_plots.csv", index=False)


def _save_run_outputs(
    target_dir: Path,
    aggregate_rows: list[dict[str, Any]],
    skipped_pairs: list[dict[str, Any]],
    skipped_plots: list[dict[str, Any]],
    *,
    paths: ProjectPaths,
) -> None:
    _save_summary_bundle(target_dir, aggregate_rows, paths=paths)
    _save_failure_reports(target_dir, skipped_pairs, skipped_plots)


def _skip_pair_record(*, detector: str, lodo_dataset: str, stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "detector": detector,
        "lodo_dataset": lodo_dataset,
        "target_mode": TARGET_MODE,
        "single_target_formulation": SINGLE_TARGET_METHOD if TARGET_MODE == "single" else "not_applicable",
        "separate_selection_method": "pareto_layer" if TARGET_MODE == "separate" else "not_applicable",
        "use_metadata": USE_METADATA,
        "metadata_variant": METADATA_VARIANT,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _skip_plot_record(
    *,
    detector: str,
    lodo_dataset: str,
    plot_stage: str,
    reason: str,
    n_heldout_configs: int | float,
    exc: Exception | None = None,
) -> dict[str, Any]:
    return {
        "detector": detector,
        "lodo_dataset": lodo_dataset,
        "target_mode": TARGET_MODE,
        "single_target_formulation": SINGLE_TARGET_METHOD if TARGET_MODE == "single" else "not_applicable",
        "metadata_variant": METADATA_VARIANT,
        "plot_stage": plot_stage,
        "exception_type": type(exc).__name__ if exc is not None else "",
        "error_message": str(exc) if exc is not None else "",
        "reason": reason,
        "n_heldout_configs": n_heldout_configs,
    }


def _details_csv_path(paths: ProjectPaths, run_dir: Path, detector: str, lodo_dataset: str) -> Path:
    parent, child = run_dir.relative_to(paths.held_out_evaluation_dir).parts[:2]
    return paths.phase1_details_file(parent, child, detector, lodo_dataset)


def _available_plot_ks(summary_row: dict[str, Any]) -> list[int]:
    return [
        k
        for k in TOP_K_VALUES
        if bool(summary_row.get(f"metrics_available_at_{k}", False))
        and int(summary_row.get(f"n_selected_at_{k}", 0)) > 0
    ]


def _selected_column_has_values(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values):
        return bool(values.fillna(False).any())
    if pd.api.types.is_numeric_dtype(values):
        return bool(pd.to_numeric(values, errors="coerce").fillna(0).ne(0).any())
    normalized = values.astype(str).str.strip().str.lower()
    return bool(normalized.isin({"true", "1", "yes", "y"}).any())


def _available_plot_ks_from_details(details_df: pd.DataFrame) -> list[int]:
    available: list[int] = []
    for k in TOP_K_VALUES:
        selected_column = f"selected_at_{k}"
        if selected_column in details_df.columns and _selected_column_has_values(details_df[selected_column]):
            available.append(k)
    return available


def _plot_distance_column() -> str:
    if TARGET_MODE == "separate":
        return "pareto_layer"
    return SINGLE_TARGET_METHOD


def _plot_exports(
    *,
    details_csv_path: Path,
    plots_root: Path,
    detector: str,
    dataset: str,
    distance_column: str,
    recommended_limit: int,
    context_csv_path: Path | None = None,
    selected_column: str | None = None,
) -> None:
    if PLOT_PARETO:
        pareto_dir = ensure_dir(plots_root / "plots_pareto" / detector)
        plot_details_csv(
            details_csv_path=details_csv_path,
            output_path=pareto_dir / f"{detector}_LODO_{dataset}_pareto.png",
            detector=detector,
            dataset=dataset,
            distance_column=distance_column,
            context_csv_path=context_csv_path,
            recommended_limit=recommended_limit,
            selected_column=selected_column,
        )
    if PLOT_PARETO_LOG:
        pareto_log_dir = ensure_dir(plots_root / "plots_pareto_log" / detector)
        plot_details_csv_log(
            details_csv_path=details_csv_path,
            output_path=pareto_log_dir / f"{detector}_LODO_{dataset}_pareto_log.png",
            detector=detector,
            dataset=dataset,
            distance_column=distance_column,
            context_csv_path=context_csv_path,
            recommended_limit=recommended_limit,
            selected_column=selected_column,
        )
def _plot_available_details(
    *,
    details_csv_path: Path,
    plots_dir: Path,
    detector: str,
    lodo_dataset: str,
    distance_column: str,
    available_plot_ks: list[int],
    skipped_plots: list[dict[str, Any]],
    n_heldout_configs: int | float,
    paths: ProjectPaths,
) -> None:
    if not available_plot_ks:
        reason = "No configured top-k recommendation set is available."
        print(f"    Skipping plots for {detector} / {lodo_dataset}: {reason}")
        skipped_plots.append(
            _skip_plot_record(
                detector=detector,
                lodo_dataset=lodo_dataset,
                plot_stage="plotting",
                reason=reason,
                n_heldout_configs=n_heldout_configs,
            )
        )
        return

    plot_k = max(available_plot_ks)
    try:
        _plot_exports(
            details_csv_path=details_csv_path,
            plots_root=plots_dir,
            detector=detector,
            dataset=lodo_dataset,
            distance_column=distance_column,
            recommended_limit=plot_k,
            context_csv_path=paths.processed_benchmark_file(detector, lodo_dataset),
            selected_column=f"selected_at_{plot_k}",
        )
    except Exception as exc:
        print(f"    Plotting failed for {detector} / {lodo_dataset}: {exc}")
        traceback.print_exc()
        skipped_plots.append(
            _skip_plot_record(
                detector=detector,
                lodo_dataset=lodo_dataset,
                plot_stage="plotting",
                reason="Plotting failed.",
                n_heldout_configs=n_heldout_configs,
                exc=exc,
            )
        )


def _has_single_test_split(bundle) -> bool:
    return bundle.X_test is not None and bundle.test_df is not None and bundle.y_test_single is not None


def _has_separate_test_split(bundle) -> bool:
    return bundle.X_test is not None and bundle.test_df is not None and bundle.y_test_separate is not None


def _phase1_completed_count(test_df: pd.DataFrame | None) -> int:
    """Count completed held-out configurations available to Phase 1 evaluation."""
    if test_df is None:
        return 0
    return int(len(test_df))


def _ensure_phase1_pair_eligible(*, detector: str, lodo_dataset: str, test_df: pd.DataFrame | None) -> int:
    """Apply the held-out-only minimum completed-configuration threshold."""
    completed_count = _phase1_completed_count(test_df)
    if completed_count <= 0:
        raise Phase1EligibilitySkip(
            f"Skipping Phase 1 evaluation for {detector}-{lodo_dataset}: "
            "0 completed configurations are available."
        )
    if bool(PHASE1_MIN_COMPLETED_ENABLED) and completed_count < int(PHASE1_MIN_COMPLETED_CONFIGS):
        raise Phase1EligibilitySkip(
            f"Skipping Phase 1 evaluation for {detector}-{lodo_dataset}: "
            f"{completed_count} completed configurations < configured minimum {int(PHASE1_MIN_COMPLETED_CONFIGS)}."
        )
    return completed_count


def _regional_alignment_keys(df: pd.DataFrame, configuration_columns: list[str]) -> pd.Series:
    key_columns = ["dataset_name", *configuration_columns]
    missing = [column for column in key_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Regional LODO alignment key is missing column(s): {missing}")
    return _stable_config_keys(df, key_columns).reset_index(drop=True)


def _first_differing_row(left: pd.Series, right: pd.Series) -> int:
    left_values = left.reset_index(drop=True)
    right_values = right.reset_index(drop=True)
    limit = min(len(left_values), len(right_values))
    if limit:
        differs = left_values.iloc[:limit].ne(right_values.iloc[:limit])
        if bool(differs.any()):
            return int(np.flatnonzero(differs.to_numpy())[0])
    return int(limit)


def _regional_alignment_error(*, detector: str, lodo_dataset: str, region_name: str, mismatch: str) -> ValueError:
    return ValueError(
        "Regional LODO alignment failure for "
        f"{SINGLE_TARGET_METHOD.upper()} / {detector} / {lodo_dataset} / {region_name}: {mismatch}."
    )


def _validate_regional_bundle_alignment(
    *,
    detector: str,
    lodo_dataset: str,
    region_name: str,
    bundle: Any,
    baseline_feature_columns: list[str],
    baseline_configuration_columns: list[str],
    baseline_numeric_columns: list[str],
    baseline_categorical_columns: list[str],
    baseline_train_keys: pd.Series,
    baseline_test_keys: pd.Series,
    baseline_train_length: int,
    baseline_test_length: int,
    baseline_x_train_columns: list[str],
    baseline_x_test_columns: list[str],
) -> None:
    checks = [
        (list(bundle.feature_columns), baseline_feature_columns, "feature columns differ from the first regional bundle"),
        (list(bundle.configuration_feature_columns), baseline_configuration_columns, "configuration feature columns differ from the first regional bundle"),
        (list(bundle.numeric_feature_columns), baseline_numeric_columns, "numeric feature columns differ from the first regional bundle"),
        (list(bundle.categorical_feature_columns), baseline_categorical_columns, "categorical feature columns differ from the first regional bundle"),
        (list(bundle.X_train.columns), baseline_x_train_columns, "X_train columns differ from the first regional bundle"),
        (list(bundle.X_test.columns), baseline_x_test_columns, "X_test columns differ from the first regional bundle"),
    ]
    for actual, expected, mismatch in checks:
        if actual != expected:
            raise _regional_alignment_error(detector=detector, lodo_dataset=lodo_dataset, region_name=region_name, mismatch=mismatch)
    if len(bundle.train_df) != baseline_train_length:
        raise _regional_alignment_error(detector=detector, lodo_dataset=lodo_dataset, region_name=region_name, mismatch=f"training row count differs from the first regional bundle ({len(bundle.train_df)} != {baseline_train_length})")
    if len(bundle.test_df) != baseline_test_length:
        raise _regional_alignment_error(detector=detector, lodo_dataset=lodo_dataset, region_name=region_name, mismatch=f"held-out row count differs from the first regional bundle ({len(bundle.test_df)} != {baseline_test_length})")
    regional_train_keys = _regional_alignment_keys(bundle.train_df, baseline_configuration_columns)
    if not regional_train_keys.equals(baseline_train_keys):
        row = _first_differing_row(regional_train_keys, baseline_train_keys)
        raise _regional_alignment_error(detector=detector, lodo_dataset=lodo_dataset, region_name=region_name, mismatch=f"training configuration order differs from the first regional bundle at row {row}")
    regional_test_keys = _regional_alignment_keys(bundle.test_df, baseline_configuration_columns)
    if not regional_test_keys.equals(baseline_test_keys):
        row = _first_differing_row(regional_test_keys, baseline_test_keys)
        raise _regional_alignment_error(detector=detector, lodo_dataset=lodo_dataset, region_name=region_name, mismatch=f"held-out configuration order differs from the first regional bundle at row {row}")


def _build_bundle(paths: ProjectPaths, detector: str, lodo_dataset: str, *, target_mode: str | None = None, single_target_column: str | None = None):
    active_target_mode = target_mode or TARGET_MODE
    return build_lodo_data_bundle(
        paths=paths,
        detector=detector,
        lodo_dataset=lodo_dataset,
        datasets=DATASETS,
        target_mode=active_target_mode,
        distance_method=SINGLE_TARGET_METHOD,
        lambda_value=DEFAULT_ACCURACY_WEIGHT,
        use_metadata=USE_METADATA,
        metadata_variant=METADATA_VARIANT,
        scale_metadata=SCALE_METADATA,
        metadata_scale_method=METADATA_SCALE_METHOD,
        single_target_column=single_target_column,
    )


def _run_plain_single_or_separate(
    *,
    paths: ProjectPaths,
    run_dir: Path,
    selected_detectors: list[str],
    selected_datasets: list[str],
    generate_plots: bool,
) -> None:
    aggregate_rows: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, Any]] = []
    skipped_plots: list[dict[str, Any]] = []
    plots_dir = ensure_dir(run_dir / "all_plots") if generate_plots else run_dir / "all_plots"
    try:
        for detector in selected_detectors:
            print(f"\n=== Detector: {detector} ===")
            for lodo_dataset in selected_datasets:
                print(f"  -> LODO: {lodo_dataset}")
                stage = "build_lodo_data"
                try:
                    bundle = _build_bundle(paths, detector, lodo_dataset)
                    _ensure_phase1_pair_eligible(detector=detector, lodo_dataset=lodo_dataset, test_df=bundle.test_df)
                    if TARGET_MODE == "single":
                        if not _has_single_test_split(bundle):
                            raise FileNotFoundError("Missing processed LODO test split for single-target evaluation.")
                        target_method = bundle.train_target_name or SINGLE_TARGET_METHOD
                        stage = "load_model"
                        model = _load_single_model(paths, detector, lodo_dataset, target_method=target_method)
                        stage = "prediction"
                        preds, tree_std, tree_iqr = _predict_for_evaluation(model, bundle.X_test)
                        stage = "evaluation"
                        evaluation = evaluate_single_regression(
                            detector=detector,
                            lodo_dataset=lodo_dataset,
                            test_df=bundle.test_df,
                            y_true=bundle.y_test_single,
                            y_pred=preds,
                            target_column=target_method,
                            top_k_values=TOP_K_VALUES,
                            tree_std=(tree_std if NEEDS_UNCERTAINTY else None),
                            tree_iqr=(tree_iqr if NEEDS_UNCERTAINTY else None),
                        )
                        distance_for_plots = target_method
                    else:
                        if not _has_separate_test_split(bundle):
                            raise FileNotFoundError("Missing processed LODO test split for separate-objective evaluation.")
                        stage = "load_models"
                        acc_model, rt_model = _load_separate_models(paths, detector, lodo_dataset)
                        stage = "prediction"
                        pred_acc, acc_std, acc_iqr = _predict_for_evaluation(acc_model, bundle.X_test)
                        pred_rt, rt_std, rt_iqr = _predict_for_evaluation(rt_model, bundle.X_test)
                        stage = "evaluation"
                        evaluation = evaluate_separate_pareto(
                            detector=detector,
                            lodo_dataset=lodo_dataset,
                            test_df=bundle.test_df,
                            pred_transformed_accuracy=pred_acc,
                            pred_transformed_runtime=pred_rt,
                            distance_method="pareto_layer",
                            ranking_method="pareto_layer",
                            top_k_values=TOP_K_VALUES,
                            configuration_columns=bundle.configuration_feature_columns,
                            accuracy_tree_std=(acc_std if NEEDS_UNCERTAINTY else None),
                            accuracy_tree_iqr=(acc_iqr if NEEDS_UNCERTAINTY else None),
                            runtime_tree_std=(rt_std if NEEDS_UNCERTAINTY else None),
                            runtime_tree_iqr=(rt_iqr if NEEDS_UNCERTAINTY else None),
                        )
                        distance_for_plots = "pareto_layer"

                    details_df = evaluation.details_df
                    details_csv_path = _details_csv_path(paths, run_dir, detector, lodo_dataset)
                    stage = "save_details"
                    save_dataframe(details_df, details_csv_path, index=False)
                    aggregate_rows.append(evaluation.summary_row)
                    if generate_plots:
                        _plot_available_details(
                            details_csv_path=details_csv_path,
                            plots_dir=plots_dir,
                            detector=detector,
                            lodo_dataset=lodo_dataset,
                            distance_column=distance_for_plots,
                            available_plot_ks=_available_plot_ks(evaluation.summary_row),
                            skipped_plots=skipped_plots,
                            n_heldout_configs=evaluation.summary_row.get("n_heldout_configs", len(details_df)),
                            paths=paths,
                        )
                except Phase1EligibilitySkip as exc:
                    print(f"    {exc}")
                    skipped_pairs.append(_skip_pair_record(detector=detector, lodo_dataset=lodo_dataset, stage="phase1_eligibility", exc=exc))
                except Exception as exc:
                    print(f"    Skipping pair {detector} / {lodo_dataset} at stage '{stage}': {exc}")
                    traceback.print_exc()
                    skipped_pairs.append(_skip_pair_record(detector=detector, lodo_dataset=lodo_dataset, stage=stage, exc=exc))
                finally:
                    _save_run_outputs(run_dir, aggregate_rows, skipped_pairs, skipped_plots, paths=paths)
    finally:
        _save_run_outputs(run_dir, aggregate_rows, skipped_pairs, skipped_plots, paths=paths)


def _run_regional_preference_single(
    *,
    paths: ProjectPaths,
    run_dir: Path,
    selected_detectors: list[str],
    selected_datasets: list[str],
    generate_plots: bool,
) -> None:
    aggregate_rows: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, Any]] = []
    skipped_plots: list[dict[str, Any]] = []
    ordered_regions = _ordered_preference_regions()
    plots_dir = ensure_dir(run_dir / "all_plots") if generate_plots else run_dir / "all_plots"
    try:
        for detector in selected_detectors:
            print(f"\n=== Detector: {detector} ===")
            for lodo_dataset in selected_datasets:
                print(f"  -> LODO: {lodo_dataset}")
                stage = "build_lodo_data"
                try:
                    predictions_by_region: dict[str, np.ndarray] = {}
                    tree_std_by_region: dict[str, np.ndarray] = {}
                    tree_iqr_by_region: dict[str, np.ndarray] = {}
                    common_test_df: pd.DataFrame | None = None
                    baseline_feature_columns: list[str] | None = None
                    baseline_configuration_columns: list[str] | None = None
                    baseline_numeric_columns: list[str] | None = None
                    baseline_categorical_columns: list[str] | None = None
                    baseline_train_keys: pd.Series | None = None
                    baseline_test_keys: pd.Series | None = None
                    baseline_train_length: int | None = None
                    baseline_test_length: int | None = None
                    baseline_x_train_columns: list[str] | None = None
                    baseline_x_test_columns: list[str] | None = None

                    for region_name, _, _ in ordered_regions:
                        region_target = f"{SINGLE_TARGET_METHOD}_{region_name}"
                        bundle = _build_bundle(paths, detector, lodo_dataset, target_mode="single", single_target_column=region_target)
                        if baseline_feature_columns is None:
                            _ensure_phase1_pair_eligible(detector=detector, lodo_dataset=lodo_dataset, test_df=bundle.test_df)
                        if not _has_single_test_split(bundle):
                            raise FileNotFoundError("Missing processed LODO test split for regional evaluation.")
                        if baseline_feature_columns is None:
                            baseline_feature_columns = list(bundle.feature_columns)
                            baseline_configuration_columns = list(bundle.configuration_feature_columns)
                            baseline_numeric_columns = list(bundle.numeric_feature_columns)
                            baseline_categorical_columns = list(bundle.categorical_feature_columns)
                            baseline_train_keys = _regional_alignment_keys(bundle.train_df, baseline_configuration_columns)
                            baseline_test_keys = _regional_alignment_keys(bundle.test_df, baseline_configuration_columns)
                            baseline_train_length = len(bundle.train_df)
                            baseline_test_length = len(bundle.test_df)
                            baseline_x_train_columns = list(bundle.X_train.columns)
                            baseline_x_test_columns = list(bundle.X_test.columns)
                        else:
                            _validate_regional_bundle_alignment(
                                detector=detector,
                                lodo_dataset=lodo_dataset,
                                region_name=region_name,
                                bundle=bundle,
                                baseline_feature_columns=baseline_feature_columns,
                                baseline_configuration_columns=baseline_configuration_columns,
                                baseline_numeric_columns=baseline_numeric_columns,
                                baseline_categorical_columns=baseline_categorical_columns,
                                baseline_train_keys=baseline_train_keys,
                                baseline_test_keys=baseline_test_keys,
                                baseline_train_length=baseline_train_length,
                                baseline_test_length=baseline_test_length,
                                baseline_x_train_columns=baseline_x_train_columns,
                                baseline_x_test_columns=baseline_x_test_columns,
                            )

                        stage = f"load_model_{region_name}"
                        model = _load_region_model(paths, detector, lodo_dataset, region_name=region_name)
                        stage = f"prediction_{region_name}"
                        preds, tree_std, tree_iqr = _predict_for_evaluation(model, bundle.X_test)
                        predictions_by_region[region_name] = preds
                        if NEEDS_UNCERTAINTY:
                            tree_std_by_region[region_name] = tree_std
                            tree_iqr_by_region[region_name] = tree_iqr
                        if common_test_df is None:
                            common_test_df = bundle.test_df.copy()

                    if common_test_df is None:
                        raise FileNotFoundError("Missing processed LODO test split for regional evaluation.")

                    stage = "evaluation"
                    evaluation = evaluate_regional_preference_regression(
                        detector=detector,
                        lodo_dataset=lodo_dataset,
                        test_df=common_test_df,
                        target_column=SINGLE_TARGET_METHOD,
                        region_names=tuple(region for region, _, _ in ordered_regions),
                        region_accuracy_weights=TRAIN_PREFERENCE_REGION_ACCURACY_WEIGHTS,
                        configuration_columns=baseline_configuration_columns,
                        predictions_by_region=predictions_by_region,
                        top_k_values=TOP_K_VALUES,
                        tree_std_by_region=(tree_std_by_region if NEEDS_UNCERTAINTY else None),
                        tree_iqr_by_region=(tree_iqr_by_region if NEEDS_UNCERTAINTY else None),
                    )
                    details_df = evaluation.details_df
                    details_csv_path = _details_csv_path(paths, run_dir, detector, lodo_dataset)
                    stage = "save_details"
                    save_dataframe(details_df, details_csv_path, index=False)
                    aggregate_rows.append(evaluation.summary_row)
                    if generate_plots:
                        _plot_available_details(
                            details_csv_path=details_csv_path,
                            plots_dir=plots_dir,
                            detector=detector,
                            lodo_dataset=lodo_dataset,
                            distance_column=SINGLE_TARGET_METHOD,
                            available_plot_ks=_available_plot_ks(evaluation.summary_row),
                            skipped_plots=skipped_plots,
                            n_heldout_configs=evaluation.summary_row.get("n_heldout_configs", len(details_df)),
                            paths=paths,
                        )
                except Phase1EligibilitySkip as exc:
                    print(f"    {exc}")
                    skipped_pairs.append(_skip_pair_record(detector=detector, lodo_dataset=lodo_dataset, stage="phase1_eligibility", exc=exc))
                except Exception as exc:
                    print(f"    Skipping pair {detector} / {lodo_dataset} at stage '{stage}': {exc}")
                    traceback.print_exc()
                    skipped_pairs.append(_skip_pair_record(detector=detector, lodo_dataset=lodo_dataset, stage=stage, exc=exc))
                finally:
                    _save_run_outputs(run_dir, aggregate_rows, skipped_pairs, skipped_plots, paths=paths)
    finally:
        _save_run_outputs(run_dir, aggregate_rows, skipped_pairs, skipped_plots, paths=paths)


def _run_plotting_from_existing_details(*, paths: ProjectPaths, run_dir: Path, selected_detectors: list[str], selected_datasets: list[str]) -> None:
    skipped_plots: list[dict[str, Any]] = []
    plots_dir = ensure_dir(run_dir / "all_plots")
    distance_column = _plot_distance_column()

    for detector in selected_detectors:
        print(f"\n=== Plotting detector: {detector} ===")
        for lodo_dataset in selected_datasets:
            print(f"  -> LODO: {lodo_dataset}")
            details_csv_path = _details_csv_path(paths, run_dir, detector, lodo_dataset)
            try:
                if not details_csv_path.exists():
                    raise FileNotFoundError(f"Details CSV not found: {details_csv_path}")
                details_df = pd.read_csv(details_csv_path)
                _plot_available_details(
                    details_csv_path=details_csv_path,
                    plots_dir=plots_dir,
                    detector=detector,
                    lodo_dataset=lodo_dataset,
                    distance_column=distance_column,
                    available_plot_ks=_available_plot_ks_from_details(details_df),
                    skipped_plots=skipped_plots,
                    n_heldout_configs=len(details_df),
                    paths=paths,
                )
            except Exception as exc:
                print(f"    Skipping plots for {detector} / {lodo_dataset}: {exc}")
                traceback.print_exc()
                skipped_plots.append(
                    _skip_plot_record(
                        detector=detector,
                        lodo_dataset=lodo_dataset,
                        plot_stage="load_details",
                        reason="Existing evaluation details could not be loaded.",
                        n_heldout_configs=0,
                        exc=exc,
                    )
                )
    save_dataframe(pd.DataFrame(skipped_plots, columns=SKIPPED_PLOT_COLUMNS), run_dir / "skipped_plots.csv", index=False)


def main() -> None:
    args = parse_args()
    _apply_pipeline_setup(args)
    selected_detectors = _resolve_cli_selection(args.detector, DETECTORS, "detector")
    selected_datasets = _resolve_cli_selection(args.dataset, DATASETS, "dataset")
    validate_evaluation_configuration()

    paths = get_paths_from_script(__file__)
    paths.ensure_core_directories()
    result_parent, result_child = _experiment_subfolders()
    run_dir = paths.phase1_result_dir(result_parent, result_child)
    ensure_dir(run_dir)

    save_json(
        _effective_run_config(
            paths=paths,
            run_dir=run_dir,
            experiment_parent=result_parent,
            experiment_tag=result_child,
            selected_detectors=selected_detectors,
            selected_datasets=selected_datasets,
        ),
        run_dir / "config.json",
    )

    if args.plots_only:
        _run_plotting_from_existing_details(
            paths=paths,
            run_dir=run_dir,
            selected_detectors=selected_detectors,
            selected_datasets=selected_datasets,
        )
        print(f"\nFinished Phase 1 plot export. Results directory: {run_dir}")
        return

    if _regional_active():
        _run_regional_preference_single(
            paths=paths,
            run_dir=run_dir,
            selected_detectors=selected_detectors,
            selected_datasets=selected_datasets,
            generate_plots=not args.skip_plots,
        )
    else:
        _run_plain_single_or_separate(
            paths=paths,
            run_dir=run_dir,
            selected_detectors=selected_detectors,
            selected_datasets=selected_datasets,
            generate_plots=not args.skip_plots,
        )

    print(f"\nFinished Phase 1 evaluation. Results directory: {run_dir}")


if __name__ == "__main__":
    main()
