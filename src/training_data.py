"""LODO training-data construction utilities.

Builds detector-specific train/test folds from processed benchmark files,
adds the configured metadata representation, constructs learning targets, and
returns feature matrices plus fitted metadata/PCA artifacts required later by
Phase 1 evaluation and configuration recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    PCA_LODO_VARIANCE,
    PCA_RANKED_NUMBER_OF_FEATURES,
    PCA_VARIANCES,
    TRAIN_APD_ALPHA,
    TRAIN_APD_EVAL_RATIO,
    DEFAULT_ACCURACY_WEIGHT,
    TRAIN_PREFERENCE_REGION_ACCURACY_WEIGHTS,
    TRAIN_PREFERENCE_REGION_NAMES,
    TRAIN_PBI_THETA,
    TRAIN_SCALARIZATION_IDEAL_POINT,
    TRAIN_USE_PREFERENCE_REGIONS,
)
from src.sweeper_setup import PIPELINE_PCA_VARIANCE_OVERRIDE_ENV
from src.paths import ProjectPaths
from src.target_utils import REGIONAL_TARGETS, compute_dataset_targets, compute_preference_region_targets, pareto_layer_rank


SCALAR_TARGET_COLUMNS = {
    "euc_dist",
    "mod_dist",
    "pareto_score",
    "pareto_loss",
    "pareto_rank",
    "tchebycheff",
    "pbi",
    "apd",
}
PREFERENCE_REGION_TARGET_COLUMNS = {
    f"{method}_{region}"
    for method in REGIONAL_TARGETS
    for region in TRAIN_PREFERENCE_REGION_NAMES
}
RESULT_AND_AUXILIARY_COLUMNS = {
    "index",
    "source",
    "Status",
    "detector",
    "dataset_name",
    "ACCURACY",
    "RUNTIME",
    "N_RUNS",
    "ACCURACY_STD",
    "RUNTIME_STD",
    "transformed_accuracy",
    "transformed_runtime",
    "is_pareto",
    "pareto_layer",
    "real_pareto_layer",
    "normalized_pareto_depth",
    "reward_factor",
    "helper_distance_to_predicted_front",
}
EVALUATION_ONLY_COLUMNS = {
    "tree_std",
    "tree_iqr",
    "accuracy_tree_std",
    "accuracy_tree_iqr",
    "runtime_tree_std",
    "runtime_tree_iqr",
    "ranking_distance",
    "predicted_order_rank",
    "best_region_rank_pct",
    "best_region_name",
    "mean_region_rank_pct",
    "recommendation_rank",
}
PROHIBITED_FEATURE_COLUMNS = {
    *RESULT_AND_AUXILIARY_COLUMNS,
    *SCALAR_TARGET_COLUMNS,
    *PREFERENCE_REGION_TARGET_COLUMNS,
    *EVALUATION_ONLY_COLUMNS,
}
PROHIBITED_FEATURE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^ACCURACY_RUN_[0-9]+$",
        r"^RUNTIME_RUN_[0-9]+$",
        r"^real_",
        r"^pred_",
        r"^abs_error_",
        r"^rank_",
        r"^rank_pct_",
        r"^selected_",
        r"^selection_source_region_",
        r"^final_selection_rank_",
        r"^tree_std_",
        r"^tree_iqr_",
        r"^accuracy_tree_std_",
        r"^accuracy_tree_iqr_",
        r"^runtime_tree_std_",
        r"^runtime_tree_iqr_",
    )
)


def _minmax_normalize_series(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce").astype(float)
    mask = series.notna()
    if not mask.any():
        return series
    lo = float(series.loc[mask].min())
    hi = float(series.loc[mask].max())
    den = hi - lo
    out = series.copy()
    if den == 0.0:
        out.loc[mask] = 0.0
    else:
        out.loc[mask] = (out.loc[mask] - lo) / den
    return out


@dataclass
class LODODataBundle:
    detector: str
    lodo_dataset: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame | None
    X_train: pd.DataFrame
    X_test: pd.DataFrame | None
    configuration_feature_columns: list[str]
    feature_columns: list[str]
    numeric_feature_columns: list[str]
    categorical_feature_columns: list[str]
    train_target_name: str | None = None
    y_train_single: pd.Series | None = None
    y_test_single: pd.Series | None = None
    separate_target_names: tuple[str, str] | None = None
    y_train_separate: pd.DataFrame | None = None
    y_test_separate: pd.DataFrame | None = None


def find_column_case_insensitive(df: pd.DataFrame, name: str) -> str | None:
    mapping = {str(c).lower(): c for c in df.columns}
    return mapping.get(str(name).lower())


def coerce_numeric_like_columns(df: pd.DataFrame, min_convert_frac: float = 0.95) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            series = out[col].astype(str).str.strip()
            series = series.replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan})
            numeric = pd.to_numeric(series, errors="coerce")
            non_na = series.notna().sum()
            if non_na == 0:
                continue
            if (numeric.notna().sum() / non_na) >= min_convert_frac:
                out[col] = numeric
    return out


def _read_processed_benchmark_file(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return coerce_numeric_like_columns(pd.read_csv(path))


def get_pca_ranked_number_of_features() -> int:
    raw = os.environ.get("PCA_RANKED_NUMBER_OF_FEATURES_OVERRIDE", "").strip()
    if not raw:
        return int(PCA_RANKED_NUMBER_OF_FEATURES)
    return max(1, int(raw))


def get_pca_ranked_selected_positions() -> list[int] | None:
    raw = os.environ.get("PCA_RANKED_SELECTED_POSITIONS_OVERRIDE", "").strip()
    if not raw:
        return None
    positions: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("PCA_RANKED_SELECTED_POSITIONS_OVERRIDE positions must be >= 1.")
        if value not in positions:
            positions.append(value)
    return positions or None


def _validate_pca_lodo_variance(value: float) -> float:
    numeric = float(value)
    allowed = [float(v) for v in PCA_VARIANCES]
    if not any(np.isclose(numeric, candidate, atol=1e-12, rtol=0.0) for candidate in allowed):
        allowed_text = ", ".join(f"{candidate:.2f}" for candidate in allowed)
        raise ValueError(f"PCA_LODO_VARIANCE must be one of [{allowed_text}]. Got {value!r}.")
    return numeric


def get_pca_lodo_variance() -> float:
    """Return the effective LODO PCA variance for this process."""
    raw = os.environ.get(PIPELINE_PCA_VARIANCE_OVERRIDE_ENV, "").strip()
    if not raw:
        return _validate_pca_lodo_variance(float(PCA_LODO_VARIANCE))
    return _validate_pca_lodo_variance(float(raw))


def is_prohibited_model_feature_column(column: str) -> bool:
    name = str(column)
    if name in PROHIBITED_FEATURE_COLUMNS:
        return True
    return any(pattern.match(name) is not None for pattern in PROHIBITED_FEATURE_PATTERNS)


def _detect_config_columns(df: pd.DataFrame, *, metadata_columns: Iterable[str] | None = None) -> list[str]:
    metadata_set = {str(col) for col in (metadata_columns or [])}
    cols = []
    for col in df.columns:
        name = str(col)
        if name in metadata_set:
            continue
        if is_prohibited_model_feature_column(name):
            continue
        cols.append(col)
    return cols


def _resolve_metadata_path(paths: ProjectPaths, metadata_variant: str, lodo_dataset: str | None = None) -> Path:
    variant = str(metadata_variant).strip().lower()
    if variant == "all":
        return paths.all_meta_features_file
    if variant == "cleaned":
        return paths.cleaned_meta_features_file
    if variant == "pruned":
        return paths.pruned_meta_features_file
    if variant == "lodo_pca":
        if not lodo_dataset:
            raise ValueError("metadata_variant='lodo_pca' requires lodo_dataset.")
        return paths.metadata_pca_lodo_file(lodo_dataset, get_pca_lodo_variance())
    if variant == "lodo_pca_ranked":
        if not lodo_dataset:
            raise ValueError("metadata_variant='lodo_pca_ranked' requires lodo_dataset.")
        return paths.metadata_pca_ranked_lodo_file(lodo_dataset, get_pca_lodo_variance())
    raise ValueError(f"Unknown metadata_variant '{metadata_variant}'.")


def _simple_scale(values: pd.Series, *, method: str) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    if method == "identity":
        return arr
    if method == "minmax":
        mask = np.isfinite(arr)
        out = arr.astype(float).copy()
        if mask.any():
            lo = float(np.nanmin(out[mask]))
            hi = float(np.nanmax(out[mask]))
            den = hi - lo
            if den == 0.0:
                out[mask] = 1.0
            else:
                out[mask] = (out[mask] - lo) / den
        return out
    raise ValueError(f"Unsupported metadata scale method '{method}'.")


def load_metadata_table(paths: ProjectPaths, metadata_variant: str, *, lodo_dataset: str | None = None) -> pd.DataFrame:
    path = _resolve_metadata_path(paths, metadata_variant, lodo_dataset)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found for variant '{metadata_variant}': {path}")
    df = pd.read_csv(path)
    if "dataset_name" not in df.columns:
        raise ValueError(f"Metadata file must contain 'dataset_name': {path}")
    variant = str(metadata_variant).strip().lower()
    if variant == "lodo_pca":
        feature_cols = [c for c in df.columns if c != "dataset_name"]
        df = df[["dataset_name"] + feature_cols].copy()
    if variant == "lodo_pca_ranked":
        n_keep = get_pca_ranked_number_of_features()
        feature_cols = [c for c in df.columns if c != "dataset_name"][:n_keep]
        selected_positions = get_pca_ranked_selected_positions()
        if selected_positions is not None:
            feature_cols = [feature_cols[pos - 1] for pos in selected_positions if 1 <= pos <= len(feature_cols)]
        df = df[["dataset_name"] + feature_cols].copy()
    return coerce_numeric_like_columns(df)


def metadata_variant_tag(metadata_variant: str, *, use_metadata: bool) -> str:
    if not use_metadata:
        return "CFGONLY"
    variant = str(metadata_variant).strip()
    lower = variant.lower()
    if lower == "lodo_pca":
        return f"META_{variant}_{get_pca_lodo_variance():.2f}"
    if lower == "lodo_pca_ranked":
        return f"META_{variant}_{get_pca_lodo_variance():.2f}_{get_pca_ranked_number_of_features()}"
    return f"META_{variant}"


def _merge_metadata(df: pd.DataFrame, dataset_name: str, metadata_df: pd.DataFrame) -> pd.DataFrame:
    meta_row = metadata_df.loc[metadata_df["dataset_name"].astype(str) == str(dataset_name)].copy()
    if meta_row.empty:
        raise ValueError(f"Metadata for dataset '{dataset_name}' not found.")
    meta_row = meta_row.iloc[0]
    meta_features = {c: meta_row[c] for c in metadata_df.columns if c != "dataset_name"}
    meta_df = pd.DataFrame([meta_features] * len(df), index=df.index)
    return pd.concat([df.copy(), meta_df], axis=1)


def _apply_dataset_targets(
    df: pd.DataFrame,
    *,
    distance_method: str,
    lambda_value: float,
) -> pd.DataFrame:
    out = df.copy()
    initial_acc = pd.to_numeric(out["transformed_accuracy"], errors="coerce").to_numpy(dtype=float)
    initial_rt = pd.to_numeric(out["transformed_runtime"], errors="coerce").to_numpy(dtype=float)

    if distance_method == "pareto_rank" and "pareto_layer" in out.columns:
        layers = pd.to_numeric(out["pareto_layer"], errors="coerce")
        out["pareto_layer"] = layers
        out["is_pareto"] = layers.eq(1).fillna(False).astype(int)
        out[distance_method] = _minmax_normalize_series(layers)
        return out

    artifacts = compute_dataset_targets(
        initial_acc,
        initial_rt,
        distance_method=distance_method,
        lambda_value=lambda_value,
        scalarization_accuracy_weight=DEFAULT_ACCURACY_WEIGHT,
        scalarization_ideal_point=TRAIN_SCALARIZATION_IDEAL_POINT,
        pbi_theta=TRAIN_PBI_THETA,
        apd_alpha=TRAIN_APD_ALPHA,
        apd_eval_ratio=TRAIN_APD_EVAL_RATIO,
    )
    out["transformed_accuracy"] = artifacts.transformed_accuracy
    out["transformed_runtime"] = artifacts.transformed_runtime
    out["is_pareto"] = artifacts.is_pareto
    out[distance_method] = artifacts.active_distance
    if artifacts.active_reward_factor is not None:
        out["reward_factor"] = artifacts.active_reward_factor
    if TRAIN_USE_PREFERENCE_REGIONS and distance_method in REGIONAL_TARGETS:
        pref_region_targets = compute_preference_region_targets(
            out["transformed_accuracy"].to_numpy(dtype=float),
            out["transformed_runtime"].to_numpy(dtype=float),
            base_method=distance_method,
            lambda_value=lambda_value,
            region_names=TRAIN_PREFERENCE_REGION_NAMES,
            region_accuracy_weights=TRAIN_PREFERENCE_REGION_ACCURACY_WEIGHTS,
            scalarization_ideal_point=TRAIN_SCALARIZATION_IDEAL_POINT,
            pbi_theta=TRAIN_PBI_THETA,
            apd_alpha=TRAIN_APD_ALPHA,
            apd_eval_ratio=TRAIN_APD_EVAL_RATIO,
        )
        for col_name, values in pref_region_targets.items():
            out[col_name] = values
    return out


def _prepare_separate_objective_reference(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    points = (
        out[["transformed_accuracy", "transformed_runtime"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(points).all():
        raise ValueError("Separate-objective data contain non-finite transformed objectives.")
    layers = pareto_layer_rank(points).astype(int)
    out["pareto_layer"] = layers
    out["is_pareto"] = (layers == 1).astype(int)
    return out


def _build_feature_matrix(
    df: pd.DataFrame,
    *,
    use_metadata: bool,
    metadata_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    metadata_cols = list(metadata_columns or []) if use_metadata else []
    config_cols = _detect_config_columns(df, metadata_columns=metadata_cols)

    feature_columns = list(config_cols)
    for col in metadata_cols:
        if col not in feature_columns and col in df.columns:
            feature_columns.append(col)

    prohibited = [col for col in feature_columns if is_prohibited_model_feature_column(col)]
    if prohibited:
        raise ValueError(
            "Model feature matrix contains prohibited result/target/evaluation column(s): "
            f"{prohibited}"
        )

    X = coerce_numeric_like_columns(df[feature_columns].copy())
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, config_cols, feature_columns, numeric_cols, categorical_cols


def build_lodo_data_bundle(
    *,
    paths: ProjectPaths,
    detector: str,
    lodo_dataset: str,
    datasets: list[str],
    target_mode: str,
    distance_method: str,
    lambda_value: float,
    use_metadata: bool = False,
    metadata_variant: str = "pruned",
    scale_metadata: bool = False,
    metadata_scale_method: str = "identity",
    single_target_column: str | None = None,
) -> LODODataBundle:
    """
    Build one detector-specific leave-one-dataset-out training bundle.

    The held-out dataset is excluded from training rows but retained as the
    optional test split. Metadata variants are resolved with the same fold
    identity used for model artifact paths.
    """
    target_mode = str(target_mode).strip().lower()
    if target_mode not in {"single", "separate"}:
        raise ValueError("target_mode must be 'single' or 'separate'.")

    metadata_df: pd.DataFrame | None = None
    metadata_cols: list[str] = []
    if use_metadata:
        metadata_df = load_metadata_table(paths, metadata_variant, lodo_dataset=lodo_dataset)
        metadata_cols = [c for c in metadata_df.columns if c != "dataset_name"]
        if scale_metadata and metadata_cols:
            metadata_df = metadata_df.copy()
            for col in metadata_cols:
                metadata_df[col] = _simple_scale(metadata_df[col], method=metadata_scale_method)

    train_frames: list[pd.DataFrame] = []
    test_df: pd.DataFrame | None = None

    for dataset_name in datasets:
        file_path = paths.processed_benchmark_file(detector, dataset_name)
        df = _read_processed_benchmark_file(file_path)
        if df is None:
            continue
        required_cols = ["transformed_accuracy", "transformed_runtime"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Processed file {file_path} is missing required columns {missing_cols}. "
                "Rerun preprocessing with the current config."
            )
        if target_mode == "single":
            df = _apply_dataset_targets(
                df,
                distance_method=distance_method,
                lambda_value=lambda_value,
            )
        else:
            df = _prepare_separate_objective_reference(df)
        if use_metadata and metadata_df is not None:
            df = _merge_metadata(df, dataset_name, metadata_df)
        df.insert(0, "dataset_name", dataset_name)
        df.insert(0, "detector", detector)
        if dataset_name == lodo_dataset:
            test_df = df.copy()
        else:
            train_frames.append(df.copy())

    if not train_frames:
        raise ValueError(f"No training data available for detector '{detector}' with LODO '{lodo_dataset}'.")

    train_df = pd.concat(train_frames, axis=0, ignore_index=True)
    X_train, config_feature_cols, feature_cols, num_cols, cat_cols = _build_feature_matrix(
        train_df, use_metadata=use_metadata, metadata_columns=metadata_cols
    )

    X_test = None
    if test_df is not None:
        X_test = coerce_numeric_like_columns(test_df[feature_cols].copy())

    if target_mode == "single":
        target_col = single_target_column or distance_method
        if target_col not in train_df.columns:
            raise ValueError(f"target column '{target_col}' not found in training data.")
        y_train = pd.to_numeric(train_df[target_col], errors="coerce")
        valid_train = y_train.notna()
        train_df = train_df.loc[valid_train].reset_index(drop=True)
        X_train = X_train.loc[valid_train].reset_index(drop=True)
        y_train = y_train.loc[valid_train].reset_index(drop=True)

        y_test = None
        if test_df is not None:
            y_test = pd.to_numeric(test_df[target_col], errors="coerce")
            valid_test = y_test.notna()
            test_df = test_df.loc[valid_test].reset_index(drop=True)
            X_test = X_test.loc[valid_test].reset_index(drop=True)
            y_test = y_test.loc[valid_test].reset_index(drop=True)

        return LODODataBundle(
            detector=detector,
            lodo_dataset=lodo_dataset,
            train_df=train_df,
            test_df=test_df,
            X_train=X_train,
            X_test=X_test,
            configuration_feature_columns=config_feature_cols,
            feature_columns=feature_cols,
            numeric_feature_columns=num_cols,
            categorical_feature_columns=cat_cols,
            train_target_name=target_col,
            y_train_single=y_train,
            y_test_single=y_test,
        )

    separate_targets = ("transformed_accuracy", "transformed_runtime")
    y_train_separate = train_df[list(separate_targets)].apply(pd.to_numeric, errors="coerce")
    valid_train = y_train_separate.notna().all(axis=1)
    train_df = train_df.loc[valid_train].reset_index(drop=True)
    X_train = X_train.loc[valid_train].reset_index(drop=True)
    y_train_separate = y_train_separate.loc[valid_train].reset_index(drop=True)

    y_test_separate = None
    if test_df is not None:
        y_test_separate = test_df[list(separate_targets)].apply(pd.to_numeric, errors="coerce")
        valid_test = y_test_separate.notna().all(axis=1)
        test_df = test_df.loc[valid_test].reset_index(drop=True)
        X_test = X_test.loc[valid_test].reset_index(drop=True)
        y_test_separate = y_test_separate.loc[valid_test].reset_index(drop=True)

    return LODODataBundle(
        detector=detector,
        lodo_dataset=lodo_dataset,
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        X_test=X_test,
        configuration_feature_columns=config_feature_cols,
        feature_columns=feature_cols,
        numeric_feature_columns=num_cols,
        categorical_feature_columns=cat_cols,
        separate_target_names=separate_targets,
        y_train_separate=y_train_separate,
        y_test_separate=y_test_separate,
    )
