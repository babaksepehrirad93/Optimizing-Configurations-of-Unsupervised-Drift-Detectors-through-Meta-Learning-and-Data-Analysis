"""
Phase 1 model training.

Purpose
-------
Train detector-specific LODO Extra Trees models for the configured target and
metadata representation.

Inputs
------
- processed benchmark CSVs
- configured metadata representation

Outputs
-------
- setup-specific Phase 1 model artifacts and sidecar metadata
- optional compatibility model copies for older consumers

Important behavior
------------------
Training excludes the held-out dataset. This script does not perform Phase 1
evaluation, recommendation selection, metric computation, or plotting; use
phase1_evaluation.py for those responsibilities.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    DEFAULT_ACCURACY_WEIGHT,
    FINAL_SINGLE_TARGET_FORMULATIONS,
    LEGACY_SINGLE_TARGET_FORMULATIONS,
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
    TRAIN_OVERWRITE_ARTIFACTS,
    TRAIN_PREFERENCE_REGION_ACCURACY_WEIGHTS,
    TRAIN_PREFERENCE_REGION_NAMES,
    TRAIN_PBI_THETA,
    TRAIN_RANDOM_STATE,
    TRAIN_SAVE_MODELS,
    TRAIN_SAVE_TRAINING_DATA_SNAPSHOT,
    TRAIN_SCALE_METADATA,
    TRAIN_SCALARIZATION_IDEAL_POINT,
    TRAIN_TARGET_MODE,
    TRAIN_USE_METADATA,
    TRAIN_USE_PREFERENCE_REGIONS,
)
from src.model_factory import build_regressor, merge_model_params, model_family_tag
from src.paths import ProjectPaths, get_paths_from_script
from src.sweeper_setup import add_pipeline_setup_args, resolve_pipeline_setup
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
from src.utils import ensure_dir, save_dataframe, save_json


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
SAVE_TRAINING_DATA_SNAPSHOT = TRAIN_SAVE_TRAINING_DATA_SNAPSHOT
SAVE_MODELS = TRAIN_SAVE_MODELS
OVERWRITE_ARTIFACTS = TRAIN_OVERWRITE_ARTIFACTS
DETECTORS = TRAIN_DETECTORS
DATASETS = TRAIN_DATASETS


def validate_supported_single_target(target_method: str) -> str:
    value = str(target_method).strip()
    if value in FINAL_SINGLE_TARGET_FORMULATIONS or value in LEGACY_SINGLE_TARGET_FORMULATIONS:
        return value
    raise ValueError(
        f"Unknown single-target formulation '{value}'. "
        f"Supported formulations: {list(FINAL_SINGLE_TARGET_FORMULATIONS + LEGACY_SINGLE_TARGET_FORMULATIONS)}."
    )


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


def validate_training_configuration() -> None:
    if TARGET_MODE == "single":
        validate_distance_method(SINGLE_TARGET_METHOD)
        validate_supported_single_target(SINGLE_TARGET_METHOD)
        if _regional_active():
            _ordered_preference_regions()
    elif TARGET_MODE == "separate":
        return
    else:
        raise ValueError("TRAIN_TARGET_MODE must be 'single' or 'separate'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1 detector recommendation models only.")
    parser.add_argument("--detector", type=str, default="ALL", help="Detector name, ALL, or comma-separated exact names.")
    parser.add_argument("--dataset", type=str, default="ALL", help="Dataset name, ALL, or comma-separated exact names.")
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


def build_feature_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        transformers.append(("num", "passthrough", numeric_columns))
    if categorical_columns:
        cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("cat", cat_pipe, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    *,
    random_state: int,
    model_params: dict[str, Any],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Build the fitted-model pipeline with passthrough numeric and one-hot categorical features."""
    preprocessor = build_feature_preprocessor(numeric_columns, categorical_columns)
    estimator = build_regressor(MODEL_FAMILY, random_state=random_state, params=model_params)
    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def _metadata_artifact_tag() -> str:
    return metadata_variant_tag(METADATA_VARIANT, use_metadata=USE_METADATA)


def _phase1_model_path(
    paths: ProjectPaths,
    detector: str,
    lodo_dataset: str,
    *,
    target_method: str,
    preference_region: str | None = None,
    objective: str | None = None,
) -> Path:
    """Resolve the unique Phase 1 model path shared with phase1_evaluation.py."""
    return paths.phase1_model_file(
        detector,
        lodo_dataset,
        target_mode=TARGET_MODE,
        target_method=target_method,
        metadata_tag=_metadata_artifact_tag(),
        preference_region=preference_region,
        objective=objective,
    )


def _phase1_model_metadata_path(
    paths: ProjectPaths,
    detector: str,
    lodo_dataset: str,
    *,
    target_method: str,
    preference_region: str | None = None,
    objective: str | None = None,
) -> Path:
    return paths.phase1_model_metadata_file(
        detector,
        lodo_dataset,
        target_mode=TARGET_MODE,
        target_method=target_method,
        metadata_tag=_metadata_artifact_tag(),
        preference_region=preference_region,
        objective=objective,
    )


def _training_snapshot_path(
    paths: ProjectPaths,
    detector: str,
    lodo_dataset: str,
    *,
    target_method: str,
    preference_region: str | None = None,
) -> Path:
    return paths.phase1_training_snapshot_file(
        detector,
        lodo_dataset,
        target_mode=TARGET_MODE,
        target_method=target_method,
        metadata_tag=_metadata_artifact_tag(),
        preference_region=preference_region,
    )


def _legacy_training_snapshot_path(paths: ProjectPaths, detector: str, lodo_dataset: str) -> Path:
    return ensure_dir(paths.training_data_dir / detector) / f"{detector}_LODO_{lodo_dataset}_training_snapshot.csv"


def _save_training_snapshot(
    paths: ProjectPaths,
    detector: str,
    lodo_dataset: str,
    bundle,
    *,
    target_method: str,
    preference_region: str | None = None,
    write_legacy: bool = False,
) -> None:
    if not SAVE_TRAINING_DATA_SNAPSHOT:
        return
    snapshot_df = bundle.train_df.reset_index(drop=True).copy()
    save_dataframe(
        snapshot_df,
        _training_snapshot_path(
            paths,
            detector,
            lodo_dataset,
            target_method=target_method,
            preference_region=preference_region,
        ),
        index=False,
    )
    if write_legacy:
        save_dataframe(snapshot_df, _legacy_training_snapshot_path(paths, detector, lodo_dataset), index=False)


def _model_sidecar_payload(
    *,
    detector: str,
    lodo_dataset: str,
    target_mode: str,
    target_method: str,
    bundle,
    preference_region: str | None = None,
    objective: str | None = None,
) -> dict[str, Any]:
    """Capture the fitted feature representation needed for held-out prediction."""
    return {
        "detector": detector,
        "lodo_dataset": lodo_dataset,
        "target_mode": target_mode,
        "target_method": target_method,
        "preference_region": preference_region,
        "objective": objective,
        "model_family": MODEL_FAMILY,
        "model_params": merge_model_params(
            MODEL_FAMILY,
            random_state=RANDOM_STATE,
            override_params=MODEL_PARAMS,
        ),
        "use_metadata": USE_METADATA,
        "metadata_variant": METADATA_VARIANT,
        "metadata_tag": _metadata_artifact_tag(),
        "scale_metadata": SCALE_METADATA,
        "metadata_scale_method": METADATA_SCALE_METHOD,
        "feature_columns": list(bundle.feature_columns),
        "configuration_feature_columns": list(bundle.configuration_feature_columns),
        "numeric_feature_columns": list(bundle.numeric_feature_columns),
        "categorical_feature_columns": list(bundle.categorical_feature_columns),
        "train_row_count": int(len(bundle.train_df)),
        "PCA_LODO_VARIANCE": float(get_pca_lodo_variance()),
        "pca_ranked_number_of_features": int(get_pca_ranked_number_of_features()),
        "pca_ranked_selected_positions": get_pca_ranked_selected_positions(),
        "PCA_VARIANCES": list(PCA_VARIANCES),
        "default_accuracy_weight": DEFAULT_ACCURACY_WEIGHT,
        "scalarization_ideal_point": TRAIN_SCALARIZATION_IDEAL_POINT,
        "pbi_theta": TRAIN_PBI_THETA,
        "apd_alpha": TRAIN_APD_ALPHA,
        "apd_eval_ratio": TRAIN_APD_EVAL_RATIO,
        "preprocessing_accuracy_mode": PREPROCESSING_ACCURACY_MODE,
        "preprocessing_runtime_mode": PREPROCESSING_RUNTIME_MODE,
        "preprocessing_runtime_upper_clipping": PREPROCESSING_RUNTIME_UPPER_CLIPPING,
        "preprocessing_runtime_clip_method": PREPROCESSING_RUNTIME_CLIP_METHOD,
        "preprocessing_runtime_clip_iqr_multiplier": PREPROCESSING_RUNTIME_CLIP_IQR_MULTIPLIER,
        "preprocessing_runtime_clip_percentile": PREPROCESSING_RUNTIME_CLIP_PERCENTILE,
    }


def _save_model(
    model: Pipeline,
    paths: ProjectPaths,
    detector: str,
    lodo_dataset: str,
    *,
    target_method: str,
    bundle,
    preference_region: str | None = None,
    objective: str | None = None,
    legacy_paths: list[Path] | None = None,
) -> Path:
    """Save a fitted model to the unique Phase 1 path and optional compatibility paths."""
    model_path = _phase1_model_path(
        paths,
        detector,
        lodo_dataset,
        target_method=target_method,
        preference_region=preference_region,
        objective=objective,
    )
    if SAVE_MODELS:
        ensure_dir(model_path.parent)
        dump(model, model_path)
        save_json(
            _model_sidecar_payload(
                detector=detector,
                lodo_dataset=lodo_dataset,
                target_mode=TARGET_MODE,
                target_method=target_method,
                preference_region=preference_region,
                objective=objective,
                bundle=bundle,
            ),
            _phase1_model_metadata_path(
                paths,
                detector,
                lodo_dataset,
                target_method=target_method,
                preference_region=preference_region,
                objective=objective,
            ),
        )
        for legacy_path in legacy_paths or []:
            ensure_dir(legacy_path.parent)
            dump(model, legacy_path)
    return model_path


def _fit_model(bundle, y_train: pd.Series) -> Pipeline:
    model = build_model_pipeline(
        random_state=RANDOM_STATE,
        model_params=MODEL_PARAMS,
        numeric_columns=bundle.numeric_feature_columns,
        categorical_columns=bundle.categorical_feature_columns,
    )
    model.fit(bundle.X_train, y_train)
    return model


def _build_bundle(paths: ProjectPaths, detector: str, lodo_dataset: str, *, single_target_column: str | None = None):
    return build_lodo_data_bundle(
        paths=paths,
        detector=detector,
        lodo_dataset=lodo_dataset,
        datasets=DATASETS,
        target_mode=TARGET_MODE if single_target_column is None else "single",
        distance_method=SINGLE_TARGET_METHOD,
        lambda_value=DEFAULT_ACCURACY_WEIGHT,
        use_metadata=USE_METADATA,
        metadata_variant=METADATA_VARIANT,
        scale_metadata=SCALE_METADATA,
        metadata_scale_method=METADATA_SCALE_METHOD,
        single_target_column=single_target_column,
    )


def _train_plain_single_or_separate(*, paths: ProjectPaths, selected_detectors: list[str], selected_datasets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detector in selected_detectors:
        print(f"\n=== Detector: {detector} ===")
        for lodo_dataset in selected_datasets:
            print(f"  -> LODO: {lodo_dataset}")
            bundle = _build_bundle(paths, detector, lodo_dataset)
            if TARGET_MODE == "single":
                target_method = bundle.train_target_name or SINGLE_TARGET_METHOD
                _save_training_snapshot(
                    paths,
                    detector,
                    lodo_dataset,
                    bundle,
                    target_method=target_method,
                    write_legacy=True,
                )
                model = _fit_model(bundle, bundle.y_train_single)
                model_path = _save_model(
                    model,
                    paths,
                    detector,
                    lodo_dataset,
                    target_method=target_method,
                    bundle=bundle,
                    legacy_paths=[paths.lodo_model_file(detector, lodo_dataset)],
                )
                rows.append({"detector": detector, "lodo_dataset": lodo_dataset, "target_method": target_method, "model_path": str(model_path)})
            else:
                _save_training_snapshot(
                    paths,
                    detector,
                    lodo_dataset,
                    bundle,
                    target_method="separate",
                    write_legacy=True,
                )
                acc_model = _fit_model(bundle, bundle.y_train_separate["transformed_accuracy"])
                rt_model = _fit_model(bundle, bundle.y_train_separate["transformed_runtime"])
                acc_path = _save_model(
                    acc_model,
                    paths,
                    detector,
                    lodo_dataset,
                    target_method="separate",
                    objective="accuracy",
                    bundle=bundle,
                    legacy_paths=[paths.lodo_model_file(detector, lodo_dataset, suffix="acc")],
                )
                rt_path = _save_model(
                    rt_model,
                    paths,
                    detector,
                    lodo_dataset,
                    target_method="separate",
                    objective="runtime",
                    bundle=bundle,
                    legacy_paths=[paths.lodo_model_file(detector, lodo_dataset, suffix="runtime")],
                )
                rows.append({"detector": detector, "lodo_dataset": lodo_dataset, "target_method": "separate", "model_path": str(acc_path), "objective": "accuracy"})
                rows.append({"detector": detector, "lodo_dataset": lodo_dataset, "target_method": "separate", "model_path": str(rt_path), "objective": "runtime"})
    return rows


def _train_regional_preference_single(*, paths: ProjectPaths, selected_detectors: list[str], selected_datasets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered_regions = _ordered_preference_regions()
    for detector in selected_detectors:
        print(f"\n=== Detector: {detector} ===")
        for lodo_dataset in selected_datasets:
            print(f"  -> LODO: {lodo_dataset}")
            region_bundle_for_recommendation: dict[str, dict[str, Any]] = {}
            snapshot_saved = False
            for region_name, _, _ in ordered_regions:
                region_target = f"{SINGLE_TARGET_METHOD}_{region_name}"
                bundle = _build_bundle(paths, detector, lodo_dataset, single_target_column=region_target)
                _save_training_snapshot(
                    paths,
                    detector,
                    lodo_dataset,
                    bundle,
                    target_method=SINGLE_TARGET_METHOD,
                    preference_region=region_name,
                    write_legacy=not snapshot_saved,
                )
                snapshot_saved = True
                model = _fit_model(bundle, bundle.y_train_single)
                model_path = _save_model(
                    model,
                    paths,
                    detector,
                    lodo_dataset,
                    target_method=SINGLE_TARGET_METHOD,
                    preference_region=region_name,
                    bundle=bundle,
                    legacy_paths=[paths.lodo_model_file(detector, lodo_dataset)],
                )
                region_bundle_for_recommendation[region_name] = {
                    "target_column": region_target,
                    "regressor": model,
                    "model_path": str(model_path),
                }
                rows.append(
                    {
                        "detector": detector,
                        "lodo_dataset": lodo_dataset,
                        "target_method": SINGLE_TARGET_METHOD,
                        "preference_region": region_name,
                        "model_path": str(model_path),
                    }
                )
            if SAVE_MODELS:
                bundle_path = paths.lodo_model_file(detector, lodo_dataset, suffix="preference_regions")
                ensure_dir(bundle_path.parent)
                dump(region_bundle_for_recommendation, bundle_path)
    return rows


def _training_config_payload(*, paths: ProjectPaths, selected_detectors: list[str], selected_datasets: list[str]) -> dict[str, Any]:
    return {
        "script": "train_models.py",
        "responsibility": "training_only",
        "target_mode": TARGET_MODE,
        "single_target_formulation": SINGLE_TARGET_METHOD if TARGET_MODE == "single" else "not_applicable",
        "separate_selection_method": "pareto_layer" if TARGET_MODE == "separate" else "not_applicable",
        "model_family": MODEL_FAMILY,
        "model_params": merge_model_params(MODEL_FAMILY, random_state=RANDOM_STATE, override_params=MODEL_PARAMS),
        "use_metadata": USE_METADATA,
        "metadata_variant": METADATA_VARIANT,
        "metadata_tag": _metadata_artifact_tag(),
        "scale_metadata": SCALE_METADATA,
        "metadata_scale_method": METADATA_SCALE_METHOD,
        "selected_detectors": selected_detectors,
        "selected_datasets": selected_datasets,
        "regional_active": _regional_active(),
        "project_root": str(paths.project_root),
        "overwrite_artifacts": OVERWRITE_ARTIFACTS,
        "save_models": SAVE_MODELS,
        "save_training_data_snapshot": SAVE_TRAINING_DATA_SNAPSHOT,
    }


def main() -> None:
    args = parse_args()
    _apply_pipeline_setup(args)
    selected_detectors = _resolve_cli_selection(args.detector, DETECTORS, "detector")
    selected_datasets = _resolve_cli_selection(args.dataset, DATASETS, "dataset")
    validate_training_configuration()

    paths = get_paths_from_script(__file__)
    paths.ensure_core_directories()
    training_run_dir = ensure_dir(paths.training_data_dir / "_phase1_training_runs")
    save_json(
        _training_config_payload(
            paths=paths,
            selected_detectors=selected_detectors,
            selected_datasets=selected_datasets,
        ),
        training_run_dir / "last_training_config.json",
    )

    if _regional_active():
        rows = _train_regional_preference_single(
            paths=paths,
            selected_detectors=selected_detectors,
            selected_datasets=selected_datasets,
        )
    else:
        rows = _train_plain_single_or_separate(
            paths=paths,
            selected_detectors=selected_detectors,
            selected_datasets=selected_datasets,
        )

    save_dataframe(pd.DataFrame(rows), training_run_dir / "last_training_artifacts.csv", index=False)
    print(f"\nTraining finished. Saved artifact index: {training_run_dir / 'last_training_artifacts.csv'}")


if __name__ == "__main__":
    main()
