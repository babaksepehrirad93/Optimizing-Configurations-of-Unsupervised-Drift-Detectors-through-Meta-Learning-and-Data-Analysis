"""
Phase 2 configuration recommendation.

Purpose
-------
Generate recommended detector configurations from saved setup-specific Phase 1
models and manually maintained detector search-space CSVs.

Inputs
------
- data/raw_data/search_space/<detector>_search_space.csv
- setup-specific Phase 1 model artifacts

Outputs
-------
- recommendation CSVs and candidate audit CSVs under results/phase_1/system_recommendations

Important behavior
------------------
Recommendation settings are shared across modes. Static mode means one
refinement stage; dynamic mode means the configured three-stage TPE refinement
with seed reselection. The script predicts promising candidates; real detector
execution happens outside this repository pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable
import sys

import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import qmc

from src.config import (
    CONFIG_RECOMMENDATION_MODE,
    CONFIG_RECOMMENDATION_DATASETS,
    CONFIG_RECOMMENDATION_DETECTORS,
    CONFIG_RECOMMENDATION_DROP_EXACT_DUPLICATE_CONFIGS_AFTER_OPTUNA,
    CONFIG_RECOMMENDATION_DROP_EXACT_DUPLICATE_CONFIGS_AFTER_SOBOL,
    CONFIG_RECOMMENDATION_EXPAND_CATEGORICAL_BOOL_COMBINATIONS_AFTER_OPTUNA,
    CONFIG_RECOMMENDATION_GLOBAL_SOBOL_SAMPLE_COUNT,
    CONFIG_RECOMMENDATION_LOCAL_NUMERIC_QUANTILE_HIGH,
    CONFIG_RECOMMENDATION_LOCAL_NUMERIC_QUANTILE_LOW,
    CONFIG_RECOMMENDATION_N_JOBS,
    CONFIG_RECOMMENDATION_OPTUNA_STARTUP_TRIALS,
    CONFIG_RECOMMENDATION_OPTUNA_STUDY_N_JOBS,
    CONFIG_RECOMMENDATION_RANDOM_SEED,
    CONFIG_RECOMMENDATION_RESTRICT_BOOLEANS_TO_SELECTED_SEEDS,
    CONFIG_RECOMMENDATION_RESTRICT_CATEGORICALS_TO_SELECTED_SEEDS,
    CONFIG_RECOMMENDATION_RESULTS_DIR_NAME,
    CONFIG_RECOMMENDATION_SOBOL_MAX_CONSTRAINT_ATTEMPTS,
    CONFIG_RECOMMENDATION_TARGET_MODE,
    CONFIG_RECOMMENDATION_TPE_CONSTANT_LIAR,
    CONFIG_RECOMMENDATION_TPE_GROUP,
    CONFIG_RECOMMENDATION_TPE_MULTIVARIATE,
    CONFIG_RECOMMENDATION_USE_SEMI_LOCAL_OPTUNA_SPACE,
    CONFIG_RECOMMENDATION_USE_FRESH_STUDY_PER_MARGIN_STAGE,
    DYNAMIC_TPE_MARGIN_SCHEDULE,
    DYNAMIC_TPE_TRIAL_SHARE_SCHEDULE,
    DYNAMIC_TPE_SEED_COUNT_SCHEDULE,
    DYNAMIC_TPE_TOTAL_TRIALS,
    COMPUTE_PREDICTION_UNCERTAINTY,
    NSGAII_TRIALS,
    NSGAII_NUMERIC_MARGIN,
    NSGAII_POPULATION_SIZE,
    NSGAII_SEED_COUNT,
    PHASE2_RECOMMENDATION_BUDGET,
    STATIC_TPE_TRIALS,
    STATIC_TPE_NUMERIC_MARGIN,
    STATIC_TPE_SEED_COUNT,
    SINGLE_TARGET_FORMULATION,
    TRAIN_METADATA_SCALE_METHOD,
    TRAIN_METADATA_VARIANT,
    TRAIN_PREFERENCE_REGION_NAMES,
    TRAIN_SCALE_METADATA,
    TRAIN_USE_METADATA,
    TRAIN_USE_PREFERENCE_REGIONS,
)
from src.paths import get_paths_from_script
from src.sweeper_setup import add_pipeline_setup_args, resolve_pipeline_setup
from src.selection import select_pareto_budget
from src.target_utils import REGIONAL_TARGETS, pareto_layer_rank
from src.training_data import _simple_scale, load_metadata_table, metadata_variant_tag
from src.utils import ensure_dir, save_dataframe

try:
    import optuna
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("Optuna is required for config recommendation. Please install Optuna before running this script.") from exc

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message=r"Argument ``multivariate`` is an experimental feature\. The interface can change in the future\.",
    category=optuna.exceptions.ExperimentalWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Argument ``group`` is an experimental feature\. The interface can change in the future\.",
    category=optuna.exceptions.ExperimentalWarning,
)


# =============================================================================
# Shared config bindings
# =============================================================================
DETECTORS = CONFIG_RECOMMENDATION_DETECTORS
DATASETS = CONFIG_RECOMMENDATION_DATASETS
SOBOL_MAX_CONSTRAINT_ATTEMPTS = CONFIG_RECOMMENDATION_SOBOL_MAX_CONSTRAINT_ATTEMPTS
RANDOM_SEED = CONFIG_RECOMMENDATION_RANDOM_SEED
RECOMMENDATION_N_JOBS = CONFIG_RECOMMENDATION_N_JOBS
RESULTS_DIR_NAME = CONFIG_RECOMMENDATION_RESULTS_DIR_NAME
TARGET_MODE = CONFIG_RECOMMENDATION_TARGET_MODE

DROP_EXACT_DUPLICATE_CONFIGS_AFTER_SOBOL = CONFIG_RECOMMENDATION_DROP_EXACT_DUPLICATE_CONFIGS_AFTER_SOBOL
DROP_EXACT_DUPLICATE_CONFIGS_AFTER_OPTUNA = CONFIG_RECOMMENDATION_DROP_EXACT_DUPLICATE_CONFIGS_AFTER_OPTUNA
OPTUNA_STARTUP_TRIALS = CONFIG_RECOMMENDATION_OPTUNA_STARTUP_TRIALS
OPTUNA_STUDY_N_JOBS = CONFIG_RECOMMENDATION_OPTUNA_STUDY_N_JOBS
TPE_MULTIVARIATE = CONFIG_RECOMMENDATION_TPE_MULTIVARIATE
TPE_GROUP = CONFIG_RECOMMENDATION_TPE_GROUP
TPE_CONSTANT_LIAR = CONFIG_RECOMMENDATION_TPE_CONSTANT_LIAR
LOCAL_NUMERIC_QUANTILE_LOW = CONFIG_RECOMMENDATION_LOCAL_NUMERIC_QUANTILE_LOW
LOCAL_NUMERIC_QUANTILE_HIGH = CONFIG_RECOMMENDATION_LOCAL_NUMERIC_QUANTILE_HIGH
RESTRICT_CATEGORICALS_TO_SELECTED_SEEDS = CONFIG_RECOMMENDATION_RESTRICT_CATEGORICALS_TO_SELECTED_SEEDS
RESTRICT_BOOLEANS_TO_SELECTED_SEEDS = CONFIG_RECOMMENDATION_RESTRICT_BOOLEANS_TO_SELECTED_SEEDS

# =============================================================================
# Single-target config bindings
# =============================================================================
USE_PREFERENCE_REGIONS = TRAIN_USE_PREFERENCE_REGIONS
PREFERENCE_REGION_NAMES = list(TRAIN_PREFERENCE_REGION_NAMES)
SINGLE_TARGET_METHOD = SINGLE_TARGET_FORMULATION


# =============================================================================
# =============================================================================
# Shared helpers
# =============================================================================
def _preference_region_budget() -> tuple[int, int]:
    """Derive equal per-region allocation from the unified Phase 2 budget."""
    region_count = len(PREFERENCE_REGION_NAMES)
    if region_count <= 0:
        raise ValueError("At least one preference region is required.")
    total_budget = int(PHASE2_RECOMMENDATION_BUDGET)
    if total_budget % region_count != 0:
        raise ValueError(
            "PHASE2_RECOMMENDATION_BUDGET must be divisible by the number of "
            f"preference regions ({region_count}) for regional recommendation."
        )
    return total_budget // region_count, total_budget


@dataclass(frozen=True)
class SearchDimension:
    name: str
    kind: str
    lower: float | None = None
    upper: float | None = None
    choices: tuple[str, ...] | None = None
    value_mode: str = "range"


@dataclass(frozen=True)
class RecommendationSettings:
    """Shared recommendation settings plus mode-specific refinement structure."""

    mode: str
    sobol_sample_count: int
    use_fresh_study_per_margin_stage: bool
    expand_categorical_bool_after_optuna: bool
    use_semi_local_optuna_space: bool
    static_tpe_seed_count: int
    static_tpe_numeric_margin: float
    static_tpe_trials: int
    dynamic_tpe_margins: tuple[float, ...]
    dynamic_tpe_seed_counts: tuple[int, ...]
    dynamic_tpe_trial_shares: tuple[float, ...]
    dynamic_tpe_total_trials: int
    nsgaii_seed_count: int
    nsgaii_numeric_margin: float
    nsgaii_trials: int
    nsgaii_population_size: int


def resolve_recommendation_settings(mode: str = CONFIG_RECOMMENDATION_MODE) -> RecommendationSettings:
    """
    Resolve effective recommendation settings for static or dynamic refinement.

    Static uses one TPE stage. Dynamic uses three TPE stages with stage-wise
    seed reselection. Separate additionally uses one independent NSGA-II branch.
    """
    normalized = str(mode).strip().lower()
    if normalized not in {"static", "dynamic"}:
        raise ValueError("CONFIG_RECOMMENDATION_MODE must be 'static' or 'dynamic'.")
    dynamic_lengths = {
        len(DYNAMIC_TPE_MARGIN_SCHEDULE),
        len(DYNAMIC_TPE_SEED_COUNT_SCHEDULE),
        len(DYNAMIC_TPE_TRIAL_SHARE_SCHEDULE),
    }
    if dynamic_lengths != {3}:
        raise ValueError(
            "Dynamic TPE schedules must each have exactly three entries: "
            "margins, seed counts, and trial shares."
        )
    share_sum = sum(float(share) for share in DYNAMIC_TPE_TRIAL_SHARE_SCHEDULE)
    if not math.isclose(share_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "DYNAMIC_TPE_TRIAL_SHARE_SCHEDULE must sum to 1.0; "
            f"got {share_sum}."
        )
    return RecommendationSettings(
        mode=normalized,
        sobol_sample_count=int(CONFIG_RECOMMENDATION_GLOBAL_SOBOL_SAMPLE_COUNT),
        use_fresh_study_per_margin_stage=bool(CONFIG_RECOMMENDATION_USE_FRESH_STUDY_PER_MARGIN_STAGE),
        expand_categorical_bool_after_optuna=bool(CONFIG_RECOMMENDATION_EXPAND_CATEGORICAL_BOOL_COMBINATIONS_AFTER_OPTUNA),
        use_semi_local_optuna_space=bool(CONFIG_RECOMMENDATION_USE_SEMI_LOCAL_OPTUNA_SPACE),
        static_tpe_seed_count=int(STATIC_TPE_SEED_COUNT),
        static_tpe_numeric_margin=float(STATIC_TPE_NUMERIC_MARGIN),
        static_tpe_trials=int(STATIC_TPE_TRIALS),
        dynamic_tpe_margins=tuple(float(value) for value in DYNAMIC_TPE_MARGIN_SCHEDULE),
        dynamic_tpe_seed_counts=tuple(int(value) for value in DYNAMIC_TPE_SEED_COUNT_SCHEDULE),
        dynamic_tpe_trial_shares=tuple(float(value) for value in DYNAMIC_TPE_TRIAL_SHARE_SCHEDULE),
        dynamic_tpe_total_trials=int(DYNAMIC_TPE_TOTAL_TRIALS),
        nsgaii_seed_count=int(NSGAII_SEED_COUNT),
        nsgaii_numeric_margin=float(NSGAII_NUMERIC_MARGIN),
        nsgaii_trials=int(NSGAII_TRIALS),
        nsgaii_population_size=int(NSGAII_POPULATION_SIZE),
    )


SETTINGS = resolve_recommendation_settings()


@dataclass(frozen=True)
class RefinementStage:
    """One Optuna refinement stage with exact seed and refinement-trial counts."""

    index: int
    margin_ratio: float
    seed_count: int
    trials: int


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    (hours, rem) = divmod(total, 3600)
    (minutes, secs) = divmod(rem, 60)
    if hours > 0:
        return f'{hours:d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'

def _resolve_cli_selection(
    selected: str,
    *,
    available: list[str],
    what: str,
) -> list[str]:
    token = str(selected).strip()
    if not token:
        raise ValueError(f'Empty {what} selection is not allowed.')
    upper = token.upper()
    if upper == 'ALL':
        return list(available)
    requested = [part.strip() for part in token.split(',') if part.strip()]
    if not requested:
        raise ValueError(f'Empty {what} selection is not allowed.')
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise ValueError(f'Unknown {what}(s) {unknown}. Available: {available}')
    seen: set[str] = set()
    ordered: list[str] = []
    for item in requested:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered

def _next_power_of_two_at_least(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))

def _normalize_bool_token(value: Any) -> bool:
    text = str(value).strip().upper()
    if text in {'TRUE', '1', 'YES'}:
        return True
    if text in {'FALSE', '0', 'NO'}:
        return False
    raise ValueError(f"Unsupported boolean token '{value}'.")

def _split_choice_payload(payload: str) -> tuple[str, ...]:
    separator = '|' if '|' in payload else ','
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escape_next = False
    bracket_depth = 0
    bracket_pairs = {'{': '}', '[': ']', '(': ')'}
    closing_brackets = set(bracket_pairs.values())

    for char in payload:
        if escape_next:
            current.append(char)
            escape_next = False
            continue
        if char == '\\':
            current.append(char)
            escape_next = True
            continue
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            current.append(char)
            quote = char
            continue
        if char in bracket_pairs:
            current.append(char)
            bracket_depth += 1
            continue
        if char in closing_brackets:
            current.append(char)
            bracket_depth = max(0, bracket_depth - 1)
            continue
        if char == separator and bracket_depth == 0:
            part = ''.join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)

    part = ''.join(current).strip()
    if part:
        parts.append(part)
    return tuple(parts)

def _search_space_missing_error(detector: str, search_space_csv: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"Search-space CSV not found for detector '{detector}':\n"
        f"{search_space_csv}\n\n"
        "Provide the manually maintained detector search-space CSV before "
        "running configuration recommendation."
    )

def _load_search_space_csv(search_space_csv: Path, *, detector: str) -> pd.DataFrame:
    """Load the manually maintained detector search-space CSV as fixed input."""
    if not search_space_csv.exists():
        raise _search_space_missing_error(detector, search_space_csv)
    with search_space_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"Search-space CSV is empty for detector '{detector}': {search_space_csv}") from exc
    normalized_header = [str(column).strip() for column in header]
    duplicate_header = sorted(
        {column for column in normalized_header if column and normalized_header.count(column) > 1}
    )
    if duplicate_header:
        raise ValueError(f"Search-space CSV has duplicate hyperparameter column(s) {duplicate_header}: {search_space_csv}")
    try:
        df = pd.read_csv(search_space_csv)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Search-space CSV is empty for detector '{detector}': {search_space_csv}") from exc
    if df.empty:
        raise ValueError(f"Search-space CSV has no rows for detector '{detector}': {search_space_csv}")
    if "Dataset" not in df.columns:
        raise ValueError(f"Search-space CSV must contain a 'Dataset' column: {search_space_csv}")
    if len(df.columns) <= 1:
        raise ValueError(f"Search-space CSV must define at least one hyperparameter column: {search_space_csv}")
    return df

def _validate_dimension(dim: SearchDimension, *, search_space_csv: Path) -> SearchDimension:
    if dim.kind not in {"int", "float", "bool", "cat"}:
        raise ValueError(f"Unsupported search-space type '{dim.kind}' for hyperparameter '{dim.name}'.")
    if dim.value_mode not in {"range", "choice"}:
        raise ValueError(f"Unsupported search-space value mode '{dim.value_mode}' for hyperparameter '{dim.name}'.")
    if dim.kind in {"int", "float"} and dim.value_mode == "range":
        if dim.lower is None or dim.upper is None:
            raise ValueError(f"Numeric range hyperparameter '{dim.name}' is missing bounds in {search_space_csv}.")
        lo = float(dim.lower)
        hi = float(dim.upper)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Numeric range hyperparameter '{dim.name}' has non-finite bounds in {search_space_csv}.")
        if lo > hi:
            raise ValueError(f"Numeric range hyperparameter '{dim.name}' has lower bound > upper bound in {search_space_csv}.")
    if dim.value_mode == "choice":
        choices = tuple(str(choice).strip() for choice in (dim.choices or ()) if str(choice).strip())
        if not choices and dim.kind != "bool":
            raise ValueError(f"Choice hyperparameter '{dim.name}' has no choices in {search_space_csv}.")
        if dim.kind in {"int", "float"}:
            numeric_choices = pd.to_numeric(pd.Series(choices), errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(numeric_choices).all():
                raise ValueError(f"Numeric choice hyperparameter '{dim.name}' has non-numeric choice(s) in {search_space_csv}.")
        if dim.kind == "bool":
            for choice in choices or ("TRUE", "FALSE"):
                _normalize_bool_token(choice)
    return dim

def _load_search_space_dimensions(paths, detector: str, dataset_name: str) -> list[SearchDimension]:
    """Load one dataset row from the authoritative manual detector search-space CSV."""
    return _parse_search_space_row(paths.detector_search_space_file(detector), dataset_name, detector=detector)

def _numeric_choice_values(dim: SearchDimension) -> list[int] | list[float]:
    raw_choices = list(dim.choices or ())
    if not raw_choices:
        raise ValueError(f"Numeric choice hyperparameter '{dim.name}' has no choices.")
    if dim.kind == 'int':
        return [int(round(float(choice))) for choice in raw_choices]
    if dim.kind == 'float':
        return [float(choice) for choice in raw_choices]
    raise ValueError(f"Hyperparameter '{dim.name}' is not a numeric choice dimension.")

CONSTRAINT_INVALID_COUNTS: dict[tuple[str, str], int] = {}
HDDDM_TEST_PAIR_PARAM = "__HDDDM_use_mmd2_use_k2s_test"

def _record_invalid_configs(detector: str, stage: str, count: int) -> None:
    if int(count) <= 0:
        return
    key = (str(detector), str(stage))
    CONSTRAINT_INVALID_COUNTS[key] = CONSTRAINT_INVALID_COUNTS.get(key, 0) + int(count)

def _print_constraint_summary() -> None:
    total = sum(CONSTRAINT_INVALID_COUNTS.values())
    print(f"\nConstraint audit: invalid generated configs={total:,}")
    if not CONSTRAINT_INVALID_COUNTS:
        return
    for (detector, stage), count in sorted(CONSTRAINT_INVALID_COUNTS.items()):
        print(f"  {detector} / {stage}: {count:,}")

def _bool_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.upper()
    return text.isin({"TRUE", "1", "YES"})

def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None

def _constraint_allowed_mask(detector: str, candidates: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=candidates.index, dtype=bool)
    if candidates.empty:
        return mask

    if detector == "HDDDM":
        mmd_col = _first_existing_column(candidates, ["use_mmd2", "mmd2"])
        k2s_col = _first_existing_column(candidates, ["use_k2s_test", "k2s_test"])
        if mmd_col is not None and k2s_col is not None:
            mask &= ~(_bool_series(candidates[mmd_col]) & ~_bool_series(candidates[k2s_col]))

    elif detector == "PCACD":
        if {"sample_period", "window_size"}.issubset(candidates.columns):
            sample_period = pd.to_numeric(candidates["sample_period"], errors="coerce")
            window_size = pd.to_numeric(candidates["window_size"], errors="coerce")
            mask &= (sample_period * window_size) >= 1.0

    elif detector == "NNDVI":
        if {"n_samples", "k_neighbors"}.issubset(candidates.columns):
            n_samples = pd.to_numeric(candidates["n_samples"], errors="coerce")
            k_neighbors = pd.to_numeric(candidates["k_neighbors"], errors="coerce")
            mask &= (2.0 * n_samples) >= (k_neighbors + 1.0)

    elif detector == "SlidShaps":
        if {"overlap", "batch_size"}.issubset(candidates.columns):
            overlap = pd.to_numeric(candidates["overlap"], errors="coerce")
            batch_size = pd.to_numeric(candidates["batch_size"], errors="coerce")
            mask &= (overlap * batch_size) >= 1.0

    elif detector == "IBDD":
        required = {"n_permutations", "update_interval", "n_consecutive_deviations"}
        if required.issubset(candidates.columns):
            n_permutations = pd.to_numeric(candidates["n_permutations"], errors="coerce")
            update_interval = pd.to_numeric(candidates["update_interval"], errors="coerce")
            n_consecutive = pd.to_numeric(candidates["n_consecutive_deviations"], errors="coerce")
            mask &= n_permutations >= n_consecutive
            mask &= update_interval >= (n_consecutive + 1.0)

    elif detector == "WindowKDE":
        required = {"small_windowSize", "big_windowSize", "recent_samples_size"}
        if required.issubset(candidates.columns):
            small = pd.to_numeric(candidates["small_windowSize"], errors="coerce")
            big = pd.to_numeric(candidates["big_windowSize"], errors="coerce")
            recent = pd.to_numeric(candidates["recent_samples_size"], errors="coerce")
            mask &= (small + 1.0) <= (big + 1.0)
            mask &= (big + 1.0) <= recent
            mask &= big >= (small * 4.0)

    return mask.fillna(False)

def _validate_generated_candidates(detector: str, candidates: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    allowed = _constraint_allowed_mask(detector, candidates)
    invalid_count = int((~allowed).sum())
    _record_invalid_configs(detector, stage, invalid_count)
    return candidates.loc[allowed].copy().reset_index(drop=True)

def _parse_search_space_row(search_space_csv: Path, dataset_name: str, *, detector: str | None = None) -> list[SearchDimension]:
    # Detector search-space CSVs are manually maintained input files. The
    # recommendation pipeline reads them as fixed search-domain definitions and
    # never derives or modifies them from benchmark performance.
    df = _load_search_space_csv(search_space_csv, detector=detector or search_space_csv.stem.replace("_search_space", ""))
    dataset_rows = df.loc[df['Dataset'].astype(str) == str(dataset_name)].copy()
    if dataset_rows.empty:
        raise ValueError(
            f"Search space file {search_space_csv} does not contain a Dataset='{dataset_name}' row."
        )
    if len(dataset_rows) > 1:
        dedup = dataset_rows.drop_duplicates()
        if len(dedup) > 1:
            raise ValueError(
                f"Search-space CSV has conflicting duplicate Dataset='{dataset_name}' rows: {search_space_csv}"
            )
        raise ValueError(
            f"Search-space CSV has duplicate Dataset='{dataset_name}' rows: {search_space_csv}"
        )
    row = dataset_rows.iloc[0]
    dimensions: list[SearchDimension] = []
    seen_names: set[str] = set()
    for column in df.columns:
        if column == 'Dataset':
            continue
        raw = str(row[column]).strip()
        if not raw or raw.lower() == 'nan':
            continue
        parts = raw.split(':', 2)
        if len(parts) == 3 and parts[0].strip().lower() in {'range', 'choice'}:
            value_mode = parts[0].strip().lower()
            prefix = parts[1].strip().lower()
            payload = parts[2].strip()
        else:
            value_mode = ''
            if ':' not in raw:
                raise ValueError(
                    f"Search-space cell for hyperparameter '{column}' must contain a type prefix in {search_space_csv}."
                )
            (prefix, payload) = raw.split(':', 1)
            prefix = prefix.strip().lower()
            payload = payload.strip()
            value_mode = 'range' if prefix in {'int', 'float'} else 'choice'
        prefix = prefix.strip().lower()
        payload = payload.strip()
        if prefix == 'int':
            if value_mode == 'choice':
                choices = _split_choice_payload(payload)
                if not choices:
                    raise ValueError(f"Numeric choice hyperparameter '{column}' has no choices in {search_space_csv}.")
                dimensions.append(_validate_dimension(SearchDimension(name=column, kind='int', choices=choices, value_mode='choice'), search_space_csv=search_space_csv))
            else:
                (lo_text, hi_text) = payload.split('|', 1)
                dimensions.append(_validate_dimension(SearchDimension(name=column, kind='int', lower=float(lo_text), upper=float(hi_text), value_mode='range'), search_space_csv=search_space_csv))
        elif prefix == 'float':
            if value_mode == 'choice':
                choices = _split_choice_payload(payload)
                if not choices:
                    raise ValueError(f"Numeric choice hyperparameter '{column}' has no choices in {search_space_csv}.")
                dimensions.append(_validate_dimension(SearchDimension(name=column, kind='float', choices=choices, value_mode='choice'), search_space_csv=search_space_csv))
            else:
                (lo_text, hi_text) = payload.split('|', 1)
                dimensions.append(_validate_dimension(SearchDimension(name=column, kind='float', lower=float(lo_text), upper=float(hi_text), value_mode='range'), search_space_csv=search_space_csv))
        elif prefix == 'bool':
            choices = tuple((part.strip().upper() for part in _split_choice_payload(payload)))
            if not choices:
                choices = ('TRUE', 'FALSE')
            dimensions.append(_validate_dimension(SearchDimension(name=column, kind='bool', choices=choices, value_mode='choice'), search_space_csv=search_space_csv))
        elif prefix == 'cat':
            choices = _split_choice_payload(payload)
            if not choices:
                raise ValueError(f"Categorical hyperparameter '{column}' has no choices in {search_space_csv}.")
            dimensions.append(_validate_dimension(SearchDimension(name=column, kind='cat', choices=choices, value_mode='choice'), search_space_csv=search_space_csv))
        else:
            raise ValueError(f"Unsupported search-space type '{prefix}' for hyperparameter '{column}'.")
        if column in seen_names:
            raise ValueError(f"Duplicate hyperparameter definition for '{column}' in {search_space_csv}.")
        seen_names.add(column)
    if not dimensions:
        raise ValueError(f'No hyperparameters found in search space file {search_space_csv}.')
    return dimensions

def _unit_to_candidate_frame(unit: np.ndarray, dimensions: list[SearchDimension]) -> pd.DataFrame:
    data: dict[str, Any] = {}
    for (idx, dim) in enumerate(dimensions):
        values = unit[:, idx]
        if dim.kind in {'int', 'float'} and dim.value_mode == 'choice':
            choices = _numeric_choice_values(dim)
            indices = np.floor(values * len(choices)).astype(int)
            indices = np.clip(indices, 0, len(choices) - 1)
            data[dim.name] = [choices[i] for i in indices]
        elif dim.kind == 'int':
            lo = int(round(float(dim.lower)))
            hi = int(round(float(dim.upper)))
            span = hi - lo + 1
            mapped = lo + np.floor(values * span).astype(int)
            mapped = np.clip(mapped, lo, hi)
            data[dim.name] = mapped.astype(int)
        elif dim.kind == 'float':
            lo = float(dim.lower)
            hi = float(dim.upper)
            data[dim.name] = lo + values * (hi - lo)
        elif dim.kind in {'cat', 'bool'}:
            choices = list(dim.choices or ())
            indices = np.floor(values * len(choices)).astype(int)
            indices = np.clip(indices, 0, len(choices) - 1)
            if dim.kind == 'bool':
                data[dim.name] = [_normalize_bool_token(choices[i]) for i in indices]
            else:
                data[dim.name] = [choices[i] for i in indices]
        else:
            raise ValueError(f"Unsupported dimension kind '{dim.kind}'.")
    return pd.DataFrame(data)

def _sample_sobol_candidates(
    dimensions: list[SearchDimension],
    *,
    count: int,
    random_seed: int,
    detector: str,
) -> pd.DataFrame:
    if count <= 0:
        return pd.DataFrame(columns=[dim.name for dim in dimensions])

    collected: list[pd.DataFrame] = []
    collected_count = 0
    attempt = 0
    draw_count = _next_power_of_two_at_least(max(count, 256))

    while collected_count < count:
        sampler = qmc.Sobol(d=len(dimensions), scramble=True, seed=random_seed + attempt)
        unit = sampler.random_base2(m=int(math.log2(draw_count)))
        batch = _unit_to_candidate_frame(unit, dimensions)
        valid_batch = _validate_generated_candidates(detector, batch, stage="sobol")
        if not valid_batch.empty:
            needed = count - collected_count
            collected.append(valid_batch.head(needed).copy())
            collected_count += min(needed, len(valid_batch))
        attempt += 1
        if attempt > int(SOBOL_MAX_CONSTRAINT_ATTEMPTS) and collected_count < count:
            raise RuntimeError(
                f"Could not generate {count:,} valid Sobol candidates for detector '{detector}' "
                f"after {attempt} attempts. Check detector constraints and search-space bounds."
            )
        if collected_count < count:
            draw_count = _next_power_of_two_at_least(min(max(draw_count * 2, count), count * 16))

    return pd.concat(collected, axis=0, ignore_index=True).head(count).reset_index(drop=True)

def _drop_exact_duplicate_configs(df: pd.DataFrame, hyperparameter_names: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.drop_duplicates(subset=hyperparameter_names, keep='first').reset_index(drop=True)


def _deduplicate_final_candidate_pool(
    df: pd.DataFrame,
    *,
    dimensions: list[SearchDimension],
) -> pd.DataFrame:
    """Remove duplicate detector hyperparameter configurations before final selection."""
    if df.empty:
        return df.copy()
    out = df.copy().reset_index(drop=True)
    out["config_key"] = _config_key_series(out, dimensions)
    source_column = (
        "candidate_source"
        if "candidate_source" in out.columns
        else "sampler"
        if "sampler" in out.columns
        else None
    )
    if source_column is not None:
        source_map = (
            out.groupby("config_key", sort=False)[source_column]
            .apply(lambda values: "|".join(dict.fromkeys(str(value) for value in values if pd.notna(value))))
            .to_dict()
        )
    else:
        source_map = {}
    out = out.drop_duplicates(subset=["config_key"], keep="first").reset_index(drop=True)
    if source_map:
        out["candidate_sources"] = out["config_key"].map(source_map)
    return out.drop(columns=["config_key"])


def _prepare_model_input(candidates: pd.DataFrame, *, required_feature_names: list[str], metadata_row: pd.Series | None) -> pd.DataFrame:
    features = candidates.copy()
    if metadata_row is not None:
        for (column, value) in metadata_row.items():
            if column not in features.columns:
                features[column] = value
    missing = [column for column in required_feature_names if column not in features.columns]
    if missing:
        raise ValueError(f'Model expects features that are not available for inference: {missing}')
    ordered = features[required_feature_names].copy()
    for column in ordered.columns:
        series = ordered[column]
        if pd.api.types.is_object_dtype(series):
            numeric = pd.to_numeric(series, errors='coerce')
            non_na = series.notna().sum()
            if non_na > 0 and numeric.notna().sum() / non_na >= 0.95:
                ordered[column] = numeric
    return ordered

def _model_feature_names(model, *, label: str='model') -> list[str]:
    names = list(getattr(model, 'feature_names_in_', []))
    if names:
        return names
    preprocessor = model.named_steps.get('preprocessor') if hasattr(model, 'named_steps') else None
    names = list(getattr(preprocessor, 'feature_names_in_', [])) if preprocessor is not None else []
    if not names:
        raise ValueError(f'Could not infer expected feature names from the saved {label}.')
    return names

def _prepare_model_for_inference(model):
    if hasattr(model, 'named_steps'):
        estimator = model.named_steps.get('estimator')
        if estimator is not None and hasattr(estimator, 'n_jobs'):
            estimator.n_jobs = RECOMMENDATION_N_JOBS
    return model

def _predict_model_scores(candidates: pd.DataFrame, *, model, metadata_row: pd.Series | None) -> np.ndarray:
    feature_names = _model_feature_names(model)
    X = _prepare_model_input(candidates, required_feature_names=feature_names, metadata_row=metadata_row)
    return np.asarray(model.predict(X), dtype=float)

def _predict_model_uncertainty(candidates: pd.DataFrame, *, model, metadata_row: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(candidates)
    nan_values = np.full(n_rows, np.nan, dtype=float)
    if n_rows == 0:
        return nan_values, nan_values
    if not hasattr(model, 'named_steps'):
        return nan_values, nan_values
    estimator = model.named_steps.get('estimator')
    preprocessor = model.named_steps.get('preprocessor')
    if estimator is None or preprocessor is None or not hasattr(estimator, 'estimators_'):
        return nan_values, nan_values
    feature_names = _model_feature_names(model)
    X = _prepare_model_input(candidates, required_feature_names=feature_names, metadata_row=metadata_row)
    Xt = preprocessor.transform(X)
    member_preds = np.column_stack([tree.predict(Xt) for tree in estimator.estimators_])
    tree_std = np.nanstd(member_preds, axis=1)
    tree_iqr = np.nanpercentile(member_preds, 75, axis=1) - np.nanpercentile(member_preds, 25, axis=1)
    return tree_std, tree_iqr

def _add_single_uncertainty_features(df: pd.DataFrame, *, model, metadata_row: pd.Series | None) -> pd.DataFrame:
    out = df.copy()
    if not COMPUTE_PREDICTION_UNCERTAINTY:
        return out
    tree_std, tree_iqr = _predict_model_uncertainty(out, model=model, metadata_row=metadata_row)
    out['tree_std'] = tree_std
    out['tree_iqr'] = tree_iqr
    return out

def _add_separate_uncertainty_features(df: pd.DataFrame, *, bundle: 'SeparateModelBundle', metadata_row: pd.Series | None) -> pd.DataFrame:
    out = df.copy()
    if not COMPUTE_PREDICTION_UNCERTAINTY:
        return out
    acc_std, acc_iqr = _predict_model_uncertainty(out, model=bundle.accuracy_model, metadata_row=metadata_row)
    runtime_std, runtime_iqr = _predict_model_uncertainty(out, model=bundle.runtime_model, metadata_row=metadata_row)
    out['accuracy_tree_std'] = acc_std
    out['accuracy_tree_iqr'] = acc_iqr
    out['runtime_tree_std'] = runtime_std
    out['runtime_tree_iqr'] = runtime_iqr
    return out

def _seed_params_for_optuna(seed_row: pd.Series, dimensions: list[SearchDimension], *, detector: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if detector == "HDDDM" and {"use_mmd2", "use_k2s_test"}.issubset(seed_row.index):
        use_mmd2 = bool(_normalize_bool_token(seed_row["use_mmd2"]))
        use_k2s = bool(_normalize_bool_token(seed_row["use_k2s_test"]))
        if use_mmd2 and not use_k2s:
            use_k2s = True
        params[HDDDM_TEST_PAIR_PARAM] = f"{int(use_mmd2)}|{int(use_k2s)}"
    for dim in dimensions:
        if detector == "HDDDM" and dim.name in {"use_mmd2", "use_k2s_test"}:
            continue
        value = seed_row[dim.name]
        if dim.kind == 'int':
            params[dim.name] = int(value)
        elif dim.kind == 'float':
            params[dim.name] = float(value)
        elif dim.kind == 'bool':
            params[dim.name] = 'TRUE' if _normalize_bool_token(value) else 'FALSE'
        elif dim.kind == 'cat':
            params[dim.name] = str(value)
        else:
            raise ValueError(f"Unsupported dimension kind '{dim.kind}'.")
    return params

def _row_from_trial_params(*, params: dict[str, Any], dimensions: list[SearchDimension], detector: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if detector == "HDDDM" and HDDDM_TEST_PAIR_PARAM in params:
        use_mmd2_text, use_k2s_text = str(params[HDDDM_TEST_PAIR_PARAM]).split("|", 1)
        row["use_mmd2"] = bool(int(use_mmd2_text))
        row["use_k2s_test"] = bool(int(use_k2s_text))
    for dim in dimensions:
        if dim.name in row:
            continue
        value = params[dim.name]
        if dim.kind == 'int':
            row[dim.name] = int(value)
        elif dim.kind == 'float':
            row[dim.name] = float(value)
        elif dim.kind == 'bool':
            row[dim.name] = _normalize_bool_token(value)
        elif dim.kind == 'cat':
            row[dim.name] = str(value)
        else:
            raise ValueError(f"Unsupported dimension kind '{dim.kind}'.")
    return row


def _partial_row_from_trial_params(
    *,
    params: dict[str, Any],
    dimensions: list[SearchDimension],
    detector: str,
) -> dict[str, Any]:
    """Convert complete or partial Optuna parameters into exportable columns."""
    row: dict[str, Any] = {dim.name: np.nan for dim in dimensions}
    if detector == "HDDDM" and HDDDM_TEST_PAIR_PARAM in params:
        use_mmd2_text, use_k2s_text = str(params[HDDDM_TEST_PAIR_PARAM]).split("|", 1)
        row["use_mmd2"] = bool(int(use_mmd2_text))
        row["use_k2s_test"] = bool(int(use_k2s_text))
    for dim in dimensions:
        if dim.name in row and not pd.isna(row[dim.name]):
            continue
        if dim.name not in params:
            continue
        value = params[dim.name]
        if dim.kind == "int":
            row[dim.name] = int(value)
        elif dim.kind == "float":
            row[dim.name] = float(value)
        elif dim.kind == "bool":
            row[dim.name] = _normalize_bool_token(value)
        elif dim.kind == "cat":
            row[dim.name] = str(value)
    return row


def _search_space_audit_fields(
    *,
    original_dimensions: list[SearchDimension],
    local_dimensions: list[SearchDimension],
) -> dict[str, Any]:
    """Describe the original and stage-local search space on every audit row."""
    original_by_name = _dimension_map(original_dimensions)
    fields: dict[str, Any] = {}
    for local_dim in local_dimensions:
        original_dim = original_by_name[local_dim.name]
        if local_dim.kind in {"int", "float"} and local_dim.value_mode == "range":
            fields[f"original_lower__{local_dim.name}"] = original_dim.lower
            fields[f"original_upper__{local_dim.name}"] = original_dim.upper
            fields[f"local_lower__{local_dim.name}"] = local_dim.lower
            fields[f"local_upper__{local_dim.name}"] = local_dim.upper
        else:
            fields[f"local_choices__{local_dim.name}"] = "|".join(
                str(choice) for choice in (local_dim.choices or ())
            )
    return fields

def _dimension_map(dimensions: list[SearchDimension]) -> dict[str, SearchDimension]:
    return {dim.name: dim for dim in dimensions}

def _suggest_dimension(
    trial: optuna.trial.Trial,
    dim: SearchDimension,
    *,
    lower: float | None = None,
    upper: float | None = None,
    choices: list[Any] | None = None,
) -> Any:
    if dim.kind in {'int', 'float'} and dim.value_mode == 'choice':
        return trial.suggest_categorical(dim.name, choices if choices is not None else _numeric_choice_values(dim))
    if dim.kind == 'int':
        lo = int(round(float(dim.lower if lower is None else lower)))
        hi = int(round(float(dim.upper if upper is None else upper)))
        if lo > hi:
            raise optuna.TrialPruned(f"No feasible integer range for {dim.name}: {lo}>{hi}")
        return trial.suggest_int(dim.name, lo, hi)
    if dim.kind == 'float':
        lo = float(dim.lower if lower is None else lower)
        hi = float(dim.upper if upper is None else upper)
        if lo > hi:
            raise optuna.TrialPruned(f"No feasible float range for {dim.name}: {lo}>{hi}")
        return trial.suggest_float(dim.name, lo, hi)
    if dim.kind == 'bool':
        choice = trial.suggest_categorical(dim.name, choices if choices is not None else list(dim.choices or ('TRUE', 'FALSE')))
        return _normalize_bool_token(choice)
    if dim.kind == 'cat':
        return trial.suggest_categorical(dim.name, choices if choices is not None else list(dim.choices or ()))
    raise ValueError(f"Unsupported dimension kind '{dim.kind}'.")

def _suggest_remaining_dimensions(
    trial: optuna.trial.Trial,
    dimensions: list[SearchDimension],
    params: dict[str, Any],
) -> dict[str, Any]:
    for dim in dimensions:
        if dim.name not in params:
            params[dim.name] = _suggest_dimension(trial, dim)
    return params

def _suggest_params(trial: optuna.trial.Trial, dimensions: list[SearchDimension], *, detector: str) -> dict[str, Any]:
    by_name = _dimension_map(dimensions)

    if detector == "HDDDM" and {"use_mmd2", "use_k2s_test"}.issubset(by_name):
        params: dict[str, Any] = {}
        pair = trial.suggest_categorical(HDDDM_TEST_PAIR_PARAM, ["0|0", "0|1", "1|1"])
        use_mmd2_text, use_k2s_text = str(pair).split("|", 1)
        params["use_mmd2"] = bool(int(use_mmd2_text))
        params["use_k2s_test"] = bool(int(use_k2s_text))
        return _suggest_remaining_dimensions(trial, dimensions, params)

    if detector == "PCACD" and {"window_size", "sample_period"}.issubset(by_name):
        params = {}
        window_dim = by_name["window_size"]
        sample_dim = by_name["sample_period"]
        min_window = max(float(window_dim.lower), math.ceil(1.0 / float(sample_dim.upper)))
        window_size = int(_suggest_dimension(trial, window_dim, lower=min_window))
        params["window_size"] = window_size
        min_sample_period = max(float(sample_dim.lower), 1.0 / float(window_size))
        params["sample_period"] = _suggest_dimension(trial, sample_dim, lower=min_sample_period)
        return _suggest_remaining_dimensions(trial, dimensions, params)

    if detector == "NNDVI" and {"n_samples", "k_neighbors"}.issubset(by_name):
        params = {}
        n_samples_dim = by_name["n_samples"]
        k_neighbors_dim = by_name["k_neighbors"]
        min_n_samples = max(float(n_samples_dim.lower), math.ceil((float(k_neighbors_dim.lower) + 1.0) / 2.0))
        n_samples = int(_suggest_dimension(trial, n_samples_dim, lower=min_n_samples))
        params["n_samples"] = n_samples
        max_k_neighbors = min(float(k_neighbors_dim.upper), (2.0 * n_samples) - 1.0)
        params["k_neighbors"] = _suggest_dimension(trial, k_neighbors_dim, upper=max_k_neighbors)
        return _suggest_remaining_dimensions(trial, dimensions, params)

    if detector == "SlidShaps" and {"batch_size", "overlap"}.issubset(by_name):
        params = {}
        batch_dim = by_name["batch_size"]
        overlap_dim = by_name["overlap"]
        min_batch = max(float(batch_dim.lower), math.ceil(1.0 / float(overlap_dim.upper)))
        batch_size = int(_suggest_dimension(trial, batch_dim, lower=min_batch))
        params["batch_size"] = batch_size
        min_overlap = max(float(overlap_dim.lower), 1.0 / float(batch_size))
        params["overlap"] = _suggest_dimension(trial, overlap_dim, lower=min_overlap)
        return _suggest_remaining_dimensions(trial, dimensions, params)

    if detector == "IBDD" and {"n_permutations", "update_interval", "n_consecutive_deviations"}.issubset(by_name):
        params = {}
        n_perm_dim = by_name["n_permutations"]
        update_dim = by_name["update_interval"]
        consec_dim = by_name["n_consecutive_deviations"]
        max_consec = min(float(consec_dim.upper), float(n_perm_dim.upper), float(update_dim.upper) - 1.0)
        n_consec = int(_suggest_dimension(trial, consec_dim, upper=max_consec))
        params["n_consecutive_deviations"] = n_consec
        params["n_permutations"] = _suggest_dimension(trial, n_perm_dim, lower=max(float(n_perm_dim.lower), float(n_consec)))
        params["update_interval"] = _suggest_dimension(trial, update_dim, lower=max(float(update_dim.lower), float(n_consec + 1)))
        return _suggest_remaining_dimensions(trial, dimensions, params)

    if detector == "WindowKDE" and {"recent_samples_size", "big_windowSize", "small_windowSize"}.issubset(by_name):
        params = {}
        recent_dim = by_name["recent_samples_size"]
        big_dim = by_name["big_windowSize"]
        small_dim = by_name["small_windowSize"]
        min_recent = max(float(recent_dim.lower), float(big_dim.lower) + 1.0, (4.0 * float(small_dim.lower)) + 1.0)
        recent = int(_suggest_dimension(trial, recent_dim, lower=min_recent))
        params["recent_samples_size"] = recent
        max_small = min(float(small_dim.upper), math.floor((float(recent) - 1.0) / 4.0))
        small = int(_suggest_dimension(trial, small_dim, upper=max_small))
        params["small_windowSize"] = small
        min_big = max(float(big_dim.lower), float(small * 4))
        max_big = min(float(big_dim.upper), float(recent - 1))
        params["big_windowSize"] = _suggest_dimension(trial, big_dim, lower=min_big, upper=max_big)
        return _suggest_remaining_dimensions(trial, dimensions, params)

    params: dict[str, Any] = {}
    for dim in dimensions:
        params[dim.name] = _suggest_dimension(trial, dim)
    return params

def _parse_args(mode: str) -> argparse.Namespace:
    description = 'Recommend detector configurations from separate LODO models.' if mode == 'separate' else 'Recommend detector configurations from trained LODO models.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--detector', type=str, default='ALL', help='Detector selection: ALL, exact name, or comma-separated exact names.')
    parser.add_argument('--dataset', type=str, default='ALL', help='Dataset selection: ALL, exact name, or comma-separated exact names.')
    add_pipeline_setup_args(parser)
    return parser.parse_args()


def _apply_pipeline_setup(args: argparse.Namespace) -> None:
    """Apply sweep setup overrides when supplied; otherwise keep config defaults."""
    global TARGET_MODE, SINGLE_TARGET_METHOD, TRAIN_USE_METADATA, TRAIN_METADATA_VARIANT, USE_PREFERENCE_REGIONS
    setup = resolve_pipeline_setup(
        args,
        default_target_mode=CONFIG_RECOMMENDATION_TARGET_MODE,
        default_single_target_formulation=SINGLE_TARGET_FORMULATION,
        default_use_metadata=TRAIN_USE_METADATA,
        default_metadata_variant=TRAIN_METADATA_VARIANT,
    )
    TARGET_MODE = setup.target_mode
    SINGLE_TARGET_METHOD = setup.single_target_formulation
    TRAIN_USE_METADATA = setup.use_metadata
    TRAIN_METADATA_VARIANT = setup.metadata_variant
    USE_PREFERENCE_REGIONS = bool(
        TRAIN_USE_PREFERENCE_REGIONS
        and TARGET_MODE == "single"
        and SINGLE_TARGET_METHOD in REGIONAL_TARGETS
    )

def _selected_scope(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    selected_detectors = _resolve_cli_selection(args.detector, available=DETECTORS, what='detector')
    selected_datasets = _resolve_cli_selection(args.dataset, available=DATASETS, what='dataset')
    return selected_detectors, selected_datasets

def _save_with_context(df: pd.DataFrame, path: Path, *, detector: str, lodo_dataset: str) -> None:
    export_df = df.copy()
    if not export_df.empty:
        export_df.insert(0, 'lodo_dataset', lodo_dataset)
        export_df.insert(0, 'detector', detector)
    save_dataframe(export_df, path, index=False)

def _load_metadata_row_if_needed(paths, *, lodo_dataset: str, required_feature_names: list[str]) -> pd.Series | None:
    if not TRAIN_USE_METADATA or not required_feature_names:
        return None
    metadata_df = load_metadata_table(paths, TRAIN_METADATA_VARIANT, lodo_dataset=lodo_dataset)
    row_df = metadata_df.loc[metadata_df["dataset_name"].astype(str) == str(lodo_dataset)]
    if row_df.empty:
        raise ValueError(f"No metadata row found for LODO dataset '{lodo_dataset}'.")
    metadata_row = row_df.iloc[0].drop(labels=["dataset_name"], errors="ignore")
    if TRAIN_SCALE_METADATA:
        numeric_cols = [col for col in metadata_row.index if col in metadata_df.columns and pd.api.types.is_numeric_dtype(metadata_df[col])]
        for col in numeric_cols:
            scaled = _simple_scale(metadata_df[col], method=TRAIN_METADATA_SCALE_METHOD)
            metadata_row[col] = scaled.loc[row_df.index[0]]
    return metadata_row


def _metadata_config_summary() -> str:
    return (
        f"TRAIN_USE_METADATA={TRAIN_USE_METADATA}, "
        f"TRAIN_METADATA_VARIANT={TRAIN_METADATA_VARIANT!r}, "
        f"TRAIN_SCALE_METADATA={TRAIN_SCALE_METADATA}, "
        f"TRAIN_METADATA_SCALE_METHOD={TRAIN_METADATA_SCALE_METHOD!r}"
    )


def _metadata_artifact_tag() -> str:
    """Return the Phase 1 metadata artifact tag shared with training/evaluation."""
    return metadata_variant_tag(
        TRAIN_METADATA_VARIANT,
        use_metadata=TRAIN_USE_METADATA,
    )


def _phase1_model_path(
    paths,
    detector: str,
    lodo_dataset: str,
    *,
    target_mode: str,
    target_method: str,
    preference_region: str | None = None,
    objective: str | None = None,
) -> Path:
    """Resolve the exact setup-specific Phase 1 model artifact for recommendation."""
    return paths.phase1_model_file(
        detector,
        lodo_dataset,
        target_mode=target_mode,
        target_method=target_method,
        metadata_tag=_metadata_artifact_tag(),
        preference_region=preference_region,
        objective=objective,
    )


def _load_phase1_model_or_fail(path: Path, *, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}. "
            "Run train_models.py first with matching target, metadata, PCA, "
            "and preference/objective configuration settings."
        )
    return _prepare_model_for_inference(load(path))


def _metadata_row_for_model_features(
    paths,
    *,
    lodo_dataset: str,
    dimensions: list[SearchDimension],
    required_feature_names: set[str],
    model_path: Path | None = None,
    model_path_for_error: Path | None = None,
) -> pd.Series | None:
    search_space_names = {dim.name for dim in dimensions}
    missing_from_search_space = sorted(required_feature_names - search_space_names)
    metadata_row = _load_metadata_row_if_needed(
        paths,
        lodo_dataset=lodo_dataset,
        required_feature_names=missing_from_search_space,
    )
    if metadata_row is None:
        missing = missing_from_search_space
    else:
        missing = [column for column in missing_from_search_space if column not in metadata_row.index]
    if missing:
        source = model_path_for_error or model_path or "model"
        raise ValueError(
            f"Model {source} requires features not available in the detector search space "
            f"or the metadata loaded by the current config: {missing}. "
            f"Current metadata config: {_metadata_config_summary()}. "
            "Set config.py to the same metadata settings used by train_models.py for this model, "
            "or retrain the model with the current config."
        )
    return metadata_row


def _allocate_by_shares(total_trials: int, shares: tuple[float, ...]) -> list[int]:
    """Allocate an exact integer total across fractional stage shares."""
    total = int(total_trials)
    exact = [total * float(share) for share in shares]
    allocated = [int(math.floor(value)) for value in exact]
    remainder = total - sum(allocated)
    remainder_order = sorted(
        range(len(exact)),
        key=lambda idx: (exact[idx] - allocated[idx], -idx),
        reverse=True,
    )
    for idx in remainder_order[:remainder]:
        allocated[idx] += 1
    return allocated


def _refinement_stage_plan(sampler_name: str) -> list[RefinementStage]:
    """Return the exact one-stage or multi-stage plan for the active mode."""
    normalized = str(sampler_name).strip().upper()
    if normalized not in {'TPE', 'NSGAII'}:
        raise ValueError(f"Unsupported sampler margin schedule '{sampler_name}'.")
    if SETTINGS.mode == "static":
        if normalized == "NSGAII":
            return [
                RefinementStage(
                    index=1,
                    margin_ratio=SETTINGS.nsgaii_numeric_margin,
                    seed_count=SETTINGS.nsgaii_seed_count,
                    trials=SETTINGS.nsgaii_trials,
                )
            ]
        return [
            RefinementStage(
                index=1,
                margin_ratio=SETTINGS.static_tpe_numeric_margin,
                seed_count=SETTINGS.static_tpe_seed_count,
                trials=SETTINGS.static_tpe_trials,
            )
        ]
    if normalized == "TPE":
        stage_totals = _allocate_by_shares(
            SETTINGS.dynamic_tpe_total_trials,
            SETTINGS.dynamic_tpe_trial_shares,
        )
        return [
            RefinementStage(
                index=stage_index,
                margin_ratio=margin_ratio,
                seed_count=seed_count,
                trials=stage_total,
            )
            for stage_index, (margin_ratio, seed_count, stage_total) in enumerate(
                zip(
                    SETTINGS.dynamic_tpe_margins,
                    SETTINGS.dynamic_tpe_seed_counts,
                    stage_totals,
                ),
                start=1,
            )
        ]
    return [
        RefinementStage(
            index=1,
            margin_ratio=SETTINGS.nsgaii_numeric_margin,
            seed_count=SETTINGS.nsgaii_seed_count,
            trials=SETTINGS.nsgaii_trials,
        )
    ]


def _format_margin_schedule(*, sampler_name: str = 'TPE') -> str:
    return ", ".join(
        (
            f"{stage.margin_ratio:.2f}={stage.trials:,} trials"
            f"/{stage.seed_count:,} seeds"
        )
        for stage in _refinement_stage_plan(sampler_name)
    )


def _build_semi_local_dimensions(
    seed_rows: pd.DataFrame,
    dimensions: list[SearchDimension],
    *,
    margin_ratio: float,
) -> list[SearchDimension]:
    """
    Rebuild the Optuna search space around selected stage seeds.

    Numeric bounds are narrowed by configured quantiles and margins. Categorical
    and Boolean domains remain global unless their restriction toggles are set.
    """
    if not SETTINGS.use_semi_local_optuna_space:
        return dimensions
    active_margin_ratio = float(margin_ratio)
    if active_margin_ratio < 0.0:
        raise ValueError("The local numeric margin ratio cannot be negative.")
    local_dimensions: list[SearchDimension] = []
    for dim in dimensions:
        if dim.kind in {"int", "float"} and dim.value_mode == "choice":
            local_dimensions.append(dim)
        elif dim.kind in {"int", "float"}:
            values = pd.to_numeric(seed_rows[dim.name], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                local_dimensions.append(dim)
                continue
            original_lo = float(dim.lower)
            original_hi = float(dim.upper)
            original_span = original_hi - original_lo
            seed_min = float(np.nanmin(values))
            seed_max = float(np.nanmax(values))
            q_low = float(np.nanquantile(values, LOCAL_NUMERIC_QUANTILE_LOW))
            q_high = float(np.nanquantile(values, LOCAL_NUMERIC_QUANTILE_HIGH))
            margin = active_margin_ratio * original_span
            local_lo = max(original_lo, min(q_low - margin, seed_min))
            local_hi = min(original_hi, max(q_high + margin, seed_max))
            if local_lo > local_hi:
                local_lo, local_hi = original_lo, original_hi
            if dim.kind == "int":
                int_lo = max(int(round(original_lo)), int(math.floor(local_lo)))
                int_hi = min(int(round(original_hi)), int(math.ceil(local_hi)))
                if int_lo > int_hi:
                    int_lo, int_hi = int(round(original_lo)), int(round(original_hi))
                local_dimensions.append(SearchDimension(dim.name, "int", float(int_lo), float(int_hi), value_mode="range"))
            else:
                if local_lo >= local_hi:
                    local_lo, local_hi = original_lo, original_hi
                local_dimensions.append(SearchDimension(dim.name, "float", float(local_lo), float(local_hi), value_mode="range"))
        elif dim.kind == "bool":
            choices = dim.choices or ("TRUE", "FALSE")
            if RESTRICT_BOOLEANS_TO_SELECTED_SEEDS:
                seen: list[str] = []
                for value in seed_rows[dim.name].tolist():
                    token = "TRUE" if _normalize_bool_token(value) else "FALSE"
                    if token not in seen:
                        seen.append(token)
                choices = tuple(seen) if seen else choices
            local_dimensions.append(SearchDimension(name=dim.name, kind="bool", choices=choices, value_mode="choice"))
        elif dim.kind == "cat":
            original_choices = list(dim.choices or ())
            if RESTRICT_CATEGORICALS_TO_SELECTED_SEEDS:
                observed: list[str] = []
                for value in seed_rows[dim.name].astype(str).tolist():
                    if value in original_choices and value not in observed:
                        observed.append(value)
                choices = tuple(observed) if observed else tuple(original_choices)
            else:
                choices = tuple(original_choices)
            if not choices:
                raise ValueError(f"Categorical hyperparameter '{dim.name}' has no local choices.")
            local_dimensions.append(SearchDimension(name=dim.name, kind="cat", choices=choices, value_mode="choice"))
        else:
            raise ValueError(f"Unsupported dimension kind '{dim.kind}'.")
    return local_dimensions


def _rows_within_search_dimensions(
    rows: pd.DataFrame,
    dimensions: list[SearchDimension],
) -> pd.DataFrame:
    """Keep warm-start rows representable by the current stage search space."""
    if rows.empty:
        return rows.copy()
    mask = pd.Series(True, index=rows.index, dtype=bool)
    for dim in dimensions:
        values = rows[dim.name]
        if dim.kind in {"int", "float"} and dim.value_mode == "range":
            numeric = pd.to_numeric(values, errors="coerce")
            mask &= numeric.notna()
            mask &= numeric.ge(float(dim.lower) - 1e-12)
            mask &= numeric.le(float(dim.upper) + 1e-12)
        elif dim.kind in {"int", "float"} and dim.value_mode == "choice":
            allowed = _numeric_choice_values(dim)
            numeric = pd.to_numeric(values, errors="coerce")
            mask &= numeric.isin(allowed)
        elif dim.kind == "bool":
            allowed = {
                _normalize_bool_token(choice)
                for choice in (dim.choices or ("TRUE", "FALSE"))
            }
            mask &= values.map(_normalize_bool_token).isin(allowed)
        elif dim.kind == "cat":
            allowed = {str(choice) for choice in (dim.choices or ())}
            mask &= values.astype(str).isin(allowed)
    return rows.loc[mask].copy().reset_index(drop=True)


# =============================================================================
# Single-target flow
# =============================================================================
def _rank_single_sobol_candidates(sobol_df: pd.DataFrame, *, dimensions: list[SearchDimension], model, metadata_row: pd.Series | None) -> pd.DataFrame:
    hyperparameter_names = [dim.name for dim in dimensions]
    ranked_df = sobol_df.copy()
    ranked_df['predicted_score'] = _predict_model_scores(ranked_df, model=model, metadata_row=metadata_row)
    ranked_df = ranked_df.sort_values('predicted_score', ascending=True).reset_index(drop=True)
    if DROP_EXACT_DUPLICATE_CONFIGS_AFTER_SOBOL:
        ranked_df = _drop_exact_duplicate_configs(ranked_df, hyperparameter_names)
    return ranked_df

def _select_top_seed_rows(
    ranked_sobol_df: pd.DataFrame,
    *,
    dimensions: list[SearchDimension],
    seed_count: int,
    stage_label: str = "sobol_top_seed",
) -> pd.DataFrame:
    if ranked_sobol_df.empty:
        return ranked_sobol_df.copy()
    hyperparameter_names = [dim.name for dim in dimensions]
    seed_rows = (
        ranked_sobol_df
        .sort_values('predicted_score', ascending=True)
        .drop_duplicates(subset=hyperparameter_names, keep='first')
        .head(int(seed_count))
    )
    if len(seed_rows) != int(seed_count):
        raise ValueError(
            f"Single-target refinement requires exactly {int(seed_count)} unique seed configurations, "
            f"but only {len(seed_rows)} unique predicted candidates are available."
        )
    seed_rows = seed_rows.copy().reset_index(drop=True)
    seed_rows = seed_rows.drop(
        columns=['source_seed_rank', 'source_seed_predicted_score'],
        errors='ignore',
    )
    seed_rows.insert(0, 'source_seed_rank', np.arange(1, len(seed_rows) + 1, dtype=int))
    seed_rows.insert(1, 'source_seed_predicted_score', seed_rows['predicted_score'].astype(float))
    seed_rows['stage_label'] = stage_label
    seed_rows['stage_trials'] = 0
    seed_rows['optuna_trial_number'] = -1
    return seed_rows

def _single_objective_factory(*, detector: str, dimensions: list[SearchDimension], model, metadata_row: pd.Series | None):

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(trial, dimensions, detector=detector)
        candidate_df = pd.DataFrame([params])
        candidate_df = _validate_generated_candidates(detector, candidate_df, stage="optuna")
        if candidate_df.empty:
            raise optuna.TrialPruned("Detector constraints rejected Optuna suggestion.")
        pred = _predict_model_scores(candidate_df, model=model, metadata_row=metadata_row)
        return float(pred[0])
    return objective

def _run_single_optuna_refinement(*, detector: str, seed_rows: pd.DataFrame, dimensions: list[SearchDimension], model, metadata_row: pd.Series | None, stage_label: str) -> pd.DataFrame:
    """
    Run TPE refinement using the active one-stage or dynamic stage plan.

    Stage trial budgets are Optuna refinement suggestions; warm-start seed
    trials are enqueued in addition and consumed before those suggestions.
    """
    if seed_rows.empty:
        return seed_rows.copy()
    hyperparameter_names = [dim.name for dim in dimensions]
    archive_frames: list[pd.DataFrame] = [seed_rows.copy()]
    archive = seed_rows.copy().reset_index(drop=True)
    protected_seed_rows = seed_rows.copy().reset_index(drop=True)
    study: optuna.study.Study | None = None
    executed_trial_total = 0

    for stage in _refinement_stage_plan('TPE'):
        if stage.trials < 0:
            raise ValueError("TPE refinement trial budgets cannot be negative.")
        if stage.trials == 0:
            continue
        if SETTINGS.mode == "dynamic":
            protected_seed_rows = _select_top_seed_rows(
                archive,
                dimensions=dimensions,
                seed_count=stage.seed_count,
                stage_label=f'dynamic_tpe_stage_{stage.index}_scalar_seed',
            )
        else:
            protected_seed_rows = seed_rows.copy().reset_index(drop=True)
        if len(protected_seed_rows) != int(stage.seed_count):
            raise RuntimeError(
                f"TPE stage {stage.index} selected {len(protected_seed_rows)} seeds; "
                f"expected {stage.seed_count}."
            )
        # Rebuild the local search space around the selected stage seeds.
        local_dimensions = _build_semi_local_dimensions(
            protected_seed_rows,
            dimensions,
            margin_ratio=stage.margin_ratio,
        )
        warm_seed_rows = _rows_within_search_dimensions(
            protected_seed_rows,
            local_dimensions,
        )
        if len(warm_seed_rows) != int(stage.seed_count):
            raise RuntimeError(
                f"TPE stage {stage.index} enqueued {len(warm_seed_rows)} warm-start seeds; "
                f"expected {stage.seed_count}. Check local-space construction and seed restrictions."
            )
        if study is None or SETTINGS.use_fresh_study_per_margin_stage:
            sampler = optuna.samplers.TPESampler(
                seed=RANDOM_SEED + stage.index - 1,
                multivariate=TPE_MULTIVARIATE,
                group=TPE_GROUP,
                constant_liar=TPE_CONSTANT_LIAR,
                n_startup_trials=OPTUNA_STARTUP_TRIALS,
                warn_independent_sampling=False,
            )
            study = optuna.create_study(direction='minimize', sampler=sampler)
        stage_trial_start = len(study.trials)
        for _, seed_row in warm_seed_rows.iterrows():
            study.enqueue_trial(
                _seed_params_for_optuna(seed_row, local_dimensions, detector=detector)
            )
        objective = _single_objective_factory(
            detector=detector,
            dimensions=local_dimensions,
            model=model,
            metadata_row=metadata_row,
        )
        study.optimize(
            objective,
            n_trials=len(warm_seed_rows) + int(stage.trials),
            n_jobs=OPTUNA_STUDY_N_JOBS,
            show_progress_bar=False,
        )

        rows: list[dict[str, Any]] = []
        stage_study_trials = study.trials[stage_trial_start:]
        executed_trial_total += len(stage_study_trials)
        warm_start_count = min(len(warm_seed_rows), len(stage_study_trials))
        if warm_start_count != int(stage.seed_count):
            raise RuntimeError(
                f"TPE stage {stage.index} consumed {warm_start_count} warm-start trials; "
                f"expected {stage.seed_count}."
            )
        for stage_trial_position, trial in enumerate(stage_study_trials):
            if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
                continue
            params_row = _row_from_trial_params(
                params=trial.params,
                dimensions=local_dimensions,
                detector=detector,
            )
            row: dict[str, Any] = {
                'source_seed_rank': -1,
                'source_seed_predicted_score': np.nan,
                'stage_label': f'{stage_label}_margin_{stage.margin_ratio:.2f}',
                'stage_trials': int(stage.trials),
                'margin_stage': int(stage.index),
                'margin_ratio': float(stage.margin_ratio),
                'optuna_trial_number': int(trial.number),
                'predicted_score': float(trial.value),
            }
            row.update(params_row)
            if stage_trial_position < warm_start_count:
                protected_row = warm_seed_rows.iloc[stage_trial_position]
                row['source_seed_rank'] = int(protected_row.get('source_seed_rank', -1))
                row['source_seed_predicted_score'] = float(
                    protected_row.get('source_seed_predicted_score', np.nan)
                )
                row['stage_label'] = 'sobol_or_archive_warm_start'
                row['stage_trials'] = 0
                row['margin_stage'] = 0
                row['margin_ratio'] = np.nan
            rows.append(row)

        if rows:
            archive_frames.append(pd.DataFrame(rows))
        archive = pd.concat(archive_frames, axis=0, ignore_index=True)
        archive = _drop_exact_duplicate_configs(archive, hyperparameter_names)
        print(
            f'    Optuna margin stage {stage.index}: ratio={stage.margin_ratio:.2f} '
            f'| trials={stage.trials:,} | protected_seeds={len(protected_seed_rows):,} '
            f'| warm_starts_in_bounds={len(warm_seed_rows):,}'
        )

    expected_total = sum(stage.seed_count + stage.trials for stage in _refinement_stage_plan('TPE'))
    if executed_trial_total != expected_total:
        raise RuntimeError(
            f"TPE refinement executed {executed_trial_total} trials; "
            f"expected {expected_total} warm-start plus refinement trials across stage(s)."
        )

    refined_df = pd.concat(archive_frames, axis=0, ignore_index=True)
    refined_df = refined_df.sort_values('predicted_score', ascending=True).reset_index(drop=True)
    if DROP_EXACT_DUPLICATE_CONFIGS_AFTER_OPTUNA:
        refined_df = _drop_exact_duplicate_configs(refined_df, hyperparameter_names)
    return refined_df.sort_values('predicted_score', ascending=True).reset_index(drop=True)

def _build_single_candidate_pool_audit(
    candidate_pool: pd.DataFrame,
    *,
    recommendations_df: pd.DataFrame,
    dimensions: list[SearchDimension],
) -> pd.DataFrame:
    """Annotate the single-target Sobol-seed plus TPE pool for export."""
    if candidate_pool.empty:
        return candidate_pool.copy()
    audit = candidate_pool.copy().reset_index(drop=True)
    audit["config_key"] = _config_key_series(audit, dimensions)
    final_keys: set[str] = set()
    rank_map: dict[str, Any] = {}
    if not recommendations_df.empty:
        final_info = recommendations_df.copy()
        final_info["config_key"] = _config_key_series(final_info, dimensions)
        final_keys = set(final_info["config_key"].astype(str))
        rank_map = (
            final_info.drop_duplicates("config_key")
            .set_index("config_key")["recommendation_rank"]
            .to_dict()
        )
    audit["is_final_recommendation"] = audit["config_key"].astype(str).isin(final_keys)
    audit["final_recommendation_rank"] = audit["config_key"].map(rank_map)
    sort_columns = ["predicted_score", "config_key"]
    missing_sort_columns = [column for column in sort_columns if column not in audit.columns]
    if missing_sort_columns:
        raise ValueError(f"Single candidate-pool audit is missing sort column(s): {missing_sort_columns}")
    audit = audit.sort_values(
        sort_columns,
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    audit.insert(0, "candidate_audit_id", np.arange(1, len(audit) + 1, dtype=int))
    return audit.reset_index(drop=True)


def _build_single_recommendation_output(*, detector: str, sobol_df: pd.DataFrame, model, dimensions: list[SearchDimension], metadata_row: pd.Series | None, budget: int = PHASE2_RECOMMENDATION_BUDGET) -> tuple[pd.DataFrame, pd.DataFrame]:
    hyperparameter_names = [dim.name for dim in dimensions]
    ranked_sobol_df = _rank_single_sobol_candidates(sobol_df, dimensions=dimensions, model=model, metadata_row=metadata_row)
    first_stage = _refinement_stage_plan('TPE')[0]
    seed_rows = _select_top_seed_rows(
        ranked_sobol_df,
        dimensions=dimensions,
        seed_count=first_stage.seed_count,
    )
    effective_trials = sum(stage.trials for stage in _refinement_stage_plan('TPE'))
    print(f'  Sobol ranked={len(ranked_sobol_df):,} | top_seeds={len(seed_rows):,} | optuna_trials={effective_trials:,}')
    candidate_pool = _run_single_optuna_refinement(detector=detector, seed_rows=seed_rows, dimensions=dimensions, model=model, metadata_row=metadata_row, stage_label='optuna_tpe')
    unique_candidates = int(candidate_pool.drop_duplicates(subset=hyperparameter_names).shape[0])
    if unique_candidates < int(budget):
        raise ValueError(
            f"Cannot select {int(budget)} single-target recommendations for detector={detector}: "
            f"only {unique_candidates} unique candidate configurations are available."
        )
    final_df = (
        candidate_pool
        .sort_values('predicted_score', ascending=True)
        .drop_duplicates(subset=hyperparameter_names, keep='first')
        .head(int(budget))
        .reset_index(drop=True)
    )
    if final_df.empty:
        empty_columns = ['recommendation_rank', 'source_seed_rank', 'source_seed_predicted_score', 'stage_label', 'stage_trials', 'optuna_trial_number', 'predicted_score'] + hyperparameter_names
        empty_final = pd.DataFrame(columns=empty_columns)
        return empty_final, _build_single_candidate_pool_audit(
            candidate_pool,
            recommendations_df=empty_final,
            dimensions=dimensions,
        )
    final_df = final_df.sort_values('predicted_score', ascending=True).reset_index(drop=True)
    final_df = _add_single_uncertainty_features(final_df, model=model, metadata_row=metadata_row)
    final_df.insert(0, 'recommendation_rank', np.arange(1, len(final_df) + 1, dtype=int))
    all_candidates_df = _build_single_candidate_pool_audit(
        candidate_pool,
        recommendations_df=final_df,
        dimensions=dimensions,
    )
    return final_df, all_candidates_df

def _save_single_recommendation_results(*, base_dir: Path, detector: str, lodo_dataset: str, recommendations_df: pd.DataFrame, all_candidates_df: pd.DataFrame) -> None:
    output_dir = ensure_dir(base_dir / 'single')
    output_path = output_dir / f'{detector}_{lodo_dataset}_single_recommended_configs.csv'
    _save_with_context(recommendations_df, output_path, detector=detector, lodo_dataset=lodo_dataset)
    all_path = output_dir / f'{detector}_{lodo_dataset}_single_all_candidates.csv'
    _save_with_context(all_candidates_df, all_path, detector=detector, lodo_dataset=lodo_dataset)


def _load_preference_region_models(paths, detector: str, lodo_dataset: str) -> tuple[dict[str, Path], dict[str, Any]]:
    model_paths: dict[str, Path] = {}
    region_bundle: dict[str, Any] = {}
    for region_name in PREFERENCE_REGION_NAMES:
        model_path = _phase1_model_path(
            paths,
            detector,
            lodo_dataset,
            target_mode="single",
            target_method=SINGLE_TARGET_METHOD,
            preference_region=region_name,
        )
        regressor = _load_phase1_model_or_fail(
            model_path,
            label=f"Preference-region recommendation model ({region_name})",
        )
        model_paths[region_name] = model_path
        region_bundle[region_name] = {
            "target_column": f"{SINGLE_TARGET_METHOD}_{region_name}",
            "regressor": regressor,
            "model_path": str(model_path),
        }
    return model_paths, region_bundle


def _preference_region_model_feature_names(region_bundle: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for region_name in PREFERENCE_REGION_NAMES:
        names.update(_model_feature_names(region_bundle[region_name]["regressor"], label=f"{region_name} regressor"))
    return names


def _build_single_preference_region_output(
    *,
    detector: str,
    sobol_df: pd.DataFrame,
    region_name: str,
    regressor,
    dimensions: list[SearchDimension],
    metadata_row: pd.Series | None,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_df, all_candidates_df = _build_single_recommendation_output(
        detector=detector,
        sobol_df=sobol_df,
        model=regressor,
        dimensions=dimensions,
        metadata_row=metadata_row,
        budget=top_k,
    )
    if not region_df.empty:
        region_df.insert(1, "preference_region", region_name)
    if not all_candidates_df.empty:
        all_candidates_df.insert(1, "preference_region", region_name)
    return region_df, all_candidates_df


def _save_single_preference_region_results(
    *,
    base_dir: Path,
    detector: str,
    lodo_dataset: str,
    region_results: dict[str, pd.DataFrame],
    region_candidate_pools: dict[str, pd.DataFrame],
    dimensions: list[SearchDimension],
) -> None:
    per_region_budget, combined_budget = _preference_region_budget()
    chosen_parts: list[pd.DataFrame] = []
    for region_name in PREFERENCE_REGION_NAMES:
        region_df = region_results.get(region_name, pd.DataFrame()).copy()
        region_dir = ensure_dir(base_dir / "preference_regions" / region_name)
        region_path = region_dir / f"{detector}_{lodo_dataset}_{region_name}_recommended_configs.csv"
        _save_with_context(region_df, region_path, detector=detector, lodo_dataset=lodo_dataset)
        region_pool = region_candidate_pools.get(region_name, pd.DataFrame()).copy()
        region_pool_path = region_dir / f"{detector}_{lodo_dataset}_{region_name}_all_candidates.csv"
        _save_with_context(region_pool, region_pool_path, detector=detector, lodo_dataset=lodo_dataset)
        if not region_df.empty:
            chosen_parts.append(region_df.head(per_region_budget).copy())

    if chosen_parts:
        hyperparameter_names = [dim.name for dim in dimensions]
        combined = pd.concat(chosen_parts, axis=0, ignore_index=True)
        combined = _drop_exact_duplicate_configs(combined, hyperparameter_names)
        combined = combined.sort_values("predicted_score", ascending=True).head(combined_budget).reset_index(drop=True)
        combined["final_rank"] = np.arange(1, len(combined) + 1, dtype=int)
    else:
        combined = pd.DataFrame()

    combined_dir = ensure_dir(base_dir / "preference_regions_combined")
    combined_path = combined_dir / f"{detector}_{lodo_dataset}_final_recommended_configs.csv"
    _save_with_context(combined, combined_path, detector=detector, lodo_dataset=lodo_dataset)


def _print_single_settings() -> None:
    tpe_plan = _refinement_stage_plan('TPE')
    print('\nConfig recommendation local settings')
    _print_effective_recommendation_settings(target_mode="single")
    print("  Algorithm: TPE")
    print(f'  Top Sobol seed count: {tpe_plan[0].seed_count:,}')
    print(f'  Optuna trials: {sum(stage.trials for stage in tpe_plan):,}')
    print(f'  Margin stages: {_format_margin_schedule(sampler_name="TPE")}')
    print(f'  Optuna startup trials: {OPTUNA_STARTUP_TRIALS:,}')
    print(f'  Semi-local Optuna space: {SETTINGS.use_semi_local_optuna_space}')
    print(f'  Final recommendation budget: {PHASE2_RECOMMENDATION_BUDGET:,}')
    print(f'  Preference-region mode: {USE_PREFERENCE_REGIONS}')
    if USE_PREFERENCE_REGIONS:
        per_region_budget, combined_budget = _preference_region_budget()
        print(f'  Preference regions: {PREFERENCE_REGION_NAMES}')
        print(f'  Preference final per region: {per_region_budget}')
        print(f'  Preference combined final K: {combined_budget}')
    print(f'  Export uncertainty features: {COMPUTE_PREDICTION_UNCERTAINTY}')

def _run_single_recommendation() -> None:
    started_at = time.perf_counter()
    args = _parse_args('single')
    _apply_pipeline_setup(args)
    selected_detectors, selected_datasets = _selected_scope(args)
    if TARGET_MODE != 'single':
        raise ValueError("Single recommendation mode requires CONFIG_RECOMMENDATION_TARGET_MODE='single'.")
    if int(PHASE2_RECOMMENDATION_BUDGET) <= 0:
        raise ValueError('PHASE2_RECOMMENDATION_BUDGET must be a positive integer.')
    if USE_PREFERENCE_REGIONS:
        _preference_region_budget()
    if _refinement_stage_plan('TPE')[0].seed_count <= 0:
        raise ValueError('Resolved recommendation seed count must be positive.')
    if sum(stage.trials for stage in _refinement_stage_plan('TPE')) < 0:
        raise ValueError('Resolved TPE trial budget must be non-negative.')
    paths = get_paths_from_script(__file__)
    paths.ensure_core_directories()
    results_root = ensure_dir(paths.results_phase1_dir / RESULTS_DIR_NAME)
    _print_single_settings()
    for detector in selected_detectors:
        for lodo_dataset in selected_datasets:
            dimensions = _load_search_space_dimensions(paths, detector, lodo_dataset)
            print(f'\nGenerating Sobol candidates for detector={detector} | dataset={lodo_dataset} ...')
            sobol_df = _sample_sobol_candidates(dimensions, count=SETTINGS.sobol_sample_count, random_seed=RANDOM_SEED, detector=detector)
            print(f'  Generated Sobol candidates: {len(sobol_df):,}')
            dataset_root = ensure_dir(results_root / detector / lodo_dataset)
            if USE_PREFERENCE_REGIONS:
                per_region_budget, combined_budget = _preference_region_budget()
                model_paths, region_bundle = _load_preference_region_models(paths, detector, lodo_dataset)
                required_feature_names = _preference_region_model_feature_names(region_bundle)
                first_model_path = model_paths[PREFERENCE_REGION_NAMES[0]]
                metadata_row = _metadata_row_for_model_features(
                    paths,
                    lodo_dataset=lodo_dataset,
                    dimensions=dimensions,
                    required_feature_names=required_feature_names,
                    model_path=first_model_path,
                )
                region_results: dict[str, pd.DataFrame] = {}
                region_candidate_pools: dict[str, pd.DataFrame] = {}
                print(f'\nDetector={detector} | Dataset={lodo_dataset} | Format=preference_regions | CombinedK={combined_budget}')
                for region_name in PREFERENCE_REGION_NAMES:
                    print(f'  Preference region: {region_name} | TopK={per_region_budget}')
                    region_results[region_name], region_candidate_pools[region_name] = _build_single_preference_region_output(
                        detector=detector,
                        sobol_df=sobol_df,
                        region_name=region_name,
                        regressor=region_bundle[region_name]["regressor"],
                        dimensions=dimensions,
                        metadata_row=metadata_row,
                        top_k=per_region_budget,
                    )
                    print(f'    Built {len(region_results[region_name]):,} configs for {region_name}')
                _save_single_preference_region_results(
                    base_dir=dataset_root,
                    detector=detector,
                    lodo_dataset=lodo_dataset,
                    region_results=region_results,
                    region_candidate_pools=region_candidate_pools,
                    dimensions=dimensions,
                )
                continue

            model_path = _phase1_model_path(
                paths,
                detector,
                lodo_dataset,
                target_mode="single",
                target_method=SINGLE_TARGET_METHOD,
            )
            model = _load_phase1_model_or_fail(model_path, label="Single-target recommendation model")
            required_feature_names = set(_model_feature_names(model))
            metadata_row = _metadata_row_for_model_features(paths, lodo_dataset=lodo_dataset, dimensions=dimensions, required_feature_names=required_feature_names, model_path=model_path)
            print(f'\nDetector={detector} | Dataset={lodo_dataset} | Format=single | Budget={PHASE2_RECOMMENDATION_BUDGET}')
            recommendations_df, all_candidates_df = _build_single_recommendation_output(detector=detector, sobol_df=sobol_df, model=model, dimensions=dimensions, metadata_row=metadata_row, budget=int(PHASE2_RECOMMENDATION_BUDGET))
            _save_single_recommendation_results(base_dir=dataset_root, detector=detector, lodo_dataset=lodo_dataset, recommendations_df=recommendations_df, all_candidates_df=all_candidates_df)
            print(f'  Saved {len(recommendations_df):,} single-target configs | candidate_pool={len(all_candidates_df):,}')
    print(f'\nTotal runtime: {_format_duration(time.perf_counter() - started_at)}')


# =============================================================================
# Separate-objective flow
# =============================================================================
@dataclass(frozen=True)
class SeparateModelBundle:
    accuracy_model: Any
    runtime_model: Any
    accuracy_model_path: Path
    runtime_model_path: Path

def _load_separate_models(paths, detector: str, lodo_dataset: str) -> SeparateModelBundle:
    acc_model_path = _phase1_model_path(
        paths,
        detector,
        lodo_dataset,
        target_mode="separate",
        target_method="separate",
        objective="accuracy",
    )
    runtime_model_path = _phase1_model_path(
        paths,
        detector,
        lodo_dataset,
        target_mode="separate",
        target_method="separate",
        objective="runtime",
    )
    return SeparateModelBundle(
        accuracy_model=_load_phase1_model_or_fail(
            acc_model_path,
            label="Separate accuracy recommendation model",
        ),
        runtime_model=_load_phase1_model_or_fail(
            runtime_model_path,
            label="Separate runtime recommendation model",
        ),
        accuracy_model_path=acc_model_path,
        runtime_model_path=runtime_model_path,
    )

def _predict_separate_objectives(candidates: pd.DataFrame, *, bundle: SeparateModelBundle, metadata_row: pd.Series | None) -> tuple[np.ndarray, np.ndarray]:
    pred_acc = _predict_model_scores(candidates, model=bundle.accuracy_model, metadata_row=metadata_row)
    pred_runtime_utility = _predict_model_scores(candidates, model=bundle.runtime_model, metadata_row=metadata_row)
    return (np.asarray(pred_acc, dtype=float), np.asarray(pred_runtime_utility, dtype=float))

def _separate_model_feature_names(bundle: SeparateModelBundle) -> set[str]:
    names = set(_model_feature_names(bundle.accuracy_model, label='accuracy regressor'))
    names.update(_model_feature_names(bundle.runtime_model, label='runtime regressor'))
    return names

def _add_separate_scores(df: pd.DataFrame, *, bundle: SeparateModelBundle, metadata_row: pd.Series | None) -> pd.DataFrame:
    out = df.copy()
    (acc, runtime_utility) = _predict_separate_objectives(out, bundle=bundle, metadata_row=metadata_row)
    out['predicted_transformed_accuracy'] = acc
    out['predicted_transformed_runtime'] = runtime_utility
    return out

def _same_cat_bool_value(left: Any, right: Any, *, kind: str) -> bool:
    if kind == "bool":
        return _normalize_bool_token(left) == _normalize_bool_token(right)
    if kind == "int":
        return int(round(float(left))) == int(round(float(right)))
    if kind == "float":
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
    return str(left) == str(right)

def _alternative_cat_bool_values(dim: SearchDimension, current_value: Any) -> list[Any]:
    if dim.kind in {"int", "float"} and dim.value_mode == "choice":
        choices = _numeric_choice_values(dim)
        return [choice for choice in choices if not _same_cat_bool_value(choice, current_value, kind=dim.kind)]
    if dim.kind == "bool":
        choices = dim.choices or ("TRUE", "FALSE")
        available = []
        for choice in choices:
            token = bool(_normalize_bool_token(choice))
            if token not in available:
                available.append(token)
        current = bool(_normalize_bool_token(current_value))
        flipped = not current
        return [flipped] if flipped in available else []
    if dim.kind == "cat":
        choices = list(dim.choices or ())
        return [choice for choice in choices if not _same_cat_bool_value(choice, current_value, kind="cat")]
    return []

def _append_categorical_bool_combination_variants(
    candidates_df: pd.DataFrame,
    *,
    detector: str,
    dimensions: list[SearchDimension],
    hyperparameter_names: list[str],
    bundle: SeparateModelBundle,
    metadata_row: pd.Series | None,
) -> tuple[pd.DataFrame, int]:
    if candidates_df.empty:
        return candidates_df.copy(), 0
    cat_bool_dimensions = [
        dim
        for dim in dimensions
        if (dim.kind in {"cat", "bool"} or (dim.kind in {"int", "float"} and dim.value_mode == "choice"))
        and dim.name in candidates_df.columns
    ]
    if not cat_bool_dimensions:
        return candidates_df.copy(), 0

    expanded_rows: list[dict[str, Any]] = []
    for (_, row) in candidates_df.iterrows():
        option_groups: list[tuple[SearchDimension, list[Any]]] = []
        for dim in cat_bool_dimensions:
            current_value = row[dim.name]
            alternatives = _alternative_cat_bool_values(dim, current_value)
            if alternatives:
                option_groups.append((dim, [current_value, *alternatives]))
        if not option_groups:
            continue

        dims = [dim for (dim, _) in option_groups]
        value_options = [options for (_, options) in option_groups]
        for values in product(*value_options):
            if all(_same_cat_bool_value(value, row[dim.name], kind=dim.kind) for (dim, value) in zip(dims, values)):
                continue
            expanded_row = row.copy()
            for (dim, value) in zip(dims, values):
                expanded_row[dim.name] = value
            expanded_df = pd.DataFrame([expanded_row.to_dict()])
            if _constraint_allowed_mask(detector, expanded_df).all():
                expanded_rows.append(expanded_row.to_dict())
            else:
                _record_invalid_configs(detector, "choice_expansion", 1)

    if not expanded_rows:
        return candidates_df.copy(), 0

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df["stage_label"] = "categorical_bool_combination_expansion"
    expanded_df["sampler"] = "CAT_BOOL_EXPANSION"
    expanded_df["stage_trials"] = 0
    expanded_df["optuna_trial_number"] = -1
    expanded_df["predicted_pareto_layer"] = np.nan
    expanded_df["is_predicted_pareto"] = False
    expanded_df = _add_separate_scores(expanded_df, bundle=bundle, metadata_row=metadata_row)

    combined = pd.concat([candidates_df, expanded_df], axis=0, ignore_index=True)
    combined = _drop_exact_duplicate_configs(combined, hyperparameter_names)
    return combined.reset_index(drop=True), len(expanded_df)

def _pareto_layer_rank_max_max(points: np.ndarray) -> np.ndarray:
    return pareto_layer_rank(np.asarray(points, dtype=float))

def _rank_separate_sobol_candidates(sobol_df: pd.DataFrame, *, dimensions: list[SearchDimension], bundle: SeparateModelBundle, metadata_row: pd.Series | None) -> pd.DataFrame:
    hyperparameter_names = [dim.name for dim in dimensions]
    ranked_df = _add_separate_scores(sobol_df, bundle=bundle, metadata_row=metadata_row)
    if DROP_EXACT_DUPLICATE_CONFIGS_AFTER_SOBOL:
        ranked_df = _drop_exact_duplicate_configs(ranked_df, hyperparameter_names)
    return ranked_df.reset_index(drop=True)

def _select_exact_separate_seed_rows(
    ranked_sobol_df: pd.DataFrame,
    *,
    dimensions: list[SearchDimension],
    seed_count: int,
    stage_label: str,
) -> pd.DataFrame:
    """Select an exact prediction-only Sobol/archive seed set."""
    hyperparameter_names = [dim.name for dim in dimensions]
    selected_idx = select_pareto_budget(
        ranked_sobol_df,
        ("predicted_transformed_accuracy", "predicted_transformed_runtime"),
        int(seed_count),
        maximize=(True, True),
        config_columns=hyperparameter_names,
    )
    full_seed_layers = pd.Series(
        _pareto_layer_rank_max_max(
            ranked_sobol_df[
                ["predicted_transformed_accuracy", "predicted_transformed_runtime"]
            ].to_numpy(dtype=float)
        ).astype(float),
        index=ranked_sobol_df.index,
    )
    if selected_idx is None:
        unique_count = int(ranked_sobol_df.drop_duplicates(subset=hyperparameter_names).shape[0])
        raise ValueError(
            f"Recommendation refinement requires exactly {int(seed_count)} unique seed configurations, "
            f"but only {unique_count} unique valid candidates are available."
        )
    seeds = ranked_sobol_df.loc[selected_idx].copy().reset_index(drop=True)
    if len(seeds) != int(seed_count) or seeds.drop_duplicates(subset=hyperparameter_names).shape[0] != int(seed_count):
        raise ValueError(f"Seed selection did not produce exactly {int(seed_count)} unique configurations.")
    seed_layers = full_seed_layers.loc[selected_idx].to_numpy(dtype=float)
    seeds = seeds.drop(
        columns=[
            'source_seed_rank',
            'source_seed_predicted_transformed_accuracy',
            'source_seed_predicted_transformed_runtime',
            'source_seed_predicted_pareto_layer',
        ],
        errors='ignore',
    )
    seeds.insert(0, 'source_seed_rank', np.arange(1, len(seeds) + 1, dtype=int))
    seeds.insert(1, 'source_seed_predicted_transformed_accuracy', seeds['predicted_transformed_accuracy'].astype(float))
    seeds.insert(2, 'source_seed_predicted_transformed_runtime', seeds['predicted_transformed_runtime'].astype(float))
    seeds.insert(3, 'source_seed_predicted_pareto_layer', seed_layers.astype(float))
    seeds['predicted_pareto_layer'] = seed_layers.astype(float)
    seeds['is_predicted_pareto'] = np.isclose(seed_layers.astype(float), 1.0)
    seeds['stage_label'] = stage_label
    seeds['stage_trials'] = 0
    seeds['optuna_trial_number'] = -1
    return seeds


def _print_effective_recommendation_settings(*, target_mode: str) -> None:
    """Print shared recommendation controls separately from refinement mode."""
    print("  Shared settings")
    print(f"    Recommendation mode: {SETTINGS.mode}")
    print(f"    Sobol count: {SETTINGS.sobol_sample_count:,}")
    print(f"    Semi-local: {SETTINGS.use_semi_local_optuna_space}")
    print(f"    Restrict categoricals: {RESTRICT_CATEGORICALS_TO_SELECTED_SEEDS}")
    print(f"    Restrict booleans: {RESTRICT_BOOLEANS_TO_SELECTED_SEEDS}")
    print(f"    TPE multivariate: {TPE_MULTIVARIATE}")
    print(f"    TPE group: {TPE_GROUP}")
    print(f"    TPE constant liar: {TPE_CONSTANT_LIAR}")
    print(f"  fresh study per margin stage: {'enabled' if SETTINGS.use_fresh_study_per_margin_stage else 'disabled'}")
    print(
        "  post-Optuna categorical/Boolean expansion setting: "
        f"{'enabled' if SETTINGS.expand_categorical_bool_after_optuna else 'disabled'} "
        "(candidate augmentation before final deduplication)"
    )
    print(f"  Target mode: {target_mode}")
    if SETTINGS.mode == "static":
        print("  Static refinement")
        print(f"    TPE seeds: {SETTINGS.static_tpe_seed_count:,}")
        print(f"    TPE margin: {SETTINGS.static_tpe_numeric_margin:.2f}")
        print(f"    TPE trials: {SETTINGS.static_tpe_trials:,}")
    else:
        print("  Dynamic TPE")
        print(f"    margins: {list(SETTINGS.dynamic_tpe_margins)}")
        print(f"    seeds: {list(SETTINGS.dynamic_tpe_seed_counts)}")
        print(f"    trial shares: {list(SETTINGS.dynamic_tpe_trial_shares)}")
        print(f"    total trials: {SETTINGS.dynamic_tpe_total_trials:,}")
    if target_mode == "separate":
        print("  Shared NSGA-II")
        print(f"    seeds: {SETTINGS.nsgaii_seed_count:,}")
        print(f"    margin: {SETTINGS.nsgaii_numeric_margin:.2f}")
        print(f"    trials: {SETTINGS.nsgaii_trials:,}")
        print(f"    population: {SETTINGS.nsgaii_population_size:,}")

def _separate_objective_factory(*, detector: str, dimensions: list[SearchDimension], bundle: SeparateModelBundle, metadata_row: pd.Series | None) -> Callable[[optuna.trial.Trial], tuple[float, float]]:

    def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
        params = _suggest_params(trial, dimensions, detector=detector)
        candidate_df = pd.DataFrame([params])
        candidate_df = _validate_generated_candidates(detector, candidate_df, stage="optuna")
        if candidate_df.empty:
            raise optuna.TrialPruned("Detector constraints rejected Optuna suggestion.")
        (acc, runtime_utility) = _predict_separate_objectives(candidate_df, bundle=bundle, metadata_row=metadata_row)
        return (float(acc[0]), float(runtime_utility[0]))
    return objective

def _make_separate_sampler(sampler_name: str, *, random_seed: int = RANDOM_SEED):
    name = sampler_name.upper().strip()
    if name == 'TPE':
        return optuna.samplers.TPESampler(seed=random_seed, multivariate=TPE_MULTIVARIATE, group=TPE_GROUP, constant_liar=TPE_CONSTANT_LIAR, n_startup_trials=OPTUNA_STARTUP_TRIALS, warn_independent_sampling=False)
    if name == 'NSGAII':
        return optuna.samplers.NSGAIISampler(seed=random_seed, population_size=SETTINGS.nsgaii_population_size)
    raise ValueError(f"Unsupported sampler '{sampler_name}'. Use 'TPE' or 'NSGAII'.")

def _separate_sampler_names_for_run() -> list[str]:
    """Separate recommendation always runs independent TPE and NSGA-II branches."""
    return ['TPE', 'NSGAII']


def _separate_optuna_trials_for_sampler(sampler_name: str) -> int:
    name = sampler_name.upper().strip()
    if name == 'TPE':
        return sum(stage.trials for stage in _refinement_stage_plan('TPE'))
    if name == 'NSGAII':
        return sum(stage.trials for stage in _refinement_stage_plan('NSGAII'))
    raise ValueError(f"Unsupported sampler '{sampler_name}'.")

def _run_separate_optuna(*, detector: str, seed_rows: pd.DataFrame, dimensions: list[SearchDimension], bundle: SeparateModelBundle, metadata_row: pd.Series | None, sampler_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one Separate refinement branch using predicted transformed objectives.

    TPE and NSGA-II branches are executed independently from their Sobol seed
    sets. The objective returns predicted transformed accuracy/runtime only;
    no drift-detection execution occurs in this script.
    """
    if seed_rows.empty:
        return seed_rows.copy(), seed_rows.copy()
    hyperparameter_names = [dim.name for dim in dimensions]
    archive = seed_rows.copy().reset_index(drop=True)
    stage_candidate_frames: list[pd.DataFrame] = []
    stage_audit_frames: list[pd.DataFrame] = []
    study: optuna.study.Study | None = None
    active_sampler_random_seed: int | None = None
    executed_trial_total = 0

    for stage in _refinement_stage_plan(sampler_name):
        if stage.trials < 0:
            raise ValueError(f"{sampler_name.upper()} refinement trial budgets cannot be negative.")
        if stage.trials == 0:
            continue
        if SETTINGS.mode == "dynamic" and sampler_name.upper().strip() == "TPE" and stage.index > 1:
            stage_seed_candidates = _select_exact_separate_seed_rows(
                archive,
                dimensions=dimensions,
                seed_count=stage.seed_count,
                stage_label=f'dynamic_tpe_stage_{stage.index}_pareto_budget_seed',
            )
        else:
            stage_seed_candidates = seed_rows.copy().reset_index(drop=True)
        if len(stage_seed_candidates) != int(stage.seed_count):
            raise RuntimeError(
                f"{sampler_name.upper()} stage {stage.index} selected {len(stage_seed_candidates)} seeds; "
                f"expected {stage.seed_count}."
            )
        # Rebuild the local search space around the selected stage seeds.
        local_dimensions = _build_semi_local_dimensions(
            stage_seed_candidates,
            dimensions,
            margin_ratio=stage.margin_ratio,
        )
        protected_seed_rows = _rows_within_search_dimensions(
            stage_seed_candidates,
            local_dimensions,
        )
        if protected_seed_rows.empty:
            raise ValueError(
                f"No protected seeds remain inside margin stage {stage.index} "
                f"for detector '{detector}'."
            )
        if len(protected_seed_rows) != int(stage.seed_count):
            raise RuntimeError(
                f"{sampler_name.upper()} stage {stage.index} enqueued {len(protected_seed_rows)} warm-start seeds; "
                f"expected {stage.seed_count}. Check local-space construction and seed restrictions."
            )
        requested_stage_random_seed = RANDOM_SEED + stage.index - 1
        if study is None or SETTINGS.use_fresh_study_per_margin_stage:
            sampler = _make_separate_sampler(
                sampler_name,
                random_seed=requested_stage_random_seed,
            )
            study = optuna.create_study(directions=['maximize', 'maximize'], sampler=sampler)
            active_sampler_random_seed = requested_stage_random_seed
        stage_trial_start = len(study.trials)
        for _, seed_row in protected_seed_rows.iterrows():
            study.enqueue_trial(
                _seed_params_for_optuna(seed_row, local_dimensions, detector=detector)
            )
        objective = _separate_objective_factory(
            detector=detector,
            dimensions=local_dimensions,
            bundle=bundle,
            metadata_row=metadata_row,
        )
        study.optimize(
            objective,
            n_trials=len(protected_seed_rows) + int(stage.trials),
            n_jobs=OPTUNA_STUDY_N_JOBS,
            show_progress_bar=False,
        )

        stage_study_trials = study.trials[stage_trial_start:]
        executed_trial_total += len(stage_study_trials)
        warm_start_count = min(len(protected_seed_rows), len(stage_study_trials))
        if warm_start_count != int(stage.seed_count):
            raise RuntimeError(
                f"{sampler_name.upper()} stage {stage.index} consumed {warm_start_count} warm-start trials; "
                f"expected {stage.seed_count}."
            )
        suggestion_trials = stage_study_trials[warm_start_count:]
        completed_suggestion_count = sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.values is not None
            for trial in suggestion_trials
        )
        pruned_suggestion_count = sum(
            trial.state == optuna.trial.TrialState.PRUNED
            for trial in suggestion_trials
        )
        failed_suggestion_count = sum(
            trial.state == optuna.trial.TrialState.FAIL
            for trial in suggestion_trials
        )
        search_space_fields = _search_space_audit_fields(
            original_dimensions=dimensions,
            local_dimensions=local_dimensions,
        )

        rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for stage_trial_position, trial in enumerate(stage_study_trials):
            is_warm_start = stage_trial_position < warm_start_count
            is_complete = (
                trial.state == optuna.trial.TrialState.COMPLETE
                and trial.values is not None
            )
            partial_params_row = _partial_row_from_trial_params(
                params=trial.params,
                dimensions=local_dimensions,
                detector=detector,
            )
            has_complete_params = all(
                not pd.isna(partial_params_row[dim.name])
                for dim in dimensions
            )
            constraint_allowed: bool | Any = pd.NA
            if has_complete_params:
                constraint_allowed = bool(
                    _constraint_allowed_mask(
                        detector,
                        pd.DataFrame([partial_params_row]),
                    ).iloc[0]
                )
            prune_reason = pd.NA
            if trial.state == optuna.trial.TrialState.PRUNED:
                if has_complete_params and constraint_allowed is False:
                    prune_reason = 'detector_constraint'
                elif not has_complete_params:
                    prune_reason = 'infeasible_or_partial_parameter_space'
                else:
                    prune_reason = 'objective_pruned_other'
            audit_row: dict[str, Any] = {
                'trial_role': 'warm_start' if is_warm_start else 'suggestion',
                'trial_state': trial.state.name,
                'is_complete_valid_candidate': bool(is_complete),
                'has_complete_hyperparameters': bool(has_complete_params),
                'constraint_allowed': constraint_allowed,
                'prune_reason': prune_reason,
                'sampler': sampler_name.upper(),
                'sampler_random_seed': int(active_sampler_random_seed),
                'margin_stage': int(stage.index),
                'margin_ratio': float(stage.margin_ratio),
                'study_trial_number': int(trial.number),
                'trial_started_at': (
                    trial.datetime_start.isoformat()
                    if trial.datetime_start is not None
                    else pd.NA
                ),
                'trial_completed_at': (
                    trial.datetime_complete.isoformat()
                    if trial.datetime_complete is not None
                    else pd.NA
                ),
                'trial_duration_seconds': (
                    (trial.datetime_complete - trial.datetime_start).total_seconds()
                    if trial.datetime_start is not None
                    and trial.datetime_complete is not None
                    else np.nan
                ),
                'stage_suggestion_index': (
                    -1 if is_warm_start else int(stage_trial_position - warm_start_count)
                ),
                'stage_allocated_trials': int(stage.trials),
                'stage_allocated_total_trials': int(warm_start_count + stage.trials),
                'stage_warm_start_count': int(warm_start_count),
                'stage_completed_trials': int(completed_suggestion_count),
                'stage_pruned_trials': int(pruned_suggestion_count),
                'stage_failed_trials': int(failed_suggestion_count),
                'protected_seed_min_target': int(stage.seed_count),
                'protected_seed_max_target': int(stage.seed_count),
                'protected_seed_selected_count': int(len(stage_seed_candidates)),
                'protected_seed_actual_count': int(warm_start_count),
                'predicted_transformed_accuracy': (
                    float(trial.values[0]) if is_complete else np.nan
                ),
                'predicted_transformed_runtime': (
                    float(trial.values[1]) if is_complete else np.nan
                ),
            }
            audit_row.update(partial_params_row)
            audit_row.update(search_space_fields)
            if is_warm_start:
                protected_row = protected_seed_rows.iloc[stage_trial_position]
                audit_row['warm_source_sampler'] = protected_row.get('sampler', 'SOBOL')
                audit_row['warm_source_margin_stage'] = protected_row.get('margin_stage', np.nan)
                audit_row['warm_source_margin_ratio'] = protected_row.get('margin_ratio', np.nan)
            else:
                audit_row['warm_source_sampler'] = pd.NA
                audit_row['warm_source_margin_stage'] = np.nan
                audit_row['warm_source_margin_ratio'] = np.nan
            audit_rows.append(audit_row)

            if is_warm_start or not is_complete:
                continue
            row: dict[str, Any] = {
                'source_seed_rank': -1,
                'source_seed_predicted_transformed_accuracy': np.nan,
                'source_seed_predicted_transformed_runtime': np.nan,
                'source_seed_predicted_pareto_layer': np.nan,
                'stage_label': f'optuna_multi_objective_margin_{stage.margin_ratio:.2f}',
                'sampler': sampler_name.upper(),
                'stage_trials': int(stage.trials),
                'margin_stage': int(stage.index),
                'margin_ratio': float(stage.margin_ratio),
                'optuna_trial_number': int(trial.number),
                'predicted_transformed_accuracy': float(trial.values[0]),
                'predicted_transformed_runtime': float(trial.values[1]),
            }
            row.update(partial_params_row)
            rows.append(row)

        stage_df = pd.DataFrame(rows)
        stage_audit_frames.append(pd.DataFrame(audit_rows))
        if not stage_df.empty:
            stage_candidate_frames.append(stage_df)
            archive = pd.concat([archive, stage_df], axis=0, ignore_index=True)
            archive = _drop_exact_duplicate_configs(archive, hyperparameter_names)
        print(
            f'    {sampler_name.upper()} margin stage {stage.index}: '
            f'ratio={stage.margin_ratio:.2f} | trials={stage.trials:,} '
            f'| completed={completed_suggestion_count:,} '
            f'| pruned={pruned_suggestion_count:,} | failed={failed_suggestion_count:,} '
            f'| protected_seeds={len(stage_seed_candidates):,} '
            f'| warm_starts_in_bounds={len(protected_seed_rows):,} '
            f'(target={stage.seed_count:,})'
        )

    expected_total = sum(stage.seed_count + stage.trials for stage in _refinement_stage_plan(sampler_name))
    if executed_trial_total != expected_total:
        raise RuntimeError(
            f"{sampler_name.upper()} refinement executed {executed_trial_total} trials; "
            f"expected {expected_total} warm-start plus refinement trials across stage(s)."
        )

    out = (
        pd.concat(stage_candidate_frames, axis=0, ignore_index=True)
        if stage_candidate_frames
        else seed_rows.head(0).copy()
    )
    if out.empty:
        out = seed_rows.head(0).copy()
        out['sampler'] = sampler_name.upper()
    if DROP_EXACT_DUPLICATE_CONFIGS_AFTER_OPTUNA:
        out = _drop_exact_duplicate_configs(out, hyperparameter_names)
    audit_df = (
        pd.concat(stage_audit_frames, axis=0, ignore_index=True)
        if stage_audit_frames
        else pd.DataFrame()
    )
    return out.reset_index(drop=True), audit_df.reset_index(drop=True)

def _select_separate_final_outputs(
    candidates_df: pd.DataFrame,
    *,
    dimensions: list[SearchDimension],
    detector: str,
    lodo_dataset: str,
    budget: int = PHASE2_RECOMMENDATION_BUDGET,
) -> pd.DataFrame:
    """Select the exact final predicted Pareto budget from the merged candidate pool."""
    if candidates_df.empty:
        unique_count = 0
        raise ValueError(
            f"Cannot select {int(budget)} recommendations for detector={detector} | "
            f"dataset={lodo_dataset}: only {unique_count} unique candidate configurations are available."
        )
    candidate_pool = _annotate_separate_predicted_pareto_layers(candidates_df)
    objective_columns = ("predicted_transformed_accuracy", "predicted_transformed_runtime")

    hyperparameter_names = [dim.name for dim in dimensions]
    selected_idx = select_pareto_budget(
        candidate_pool,
        objective_columns,
        int(budget),
        maximize=(True, True),
        config_columns=hyperparameter_names,
    )
    if selected_idx is None:
        unique_count = int(candidate_pool.drop_duplicates(subset=hyperparameter_names).shape[0])
        raise ValueError(
            f"Cannot select {int(budget)} recommendations for detector={detector} | "
            f"dataset={lodo_dataset}: only {unique_count} unique candidate configurations are available."
        )
    final_df = candidate_pool.loc[selected_idx].copy().reset_index(drop=True)
    unique_count = int(final_df.drop_duplicates(subset=hyperparameter_names).shape[0])
    if len(final_df) != int(budget) or unique_count != int(budget):
        raise ValueError(
            f"Final recommendation selection for detector={detector} | dataset={lodo_dataset} "
            f"produced {len(final_df)} rows/{unique_count} unique configurations; expected {int(budget)}."
        )
    final_df.insert(0, 'recommendation_rank', np.arange(1, len(final_df) + 1, dtype=int))
    return final_df


def _annotate_separate_predicted_pareto_layers(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Compute predicted Pareto-layer diagnostics for every Separate candidate row."""
    objective_columns = ("predicted_transformed_accuracy", "predicted_transformed_runtime")
    missing = [column for column in objective_columns if column not in candidates_df.columns]
    if missing:
        raise ValueError(f"Separate candidate pool is missing prediction column(s): {missing}")

    candidate_pool = candidates_df.copy().reset_index(drop=True)
    candidate_pool["predicted_pareto_layer"] = _pareto_layer_rank_max_max(
        candidate_pool[list(objective_columns)].to_numpy(dtype=float)
    ).astype(float)
    candidate_pool["is_predicted_pareto"] = np.isclose(
        candidate_pool["predicted_pareto_layer"].to_numpy(dtype=float),
        1.0,
    )
    return candidate_pool


def _config_key_series(
    df: pd.DataFrame,
    dimensions: list[SearchDimension],
) -> pd.Series:
    if df.empty:
        return pd.Series(index=df.index, dtype='object')
    hyperparameter_names = [dim.name for dim in dimensions]
    values = df.reindex(columns=hyperparameter_names).copy()
    for dim in dimensions:
        def canonical(value: Any) -> str:
            if pd.isna(value):
                return '<NA>'
            if dim.kind == 'int':
                return f'i:{int(round(float(value)))}'
            if dim.kind == 'float':
                return f'f:{float(value):.17g}'
            if dim.kind == 'bool':
                return f'b:{int(_normalize_bool_token(value))}'
            return f's:{str(value)}'

        values[dim.name] = values[dim.name].map(canonical)
    return values.agg('\x1f'.join, axis=1)


def _build_separate_all_candidates_audit(
    *,
    seed_rows: pd.DataFrame,
    optuna_audit_df: pd.DataFrame,
    combined_candidates: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    dimensions: list[SearchDimension],
) -> pd.DataFrame:
    """Build a complete trial/candidate audit without affecting final ranking."""
    hyperparameter_names = [dim.name for dim in dimensions]
    audit_parts: list[pd.DataFrame] = []

    if not seed_rows.empty:
        sobol_audit = seed_rows.copy()
        sobol_audit['trial_role'] = 'sobol_protected_seed'
        sobol_audit['trial_state'] = 'COMPLETE'
        sobol_audit['is_complete_valid_candidate'] = True
        sobol_audit['has_complete_hyperparameters'] = True
        sobol_audit['constraint_allowed'] = True
        sobol_audit['prune_reason'] = pd.NA
        sobol_audit['sampler'] = 'SOBOL'
        sobol_audit['sampler_random_seed'] = RANDOM_SEED
        sobol_audit['study_trial_number'] = -1
        sobol_audit['stage_suggestion_index'] = -1
        audit_parts.append(sobol_audit)

    if not optuna_audit_df.empty:
        audit_parts.append(optuna_audit_df.copy())

    if not audit_parts:
        return pd.DataFrame()

    audit = pd.concat(audit_parts, axis=0, ignore_index=True, sort=False)
    for column in hyperparameter_names:
        if column not in audit.columns:
            audit[column] = np.nan
    audit['config_key'] = _config_key_series(audit, dimensions)
    has_complete_config = (
        audit['is_complete_valid_candidate'].fillna(False).astype(bool)
        & audit[hyperparameter_names].notna().all(axis=1)
    )
    audit['duplicate_occurrence_count'] = 0
    audit.loc[has_complete_config, 'duplicate_occurrence_count'] = (
        audit.loc[has_complete_config]
        .groupby('config_key')['config_key']
        .transform('size')
        .astype(int)
    )
    audit['is_exact_duplicate'] = False
    audit.loc[has_complete_config, 'is_exact_duplicate'] = audit.loc[
        has_complete_config
    ].duplicated(subset=['config_key'], keep='first')

    combined_keys = set(_config_key_series(combined_candidates, dimensions))
    final_keys = set(_config_key_series(recommendations_df, dimensions))
    audit['config_present_in_combined_pool'] = (
        has_complete_config & audit['config_key'].isin(combined_keys)
    )
    audit['is_final_recommendation'] = (
        has_complete_config & audit['config_key'].isin(final_keys)
    )

    if not combined_candidates.empty:
        combined_info = combined_candidates.copy()
        combined_info['config_key'] = _config_key_series(combined_info, dimensions)
        if 'predicted_pareto_layer' in combined_info.columns:
            layer_map = (
                combined_info.drop_duplicates('config_key')
                .set_index('config_key')['predicted_pareto_layer']
            )
            audit['predicted_pareto_layer'] = audit['config_key'].map(layer_map).where(
                audit['config_present_in_combined_pool'],
                audit.get('predicted_pareto_layer', np.nan),
            )
        if 'is_predicted_pareto' in combined_info.columns:
            pareto_map = (
                combined_info.drop_duplicates('config_key')
                .set_index('config_key')['is_predicted_pareto']
            )
            audit['is_predicted_pareto'] = audit['config_key'].map(pareto_map).where(
                audit['config_present_in_combined_pool'],
                audit.get('is_predicted_pareto', pd.NA),
            )

    if not recommendations_df.empty:
        final_info = recommendations_df.copy()
        final_info['config_key'] = _config_key_series(final_info, dimensions)
        rank_map = final_info.drop_duplicates('config_key').set_index('config_key')[
            'recommendation_rank'
        ]
        layer_map = final_info.drop_duplicates('config_key').set_index('config_key')[
            'predicted_pareto_layer'
        ]
        audit['final_recommendation_rank'] = audit['config_key'].map(rank_map)
        audit['final_predicted_pareto_layer'] = audit['config_key'].map(layer_map)
    else:
        audit['final_recommendation_rank'] = np.nan
        audit['final_predicted_pareto_layer'] = np.nan

    for column in ("predicted_pareto_layer", "predicted_transformed_accuracy", "predicted_transformed_runtime"):
        if column not in audit.columns:
            audit[column] = np.nan
    audit = audit.sort_values(
        [
            "predicted_pareto_layer",
            "predicted_transformed_accuracy",
            "predicted_transformed_runtime",
            "config_key",
        ],
        ascending=[True, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    audit.insert(0, 'candidate_audit_id', np.arange(1, len(audit) + 1, dtype=int))
    return audit.reset_index(drop=True)

def _save_separate_results(
    *,
    base_dir: Path,
    detector: str,
    lodo_dataset: str,
    recommendations_df: pd.DataFrame,
    all_candidates_df: pd.DataFrame,
) -> None:
    output_dir = ensure_dir(base_dir / 'separate')
    final_path = output_dir / f'{detector}_{lodo_dataset}_separate_recommended_configs.csv'
    _save_with_context(recommendations_df, final_path, detector=detector, lodo_dataset=lodo_dataset)
    all_path = output_dir / f'{detector}_{lodo_dataset}_separate_all_candidates.csv'
    _save_with_context(all_candidates_df, all_path, detector=detector, lodo_dataset=lodo_dataset)

def _print_separate_settings() -> None:
    run_samplers = _separate_sampler_names_for_run()
    print('\nSeparate config recommendation local settings')
    _print_effective_recommendation_settings(target_mode="separate")
    print(f"  Algorithms: {', '.join(run_samplers)}")
    print(f'  Recommendation target mode: {TARGET_MODE}')
    print('  Sobol prefiltering for layering: disabled')
    if SETTINGS.mode == 'static':
        print('  Static seed selection: exact Pareto-budget selector')
    else:
        print('  Dynamic seed reselection: exact Pareto-budget selector at each TPE stage')
    for run_sampler in run_samplers:
        trials = _separate_optuna_trials_for_sampler(run_sampler)
        print(f'  {run_sampler} Optuna trials: {trials:,}')
        print(
            f'  {run_sampler} margin stages: '
            f'{_format_margin_schedule(sampler_name=run_sampler)}'
        )
    print(f'  Optuna startup trials: {OPTUNA_STARTUP_TRIALS:,}')
    print('  Objective directions: transformed_accuracy=maximize, transformed_runtime=maximize')
    print(f'  Semi-local Optuna space: {SETTINGS.use_semi_local_optuna_space}')
    print(f'  Final recommendation budget: {PHASE2_RECOMMENDATION_BUDGET:,}')
    print('  Final candidate pool: selected Sobol seeds + TPE candidates + NSGA-II candidates')
    print(f'  Export uncertainty features: {COMPUTE_PREDICTION_UNCERTAINTY}')

def _run_separate_recommendation() -> None:
    started_at = time.perf_counter()
    args = _parse_args('separate')
    _apply_pipeline_setup(args)
    run_samplers = _separate_sampler_names_for_run()
    selected_detectors, selected_datasets = _selected_scope(args)
    if TARGET_MODE != 'separate':
        print("\nWARNING: TRAIN_TARGET_MODE in config.py is not 'separate'. This script expects separate-mode artifacts named *_acc.joblib and *_runtime.joblib.")
    paths = get_paths_from_script(__file__)
    paths.ensure_core_directories()
    results_root = ensure_dir(paths.results_phase1_dir / RESULTS_DIR_NAME)
    _print_separate_settings()
    for detector in selected_detectors:
        for lodo_dataset in selected_datasets:
            dimensions = _load_search_space_dimensions(paths, detector, lodo_dataset)
            hyperparameter_names = [dim.name for dim in dimensions]
            print(f'\nGenerating Sobol candidates for detector={detector} | dataset={lodo_dataset} ...')
            sobol_df = _sample_sobol_candidates(dimensions, count=SETTINGS.sobol_sample_count, random_seed=RANDOM_SEED, detector=detector)
            print(f'  Generated Sobol candidates: {len(sobol_df):,}')
            dataset_started_at = time.perf_counter()
            dataset_root = ensure_dir(results_root / detector / lodo_dataset)
            bundle = _load_separate_models(paths, detector, lodo_dataset)
            required_feature_names = _separate_model_feature_names(bundle)
            metadata_row = _metadata_row_for_model_features(paths, lodo_dataset=lodo_dataset, dimensions=dimensions, required_feature_names=required_feature_names, model_path_for_error=bundle.accuracy_model_path)
            print(f'\nDetector={detector} | Dataset={lodo_dataset} | Format=separate | Algorithms=TPE+NSGAII')
            print(f'  Accuracy model: {bundle.accuracy_model_path}')
            print(f'  Runtime model:  {bundle.runtime_model_path}')
            ranked_sobol_df = _rank_separate_sobol_candidates(sobol_df, dimensions=dimensions, bundle=bundle, metadata_row=metadata_row)
            if DROP_EXACT_DUPLICATE_CONFIGS_AFTER_SOBOL:
                ranked_sobol_df = _drop_exact_duplicate_configs(ranked_sobol_df, hyperparameter_names)
            seed_rows_by_sampler: dict[str, pd.DataFrame] = {}
            if SETTINGS.mode == "static":
                static_seed_rows = _select_exact_separate_seed_rows(
                    ranked_sobol_df,
                    dimensions=dimensions,
                    seed_count=_refinement_stage_plan('TPE')[0].seed_count,
                    stage_label='static_separate_shared_pareto_budget_seed',
                )
                seed_rows_by_sampler = {
                    run_sampler: static_seed_rows.copy()
                    for run_sampler in run_samplers
                }
            else:
                for run_sampler in run_samplers:
                    first_stage = _refinement_stage_plan(run_sampler)[0]
                    seed_rows_by_sampler[run_sampler] = _select_exact_separate_seed_rows(
                        ranked_sobol_df,
                        dimensions=dimensions,
                        seed_count=first_stage.seed_count,
                        stage_label=f'dynamic_{run_sampler.lower()}_stage_1_pareto_budget_seed',
                    )
            seed_pool_for_final = _deduplicate_final_candidate_pool(
                pd.concat(seed_rows_by_sampler.values(), axis=0, ignore_index=True),
                dimensions=dimensions,
            )
            seed_summary_parts = []
            for name, rows in seed_rows_by_sampler.items():
                seed_max_layer = int(pd.to_numeric(rows['predicted_pareto_layer'], errors='coerce').max()) if len(rows) else 'NA'
                seed_layer_1 = int((pd.to_numeric(rows['predicted_pareto_layer'], errors='coerce') == 1).sum()) if len(rows) else 0
                seed_summary_parts.append(
                    f'{name}: seeds={len(rows):,}, layer1={seed_layer_1:,}, max_layer={seed_max_layer}'
                )
            trial_summary = ', '.join(
                f'{name}={_separate_optuna_trials_for_sampler(name):,}'
                for name in run_samplers
            )
            print(f'  Sobol candidates after dedup={len(ranked_sobol_df):,} | {"; ".join(seed_summary_parts)} | optuna_trials={trial_summary}')
            optuna_runs = [
                _run_separate_optuna(detector=detector, seed_rows=seed_rows_by_sampler[run_sampler], dimensions=dimensions, bundle=bundle, metadata_row=metadata_row, sampler_name=run_sampler)
                for run_sampler in run_samplers
            ]
            optuna_candidate_frames = [run_candidates for run_candidates, _ in optuna_runs]
            optuna_audit_frames = [run_audit for _, run_audit in optuna_runs]
            optuna_candidates_df = pd.concat(optuna_candidate_frames, axis=0, ignore_index=True)
            optuna_audit_df = pd.concat(optuna_audit_frames, axis=0, ignore_index=True)
            if SETTINGS.expand_categorical_bool_after_optuna:
                optuna_candidates_df, expanded_count = _append_categorical_bool_combination_variants(
                    optuna_candidates_df,
                    detector=detector,
                    dimensions=dimensions,
                    hyperparameter_names=hyperparameter_names,
                    bundle=bundle,
                    metadata_row=metadata_row,
                )
                print(f"  Categorical/Boolean expansion added {expanded_count:,} candidate variant(s)")
            if not optuna_candidates_df.empty:
                optuna_candidates_df['candidate_source'] = optuna_candidates_df['sampler'].astype(str).str.lower()
            sobol_seed_pool = seed_pool_for_final.copy()
            sobol_seed_pool['sampler'] = 'SOBOL_SEED'
            sobol_seed_pool['candidate_source'] = 'sobol_seed'
            # Merge Sobol seeds and both independent refinement branches before
            # final predicted Pareto-layer selection.
            combined_candidates = pd.concat(
                [sobol_seed_pool, optuna_candidates_df],
                axis=0,
                ignore_index=True,
            )
            combined_candidates = _deduplicate_final_candidate_pool(
                combined_candidates,
                dimensions=dimensions,
            )
            combined_candidates = _annotate_separate_predicted_pareto_layers(combined_candidates)
            recommendations_df = _select_separate_final_outputs(
                combined_candidates,
                dimensions=dimensions,
                detector=detector,
                lodo_dataset=lodo_dataset,
                budget=PHASE2_RECOMMENDATION_BUDGET,
            )
            recommendations_df = _add_separate_uncertainty_features(recommendations_df, bundle=bundle, metadata_row=metadata_row)
            all_candidates_audit = _build_separate_all_candidates_audit(
                seed_rows=seed_pool_for_final,
                optuna_audit_df=optuna_audit_df,
                combined_candidates=combined_candidates,
                recommendations_df=recommendations_df,
                dimensions=dimensions,
            )
            _save_separate_results(base_dir=dataset_root, detector=detector, lodo_dataset=lodo_dataset, recommendations_df=recommendations_df, all_candidates_df=all_candidates_audit)
            final_layer_1 = int((recommendations_df['predicted_pareto_layer'] == 1).sum()) if len(recommendations_df) else 0
            print(f'  Saved {len(recommendations_df):,} separate recommendations | final_budget={PHASE2_RECOMMENDATION_BUDGET:,} | final_layer_1={final_layer_1:,} | combined_candidates={len(combined_candidates):,} | audit_rows={len(all_candidates_audit):,} | runtime={_format_duration(time.perf_counter() - dataset_started_at)}')
    print(f'\nTotal runtime: {_format_duration(time.perf_counter() - started_at)}')


# =============================================================================
# Unified entry point
# =============================================================================

def _selected_recommendation_mode() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    add_pipeline_setup_args(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    setup = resolve_pipeline_setup(
        args,
        default_target_mode=CONFIG_RECOMMENDATION_TARGET_MODE,
        default_single_target_formulation=SINGLE_TARGET_FORMULATION,
        default_use_metadata=TRAIN_USE_METADATA,
        default_metadata_variant=TRAIN_METADATA_VARIANT,
    )
    mode = setup.target_mode
    if mode not in {"single", "separate"}:
        raise ValueError(
            "CONFIG_RECOMMENDATION_TARGET_MODE must be either 'single' or 'separate'. "
            f"Got: {CONFIG_RECOMMENDATION_TARGET_MODE!r}"
        )
    return mode


def _validate_recommendation_controls() -> None:
    if int(SOBOL_MAX_CONSTRAINT_ATTEMPTS) < 0:
        raise ValueError("CONFIG_RECOMMENDATION_SOBOL_MAX_CONSTRAINT_ATTEMPTS cannot be negative.")
    if int(PHASE2_RECOMMENDATION_BUDGET) <= 0:
        raise ValueError("PHASE2_RECOMMENDATION_BUDGET must be positive.")


def main() -> None:
    _validate_recommendation_controls()
    mode = _selected_recommendation_mode()
    global TARGET_MODE
    TARGET_MODE = mode
    try:
        if mode == "single":
            _run_single_recommendation()
            return
        _run_separate_recommendation()
    finally:
        _print_constraint_summary()


if __name__ == "__main__":
    main()

