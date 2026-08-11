"""Metric implementations for Phase 1 and Phase 2 evaluation.

Contains Pareto-distance indicators, exact continuous bi-objective R2,
dominance rates, NPD/MNPD helpers, and rank-correlation utilities. Metric
functions return numeric values without presentation rounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import (
    TRAIN_NPD_THRESHOLDS,
    TRAIN_TOP_K_VALUES,
)
from src.selection import select_pareto_budget
from src.target_utils import pareto_layer_rank


EXACT_R2_IMPLEMENTATION_VERSION = "exact_continuous_r2_biobjective_v1"


@dataclass(frozen=True)
class R2PreparedSet:
    """Cleaned point set used by the exact bi-objective R2 calculation."""

    points: np.ndarray
    unique_point_count: int
    nondominated_point_count: int


def _as_loss_points(points: np.ndarray, *, atol: float) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (n_points, 2). Got {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError("exact R2 is undefined for an empty point set.")
    if not np.isfinite(arr).all():
        raise ValueError("points contain NaN or infinite loss values.")
    if np.any(arr < -atol):
        lo = float(np.min(arr))
        raise ValueError(f"loss values must be non-negative. Minimum value: {lo}.")
    return np.where((arr < 0.0) & (arr >= -atol), 0.0, arr)


def nondominated_biobjective(points: np.ndarray, *, atol: float = 1e-12) -> R2PreparedSet:
    """Return duplicate-free nondominated minimization points ordered by objective 1."""

    arr = _as_loss_points(points, atol=atol)
    unique = np.unique(arr, axis=0)
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    sorted_points = unique[order]

    retained: list[np.ndarray] = []
    best_second = np.inf
    for point in sorted_points:
        second = float(point[1])
        if second < best_second - atol:
            retained.append(point)
            best_second = second

    if not retained:
        raise ValueError("No nondominated points remain after filtering.")

    nd = np.vstack(retained).astype(float, copy=False)
    return R2PreparedSet(
        points=nd,
        unique_point_count=int(len(unique)),
        nondominated_point_count=int(len(nd)),
    )


def _clamp_unit(value: float, *, name: str, atol: float) -> float:
    if value < -atol or value > 1.0 + atol:
        raise ValueError(f"{name}={value} is outside [0, 1] beyond tolerance.")
    return float(min(1.0, max(0.0, value)))


def exact_r2_biobjective(points: np.ndarray, *, atol: float = 1e-12) -> float:
    """Compute the exact continuous bi-objective R2 indicator.

    Input points are minimization losses with shape ``(n_points, 2)``. Lower
    values are better, and the ideal point is ``[0.0, 0.0]``.
    """

    prepared = nondominated_biobjective(points, atol=atol)
    nd = prepared.points

    if np.any(np.all(np.abs(nd) <= atol, axis=1)):
        return 0.0

    total = 0.0
    n_points = len(nd)
    a_values = nd[:, 0]
    b_values = nd[:, 1]

    for i, (a_i, b_i) in enumerate(nd):
        denom = float(a_i + b_i)
        if denom <= atol:
            return 0.0

        q_i = _clamp_unit(float(b_i / denom), name=f"q_{i}", atol=atol)

        if i == n_points - 1:
            lower = 0.0
        else:
            lower = _clamp_unit(
                float(b_i / (a_values[i + 1] + b_i)),
                name=f"L_{i}",
                atol=atol,
            )

        if i == 0:
            upper = 1.0
        else:
            upper = _clamp_unit(
                float(b_values[i - 1] / (a_i + b_values[i - 1])),
                name=f"U_{i}",
                atol=atol,
            )

        if lower > q_i + atol or q_i > upper + atol:
            raise ValueError(
                "Invalid exact-R2 interval ordering for nondominated point "
                f"{i}: L={lower}, q={q_i}, U={upper}, point=({a_i}, {b_i})."
            )
        lower = min(lower, q_i)
        upper = max(upper, q_i)

        contribution = (
            0.5 * float(b_i) * (((1.0 - lower) ** 2) - ((1.0 - q_i) ** 2))
            + 0.5 * float(a_i) * ((upper**2) - (q_i**2))
        )
        if contribution < -atol:
            raise ValueError(f"Negative exact-R2 contribution beyond tolerance: {contribution}.")
        total += max(0.0, contribution)

    result = float(total)
    if np.all(nd <= 1.0 + atol) and (result < -atol or result > 0.75 + atol):
        raise ValueError(f"exact R2 for points in [0, 1]^2 must be in [0, 0.75]. Got {result}.")
    return float(min(0.75, max(0.0, result)) if np.all(nd <= 1.0 + atol) else result)


def approximate_r2_biobjective(
    points: np.ndarray,
    *,
    n_grid: int = 100_001,
    atol: float = 1e-12,
) -> float:
    """Numerically approximate the defining integral for validation tests only."""

    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")
    arr = _as_loss_points(points, atol=atol)
    weights = np.linspace(0.0, 1.0, int(n_grid), dtype=float)
    values = np.maximum(
        weights[:, None] * arr[None, :, 0],
        (1.0 - weights[:, None]) * arr[None, :, 1],
    )
    envelope = np.min(values, axis=1)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(envelope, weights))
    return float(np.trapz(envelope, weights))


def _percent_label(value: float) -> str:
    return f"{int(round(float(value) * 100.0)):02d}"


def _build_phase1_metric_columns() -> list[str]:
    columns: list[str] = []

    for k in sorted({int(v) for v in TRAIN_TOP_K_VALUES if int(v) > 0}):
        columns.extend(
            [
                f"gd_plus_at_{k}",
                f"mnpd_at_{k}",
            ]
        )
        for threshold in sorted({float(v) for v in TRAIN_NPD_THRESHOLDS}):
            columns.append(f"npd_le_{_percent_label(threshold)}_hit_rate_at_{k}")
        columns.extend(
            [
                f"igd_plus_at_{k}",
                f"mean_exact_r2_selected_at_{k}",
                f"mean_r2_gap_selected_at_{k}",
            ]
        )

    columns.extend(
        [
            "spearman_internal_target",
            "rho_acc",
            "rho_rt",
        ]
    )
    return columns


PHASE1_METRIC_COLUMNS: list[str] = _build_phase1_metric_columns()
PHASE1_METRIC_COLUMN_SET: set[str] = set(PHASE1_METRIC_COLUMNS)


def clean_global_metric_columns(top_k_values: Iterable[int] | None = None) -> list[str]:
    del top_k_values
    return list(PHASE1_METRIC_COLUMNS)


def is_clean_global_metric_column(column: str) -> bool:
    return column in PHASE1_METRIC_COLUMN_SET


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    return _safe_rank_correlation(y_true, y_pred, method=method)


def _finite_numeric_series(values: pd.Series | np.ndarray, *, name: str) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _require_finite_columns(df: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required transformed-objective column(s): {missing}")
    points = df[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("Real transformed objectives contain non-finite values.")
    return points


def _normalized_pareto_depth(layers: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(layers, dtype=float)
    if arr.size == 0:
        return arr.astype(float)
    if not np.isfinite(arr).all():
        raise ValueError("Real Pareto layers contain non-finite values.")
    l_max = float(np.max(arr))
    if l_max < 1.0:
        raise ValueError("Real Pareto layers must be positive.")
    if np.isclose(l_max, 1.0):
        return np.zeros(arr.shape, dtype=float)
    return (arr - 1.0) / (l_max - 1.0)


def _prepare_real_reference(details_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    details = details_df.copy()
    real_points = _require_finite_columns(details, ["transformed_accuracy", "transformed_runtime"])

    layer_source = None
    if "real_pareto_layer" in details.columns:
        layer_source = "real_pareto_layer"
    elif "pareto_layer" in details.columns:
        layer_source = "pareto_layer"

    layers: np.ndarray
    if layer_source is not None:
        candidate = pd.to_numeric(details[layer_source], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(candidate).all() and np.all(candidate >= 1.0):
            layers = candidate.astype(float)
        else:
            layers = pareto_layer_rank(real_points).astype(float)
    else:
        layers = pareto_layer_rank(real_points).astype(float)

    if layers.size != len(details) or not np.isfinite(layers).all() or np.nanmin(layers) < 1.0:
        raise ValueError("Real Pareto-layer calculation failed.")

    details["real_pareto_layer"] = layers
    details["normalized_pareto_depth"] = _normalized_pareto_depth(layers)

    real_pareto_rows = details.loc[np.isclose(details["real_pareto_layer"].to_numpy(dtype=float), 1.0)].copy()
    if real_pareto_rows.empty:
        raise ValueError("The real Pareto set is empty.")

    real_front = np.unique(
        real_pareto_rows[["transformed_accuracy", "transformed_runtime"]].to_numpy(dtype=float),
        axis=0,
    )
    if real_front.size == 0:
        raise ValueError("The real Pareto set is empty.")

    return details, real_front


def _phase1_metric_nan_values(k: int, thresholds: Iterable[float]) -> dict[str, float]:
    out: dict[str, float] = {
        f"gd_plus_at_{k}": float("nan"),
        f"mnpd_at_{k}": float("nan"),
        f"igd_plus_at_{k}": float("nan"),
    }
    for threshold in thresholds:
        out[f"npd_le_{_percent_label(threshold)}_hit_rate_at_{k}"] = float("nan")
    return out


def _directional_distance_plus(selected_points: np.ndarray, front_points: np.ndarray) -> np.ndarray:
    selected = np.asarray(selected_points, dtype=float)
    front = np.asarray(front_points, dtype=float)
    if selected.size == 0 or front.size == 0:
        return np.empty((len(selected), len(front)), dtype=float)
    diff = np.maximum(front[None, :, :] - selected[:, None, :], 0.0)
    return np.sqrt(np.sum(diff * diff, axis=2))


def _gd_plus(selected_points: np.ndarray, real_front: np.ndarray) -> float:
    distances = _directional_distance_plus(selected_points, real_front)
    return float(np.mean(np.min(distances, axis=1)))


def _igd_plus(selected_points: np.ndarray, real_front: np.ndarray) -> float:
    distances = _directional_distance_plus(selected_points, real_front)
    return float(np.mean(np.min(distances, axis=0)))


def gd_plus(selected_points: np.ndarray, reference_points: np.ndarray) -> float:
    """Compute GD+ for two non-empty transformed objective point sets."""

    selected = np.asarray(selected_points, dtype=float)
    reference = np.asarray(reference_points, dtype=float)
    if selected.ndim != 2 or selected.shape[1] != 2:
        raise ValueError(f"selected_points must have shape (n_points, 2). Got {selected.shape}.")
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError(f"reference_points must have shape (n_points, 2). Got {reference.shape}.")
    if selected.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("GD+ requires non-empty selected and reference point sets.")
    if not np.isfinite(selected).all() or not np.isfinite(reference).all():
        raise ValueError("GD+ points must be finite.")
    return _gd_plus(selected, reference)


def igd_plus(selected_points: np.ndarray, reference_points: np.ndarray) -> float:
    """Compute IGD+ for two non-empty transformed objective point sets."""

    selected = np.asarray(selected_points, dtype=float)
    reference = np.asarray(reference_points, dtype=float)
    if selected.ndim != 2 or selected.shape[1] != 2:
        raise ValueError(f"selected_points must have shape (n_points, 2). Got {selected.shape}.")
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError(f"reference_points must have shape (n_points, 2). Got {reference.shape}.")
    if selected.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("IGD+ requires non-empty selected and reference point sets.")
    if not np.isfinite(selected).all() or not np.isfinite(reference).all():
        raise ValueError("IGD+ points must be finite.")
    return _igd_plus(selected, reference)


def _as_exact_r2_loss_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected transformed objective points with shape (n_points, 2). Got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Transformed objective points contain non-finite values.")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("Transformed objective points must be in [0, 1] for exact R2.")
    values = np.clip(values, 0.0, 1.0)
    return np.column_stack([1.0 - values[:, 0], 1.0 - values[:, 1]])


def transformed_objectives_to_exact_r2_losses(points: np.ndarray) -> np.ndarray:
    """Convert higher-is-better transformed objective points to exact-R2 losses."""

    return _as_exact_r2_loss_points(points)


def exact_r2_from_transformed_objectives(points: np.ndarray) -> float:
    """Compute exact continuous bi-objective R2 from transformed maximization points."""

    return exact_r2_biobjective(_as_exact_r2_loss_points(points))


def _exact_r2_metrics(selected_points: np.ndarray, real_front: np.ndarray) -> tuple[float, float]:
    exact_r2 = exact_r2_biobjective(_as_exact_r2_loss_points(selected_points))
    observed_pareto_r2 = exact_r2_biobjective(_as_exact_r2_loss_points(real_front))
    return float(exact_r2), float(exact_r2 - observed_pareto_r2)


def _select_indices_for_k(
    details: pd.DataFrame,
    k: int,
    *,
    selection_mode: str,
    configuration_columns: Iterable[str] | None = None,
) -> pd.Index | None:
    if len(details) < int(k):
        return None
    if selection_mode == "single":
        ranks = pd.to_numeric(details["recommendation_rank"], errors="coerce")
        if not np.isfinite(ranks.to_numpy(dtype=float)).all():
            raise ValueError("recommendation_rank contains non-finite values.")
        return details.assign(_rank=ranks).nsmallest(int(k), "_rank").index
    if selection_mode == "separate":
        return select_pareto_budget(
            details,
            ("pred_transformed_accuracy", "pred_transformed_runtime"),
            int(k),
            maximize=(True, True),
            config_columns=configuration_columns,
        )
    raise ValueError("selection_mode must be 'single' or 'separate'.")


def _safe_rank_correlation(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, *, method: str) -> float:
    if method != "spearman":
        raise ValueError("Only Spearman correlation is part of the official Phase 1 metric set.")
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return float("nan")
    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    if np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
        return float("nan")
    result = spearmanr(x_valid, y_valid)
    return float(result.statistic) if np.isfinite(result.statistic) else float("nan")


def safe_spearman_correlation(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Public wrapper for the existing safe Spearman helper."""

    return _safe_rank_correlation(x, y, method="spearman")


def raw_accuracy_runtime_dominates(
    recommendation: np.ndarray | pd.Series | tuple[float, float] | list[float],
    reference: np.ndarray | pd.Series | tuple[float, float] | list[float],
    *,
    atol: float = 1e-12,
) -> bool:
    """Return whether one raw accuracy/runtime point dominates another.

    Accuracy is maximized and runtime is minimized. Inputs are ordered as
    ``(accuracy, runtime)``.
    """

    r = np.asarray(recommendation, dtype=float)
    q = np.asarray(reference, dtype=float)
    if r.shape != (2,) or q.shape != (2,):
        raise ValueError("Dominance points must be one-dimensional (accuracy, runtime) pairs.")
    if not np.isfinite(r).all() or not np.isfinite(q).all():
        return False

    no_worse = bool(r[0] >= q[0] - atol and r[1] <= q[1] + atol)
    strictly_better = bool(r[0] > q[0] + atol or r[1] < q[1] - atol)
    return no_worse and strictly_better


def raw_accuracy_runtime_dominance_rates(
    recommendation_points: np.ndarray | pd.DataFrame,
    reference_points: np.ndarray | pd.DataFrame,
    *,
    atol: float = 1e-12,
) -> tuple[float, float]:
    """Return recommendation and reference dominance rates for raw objectives.

    The first rate is the share of completed recommendations that dominate at
    least one reference. The second rate is the share of references dominated by
    at least one completed recommendation.
    """

    if isinstance(recommendation_points, pd.DataFrame):
        rec = recommendation_points[["ACCURACY", "RUNTIME"]].to_numpy(dtype=float)
    else:
        rec = np.asarray(recommendation_points, dtype=float)
    if isinstance(reference_points, pd.DataFrame):
        ref = reference_points[["ACCURACY", "RUNTIME"]].to_numpy(dtype=float)
    else:
        ref = np.asarray(reference_points, dtype=float)

    if rec.ndim != 2 or rec.shape[1] != 2:
        raise ValueError(f"recommendation_points must have shape (n_points, 2). Got {rec.shape}.")
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"reference_points must have shape (n_points, 2). Got {ref.shape}.")
    if rec.shape[0] == 0 or ref.shape[0] == 0:
        raise ValueError("Dominance rates require non-empty recommendation and reference point sets.")
    if not np.isfinite(rec).all() or not np.isfinite(ref).all():
        raise ValueError("Dominance-rate points must be finite.")

    accuracy_no_worse = rec[:, None, 0] >= ref[None, :, 0] - atol
    runtime_no_worse = rec[:, None, 1] <= ref[None, :, 1] + atol
    accuracy_better = rec[:, None, 0] > ref[None, :, 0] + atol
    runtime_better = rec[:, None, 1] < ref[None, :, 1] - atol
    dominance = accuracy_no_worse & runtime_no_worse & (accuracy_better | runtime_better)

    recommendation_dominance_rate = float(np.mean(np.any(dominance, axis=1)))
    reference_dominated_rate = float(np.mean(np.any(dominance, axis=0)))
    return recommendation_dominance_rate, reference_dominated_rate


def _compute_phase1_metrics(
    details_df: pd.DataFrame,
    *,
    selection_mode: str,
    top_k_values: Iterable[int],
    predicted_order_column: str,
    npd_thresholds: Iterable[float] | None = None,
    configuration_columns: Iterable[str] | None = None,
    preselected_indices_by_k: dict[int, pd.Index] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    thresholds = sorted({float(v) for v in (npd_thresholds or TRAIN_NPD_THRESHOLDS)})
    k_values = sorted({int(v) for v in top_k_values if int(v) > 0})

    # Freeze the recommendation set from prediction-derived columns before
    # constructing any observed-performance reference information.
    selection_details = details_df.copy()
    if predicted_order_column not in selection_details.columns:
        raise ValueError(f"Missing predicted ordering column: {predicted_order_column}")
    selected_indices_by_k: dict[int, pd.Index | None] = {}
    for k in k_values:
        if preselected_indices_by_k is not None and int(k) in preselected_indices_by_k:
            selected_idx = pd.Index(preselected_indices_by_k[int(k)])
            missing_idx = selected_idx.difference(selection_details.index)
            if len(missing_idx) > 0:
                raise ValueError(f"Preselected indices for k={k} are not present in details_df.")
            if len(selected_idx) != len(selected_idx.unique()):
                raise ValueError(f"Preselected indices for k={k} contain duplicates.")
        else:
            selected_idx = _select_indices_for_k(
                selection_details,
                k,
                selection_mode=selection_mode,
                configuration_columns=configuration_columns,
            )
        selected_indices_by_k[int(k)] = selected_idx

    details, real_front = _prepare_real_reference(selection_details)
    if predicted_order_column not in details.columns:
        raise ValueError(f"Missing predicted ordering column: {predicted_order_column}")

    summary: dict[str, Any] = {
        "n_heldout_configs": int(len(details)),
        "n_real_pareto_points": int(len(real_front)),
        "n_real_pareto_layers": int(pd.to_numeric(details["real_pareto_layer"], errors="coerce").nunique(dropna=True)),
    }

    for k in k_values:
        details[f"selected_at_{k}"] = False
        selected_idx = selected_indices_by_k[int(k)]
        summary[f"metrics_available_at_{k}"] = bool(selected_idx is not None)
        r2_metric_name = f"mean_exact_r2_selected_at_{k}"
        r2_gap_name = f"mean_r2_gap_selected_at_{k}"
        summary[r2_metric_name] = float("nan")
        summary[r2_gap_name] = float("nan")
        if selected_idx is None:
            summary[f"n_selected_at_{k}"] = 0
            summary.update(_phase1_metric_nan_values(k, thresholds))
            continue

        selected_df = details.loc[selected_idx].copy()
        if selected_df.empty:
            raise ValueError(f"Selected-set construction unexpectedly returned an empty set for k={k}.")
        details.loc[selected_idx, f"selected_at_{k}"] = True
        summary[f"n_selected_at_{k}"] = int(len(selected_df))

        selected_points = selected_df[["transformed_accuracy", "transformed_runtime"]].to_numpy(dtype=float)

        summary[f"gd_plus_at_{k}"] = _gd_plus(selected_points, real_front)
        summary[f"mnpd_at_{k}"] = float(np.mean(selected_df["normalized_pareto_depth"].to_numpy(dtype=float)))
        for threshold in thresholds:
            summary[f"npd_le_{_percent_label(threshold)}_hit_rate_at_{k}"] = float(
                np.mean(selected_df["normalized_pareto_depth"].to_numpy(dtype=float) <= threshold)
            )
        summary[f"igd_plus_at_{k}"] = _igd_plus(selected_points, real_front)
        exact_r2, r2_gap = _exact_r2_metrics(selected_points, real_front)
        summary[r2_metric_name] = exact_r2
        summary[r2_gap_name] = r2_gap

    return details, summary
