"""Reusable Phase 1 evaluation helpers.

Provides prediction-based selection and metric preparation for single-target,
preference-region, and Separate held-out evaluation. Selection helpers operate on
model predictions; observed held-out objectives are joined afterward for
metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import DEFAULT_ACCURACY_WEIGHT
from src.metrics import (
    _compute_phase1_metrics,
    _finite_numeric_series,
    _require_finite_columns,
    _safe_rank_correlation,
)
from src.selection import stable_config_keys as _stable_config_keys
from src.target_utils import (
    euclidean_distance_to_front,
    ishibuchi_distance_to_front,
    novel_pareto_score,
    ordered_preference_regions,
    pareto_front_mask,
    pareto_layer_rank,
)


@dataclass
class EvaluationResult:
    details_df: pd.DataFrame
    summary_row: dict[str, Any]


def _resolve_manual_weights() -> tuple[float, float]:
    lam = float(DEFAULT_ACCURACY_WEIGHT)
    lam = min(max(lam, 0.0), 1.0)
    return lam, 1.0 - lam


def evaluate_single_regression(
    *,
    detector: str,
    lodo_dataset: str,
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_column: str,
    top_k_values: list[int],
    tree_std: np.ndarray | None = None,
    tree_iqr: np.ndarray | None = None,
) -> EvaluationResult:
    """
    Evaluate a single scalar-target model on one held-out detector-dataset pair.

    Recommendations are ranked by predicted scalar target, with lower values
    preferred. Observed held-out objectives are used only for metrics.
    """
    details = test_df.copy().reset_index(drop=True)
    real_col = f"real_{target_column}"
    pred_col = f"pred_{target_column}"
    error_col = f"abs_error_{target_column}"
    details[real_col] = np.asarray(y_true, dtype=float)
    details[pred_col] = _finite_numeric_series(y_pred, name=pred_col)
    details[error_col] = np.abs(details[pred_col] - details[real_col])

    if tree_std is not None:
        details["tree_std"] = np.asarray(tree_std, dtype=float)
    if tree_iqr is not None:
        details["tree_iqr"] = np.asarray(tree_iqr, dtype=float)

    details["ranking_distance"] = pd.to_numeric(details[pred_col], errors="coerce")
    if not np.isfinite(details["ranking_distance"].to_numpy(dtype=float)).all():
        raise ValueError("Predicted values required for sorting contain non-finite values.")

    details = details.sort_values(["ranking_distance"], ascending=[True]).reset_index(drop=True)
    details["recommendation_rank"] = np.arange(1, len(details) + 1, dtype=int)
    details["predicted_order_rank"] = rankdata(details[pred_col].to_numpy(dtype=float), method="average")

    details, phase1_summary = _compute_phase1_metrics(
        details,
        selection_mode="single",
        top_k_values=top_k_values,
        predicted_order_column="predicted_order_rank",
    )

    summary = {
        "detector": detector,
        "lodo_dataset": lodo_dataset,
        "target_mode": "single",
        "single_target_formulation": target_column,
        "separate_selection_method": "not_applicable",
        "target_column": target_column,
        "spearman_internal_target": _safe_rank_correlation(details[real_col], details[pred_col], method="spearman"),
        "rho_acc": float("nan"),
        "rho_rt": float("nan"),
    }
    summary.update(phase1_summary)
    return EvaluationResult(details_df=details, summary_row=summary)


def evaluate_single_preference_region_regression(
    *,
    detector: str,
    lodo_dataset: str,
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_column: str,
    region_name: str,
    top_k_values: list[int],
    tree_std: np.ndarray | None = None,
    tree_iqr: np.ndarray | None = None,
) -> EvaluationResult:
    result = evaluate_single_regression(
        detector=detector,
        lodo_dataset=lodo_dataset,
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        target_column=target_column,
        top_k_values=top_k_values,
        tree_std=tree_std,
        tree_iqr=tree_iqr,
    )
    result.details_df["preference_region"] = region_name
    result.summary_row["preference_region"] = region_name
    result.summary_row["preference_region_target"] = target_column
    return result


def _assign_region_ranks(details: pd.DataFrame, *, target_column: str, region_names: list[str], config_keys: pd.Series) -> pd.DataFrame:
    out = details.copy()
    n_candidates = len(out)
    denominator = max(n_candidates - 1, 1)
    for region_name in region_names:
        pred_col = f"pred_{target_column}_{region_name}"
        rank_col = f"rank_{target_column}_{region_name}"
        rank_pct_col = f"rank_pct_{target_column}_{region_name}"
        sort_frame = pd.DataFrame(
            {
                "pred": pd.to_numeric(out[pred_col], errors="coerce"),
                "config_key": config_keys,
            },
            index=out.index,
        )
        pred_values = sort_frame["pred"].to_numpy(dtype=float)
        if not np.isfinite(pred_values).all():
            raise ValueError(f"Predicted regional target contains non-finite values: {pred_col}")
        ordered_idx = sort_frame.sort_values(["pred", "config_key"], ascending=[True, True], kind="mergesort").index
        average_ranks = pd.Series(rankdata(pred_values, method="average"), index=out.index, dtype=float)
        out[rank_col] = average_ranks.loc[ordered_idx].sort_index()
        out[rank_pct_col] = (average_ranks.astype(float) - 1.0) / float(denominator)
    return out


def _select_equal_regional_indices(
    details: pd.DataFrame,
    *,
    target_column: str,
    ordered_regions: list[tuple[str, float, float]],
    config_keys: pd.Series,
    k: int,
) -> tuple[pd.Index | None, dict[str, int], dict[int, str]]:
    k = int(k)
    if len(details) < k:
        return None, {region: 0 for region, _, _ in ordered_regions}, {}
    if config_keys.nunique(dropna=False) < k:
        raise ValueError(f"Cannot select {k} unique configurations from only {config_keys.nunique(dropna=False)} unique keys.")

    region_names = [region for region, _, _ in ordered_regions]
    region_order = {region: idx for idx, region in enumerate(region_names)}
    base_quota = k // len(region_names)
    remainder = k % len(region_names)
    quotas = {region: base_quota for region in region_names}
    selected_keys: set[str] = set()
    selected_indices: list[int] = []
    selected_source: dict[int, str] = {}
    selected_counts = {region: 0 for region in region_names}

    sorted_indices: dict[str, list[int]] = {}
    pointers = {region: 0 for region in region_names}
    for region in region_names:
        pred_col = f"pred_{target_column}_{region}"
        rank_pct_col = f"rank_pct_{target_column}_{region}"
        order_df = pd.DataFrame(
            {
                "pred": pd.to_numeric(details[pred_col], errors="coerce"),
                "rank_pct": pd.to_numeric(details[rank_pct_col], errors="coerce"),
                "config_key": config_keys,
            },
            index=details.index,
        )
        sorted_indices[region] = [
            int(idx)
            for idx in order_df.sort_values(["pred", "rank_pct", "config_key"], ascending=[True, True, True], kind="mergesort").index
        ]

    def next_candidate(region: str) -> int | None:
        values = sorted_indices[region]
        pointer = pointers[region]
        while pointer < len(values) and str(config_keys.loc[values[pointer]]) in selected_keys:
            pointer += 1
        pointers[region] = pointer
        if pointer >= len(values):
            return None
        return values[pointer]

    def choose_claim(claims: list[tuple[str, int]]) -> tuple[str, int]:
        return min(
            claims,
            key=lambda item: (
                float(details.loc[item[1], f"rank_pct_{target_column}_{item[0]}"]),
                region_order[item[0]],
                str(config_keys.loc[item[1]]),
            ),
        )

    while any(selected_counts[region] < quotas[region] for region in region_names):
        proposals: dict[str, list[tuple[str, int]]] = {}
        for region in region_names:
            if selected_counts[region] >= quotas[region]:
                continue
            idx = next_candidate(region)
            if idx is None:
                raise ValueError(f"Region '{region}' cannot fill its quota for k={k}.")
            proposals.setdefault(str(config_keys.loc[idx]), []).append((region, idx))

        for key, claims in sorted(proposals.items(), key=lambda item: item[0]):
            winner_region, winner_idx = choose_claim(claims)
            if key in selected_keys or selected_counts[winner_region] >= quotas[winner_region]:
                continue
            selected_keys.add(key)
            selected_indices.append(winner_idx)
            selected_source[winner_idx] = winner_region
            selected_counts[winner_region] += 1
            pointers[winner_region] += 1

    for _ in range(remainder):
        claims = []
        for region in region_names:
            idx = next_candidate(region)
            if idx is not None:
                claims.append((region, idx))
        if not claims:
            raise ValueError(f"Could not allocate remaining regional quota for k={k}.")
        winner_region, winner_idx = choose_claim(claims)
        key = str(config_keys.loc[winner_idx])
        selected_keys.add(key)
        selected_indices.append(winner_idx)
        selected_source[winner_idx] = winner_region
        selected_counts[winner_region] += 1
        pointers[winner_region] += 1

    if len(set(selected_indices)) != k or len(selected_keys) != k:
        raise ValueError(f"Regional selection for k={k} did not produce exactly {k} unique configurations.")
    return pd.Index(selected_indices), selected_counts, selected_source


def evaluate_regional_preference_regression(
    *,
    detector: str,
    lodo_dataset: str,
    test_df: pd.DataFrame,
    target_column: str,
    region_names: tuple[str, ...],
    region_accuracy_weights: dict[str, float],
    configuration_columns: list[str],
    predictions_by_region: dict[str, np.ndarray],
    top_k_values: list[int],
    tree_std_by_region: dict[str, np.ndarray] | None = None,
    tree_iqr_by_region: dict[str, np.ndarray] | None = None,
) -> EvaluationResult:
    """
    Evaluate regional Tch/PBI/APD models with equal per-region allocation.

    Each preference region ranks configurations by its predicted scalar target.
    The combined recommendation set is deduplicated by full configuration key.
    """
    ordered_regions = ordered_preference_regions(region_names, region_accuracy_weights)
    ordered_region_names = [region for region, _, _ in ordered_regions]
    details = test_df.copy().reset_index(drop=True)
    target_columns = [f"{target_column}_{region}" for region in ordered_region_names]
    missing_targets = [column for column in target_columns if column not in details.columns]
    if missing_targets:
        raise ValueError(f"Regional details are missing target columns: {missing_targets}")
    missing_configuration = [column for column in configuration_columns if column not in details.columns]
    if missing_configuration:
        raise ValueError(f"Regional details are missing configuration columns: {missing_configuration}")
    missing_predictions = [region for region in ordered_region_names if region not in predictions_by_region]
    if missing_predictions:
        raise ValueError(f"Regional predictions are missing for region(s): {missing_predictions}")

    config_keys = _stable_config_keys(details, list(configuration_columns))

    for region in ordered_region_names:
        region_target = f"{target_column}_{region}"
        real_col = f"real_{region_target}"
        pred_col = f"pred_{region_target}"
        error_col = f"abs_error_{region_target}"
        details[real_col] = pd.to_numeric(details[region_target], errors="coerce").to_numpy(dtype=float)
        details[pred_col] = _finite_numeric_series(predictions_by_region[region], name=pred_col)
        details[error_col] = np.abs(details[pred_col] - details[real_col])
        if tree_std_by_region is not None and region in tree_std_by_region:
            details[f"tree_std_{region}"] = np.asarray(tree_std_by_region[region], dtype=float)
        if tree_iqr_by_region is not None and region in tree_iqr_by_region:
            details[f"tree_iqr_{region}"] = np.asarray(tree_iqr_by_region[region], dtype=float)

    details = _assign_region_ranks(details, target_column=target_column, region_names=ordered_region_names, config_keys=config_keys)
    rank_pct_cols = [f"rank_pct_{target_column}_{region}" for region in ordered_region_names]
    rank_pct_values = details[rank_pct_cols].to_numpy(dtype=float)
    best_positions = np.argmin(rank_pct_values, axis=1)
    details["best_region_rank_pct"] = rank_pct_values[np.arange(len(details)), best_positions]
    details["best_region_name"] = [ordered_region_names[int(pos)] for pos in best_positions]
    details["mean_region_rank_pct"] = np.mean(rank_pct_values, axis=1)
    details["__config_key"] = config_keys
    details = details.sort_values(
        ["best_region_rank_pct", "mean_region_rank_pct", "__config_key"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    details["recommendation_rank"] = np.arange(1, len(details) + 1, dtype=int)
    config_keys = details["__config_key"].copy()

    preselected_indices_by_k: dict[int, pd.Index] = {}
    selected_counts_by_k: dict[int, dict[str, int]] = {}
    for k in sorted({int(value) for value in top_k_values if int(value) > 0}):
        selected_idx, selected_counts, selected_source = _select_equal_regional_indices(
            details,
            target_column=target_column,
            ordered_regions=ordered_regions,
            config_keys=config_keys,
            k=k,
        )
        selected_counts_by_k[k] = selected_counts
        details[f"selected_at_{k}"] = False
        details[f"selection_source_region_at_{k}"] = pd.NA
        details[f"final_selection_rank_at_{k}"] = np.nan
        for region in ordered_region_names:
            details[f"selected_{region}_at_{k}"] = False
        if selected_idx is None:
            continue
        preselected_indices_by_k[k] = selected_idx
        details.loc[selected_idx, f"selected_at_{k}"] = True
        for idx, source_region in selected_source.items():
            details.loc[idx, f"selected_{source_region}_at_{k}"] = True
            details.loc[idx, f"selection_source_region_at_{k}"] = source_region
        selected_order = details.loc[selected_idx].sort_values(
            ["best_region_rank_pct", "mean_region_rank_pct", "__config_key"],
            ascending=[True, True, True],
            kind="mergesort",
        ).index
        details.loc[selected_order, f"final_selection_rank_at_{k}"] = np.arange(1, len(selected_order) + 1, dtype=int)

    details, phase1_summary = _compute_phase1_metrics(
        details,
        selection_mode="single",
        top_k_values=top_k_values,
        predicted_order_column="best_region_rank_pct",
        preselected_indices_by_k=preselected_indices_by_k,
    )

    if bool(phase1_summary.get("metrics_available_at_20", False)):
        selected_col = "selected_at_20"
        n_selected_20 = int(details[selected_col].sum())
        if n_selected_20 != 20:
            raise ValueError(f"Regional top-20 validation failed: {selected_col} contains {n_selected_20} rows, expected 20.")
        region_selected_cols = [f"selected_{region}_at_20" for region in ordered_region_names]
        missing_region_selected_cols = [column for column in region_selected_cols if column not in details.columns]
        if missing_region_selected_cols:
            raise ValueError(f"Regional top-20 validation failed: missing selection-source columns {missing_region_selected_cols}.")
        selected_region_counts = {
            region: int(details.loc[details[selected_col], f"selected_{region}_at_20"].sum())
            for region in ordered_region_names
        }
        if len(ordered_region_names) == 5 and any(count != 4 for count in selected_region_counts.values()):
            raise ValueError(
                "Regional top-20 validation failed: expected exactly four selected configurations per region, "
                f"got {selected_region_counts}."
            )
        selected_source_counts = details.loc[details[selected_col], region_selected_cols].sum(axis=1).to_numpy(dtype=int)
        if not np.all(selected_source_counts == 1):
            raise ValueError("Regional top-20 validation failed: each selected configuration must have exactly one source region.")

    summary: dict[str, Any] = {
        "detector": detector,
        "lodo_dataset": lodo_dataset,
        "target_mode": "single",
        "single_target_formulation": target_column,
        "separate_selection_method": "not_applicable",
        "target_column": "regional_targets",
        "preference_regions_enabled": True,
        "n_preference_regions": len(ordered_region_names),
        "preference_accuracy_weights": "|".join(f"{weight:g}" for _, weight, _ in ordered_regions),
        "rho_acc": float("nan"),
        "rho_rt": float("nan"),
    }
    spearman_values: list[float] = []
    for region in ordered_region_names:
        real_col = f"real_{target_column}_{region}"
        pred_col = f"pred_{target_column}_{region}"
        spearman = _safe_rank_correlation(details[real_col], details[pred_col], method="spearman")
        summary[f"spearman_internal_target_{region}"] = spearman
        if np.isfinite(spearman):
            spearman_values.append(spearman)
    summary["spearman_internal_target"] = float(np.mean(spearman_values)) if spearman_values else float("nan")
    for k, selected_counts in selected_counts_by_k.items():
        for region in ordered_region_names:
            summary[f"n_selected_from_{region}_at_{k}"] = int(selected_counts.get(region, 0))
    summary.update(phase1_summary)
    details = details.drop(columns=["__config_key"], errors="ignore")
    return EvaluationResult(details_df=details, summary_row=summary)


def summarize_candidate_set(
    *,
    detector: str,
    lodo_dataset: str,
    details_df: pd.DataFrame,
    base_target: str,
    top_k_values: list[int],
    pred_column: str,
) -> dict[str, Any]:
    details = details_df.copy()
    if "predicted_order_rank" not in details.columns:
        if pred_column in details.columns:
            details["predicted_order_rank"] = rankdata(pd.to_numeric(details[pred_column], errors="coerce"), method="average")
        elif "recommendation_rank" in details.columns:
            details["predicted_order_rank"] = pd.to_numeric(details["recommendation_rank"], errors="coerce")
        else:
            raise ValueError("summarize_candidate_set requires a predicted column or recommendation_rank.")

    if "recommendation_rank" not in details.columns:
        details["recommendation_rank"] = np.arange(1, len(details) + 1, dtype=int)

    _, phase1_summary = _compute_phase1_metrics(
        details,
        selection_mode="single",
        top_k_values=top_k_values,
        predicted_order_column="predicted_order_rank",
    )
    summary = {
        "detector": detector,
        "lodo_dataset": lodo_dataset,
        "n_imported_configs": int(len(details_df)),
        "target_column": base_target,
    }
    summary.update(phase1_summary)
    summary["spearman_internal_target"] = float("nan")
    summary["rho_acc"] = float("nan")
    summary["rho_rt"] = float("nan")
    return summary


def evaluate_separate_pareto(
    *,
    detector: str,
    lodo_dataset: str,
    test_df: pd.DataFrame,
    pred_transformed_accuracy: np.ndarray,
    pred_transformed_runtime: np.ndarray,
    distance_method: str,
    top_k_values: list[int],
    configuration_columns: list[str],
    ranking_method: str | None = None,
    accuracy_tree_std: np.ndarray | None = None,
    accuracy_tree_iqr: np.ndarray | None = None,
    runtime_tree_std: np.ndarray | None = None,
    runtime_tree_iqr: np.ndarray | None = None,
) -> EvaluationResult:
    """
    Evaluate Separate predictions using predicted Pareto-layer recommendation selection.

    Predicted transformed accuracy/runtime choose the recommendation set. Real
    transformed held-out objectives define evaluation references and metrics.
    """
    selected_ranking_method = str(ranking_method or distance_method).strip()
    if selected_ranking_method not in {"euc_dist", "mod_dist", "pareto_score", "pareto_layer"}:
        raise ValueError(
            "ranking_method must be 'euc_dist', 'mod_dist', 'pareto_score' or 'pareto_layer'."
        )

    details = test_df.copy().reset_index(drop=True)
    details["pred_transformed_accuracy"] = _finite_numeric_series(
        pred_transformed_accuracy,
        name="pred_transformed_accuracy",
    )
    details["pred_transformed_runtime"] = _finite_numeric_series(
        pred_transformed_runtime,
        name="pred_transformed_runtime",
    )
    if accuracy_tree_std is not None:
        details["accuracy_tree_std"] = np.asarray(accuracy_tree_std, dtype=float)
    if accuracy_tree_iqr is not None:
        details["accuracy_tree_iqr"] = np.asarray(accuracy_tree_iqr, dtype=float)
    if runtime_tree_std is not None:
        details["runtime_tree_std"] = np.asarray(runtime_tree_std, dtype=float)
    if runtime_tree_iqr is not None:
        details["runtime_tree_iqr"] = np.asarray(runtime_tree_iqr, dtype=float)

    pred_points = details[["pred_transformed_accuracy", "pred_transformed_runtime"]].to_numpy(dtype=float)
    pred_front_mask = pareto_front_mask(pred_points)
    pred_front = pred_points[pred_front_mask]
    details["pred_is_pareto"] = pred_front_mask.astype(int)
    details["predicted_pareto_layer"] = pareto_layer_rank(pred_points).astype(float)
    details["pred_pareto_layer"] = details["predicted_pareto_layer"]
    if not np.isfinite(details["predicted_pareto_layer"].to_numpy(dtype=float)).all():
        raise ValueError("Predicted Pareto-layer calculation failed.")

    weights = _resolve_manual_weights()
    details["pred_balanced_objective"] = np.minimum(
        details["pred_transformed_accuracy"],
        details["pred_transformed_runtime"],
    )
    details["pred_mean_objective"] = (
        details["pred_transformed_accuracy"] + details["pred_transformed_runtime"]
    ) / 2.0
    details["helper_distance_to_predicted_front"] = pd.to_numeric(details["predicted_pareto_layer"], errors="coerce")
    chosen_pred_col = "predicted_pareto_layer"
    sort_cols = ["pred_pareto_layer", "pred_balanced_objective", "pred_mean_objective"]
    ascending = [True, False, False]

    if selected_ranking_method != "pareto_layer":
        if selected_ranking_method == "euc_dist":
            details["pred_euc_dist"] = euclidean_distance_to_front(pred_points, pred_front, weights=weights)
        elif selected_ranking_method == "mod_dist":
            details["pred_mod_dist"] = ishibuchi_distance_to_front(pred_points, pred_front, weights=weights)
        else:
            details["pred_pareto_score"], details["pred_reward_factor"] = novel_pareto_score(pred_points, pred_front)
        chosen_pred_col = f"pred_{selected_ranking_method}"
        details["helper_distance_to_predicted_front"] = pd.to_numeric(details[chosen_pred_col], errors="coerce")
        sort_cols = ["pred_is_pareto", "helper_distance_to_predicted_front"]
        ascending = [False, True]

    if not np.isfinite(details[sort_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).all():
        raise ValueError("Predicted values required for sorting contain non-finite values.")

    details = details.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    details["recommendation_rank"] = np.arange(1, len(details) + 1, dtype=int)
    details["predicted_order_rank"] = pd.to_numeric(details["predicted_pareto_layer"], errors="coerce")

    # Selection is frozen from predictions only. Observed transformed objectives
    # are used below by _compute_phase1_metrics only after selected_at_k exists.
    details, phase1_summary = _compute_phase1_metrics(
        details,
        selection_mode="separate",
        top_k_values=top_k_values,
        predicted_order_column="predicted_pareto_layer",
        configuration_columns=configuration_columns,
    )

    rho_acc = _safe_rank_correlation(
        details["transformed_accuracy"],
        details["pred_transformed_accuracy"],
        method="spearman",
    )
    rho_rt = _safe_rank_correlation(
        details["transformed_runtime"],
        details["pred_transformed_runtime"],
        method="spearman",
    )
    summary = {
        "detector": detector,
        "lodo_dataset": lodo_dataset,
        "target_mode": "separate",
        "single_target_formulation": "not_applicable",
        "separate_selection_method": selected_ranking_method,
        "target_column": "separate_objectives",
        "spearman_internal_target": float("nan"),
        "rho_acc": rho_acc,
        "rho_rt": rho_rt,
    }
    summary.update(phase1_summary)
    if chosen_pred_col not in details.columns:
        details[chosen_pred_col] = details["predicted_pareto_layer"]
    return EvaluationResult(details_df=details, summary_row=summary)
