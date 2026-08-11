"""Deterministic recommendation-selection utilities.

Provides stable hyperparameter identity keys and exact-budget Pareto-layer
selection used by Phase 1 Separate evaluation and Phase 2 final recommendation.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.target_utils import pareto_layer_rank


def stable_config_keys(df: pd.DataFrame, config_columns: Iterable[str] | None) -> pd.Series:
    """Return deterministic identity keys for complete detector hyperparameter configs."""

    columns = list(config_columns or [])
    if not columns:
        return pd.Series([str(i) for i in df.index], index=df.index, dtype="object")

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Configuration identity column(s) missing from candidate dataframe: {missing}")

    def canonical(value) -> str:
        if pd.isna(value):
            return "<NA>"
        if isinstance(value, (bool, np.bool_)):
            return f"b:{int(bool(value))}"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return f"s:{str(value)}"
        if np.isfinite(numeric):
            if np.isclose(numeric, round(numeric), rtol=0.0, atol=1e-12):
                return f"i:{int(round(numeric))}"
            return f"f:{numeric:.17g}"
        return f"s:{str(value)}"

    key_frame = df[columns].copy()
    for column in columns:
        key_frame[column] = key_frame[column].map(canonical)
    return key_frame.agg("\x1f".join, axis=1)


def select_pareto_budget(
    dataframe: pd.DataFrame,
    objective_columns: tuple[str, str] | list[str],
    budget: int,
    *,
    maximize: tuple[bool, bool] = (True, True),
    config_columns: Iterable[str] | None = None,
) -> pd.Index | None:
    """
    Select exactly ``budget`` unique configurations from predicted Pareto layers.

    The rule is thesis-style complete-layer selection: add whole Pareto layers
    while the next complete layer fits. If the next layer would overflow the
    remaining budget, sort only that boundary layer along the predicted
    trade-off, split it into as many approximately equal segments as remaining
    slots, and choose one middle representative from each segment. Selection is
    based only on the supplied objective columns and deterministic configuration
    identity keys; observed held-out objectives must not be passed here.
    """

    k = int(budget)
    if k <= 0:
        raise ValueError(f"budget must be positive. Got {budget!r}.")
    columns = list(objective_columns)
    if len(columns) != 2:
        raise ValueError("select_pareto_budget currently expects exactly two objective columns.")
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Pareto-budget objective column(s) missing: {missing}")

    if len(dataframe) < k:
        return None

    candidates = dataframe.copy()
    candidates["__config_key"] = stable_config_keys(candidates, config_columns)
    if candidates["__config_key"].nunique(dropna=False) < k:
        return None

    obj = candidates[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(obj).all():
        raise ValueError("Pareto-budget objective columns must be finite.")

    max_points = obj.copy()
    for col_idx, is_maximized in enumerate(maximize):
        if not bool(is_maximized):
            max_points[:, col_idx] *= -1.0
    candidates["__pareto_layer"] = pareto_layer_rank(max_points).astype(float)
    candidates["__obj0"] = max_points[:, 0]
    candidates["__obj1"] = max_points[:, 1]
    candidates["__original_order"] = np.arange(len(candidates), dtype=int)

    representatives = candidates.sort_values(
        ["__config_key", "__pareto_layer", "__obj0", "__obj1", "__original_order"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    ).drop_duplicates("__config_key", keep="first")

    selected_indices: list[int] = []
    selected_keys: set[str] = set()
    layers = sorted(float(layer) for layer in representatives["__pareto_layer"].dropna().unique())

    for layer in layers:
        layer_df = representatives.loc[np.isclose(representatives["__pareto_layer"].to_numpy(dtype=float), layer)].copy()
        remaining = k - len(selected_indices)
        if remaining <= 0:
            break
        if len(layer_df) <= remaining:
            ordered_layer = layer_df.sort_values(
                ["__obj0", "__obj1", "__config_key", "__original_order"],
                ascending=[False, False, True, True],
                kind="mergesort",
            )
            for idx, key in zip(ordered_layer.index, ordered_layer["__config_key"]):
                key_text = str(key)
                if key_text not in selected_keys:
                    selected_indices.append(idx)
                    selected_keys.add(key_text)
            continue

        boundary = layer_df.sort_values(
            ["__obj0", "__obj1", "__config_key", "__original_order"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        n_boundary = len(boundary)
        segment_edges = np.linspace(0, n_boundary, remaining + 1)
        chosen_positions: list[int] = []
        for segment_idx in range(remaining):
            start = int(np.floor(segment_edges[segment_idx]))
            end = int(np.floor(segment_edges[segment_idx + 1]))
            end = max(end, start + 1)
            end = min(end, n_boundary)
            midpoint = start + ((end - start - 1) // 2)
            chosen_positions.append(min(midpoint, n_boundary - 1))
        for position in chosen_positions:
            idx = boundary.index[position]
            key_text = str(boundary.loc[idx, "__config_key"])
            if key_text not in selected_keys:
                selected_indices.append(idx)
                selected_keys.add(key_text)
        break

    if len(selected_indices) != k or len(selected_keys) != k:
        raise ValueError(f"Pareto-budget selection produced {len(selected_indices)} rows/{len(selected_keys)} keys, expected {k}.")
    return pd.Index(selected_indices)
