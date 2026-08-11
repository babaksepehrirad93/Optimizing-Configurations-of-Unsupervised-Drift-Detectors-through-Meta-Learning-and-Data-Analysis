"""Objective transformation helpers.

Transforms raw benchmark objectives into training/evaluation objectives.
Accuracy remains higher-is-better. Runtime is optionally clipped, log-scaled,
min-max normalized, and reversed so transformed runtime is higher-is-better.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


AccuracyMode = Literal["identity", "minmax"]
RuntimeMode = Literal["log1p_minmax", "minmax"]


def to_numeric_array(values: pd.Series | np.ndarray | list[float], name: str = "values") -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if np.isnan(arr).all():
        raise ValueError(f"'{name}' could not be converted to numeric values.")
    return arr


def minmax_scale(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(out)
    if not mask.any():
        return np.full_like(out, np.nan, dtype=float)
    lo = float(np.nanmin(out[mask]))
    hi = float(np.nanmax(out[mask]))
    den = hi - lo
    if den == 0.0:
        out[mask] = 1.0
        return out
    out[mask] = (out[mask] - lo) / den
    return out


def upper_iqr_clip(values: np.ndarray, *, multiplier: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(out)
    if not mask.any():
        return out
    q1 = float(np.nanpercentile(out[mask], 25.0))
    q3 = float(np.nanpercentile(out[mask], 75.0))
    iqr = q3 - q1
    if iqr <= 0:
        return out
    threshold = q3 + (float(multiplier) * iqr)
    if not np.isfinite(threshold):
        return out
    out[mask] = np.minimum(out[mask], threshold)
    return out


def upper_percentile_clip(values: np.ndarray, *, percentile: float) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(out)
    if not mask.any():
        return out
    pct = float(percentile)
    if pct <= 0.0 or pct >= 100.0:
        return out
    threshold = float(np.nanpercentile(out[mask], pct))
    if not np.isfinite(threshold):
        return out
    out[mask] = np.minimum(out[mask], threshold)
    return out


def transform_accuracy(
    accuracy: pd.Series | np.ndarray | list[float],
    *,
    mode: AccuracyMode,
) -> np.ndarray:
    """Transform raw accuracy while preserving higher-is-better direction."""
    arr = to_numeric_array(accuracy, name="accuracy")
    if mode == "identity":
        return arr
    if mode == "minmax":
        return minmax_scale(arr)
    raise ValueError("PREPROCESSING_ACCURACY_MODE must be 'identity' or 'minmax'.")


def transform_runtime(
    runtime: pd.Series | np.ndarray | list[float],
    *,
    mode: RuntimeMode,
    use_upper_clipping: bool,
    clip_method: str,
    clip_iqr_multiplier: float,
    clip_percentile: float,
) -> np.ndarray:
    """Transform raw runtime into a higher-is-better runtime utility."""
    arr = to_numeric_array(runtime, name="runtime")
    if clip_method not in {"iqr_upper", "percentile_upper"}:
        raise ValueError("PREPROCESSING_RUNTIME_CLIP_METHOD must be 'iqr_upper' or 'percentile_upper'.")

    if mode == "log1p_minmax":
        current = arr.copy()
        if use_upper_clipping and clip_method == "percentile_upper":
            current = upper_percentile_clip(current, percentile=clip_percentile)
        if np.nanmin(current) < 0:
            raise ValueError("runtime contains negative values; log1p_minmax is not valid.")
        current = np.log1p(current)
        if use_upper_clipping and clip_method == "iqr_upper":
            current = upper_iqr_clip(current, multiplier=clip_iqr_multiplier)
        current = minmax_scale(current)
        return 1.0 - current

    if mode == "minmax":
        current = arr.copy()
        if use_upper_clipping:
            if clip_method == "iqr_upper":
                current = upper_iqr_clip(current, multiplier=clip_iqr_multiplier)
            else:
                current = upper_percentile_clip(current, percentile=clip_percentile)
        current = minmax_scale(current)
        return 1.0 - current

    raise ValueError("PREPROCESSING_RUNTIME_MODE must be 'log1p_minmax' or 'minmax'.")
