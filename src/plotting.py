"""Phase 1 Pareto objective-space plotting helpers.

Plots observed held-out benchmark points and selected recommendations from
Phase 1 detail CSVs. Plot settings are defined in `src.config`.
"""

from __future__ import annotations

from math import ceil, floor
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from src.config import PLOT_FIGSIZE, PLOT_SAVE_DPI, TRAIN_TOP_K_VALUES


FIGSIZE = PLOT_FIGSIZE
SAVE_DPI = PLOT_SAVE_DPI


def _floor_to_decimals(x: float, decimals: int) -> float:
    factor = 10 ** decimals
    return floor(x * factor) / factor


def _padded_linear_bounds(lo: float, hi: float, *, pad_fraction: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Cannot build plot bounds from non-finite values.")
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    if np.isclose(span, 0.0):
        pad = max(abs(lo) * pad_fraction, 1.0)
    else:
        pad = span * pad_fraction
    return lo - pad, hi + pad


def _expand_accuracy_bounds(y_min: float, y_max: float) -> tuple[float, float]:
    low, high = _padded_linear_bounds(y_min, y_max)
    low = _floor_to_decimals(low, 2)
    high = ceil(high * 100.0) / 100.0
    low = max(0.0, low)
    if low >= high:
        low = max(0.0, low - 0.01)
        high = high + 0.01
    return low, high


def _expand_runtime_bounds(x_min: float, x_max: float) -> tuple[float, float]:
    low, high = _padded_linear_bounds(x_min, x_max)
    low = max(0.0, floor(low))
    high = ceil(high)
    if low >= high:
        high = low + 1.0
    return low, high


def _selection_mask(df: pd.DataFrame, selected_column: str) -> pd.Series:
    if selected_column not in df.columns:
        raise ValueError(f"Required recommendation-selection column is missing: {selected_column}")
    return pd.to_numeric(df[selected_column], errors="coerce").fillna(0).astype(bool)


def _select_recommended_df(plot_df: pd.DataFrame, *, selected_column: str) -> pd.DataFrame:
    selected_mask = _selection_mask(plot_df, selected_column)
    recommended_df = plot_df.loc[selected_mask].copy()
    if recommended_df.empty:
        raise ValueError(f"No configurations are marked by {selected_column}.")
    return recommended_df


def _select_recommended_df_legacy(
    plot_df: pd.DataFrame,
    *,
    recommended_limit: int | None,
) -> pd.DataFrame:
    if recommended_limit is not None:
        return (
            plot_df
            .sort_values("recommendation_rank", ascending=True)
            .head(recommended_limit)
            .copy()
        )
    return plot_df.sort_values("recommendation_rank", ascending=True).copy()


def _expected_recommendation_count(details_df: pd.DataFrame, selected_column: str | None) -> int | None:
    if selected_column is None:
        return None
    if selected_column not in details_df.columns:
        raise ValueError(f"Required recommendation-selection column is missing: {selected_column}")
    return int(pd.to_numeric(details_df[selected_column], errors="coerce").fillna(0).astype(bool).sum())


def _assert_recommendation_count(
    *,
    details_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
    selected_column: str | None,
) -> None:
    expected_count = _expected_recommendation_count(details_df, selected_column)
    if expected_count is None:
        return
    plotted_count = len(recommended_df)
    if plotted_count != expected_count:
        raise AssertionError(
            f"Recommendation plotting mismatch for {selected_column}: "
            f"expected {expected_count}, plotted {plotted_count}."
        )


def _prediction_label_k() -> int:
    if not TRAIN_TOP_K_VALUES:
        raise ValueError("TRAIN_TOP_K_VALUES must contain at least one value for plotting.")
    return int(TRAIN_TOP_K_VALUES[-1])


def _prepare_details_df(details_csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(details_csv_path))
    required = ["ACCURACY", "RUNTIME"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Pareto plotting: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No valid rows left in details CSV after Pareto plotting cleanup.")

    if "recommendation_rank" not in df.columns:
        df = df.reset_index(drop=True)
        df["recommendation_rank"] = np.arange(1, len(df) + 1, dtype=int)

    return df


def _load_context_df(context_csv_path: str | Path | None) -> pd.DataFrame | None:
    if context_csv_path is None:
        return None
    df = pd.read_csv(Path(context_csv_path))
    if "ACCURACY" not in df.columns or "RUNTIME" not in df.columns:
        return None
    df["ACCURACY"] = pd.to_numeric(df["ACCURACY"], errors="coerce")
    df["RUNTIME"] = pd.to_numeric(df["RUNTIME"], errors="coerce")
    if "is_pareto" in df.columns:
        df["is_pareto"] = pd.to_numeric(df["is_pareto"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["ACCURACY", "RUNTIME"]).copy()


def _real_pareto_df(source_df: pd.DataFrame) -> pd.DataFrame:
    if "is_pareto" in source_df.columns:
        pareto_mask = pd.to_numeric(source_df["is_pareto"], errors="coerce").fillna(0).astype(int) == 1
        return source_df.loc[pareto_mask].copy()
    if "real_pareto_layer" in source_df.columns:
        layer = pd.to_numeric(source_df["real_pareto_layer"], errors="coerce")
        return source_df.loc[np.isclose(layer.to_numpy(dtype=float), 1.0)].copy()
    if "pareto_layer" in source_df.columns:
        layer = pd.to_numeric(source_df["pareto_layer"], errors="coerce")
        return source_df.loc[np.isclose(layer.to_numpy(dtype=float), 1.0)].copy()
    return source_df.iloc[[]].copy()


def _objective_source_df(
    *,
    context_csv_path: str | Path | None,
    details_df: pd.DataFrame,
) -> pd.DataFrame:
    context_df = _load_context_df(context_csv_path)
    return context_df if context_df is not None else details_df


def _scatter_objective_layers(
    ax: plt.Axes,
    *,
    all_df: pd.DataFrame,
    real_pareto_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
    recommended_label: str | None,
) -> None:
    ax.scatter(
        all_df["RUNTIME"],
        all_df["ACCURACY"],
        color="red",
        s=16,
        alpha=0.25,
        label="All Configurations",
        zorder=1,
    )
    if not real_pareto_df.empty:
        ax.scatter(
            real_pareto_df["RUNTIME"],
            real_pareto_df["ACCURACY"],
            color="black",
            s=64,
            label="Pareto Front Configurations",
            zorder=2,
        )
    ax.scatter(
        recommended_df["RUNTIME"],
        recommended_df["ACCURACY"],
        # color="blue",
        # edgecolors="black",
        # linewidths=0.4,
        # color="#0059fd",
        color="#1973fc",
        # color="deepskyblue",
        # edgecolors="white",
        edgecolors="#1f00d1",
        linewidths=0.4,
        s=32,
        label=recommended_label,
        zorder=3,
    )


def _finalize_plot(fig: plt.Figure, ax: plt.Axes, *, output_path: Path, title: str) -> Path:
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _format_runtime_tick(value: float, _position: int | None = None) -> str:
    if not np.isfinite(value):
        return ""
    value = float(value)
    if abs(value) >= 100.0:
        return f"{value:,.0f}"
    if abs(value) >= 1.0:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _round_runtime_tick(value: float) -> float:
    if not np.isfinite(value) or np.isclose(value, 0.0):
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    abs_value = abs(float(value))
    magnitude = 10.0 ** floor(np.log10(abs_value))
    mantissa = abs_value / magnitude
    nice_mantissa = min((1.0, 2.0, 3.0, 5.0, 7.0, 10.0), key=lambda candidate: abs(candidate - mantissa))
    return sign * nice_mantissa * magnitude


def _log1p_runtime_bounds(runtime_min: float, runtime_max: float, *, pad_fraction: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(runtime_min) or not np.isfinite(runtime_max):
        raise ValueError("Cannot build log1p runtime bounds from non-finite values.")
    if runtime_min < 0.0 or runtime_max < 0.0:
        raise ValueError("Cannot apply log1p runtime axis to negative runtime values.")
    if runtime_min > runtime_max:
        runtime_min, runtime_max = runtime_max, runtime_min
    log_min = float(np.log1p(runtime_min))
    log_max = float(np.log1p(runtime_max))
    log_span = log_max - log_min
    pad = max(log_span * pad_fraction, 0.05)
    low = max(0.0, float(np.expm1(log_min - pad)))
    high = float(np.expm1(log_max + pad))
    if low >= high:
        high = low + 1.0
    return low, high


def _log1p_runtime_ticks(x_min: float, x_max: float, *, n_ticks: int = 8) -> np.ndarray:
    if not np.isfinite(x_min) or not np.isfinite(x_max) or np.isclose(x_min, x_max):
        return np.array([x_min], dtype=float)
    log_ticks = np.linspace(np.log1p(x_min), np.log1p(x_max), int(n_ticks) + 2)[1:-1]
    rounded_ticks = [_round_runtime_tick(float(np.expm1(value))) for value in log_ticks]
    ticks: list[float] = []
    for tick in rounded_ticks:
        if tick <= x_min or tick >= x_max:
            continue
        if ticks and np.isclose(tick, ticks[-1]):
            continue
        if tick not in ticks:
            ticks.append(tick)
    if not ticks:
        ticks = [float(np.expm1(value)) for value in log_ticks]
    return np.array(ticks, dtype=float)


def _set_runtime_axis(ax: plt.Axes, all_df: pd.DataFrame, *, log1p_runtime_axis: bool) -> None:
    runtime_min = float(all_df["RUNTIME"].min())
    runtime_max = float(all_df["RUNTIME"].max())
    x_min, x_max = _expand_runtime_bounds(runtime_min, runtime_max)
    if log1p_runtime_axis:
        runtimes = pd.to_numeric(all_df["RUNTIME"], errors="coerce").to_numpy(dtype=float)
        if np.any(runtimes < 0.0):
            raise ValueError("Cannot apply log1p runtime axis to negative runtime values.")
        x_min, x_max = _log1p_runtime_bounds(runtime_min, runtime_max)
        ax.set_xscale("function", functions=(np.log1p, np.expm1))
        ax.set_xticks(_log1p_runtime_ticks(x_min, x_max))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_runtime_tick))
    ax.set_xlim(x_min, x_max)


def _plot_details_csv(
    details_csv_path: str | Path,
    *,
    output_path: str | Path,
    detector: str,
    dataset: str,
    distance_column: str,
    context_csv_path: str | Path | None = None,
    recommended_limit: int | None = None,
    selected_column: str | None = None,
    log1p_runtime_axis: bool = False,
) -> Path:
    del distance_column
    output_path = Path(output_path)
    details_df = _prepare_details_df(details_csv_path)
    all_df = _objective_source_df(context_csv_path=context_csv_path, details_df=details_df)
    real_pareto_df = _real_pareto_df(all_df)
    if selected_column is not None:
        recommended_df = _select_recommended_df(details_df, selected_column=selected_column)
    else:
        recommended_df = _select_recommended_df_legacy(details_df, recommended_limit=recommended_limit)
    _assert_recommendation_count(
        details_df=details_df,
        recommended_df=recommended_df,
        selected_column=selected_column,
    )
    prediction_label = f"Predicted Top-{_prediction_label_k()} Configurations"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _scatter_objective_layers(
        ax,
        all_df=all_df,
        real_pareto_df=real_pareto_df,
        recommended_df=recommended_df,
        recommended_label=prediction_label,
    )
    y_min, y_max = _expand_accuracy_bounds(float(all_df["ACCURACY"].min()), float(all_df["ACCURACY"].max()))
    _set_runtime_axis(ax, all_df, log1p_runtime_axis=log1p_runtime_axis)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("RUNTIME")
    ax.set_ylabel("ACCURACY")
    ax.legend(loc="lower right")

    # title = f"Detector: {detector}\nHeld-out Dataset: {dataset}"
    title = f"Detector: $\\bf{{{detector}}}$\nHeld-out Dataset: $\\bf{{{dataset}}}$"
    return _finalize_plot(fig, ax, output_path=output_path, title=title)


def plot_details_csv(
    details_csv_path: str | Path,
    *,
    output_path: str | Path,
    detector: str,
    dataset: str,
    distance_column: str,
    context_csv_path: str | Path | None = None,
    recommended_limit: int | None = None,
    selected_column: str | None = None,
) -> Path:
    return _plot_details_csv(
        details_csv_path=details_csv_path,
        output_path=output_path,
        detector=detector,
        dataset=dataset,
        distance_column=distance_column,
        context_csv_path=context_csv_path,
        recommended_limit=recommended_limit,
        selected_column=selected_column,
        log1p_runtime_axis=False,
    )


def plot_details_csv_log(
    details_csv_path: str | Path,
    *,
    output_path: str | Path,
    detector: str,
    dataset: str,
    distance_column: str,
    context_csv_path: str | Path | None = None,
    recommended_limit: int | None = None,
    selected_column: str | None = None,
) -> Path:
    return _plot_details_csv(
        details_csv_path=details_csv_path,
        output_path=output_path,
        detector=detector,
        dataset=dataset,
        distance_column=distance_column,
        context_csv_path=context_csv_path,
        recommended_limit=recommended_limit,
        selected_column=selected_column,
        log1p_runtime_axis=True,
    )
