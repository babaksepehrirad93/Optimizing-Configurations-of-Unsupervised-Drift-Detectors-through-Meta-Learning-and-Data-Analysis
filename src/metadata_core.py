"""Dataset metadata computation and LODO PCA construction.

Computes stream-prefix metadata from raw datasets, removes invalid or redundant
metadata columns, and builds train-only LODO PCA and LODO PCA ranked
representations. PCA scalers and components are fitted without the held-out
dataset for each fold.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CORR_PRUNE_THRESHOLD,
    EARLYSHIFT_BATCH_SIZE,
    ENTROPY_BINS,
    GLOBAL_RANDOM_SEED,
    LANDMARK_MAX_FOLDS,
    LANDMARK_MIN_PER_CLASS_FOR_CV,
    LANDMARK_MIN_SAMPLES_TOTAL,
    LOGREG_MAX_ITER,
    METADATA_ENABLE_LANDMARKING,
    METADATA_ENABLE_TREE_DESCRIPTORS,
    METADATA_N_PREFIX,
    NEAR_CONST_EPS,
    PCA_MAX_COMPONENTS,
)



DEFAULT_N_PREFIX = METADATA_N_PREFIX
DEFAULT_EARLYSHIFT_BATCH_SIZE = EARLYSHIFT_BATCH_SIZE

ENABLE_LANDMARKING_DEFAULT = METADATA_ENABLE_LANDMARKING
ENABLE_TREE_DESCRIPTORS_DEFAULT = METADATA_ENABLE_TREE_DESCRIPTORS

RANDOM_SEED = GLOBAL_RANDOM_SEED


AGG_STAT_NAMES = ("mean", "std", "min", "q1", "median", "q3", "max")

def quiet_warnings() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="The least populated class in y has only")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")


def safe_float(x) -> float:
    if x is None:
        return float("nan")
    try:
        x = float(x)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def safe_log(x: float) -> float:
    x = safe_float(x)
    return safe_float(np.log(x)) if (np.isfinite(x) and x > 0.0) else float("nan")


def shannon_entropy_binned(x: np.ndarray, bins: int = ENTROPY_BINS) -> float:
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    if np.nanmax(x) == np.nanmin(x):
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    total = hist.sum()
    if total <= 0:
        return float("nan")
    p = hist.astype(float) / total
    p = p[p > 0]
    h = -np.sum(p * np.log2(p))
    h_norm = h / np.log2(bins)
    return float(np.clip(h_norm, 0.0, 1.0))


def class_entropy_norm(y: pd.Series) -> float:
    vc = y.value_counts(dropna=False)
    if vc.size <= 1:
        return 0.0
    p = (vc / vc.sum()).to_numpy(dtype=float)
    p = p[p > 0]
    h = -np.sum(p * np.log2(p))
    h_norm = h / np.log2(vc.size)
    return float(np.clip(h_norm, 0.0, 1.0))


def ks_statistic_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)].astype(float)
    b = b[np.isfinite(b)].astype(float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    a_sort = np.sort(a)
    b_sort = np.sort(b)
    all_vals = np.sort(np.concatenate([a_sort, b_sort]))
    cdf_a = np.searchsorted(a_sort, all_vals, side="right") / a_sort.size
    cdf_b = np.searchsorted(b_sort, all_vals, side="right") / b_sort.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def wasserstein_approx_1d(a: np.ndarray, b: np.ndarray, q: int = 50) -> float:
    a = a[np.isfinite(a)].astype(float)
    b = b[np.isfinite(b)].astype(float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    qs = np.linspace(0.0, 1.0, q)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


def kl_divergence_binned_1d(a: np.ndarray, b: np.ndarray, bins: int = 20, eps: float = 1e-12) -> float:
    a = a[np.isfinite(a)].astype(float)
    b = b[np.isfinite(b)].astype(float)
    if a.size < 2 or b.size < 2:
        return float("nan")

    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)

    p = pa.astype(float) + eps
    q = pb.astype(float) + eps
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


def hellinger_distance_binned_1d(a: np.ndarray, b: np.ndarray, bins: int = 20, eps: float = 1e-12) -> float:
    a = a[np.isfinite(a)].astype(float)
    b = b[np.isfinite(b)].astype(float)
    if a.size < 2 or b.size < 2:
        return float("nan")

    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)

    p = pa.astype(float) + eps
    q = pb.astype(float) + eps
    p /= p.sum()
    q /= q.sum()

    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def agg_stats(vals: np.ndarray, prefix: str) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    out = {}
    out[f"{prefix}_mean"] = safe_float(np.nanmean(vals))
    out[f"{prefix}_std"] = safe_float(np.nanstd(vals))
    out[f"{prefix}_min"] = safe_float(np.nanmin(vals))
    out[f"{prefix}_q1"] = safe_float(np.nanpercentile(vals, 25))
    out[f"{prefix}_median"] = safe_float(np.nanmedian(vals))
    out[f"{prefix}_q3"] = safe_float(np.nanpercentile(vals, 75))
    out[f"{prefix}_max"] = safe_float(np.nanmax(vals))
    return out


def nan_agg_stats(prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{stat}": safe_float(float("nan")) for stat in AGG_STAT_NAMES}


def aggregate_transition_values(vals: np.ndarray, prefix: str) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0 or not np.isfinite(vals).any():
        return nan_agg_stats(prefix)
    return agg_stats(vals, prefix)


def make_complete_ordered_batches(n_rows: int, batch_size: int) -> list[slice]:
    try:
        n_rows = int(n_rows)
        batch_size = int(batch_size)
    except Exception:
        return []
    if n_rows <= 0 or batch_size <= 0:
        return []

    n_complete = n_rows // batch_size
    return [
        slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        for batch_idx in range(n_complete)
    ]


def compute_batch_earlyshift_features(
    X: np.ndarray,
    batch_size: int = DEFAULT_EARLYSHIFT_BATCH_SIZE,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    metric_funcs = {
        "ks": lambda a, b: ks_statistic_1d(a, b),
        "wasserstein": lambda a, b: wasserstein_approx_1d(a, b, q=50),
        "kl": lambda a, b: kl_divergence_binned_1d(a, b, bins=20),
        "hellinger": lambda a, b: hellinger_distance_binned_1d(a, b, bins=20),
    }
    metric_feature_values = {name: [] for name in metric_funcs}

    batches = make_complete_ordered_batches(X.shape[0], batch_size)
    if len(batches) >= 2:
        for prev_slice, next_slice in zip(batches[:-1], batches[1:]):
            a = X[prev_slice, :]
            b = X[next_slice, :]

            if X.shape[1] > 0:
                for metric_name, metric_func in metric_funcs.items():
                    feature_values = np.array(
                        [metric_func(a[:, j], b[:, j]) for j in range(X.shape[1])],
                        dtype=float,
                    )
                    valid_values = feature_values[np.isfinite(feature_values)]
                    metric_feature_values[metric_name].extend(valid_values.tolist())

    for metric_name, values in metric_feature_values.items():
        out.update(
            aggregate_transition_values(
                np.asarray(values, dtype=float),
                f"earlyshift_batch__{metric_name}",
            )
        )
    return out


def choose_cv_folds(y_codes: np.ndarray, max_folds: int = LANDMARK_MAX_FOLDS) -> int:
    counts = np.bincount(y_codes)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0
    min_count = int(counts.min())
    return int(min(max_folds, min_count))


def cv_acc(model, X: np.ndarray, y: np.ndarray) -> float:
    n = X.shape[0]
    if n < LANDMARK_MIN_SAMPLES_TOTAL:
        return float("nan")
    if np.unique(y).size < 2:
        return float("nan")

    n_splits = choose_cv_folds(y, max_folds=LANDMARK_MAX_FOLDS)
    if n_splits < LANDMARK_MIN_PER_CLASS_FOR_CV:
        return float("nan")

    try:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        return safe_float(np.mean(scores))
    except Exception:
        return float("nan")


def augment_minority_classes_for_landmarking(
    X: np.ndarray,
    y: np.ndarray,
    target_min_per_class: int = LANDMARK_MIN_PER_CLASS_FOR_CV,
    noise_scale: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)

    feat_std = np.nanstd(X, axis=0)
    feat_std = np.where(np.isfinite(feat_std), feat_std, 0.0)

    X_new = [X]
    y_new = [y]

    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        if cnt >= target_min_per_class:
            continue

        need = int(target_min_per_class - cnt)
        idx_cls = np.flatnonzero(y == cls)
        base_idx = rng.choice(idx_cls, size=need, replace=True)
        base = X[base_idx].copy()
        jitter = rng.normal(loc=0.0, scale=noise_scale, size=base.shape) * feat_std
        synth = base + jitter

        X_new.append(synth)
        y_new.append(np.full(need, cls, dtype=y.dtype))

    return np.vstack(X_new), np.concatenate(y_new)


# ============================================================
# Core extraction
# ============================================================
def extract_meta_features(
    df_prefix: pd.DataFrame,
    dataset_name: str,
    *,
    enable_landmarking: bool = ENABLE_LANDMARKING_DEFAULT,
    enable_tree_descriptors: bool = ENABLE_TREE_DESCRIPTORS_DEFAULT,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    meta: Dict[str, float] = {"dataset_name": dataset_name}
    ds_info: Dict[str, float] = {"dataset_name": dataset_name}

    X_all = df_prefix.iloc[:, :-1]
    y_raw = df_prefix.iloc[:, -1]

    X_df = X_all.select_dtypes(include=[np.number]).copy()
    y = y_raw.astype(str)

    ds_info["n_instances_used"] = safe_float(len(df_prefix))
    ds_info["n_features"] = safe_float(X_all.shape[1])
    ds_info["n_numeric_features"] = safe_float(X_df.shape[1])

    vc = y.value_counts(dropna=False)
    n_classes = float(vc.size)
    min_cc = float(vc.min()) if vc.size else float("nan")
    max_cc = float(vc.max()) if vc.size else float("nan")

    ds_info["n_classes"] = safe_float(n_classes)
    ds_info["min_class_count"] = safe_float(min_cc)
    ds_info["max_class_count"] = safe_float(max_cc)

    meta["n_features"] = safe_float(X_all.shape[1])
    meta["log_n_features"] = safe_log(meta["n_features"])

    meta["n_classes"] = safe_float(n_classes)
    meta["min_class_count"] = safe_float(min_cc)
    meta["max_class_count"] = safe_float(max_cc)
    meta["class_entropy_norm"] = safe_float(class_entropy_norm(y))
    if np.isfinite(min_cc) and np.isfinite(max_cc) and min_cc > 0:
        meta["imbalance_ratio"] = safe_float(max_cc / min_cc)
    else:
        meta["imbalance_ratio"] = float("nan")

    class_probs = (vc / vc.sum()).to_numpy(dtype=float) if vc.sum() > 0 else np.array([], dtype=float)
    meta.update(agg_stats(class_probs, "class_prob"))

    y_codes = pd.Categorical(y).codes
    n_splits_orig = choose_cv_folds(y_codes, max_folds=LANDMARK_MAX_FOLDS)
    ds_info["cv_folds_used"] = safe_float(n_splits_orig)
    ds_info["landmarking_possible"] = 1.0 if n_splits_orig >= LANDMARK_MIN_PER_CLASS_FOR_CV else 0.0

    X = X_df.to_numpy(dtype=float)
    n, d = X.shape

    nan_fracs = np.mean(~np.isfinite(X), axis=0) if d > 0 else np.array([], dtype=float)
    meta.update(agg_stats(nan_fracs, "feat_nan_frac"))

    zero_fracs = []
    for j in range(d):
        col = X[:, j]
        m = np.isfinite(col)
        zero_fracs.append(np.mean(col[m] == 0.0) if m.sum() else np.nan)
    meta.update(agg_stats(np.array(zero_fracs, dtype=float), "feat_zero_frac"))

    col_std = np.nanstd(X, axis=0) if d > 0 else np.array([], dtype=float)

    col_mean = np.nanmean(X, axis=0) if d > 0 else np.array([], dtype=float)
    col_min = np.nanmin(X, axis=0) if d > 0 else np.array([], dtype=float)
    col_max = np.nanmax(X, axis=0) if d > 0 else np.array([], dtype=float)
    meta.update(agg_stats(col_mean, "feat_mean"))
    meta.update(agg_stats(col_std, "feat_std"))
    meta.update(agg_stats(col_min, "feat_min"))
    meta.update(agg_stats(col_max, "feat_max"))

    X_df_num = pd.DataFrame(X, columns=X_df.columns)
    q25 = X_df_num.quantile(0.25, axis=0, interpolation="linear").to_numpy(dtype=float) if d > 0 else np.array([], dtype=float)
    q75 = X_df_num.quantile(0.75, axis=0, interpolation="linear").to_numpy(dtype=float) if d > 0 else np.array([], dtype=float)
    iqr = q75 - q25
    meta.update(agg_stats(iqr, "feat_iqr"))

    col_skew = X_df_num.skew(axis=0, skipna=True).to_numpy(dtype=float) if d > 0 else np.array([], dtype=float)
    col_kurt = X_df_num.kurt(axis=0, skipna=True).to_numpy(dtype=float) if d > 0 else np.array([], dtype=float)
    meta.update(agg_stats(col_skew, "feat_skew"))
    meta.update(agg_stats(col_kurt, "feat_kurt"))

    ent = np.array([shannon_entropy_binned(X[:, j], bins=ENTROPY_BINS) for j in range(d)], dtype=float)
    meta.update(agg_stats(ent, "feat_entropy_binned"))

    X_corr = X.copy()
    stds = X_corr.std(axis=0) if d > 0 else np.array([], dtype=float)
    keep = stds > 1e-12 if d > 0 else np.array([], dtype=bool)
    X_corr = X_corr[:, keep] if d > 0 else X_corr

    if X_corr.shape[1] >= 2:
        corr = np.corrcoef(X_corr, rowvar=False)
        iu = np.triu_indices(corr.shape[0], k=1)
        abs_vals = np.abs(corr[iu])
        abs_vals = abs_vals[np.isfinite(abs_vals)]

        meta["mean_abs_corr"] = safe_float(abs_vals.mean()) if abs_vals.size else float("nan")
        meta["max_abs_corr"] = safe_float(abs_vals.max()) if abs_vals.size else float("nan")
    else:
        meta["mean_abs_corr"] = float("nan")
        meta["max_abs_corr"] = float("nan")

    Xp = X_corr.astype(float)
    if Xp.shape[1] >= 1:
        Xp = (Xp - Xp.mean(axis=0)) / (Xp.std(axis=0) + 1e-12)

    try:
        if Xp.shape[1] >= 1 and Xp.shape[0] >= 2:
            p1 = PCA(n_components=1, svd_solver="randomized", random_state=RANDOM_SEED)
            p1.fit(Xp)
            meta["x_pca_first_component_ratio"] = safe_float(p1.explained_variance_ratio_[0])
            pc1_scores = p1.transform(Xp).ravel()
            s = pd.Series(pc1_scores)
            meta["x_pca_skewness_first_pc"] = safe_float(s.skew(skipna=True))
            meta["x_pca_kurtosis_first_pc"] = safe_float(s.kurt(skipna=True))
        else:
            meta["x_pca_first_component_ratio"] = float("nan")
            meta["x_pca_skewness_first_pc"] = float("nan")
            meta["x_pca_kurtosis_first_pc"] = float("nan")

        max_comp = min(Xp.shape[1], 50, Xp.shape[0])
        if max_comp >= 2:
            pfull = PCA(n_components=max_comp, svd_solver="randomized", random_state=RANDOM_SEED)
            pfull.fit(Xp)
            cumsum = np.cumsum(pfull.explained_variance_ratio_)
            meta["x_pca_k_for_95pct"] = safe_float(int(np.searchsorted(cumsum, 0.95) + 1))
        else:
            meta["x_pca_k_for_95pct"] = float("nan")
    except Exception:
        meta["x_pca_first_component_ratio"] = float("nan")
        meta["x_pca_k_for_95pct"] = float("nan")
        meta["x_pca_skewness_first_pc"] = float("nan")
        meta["x_pca_kurtosis_first_pc"] = float("nan")

    try:
        mi = mutual_info_classif(X_corr, y_codes, random_state=RANDOM_SEED, discrete_features=False)
        meta.update(agg_stats(mi, "mi_feat_to_label"))
    except Exception:
        for k in ["mean", "std", "min", "q1", "median", "q3", "max"]:
            meta[f"mi_feat_to_label_{k}"] = float("nan")

    if enable_landmarking:
        X_lm = X_corr
        y_lm = y_codes

        if ds_info["landmarking_possible"] < 0.5:
            X_lm, y_lm = augment_minority_classes_for_landmarking(
                X_corr,
                y_codes,
                target_min_per_class=LANDMARK_MIN_PER_CLASS_FOR_CV,
                noise_scale=1e-3,
            )

        meta["landmark__gnb_acc"] = cv_acc(GaussianNB(), X_lm, y_lm)
        meta["landmark__1nn_acc"] = cv_acc(KNeighborsClassifier(n_neighbors=1), X_lm, y_lm)
        meta["landmark__dt_stump_acc"] = cv_acc(
            DecisionTreeClassifier(max_depth=1, random_state=RANDOM_SEED), X_lm, y_lm
        )
        meta["landmark__dt_depth5_acc"] = cv_acc(
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED), X_lm, y_lm
        )

        logreg_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=LOGREG_MAX_ITER, n_jobs=1, solver="lbfgs", multi_class="auto"),
        )
        meta["landmark__logreg_acc"] = cv_acc(logreg_model, X_lm, y_lm)
    else:
        meta["landmark__gnb_acc"] = float("nan")
        meta["landmark__1nn_acc"] = float("nan")
        meta["landmark__dt_stump_acc"] = float("nan")
        meta["landmark__dt_depth5_acc"] = float("nan")
        meta["landmark__logreg_acc"] = float("nan")

    if enable_tree_descriptors:
        try:
            tree = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED)
            tree.fit(X_corr, y_codes)
            meta["tree__n_nodes"] = safe_float(tree.tree_.node_count)
            meta["tree__max_depth"] = safe_float(tree.get_depth())
            meta["tree__n_leaves"] = safe_float(tree.get_n_leaves())
        except Exception:
            meta["tree__n_nodes"] = float("nan")
            meta["tree__max_depth"] = float("nan")
            meta["tree__n_leaves"] = float("nan")
    else:
        meta["tree__n_nodes"] = float("nan")
        meta["tree__max_depth"] = float("nan")
        meta["tree__n_leaves"] = float("nan")

    k_clust = int(max(2, min(vc.size if vc.size > 0 else 2, 10)))
    try:
        km = KMeans(n_clusters=k_clust, random_state=RANDOM_SEED, n_init="auto")
        clusters = km.fit_predict(Xp if Xp.shape[1] >= 2 else X_corr)

        if len(np.unique(clusters)) >= 2 and n > len(np.unique(clusters)):
            meta["cluster__silhouette"] = safe_float(
                silhouette_score(Xp if Xp.shape[1] >= 2 else X_corr, clusters)
            )
        else:
            meta["cluster__silhouette"] = float("nan")
    except Exception:
        meta["cluster__silhouette"] = float("nan")

    meta.update(
        compute_batch_earlyshift_features(
            X,
            batch_size=DEFAULT_EARLYSHIFT_BATCH_SIZE,
        )
    )

    return meta, ds_info


def extract_from_csv(
    csv_path: Path,
    n_prefix: int = DEFAULT_N_PREFIX,
    *,
    enable_landmarking: bool = ENABLE_LANDMARKING_DEFAULT,
    enable_tree_descriptors: bool = ENABLE_TREE_DESCRIPTORS_DEFAULT,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    df_prefix = pd.read_csv(csv_path, nrows=n_prefix)
    return extract_meta_features(
        df_prefix,
        dataset_name=csv_path.stem,
        enable_landmarking=enable_landmarking,
        enable_tree_descriptors=enable_tree_descriptors,
    )


def clean_and_prune_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned metadata after NaN, constant, duplicate, and correlation pruning."""
    df2 = df.copy()
    id_col = "dataset_name"
    feat_cols = [c for c in df2.columns if c != id_col]

    any_nan = [c for c in feat_cols if df2[c].isna().any()]
    df2.drop(columns=any_nan, inplace=True)
    feat_cols = [c for c in df2.columns if c != id_col]

    drop_const = []
    for c in feat_cols:
        s = df2[c].dropna()
        if s.size <= 1:
            drop_const.append(c)
            continue
        if float(np.nanvar(s.to_numpy(dtype=float))) <= NEAR_CONST_EPS:
            drop_const.append(c)
    df2.drop(columns=drop_const, inplace=True)
    feat_cols = [c for c in df2.columns if c != id_col]

    if feat_cols:
        dup_mask = df2[feat_cols].T.duplicated(keep="first")
        dup_cols = df2[feat_cols].columns[dup_mask].tolist()
        df2.drop(columns=dup_cols, inplace=True)
        feat_cols = [c for c in df2.columns if c != id_col]

    if len(feat_cols) >= 2:
        pearson = df2[feat_cols].corr(method="pearson").abs()
        spearman = df2[feat_cols].corr(method="spearman").abs()
        corr = np.maximum(pearson, spearman)
        
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            high = upper[col][upper[col] > CORR_PRUNE_THRESHOLD].index.tolist()
            for row in high:
                if row in to_drop or col in to_drop:
                    continue
                to_drop.add(col)

        df2.drop(columns=list(to_drop), inplace=True)

    return df2


def clean_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned metadata after NaN, constant, and duplicate-column removal."""
    df2 = df.copy()
    id_col = "dataset_name"
    feat_cols = [c for c in df2.columns if c != id_col]

    any_nan = [c for c in feat_cols if df2[c].isna().any()]
    df2.drop(columns=any_nan, inplace=True)
    feat_cols = [c for c in df2.columns if c != id_col]

    drop_const = []
    for c in feat_cols:
        s = df2[c].dropna()
        if s.size <= 1:
            drop_const.append(c)
            continue
        if float(np.nanvar(s.to_numpy(dtype=float))) <= NEAR_CONST_EPS:
            drop_const.append(c)
    df2.drop(columns=drop_const, inplace=True)
    feat_cols = [c for c in df2.columns if c != id_col]

    if feat_cols:
        dup_mask = df2[feat_cols].T.duplicated(keep="first")
        dup_cols = df2[feat_cols].columns[dup_mask].tolist()
        df2.drop(columns=dup_cols, inplace=True)

    return df2


def fit_train_only_pca_metadata(
    df_all: pd.DataFrame,
    left_out_dataset: str,
    *,
    variance_target: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit scaler and PCA without the held-out dataset and transform all datasets.

    The returned metadata rows include all datasets, while PCA parameters are
    estimated only from the LODO training datasets.
    """
    id_col = "dataset_name"
    train_mask = (df_all[id_col].values != left_out_dataset)

    feat_cols = [c for c in df_all.columns if c != id_col]
    X_all = df_all[feat_cols].to_numpy(dtype=float)
    X_train = X_all[train_mask]

    if np.isnan(X_all).any():
        raise RuntimeError("NaNs found in meta-features after cleaning, but PCA has no imputation now.")

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = scaler.fit_transform(X_train)

    max_comp = min(PCA_MAX_COMPONENTS, X_train_std.shape[0], X_train_std.shape[1])
    if max_comp < 1:
        raise RuntimeError("Not enough meta-features after cleaning to run PCA in this fold.")

    pca_full = PCA(n_components=max_comp, random_state=RANDOM_SEED)
    pca_full.fit(X_train_std)

    evr = pca_full.explained_variance_ratio_
    cumsum = np.cumsum(evr)
    target = float(variance_target)
    if not np.isfinite(target) or target <= 0.0 or target > 1.0:
        raise ValueError(f"variance_target must be finite and in (0, 1]. Got {variance_target!r}.")
    k = int(np.searchsorted(cumsum, target) + 1)
    k = max(1, min(k, max_comp))

    X_all_std = scaler.transform(X_all)
    Z_all = pca_full.transform(X_all_std)[:, :k]

    pc_cols = [f"PC{i+1}" for i in range(k)]
    df_pcs_all = pd.DataFrame(Z_all, columns=pc_cols)
    df_pcs_all.insert(0, id_col, df_all[id_col].values)

    loadings = pca_full.components_[:k, :]
    top_rows = []
    for i in range(k):
        vec = loadings[i, :]
        idx = np.argsort(np.abs(vec))[::-1][:10]
        for rank, j in enumerate(idx, start=1):
            top_rows.append({
                "PC": f"PC{i+1}",
                "pc_rank_by_variance": i + 1,
                "explained_variance_ratio": float(evr[i]),
                "cumulative_explained_variance": float(np.sum(evr[: i + 1])),
                "top_loading_rank": rank,
                "meta_feature": feat_cols[j],
                "loading": float(vec[j]),
                "abs_loading": float(abs(vec[j])),
            })

    df_info = pd.DataFrame(top_rows)
    return df_pcs_all, df_info


def build_lodo_pca_ranked_features(
    df_all: pd.DataFrame,
    df_info: pd.DataFrame,
    *,
    use_all_retained_pcs: bool = True,
) -> pd.DataFrame:
    """
    Rank original metadata features by PCA importance using weighted loadings.

    Feature score:
        sum(abs(loading) * explained_variance_ratio) across retained PCs

    This produces a fold-specific ranking of the original pruned metadata
    features, ordered from most structurally important to least important in the
    LODO PCA space.
    """
    id_col = "dataset_name"
    if id_col not in df_all.columns:
        raise ValueError("df_all must contain 'dataset_name'.")
    if df_info.empty:
        return df_all[[id_col]].copy()

    working = df_info.copy()
    working["pc_rank_by_variance"] = pd.to_numeric(working["pc_rank_by_variance"], errors="coerce")
    working["explained_variance_ratio"] = pd.to_numeric(working["explained_variance_ratio"], errors="coerce")
    working["abs_loading"] = pd.to_numeric(working["abs_loading"], errors="coerce")

    if not use_all_retained_pcs:
        working = working.loc[working["pc_rank_by_variance"] <= 3].copy()

    working["weighted_loading_score"] = working["explained_variance_ratio"] * working["abs_loading"]
    feature_scores = (
        working.groupby("meta_feature", dropna=False)["weighted_loading_score"]
        .sum()
        .sort_values(ascending=False)
    )

    ranked_features = [f for f in feature_scores.index.astype(str).tolist() if f in df_all.columns and f != id_col]
    remaining_features = [c for c in df_all.columns if c != id_col and c not in ranked_features]
    ordered_features = ranked_features + remaining_features

    return df_all[[id_col] + ordered_features].copy()
