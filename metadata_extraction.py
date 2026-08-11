"""
Metadata extraction and variant export.

Purpose
-------
Compute dataset metadata from raw dataset prefixes, clean/prune the metadata,
and export the LODO PCA variants used by Phase 1 training.

Inputs
------
- data/raw_data/datasets/<dataset>.csv

Outputs
-------
- dataset_overview.csv
- all_meta_features.csv, cleaned_meta_features.csv, pruned_meta_features.csv
- LODO PCA 80/85/90/95 and PCA-ranked metadata artifacts

Important behavior
------------------
LODO PCA transformations are fitted without the held-out dataset. This script
does not build models or recommendation search spaces.
"""


from __future__ import annotations

from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from src.metadata_core import (
    build_lodo_pca_ranked_features,
    clean_and_prune_meta_features,
    clean_metadata_features,
    extract_from_csv,
    fit_train_only_pca_metadata,
    quiet_warnings,
)
from src.paths import ProjectPaths, get_paths_from_script
from src.utils import ensure_dir

from src.config import (
    METADATA_ENABLE_LANDMARKING,
    METADATA_ENABLE_TREE_DESCRIPTORS,
    METADATA_EXPORT_ALL_META_FEATURES,
    METADATA_EXPORT_DATASET_OVERVIEW,
    METADATA_EXPORT_LODO_PCA_VARIANTS,
    METADATA_EXPORT_PRUNED_META_FEATURES,
    METADATA_EXPORT_CLEANED_META_FEATURES,
    METADATA_FRESH_COMPUTATION,
    METADATA_N_PREFIX,
    METADATA_QUIET_WARNINGS,
    PCA_VARIANCES,
)

FRESH_COMPUTATION = METADATA_FRESH_COMPUTATION

EXPORT_DATASET_OVERVIEW = METADATA_EXPORT_DATASET_OVERVIEW
EXPORT_ALL_META_FEATURES = METADATA_EXPORT_ALL_META_FEATURES
EXPORT_CLEANED_META_FEATURES = METADATA_EXPORT_CLEANED_META_FEATURES
EXPORT_PRUNED_META_FEATURES = METADATA_EXPORT_PRUNED_META_FEATURES
EXPORT_LODO_PCA_VARIANTS = METADATA_EXPORT_LODO_PCA_VARIANTS

N_PREFIX = METADATA_N_PREFIX
ENABLE_LANDMARKING = METADATA_ENABLE_LANDMARKING
ENABLE_TREE_DESCRIPTORS = METADATA_ENABLE_TREE_DESCRIPTORS
QUIET_WARNINGS = METADATA_QUIET_WARNINGS



def _load_csv_or_fail(path, message: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{message}\nMissing file: {path}")
    return pd.read_csv(path)


def compute_all_metadata(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    datasets_dir = paths.raw_datasets_dir
    csv_files = sorted(datasets_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No dataset CSV files found in: {datasets_dir}")

    metas: List[Dict[str, float]] = []
    overview_rows: List[Dict[str, float]] = []
    failed: List[str] = []

    for csv_file in tqdm(csv_files, desc="Extracting metadata"):
        try:
            meta, ds_info = extract_from_csv(
                csv_file,
                n_prefix=N_PREFIX,
                enable_landmarking=ENABLE_LANDMARKING,
                enable_tree_descriptors=ENABLE_TREE_DESCRIPTORS,
            )
            metas.append(meta)
            overview_rows.append(ds_info)
        except Exception as exc:
            failed.append(f"{csv_file.name}: {exc}")

    df_overview = pd.DataFrame(overview_rows)
    df_all = pd.DataFrame(metas)

    if "dataset_name" in df_all.columns:
        ordered = ["dataset_name"] + [c for c in df_all.columns if c != "dataset_name"]
        df_all = df_all[ordered].copy()

    return df_overview, df_all, failed


def export_train_only_pca_variants(paths: ProjectPaths, df_pruned: pd.DataFrame) -> None:
    ensure_dir(paths.metadata_pca_dir)
    ensure_dir(paths.metadata_pca_lodo_dir)
    ensure_dir(paths.metadata_pca_ranked_dir)
    ensure_dir(paths.metadata_pca_variance_info_dir)

    dataset_names = df_pruned["dataset_name"].astype(str).tolist()
    for variance in PCA_VARIANCES:
        ensure_dir(paths.metadata_pca_lodo_variant_dir(variance))
        ensure_dir(paths.metadata_pca_ranked_lodo_variant_dir(variance))
        ensure_dir(paths.metadata_pca_variance_info_lodo_file("_placeholder", variance).parent)
        for left_out in dataset_names:
            df_pca, df_info = fit_train_only_pca_metadata(
                df_pruned,
                left_out_dataset=left_out,
                variance_target=float(variance),
            )
            df_ranked = build_lodo_pca_ranked_features(df_pruned, df_info)

            info_path = paths.metadata_pca_variance_info_lodo_file(left_out, variance)
            ranked_path = paths.metadata_pca_ranked_lodo_file(left_out, variance)
            pca_path = paths.metadata_pca_lodo_file(left_out, variance)
            df_info.to_csv(info_path, index=False)
            df_ranked.to_csv(ranked_path, index=False)
            df_pca.to_csv(pca_path, index=False)
            print(f"Saved: {info_path}  (rows={df_info.shape[0]}, cols={df_info.shape[1]})")
            print(f"Saved: {ranked_path}  (rows={df_ranked.shape[0]}, cols={df_ranked.shape[1]})")
            print(f"Saved: {pca_path}  (rows={df_pca.shape[0]}, cols={df_pca.shape[1]})")


def main() -> None:
    paths = get_paths_from_script(__file__)
    paths.ensure_core_directories()

    if QUIET_WARNINGS:
        quiet_warnings()

    overview_path = paths.dataset_overview_file
    all_path = paths.all_meta_features_file
    cleaned_path = paths.cleaned_meta_features_file
    pruned_path = paths.pruned_meta_features_file

    if FRESH_COMPUTATION:
        df_overview, df_all, failed = compute_all_metadata(paths)

        if EXPORT_DATASET_OVERVIEW:
            df_overview.to_csv(overview_path, index=False)
            print(f"Saved: {overview_path}  (rows={df_overview.shape[0]}, cols={df_overview.shape[1]})")

        if EXPORT_ALL_META_FEATURES:
            df_all.to_csv(all_path, index=False)
            print(f"Saved: {all_path}  (rows={df_all.shape[0]}, cols={df_all.shape[1]})")

        if failed:
            print("\nSome datasets failed during metadata extraction:")
            for msg in failed:
                print(" -", msg)
    else:
        df_all = _load_csv_or_fail(
            all_path,
            "FRESH_COMPUTATION is False, so all_meta_features.csv must already exist.",
        )

    if EXPORT_CLEANED_META_FEATURES:
        df_cleaned = clean_metadata_features(df_all)
        df_cleaned.to_csv(cleaned_path, index=False)
        print(f"Saved: {cleaned_path}  (rows={df_cleaned.shape[0]}, cols={df_cleaned.shape[1]})")

    else:
        df_cleaned = _load_csv_or_fail(
            cleaned_path,
            "cleaned_meta_features.csv is required because its export is disabled.",
        )

    if EXPORT_PRUNED_META_FEATURES:
        df_pruned = clean_and_prune_meta_features(df_all)
        df_pruned.to_csv(pruned_path, index=False)
        print(f"Saved: {pruned_path}  (rows={df_pruned.shape[0]}, cols={df_pruned.shape[1]})")

    else:
        df_pruned = _load_csv_or_fail(
            pruned_path,
            "pruned_meta_features.csv is required because its export is disabled.",
        )

    if EXPORT_LODO_PCA_VARIANTS:
        export_train_only_pca_variants(paths, df_pruned)


if __name__ == "__main__":
    main()
