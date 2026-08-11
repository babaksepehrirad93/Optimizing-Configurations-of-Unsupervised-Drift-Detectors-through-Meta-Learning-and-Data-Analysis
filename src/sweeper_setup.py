"""Shared sweep setup interface for pipeline_runner.py and child scripts.

Standalone stage scripts default to src.config. pipeline_runner.py single mode
also uses src.config directly and passes no setup-specific arguments. Only
pipeline_runner.py sweep mode passes the hidden setup arguments defined here.
This module exists solely to keep that sweep-only interface consistent between
the runner and child scripts.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from src.config import (
    PCA_LODO_VARIANCE,
    PCA_VARIANCES,
    SINGLE_TARGET_FORMULATION,
    SUPPORTED_SINGLE_TARGET_FORMULATIONS,
    TRAIN_METADATA_VARIANT,
    TRAIN_TARGET_MODE,
    TRAIN_USE_METADATA,
)


PIPELINE_PCA_VARIANCE_OVERRIDE_ENV = "PIPELINE_PCA_VARIANCE_OVERRIDE"
SUPPORTED_METADATA_VARIANTS = {"cfg", "pruned", "lodo_pca", "lodo_pca_ranked"}
PCA_METADATA_VARIANTS = {"lodo_pca", "lodo_pca_ranked"}


@dataclass(frozen=True)
class SweepSetup:
    target_mode: str
    single_target_formulation: str
    use_metadata: bool
    metadata_variant: str
    pca_variance: float | None
    has_pipeline_overrides: bool


def add_pipeline_setup_args(parser: argparse.ArgumentParser) -> None:
    """Add hidden sweep-only setup arguments to a child-stage parser."""
    parser.add_argument("--pipeline-target-mode", choices=("single", "separate"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pipeline-target-formulation", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pipeline-metadata-variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pipeline-pca-variance", type=float, default=None, help=argparse.SUPPRESS)


def _has_pipeline_overrides(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None) is not None
        for name in (
            "pipeline_target_mode",
            "pipeline_target_formulation",
            "pipeline_metadata_variant",
            "pipeline_pca_variance",
        )
    )


def _validate_pca_variance(
    value: float | None,
    *,
    metadata_variant: str,
    require_explicit: bool,
) -> float | None:
    if metadata_variant not in PCA_METADATA_VARIANTS:
        if value is not None:
            raise ValueError(f"PCA variance is only valid for {sorted(PCA_METADATA_VARIANTS)} metadata setups.")
        return None
    if value is None:
        if require_explicit:
            raise ValueError(f"Metadata setup {metadata_variant!r} requires --pipeline-pca-variance in sweep mode.")
        value = float(PCA_LODO_VARIANCE)
    allowed = [float(candidate) for candidate in PCA_VARIANCES]
    if not any(abs(float(value) - candidate) <= 1e-12 for candidate in allowed):
        raise ValueError(f"PCA variance must be one of {allowed}. Got {value!r}.")
    return float(value)


def resolve_pipeline_setup(
    args: argparse.Namespace,
    *,
    default_target_mode: str = TRAIN_TARGET_MODE,
    default_single_target_formulation: str = SINGLE_TARGET_FORMULATION,
    default_use_metadata: bool = TRAIN_USE_METADATA,
    default_metadata_variant: str = TRAIN_METADATA_VARIANT,
) -> SweepSetup:
    """Resolve hidden sweep args when present, otherwise return src.config values."""
    has_overrides = _has_pipeline_overrides(args)
    target_mode = str(args.pipeline_target_mode or default_target_mode).strip().lower()
    if target_mode not in {"single", "separate"}:
        raise ValueError(f"target mode must be 'single' or 'separate'. Got {target_mode!r}.")

    single_target = str(args.pipeline_target_formulation or default_single_target_formulation).strip()
    if target_mode == "single" and single_target not in SUPPORTED_SINGLE_TARGET_FORMULATIONS:
        raise ValueError(
            f"Unknown single-target formulation {single_target!r}. "
            f"Supported: {list(SUPPORTED_SINGLE_TARGET_FORMULATIONS)}."
        )

    if args.pipeline_metadata_variant is None:
        use_metadata = bool(default_use_metadata)
        metadata_variant = str(default_metadata_variant).strip().lower()
        if not use_metadata:
            metadata_variant = "cfg"
    else:
        metadata_variant = str(args.pipeline_metadata_variant).strip().lower()
        use_metadata = metadata_variant != "cfg"

    if metadata_variant not in SUPPORTED_METADATA_VARIANTS:
        raise ValueError(f"Unknown metadata setup {metadata_variant!r}. Supported: {sorted(SUPPORTED_METADATA_VARIANTS)}.")

    pca_variance = _validate_pca_variance(
        args.pipeline_pca_variance,
        metadata_variant=metadata_variant,
        require_explicit=has_overrides and metadata_variant in PCA_METADATA_VARIANTS,
    )
    if has_overrides and pca_variance is not None:
        os.environ[PIPELINE_PCA_VARIANCE_OVERRIDE_ENV] = f"{pca_variance:.2f}"
    else:
        os.environ.pop(PIPELINE_PCA_VARIANCE_OVERRIDE_ENV, None)

    return SweepSetup(
        target_mode=target_mode,
        single_target_formulation=single_target,
        use_metadata=use_metadata,
        metadata_variant=metadata_variant,
        pca_variance=pca_variance,
        has_pipeline_overrides=has_overrides,
    )
