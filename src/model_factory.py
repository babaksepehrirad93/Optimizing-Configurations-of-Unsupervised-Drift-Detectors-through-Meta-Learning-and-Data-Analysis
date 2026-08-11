"""Extra Trees model-construction helpers.

Builds the configured sklearn regressor used by Phase 1 training. Feature
preprocessing is assembled in `train_models.py`; this module only resolves
model-family validation and estimator parameters.
"""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.ensemble import ExtraTreesRegressor

from src.config import (
    DEFAULT_REGRESSOR_MODEL_PARAMS,
    SUPPORTED_MODEL_FAMILIES,
)



def _normalize_family(model_family: str) -> str:
    family = str(model_family).strip().upper()
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(
            f"Unsupported MODEL_FAMILY='{model_family}'. Supported: {SUPPORTED_MODEL_FAMILIES}"
        )
    return family


def get_default_regressor_params(model_family: str, *, random_state: int) -> dict[str, Any]:
    """Return configured default regressor parameters with the requested seed."""
    family = _normalize_family(model_family)
    params = dict(DEFAULT_REGRESSOR_MODEL_PARAMS[family])
    params["random_state"] = random_state
    return params


def merge_model_params(
    model_family: str,
    *,
    random_state: int,
    override_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    params = get_default_regressor_params(model_family, random_state=random_state)
    if override_params:
        params.update(dict(override_params))
    return params


def build_regressor(
    model_family: str,
    *,
    random_state: int = 42,
    params: Mapping[str, Any] | None = None,
):
    """Build one Extra Trees regressor instance for Phase 1 model training."""
    family = _normalize_family(model_family)
    merged_params = merge_model_params(
        family,
        random_state=random_state,
        override_params=params,
    )

    if family == "ET":
        return ExtraTreesRegressor(**merged_params)

    raise AssertionError("Unreachable model family branch.")


def model_family_tag(model_family: str) -> str:
    family = _normalize_family(model_family)
    return family
