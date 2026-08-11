"""Learning-target and Pareto-layer utilities.

Constructs scalar single-target formulations from transformed objectives and
provides Pareto-front/layer helpers used by Phase 1 evaluation and Phase 2
recommendation. Transformed accuracy and transformed runtime are both
higher-is-better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REGIONAL_TARGETS = {"tchebycheff", "pbi", "apd"}


@dataclass
class DatasetTargetArtifacts:
    transformed_accuracy: np.ndarray
    transformed_runtime: np.ndarray
    is_pareto: np.ndarray
    active_distance: np.ndarray
    front_points: np.ndarray
    active_reward_factor: np.ndarray | None = None


def validate_distance_method(distance_method: str) -> str:
    value = str(distance_method).strip()
    if value not in {
        "euc_dist",
        "mod_dist",
        "pareto_score",
        "pareto_loss",
        "pareto_rank",
        "tchebycheff",
        "pbi",
        "apd",
    }:
        raise ValueError(
            "SINGLE_TARGET_FORMULATION must be 'euc_dist', 'mod_dist', 'pareto_score', "
            "'pareto_loss', 'pareto_rank', 'tchebycheff', 'pbi', or 'apd'."
        )
    return value


def pareto_front_mask(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    mask = np.ones(n, dtype=bool)
    if n == 0:
        return mask
    for i in range(n):
        if not mask[i]:
            continue
        dominates = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        dominates[i] = False
        if dominates.any():
            mask[i] = False
    return mask


def pareto_layer_rank(points: np.ndarray) -> np.ndarray:
    """
    Compute raw Pareto layer index for each point.

    points[:, 0] = transformed_accuracy, higher is better
    points[:, 1] = transformed_runtime, higher is better

    Layer 1 = Pareto front.
    Layer 2 = Pareto front after removing layer 1.
    ...
    Lower layer means better configuration.

    Return raw layer numbers, not normalized values. Normalization to [0, 1]
    is handled by compute_scalar_target_by_method().
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)

    layers = np.full(n, np.nan, dtype=float)
    if n == 0:
        return layers

    finite = np.isfinite(pts).all(axis=1)
    finite_idx = np.flatnonzero(finite)

    if len(finite_idx) > 0:
        finite_pts = pts[finite_idx]
        order = np.lexsort((-finite_pts[:, 1], -finite_pts[:, 0]))
        sorted_idx = finite_idx[order]
        sorted_pts = pts[sorted_idx]

        runtime_values = np.unique(finite_pts[:, 1])[::-1]
        runtime_pos = {float(value): pos + 1 for pos, value in enumerate(runtime_values)}
        tree_size = len(runtime_values)

        def _fenwick_update(tree: np.ndarray, idx: int, value: float) -> None:
            while idx <= tree_size:
                if value > tree[idx]:
                    tree[idx] = value
                idx += idx & -idx

        def _fenwick_query(tree: np.ndarray, idx: int) -> float:
            best = 0.0
            while idx > 0:
                if tree[idx] > best:
                    best = float(tree[idx])
                idx -= idx & -idx
            return best

        global_tree = np.zeros(tree_size + 1, dtype=float)
        max_layer = 0.0
        pos = 0

        while pos < len(sorted_idx):
            acc_value = sorted_pts[pos, 0]
            acc_end = pos + 1
            while acc_end < len(sorted_idx) and sorted_pts[acc_end, 0] == acc_value:
                acc_end += 1

            same_accuracy_tree = np.zeros(tree_size + 1, dtype=float)
            pending_global_updates: list[tuple[int, float]] = []
            rt_pos = pos

            while rt_pos < acc_end:
                runtime_value = sorted_pts[rt_pos, 1]
                rt_end = rt_pos + 1
                while rt_end < acc_end and sorted_pts[rt_end, 1] == runtime_value:
                    rt_end += 1

                tree_idx = runtime_pos[float(runtime_value)]
                best_dominating_layer = max(
                    _fenwick_query(global_tree, tree_idx),
                    _fenwick_query(same_accuracy_tree, tree_idx),
                )
                rank = best_dominating_layer + 1.0
                layers[sorted_idx[rt_pos:rt_end]] = rank
                max_layer = max(max_layer, rank)

                _fenwick_update(same_accuracy_tree, tree_idx, rank)
                pending_global_updates.append((tree_idx, rank))
                rt_pos = rt_end

            for tree_idx, rank in pending_global_updates:
                _fenwick_update(global_tree, tree_idx, rank)

            pos = acc_end

    if (~finite).any():
        max_finite_layer = float(np.nanmax(layers[finite])) if finite.any() else 0.0
        layers[~finite] = max_finite_layer + 1.0

    return layers


def _resolve_manual_weights(lambda_value: float | None) -> tuple[float, float]:
    lam = 0.5 if lambda_value is None else float(lambda_value)
    lam = min(max(lam, 0.0), 1.0)
    return lam, 1.0 - lam


def euclidean_distance_to_front(
    points: np.ndarray,
    front_points: np.ndarray,
    *,
    weights: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    front = np.asarray(front_points, dtype=float)
    if len(front) == 0:
        return np.full(len(pts), np.nan, dtype=float)
    diff = pts[:, None, :] - front[None, :, :]
    weighted_sq = (diff * diff) * np.asarray(weights, dtype=float)
    dist = np.sqrt(np.sum(weighted_sq, axis=2))
    return np.min(dist, axis=1)


def ishibuchi_distance_to_front(
    points: np.ndarray,
    front_points: np.ndarray,
    *,
    weights: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    front = np.asarray(front_points, dtype=float)
    if len(front) == 0:
        return np.full(len(pts), np.nan, dtype=float)
    weighted_shortfalls = []
    for front_point in front:
        shortfall = np.maximum(front_point - pts, 0.0)
        weighted_sq = (shortfall * shortfall) * np.asarray(weights, dtype=float)
        weighted_shortfalls.append(np.sqrt(np.sum(weighted_sq, axis=1)))
    return np.min(np.column_stack(weighted_shortfalls), axis=1)



def novel_pareto_score(
    points: np.ndarray,
    front_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    front = np.asarray(front_points, dtype=float)
    if len(front) == 0:
        nan = np.full(len(pts), np.nan, dtype=float)
        return nan, nan.copy()

    out = np.full(len(pts), np.nan, dtype=float)
    reward_out = np.full(len(pts), np.nan, dtype=float)
    tol = 1e-9
    lowest_acc_idx = int(np.argmin(front[:, 0]))
    lowest_acc_front = front[lowest_acc_idx : lowest_acc_idx + 1]
    edge_fallback_score = ishibuchi_distance_to_front(pts, lowest_acc_front, weights=(2.0, 1.0))

    for idx, point in enumerate(pts):
        exact_match = np.isclose(front[:, 0], point[0], atol=tol, rtol=0.0) & np.isclose(front[:, 1], point[1], atol=tol, rtol=0.0)
        if exact_match.any():
            out[idx] = 0.0
            reward_out[idx] = 0.0
            continue

        allowed = front[front[:, 0] <= (point[0] + tol)]
        if len(allowed) > 0:
            best_idx = int(np.argmax(allowed[:, 0]))
            chosen_front = allowed[best_idx]
            rt_gap = float(chosen_front[1] - point[1])
            acc_gap = float(point[0] - chosen_front[0])
            acc_factor = 0.5 * float(np.sqrt(acc_gap))
            # acc_factor = float(np.sqrt(acc_gap))
            out[idx] = rt_gap * (1.0 - acc_factor)
            reward_out[idx] = acc_factor
        else:
            out[idx] = float(edge_fallback_score[idx])
            reward_out[idx] = 0.0

    return out, reward_out


def pareto_loss(
    points: np.ndarray,
    front_points: np.ndarray,
) -> np.ndarray:
    """
    Compute Pareto-reference loss using one dominating Pareto reference.

    Assumptions
    -----------
    points[:, 0] = transformed_accuracy  (higher is better)
    points[:, 1] = transformed_runtime   (higher is better; lower raw runtime)

    For each config x:
    - If x is exactly a Pareto point, loss = 0.
    - Otherwise, find Pareto points that are equal/better in BOTH objectives:
        A_p >= A_x and R_p >= R_x.
    - From those candidates, select the closest "first" Pareto reference:
        the Pareto point with the lowest accuracy that is still >= A_x.
      On a clean 2D Pareto front, this should also correspond to the closest
      dominating Pareto point on the frontier.
    - Use the same reference for both losses:
        accuracy_loss = A_ref - A_x
        runtime_loss  = R_ref - R_x
    - Return raw losses. Normalization is handled later by
      compute_scalar_target_by_method().
    """
    pts = np.asarray(points, dtype=float)
    front = np.asarray(front_points, dtype=float)

    if len(pts) == 0:
        return np.asarray([], dtype=float)

    if len(front) == 0:
        return np.zeros(len(pts), dtype=float)

    front = front[np.isfinite(front).all(axis=1)]

    if len(front) == 0:
        return np.zeros(len(pts), dtype=float)

    raw = np.full(len(pts), np.nan, dtype=float)
    tol = 1e-9

    for idx, point in enumerate(pts):
        ax = float(point[0])
        rx = float(point[1])

        if not np.isfinite(ax) or not np.isfinite(rx):
            continue

        # Pareto points get exactly zero loss.
        exact_match = (
            np.isclose(front[:, 0], ax, atol=tol, rtol=0.0)
            & np.isclose(front[:, 1], rx, atol=tol, rtol=0.0)
        )
        if exact_match.any():
            raw[idx] = 0.0
            continue

        # Find Pareto references that dominate x in transformed space.
        # Both objectives are higher-is-better.
        ref_candidates = front[
            (front[:, 0] >= (ax - tol))
            & (front[:, 1] >= (rx - tol))
        ]

        if len(ref_candidates) == 0:
            raise ValueError(
                "No dominating Pareto reference found for non-Pareto point. "
                f"idx={idx}, point=(A={ax}, R={rx}), "
                f"front_min_A={float(np.nanmin(front[:, 0]))}, "
                f"front_max_A={float(np.nanmax(front[:, 0]))}, "
                f"front_min_R={float(np.nanmin(front[:, 1]))}, "
                f"front_max_R={float(np.nanmax(front[:, 1]))}"
            )

        # Select the first/closest dominating Pareto point:
        # lowest accuracy among references with A_ref >= A_x.
        #
        # Tie-breaker: if multiple references have almost the same accuracy,
        # choose the one with highest transformed runtime.
        order = np.lexsort((-ref_candidates[:, 1], ref_candidates[:, 0]))
        ref = ref_candidates[int(order[0])]

        accuracy_loss = float(ref[0]) - ax
        runtime_loss = float(ref[1]) - rx

        if accuracy_loss < -tol or runtime_loss < -tol:
            raise ValueError(
                "Negative component loss detected. "
                f"idx={idx}, point=(A={ax}, R={rx}), "
                f"ref=(A={float(ref[0])}, R={float(ref[1])}), "
                f"accuracy_loss={accuracy_loss}, runtime_loss={runtime_loss}"
            )

        raw[idx] = 0.75 * accuracy_loss + 0.25 * runtime_loss

    mask = np.isfinite(raw)
    if not mask.any():
        return np.zeros(len(pts), dtype=float)

    out = np.zeros(len(pts), dtype=float)
    out[mask] = raw[mask]
    return out





def tchebycheff_loss(
    points: np.ndarray,
    *,
    accuracy_weight: float = 0.5,
    ideal_point: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    w_acc = float(np.clip(accuracy_weight, 0.0, 1.0))
    w_rt = 1.0 - w_acc
    ideal = np.asarray(ideal_point, dtype=float)

    shortfall = np.maximum(ideal[None, :] - pts, 0.0)
    weighted = shortfall * np.asarray([w_acc, w_rt], dtype=float)
    return np.max(weighted, axis=1)


def _pbi_loss_direction_from_accuracy_weight(accuracy_weight: float) -> np.ndarray:
    """
    Convert an accuracy-importance weight into a PBI reference direction
    in minimization-loss space.

    Points are internally converted to:
        [accuracy_loss, runtime_loss]

    If accuracy_weight is high, accuracy_loss should be small while more
    runtime_loss is allowed, so the direction moves toward the runtime-loss axis.
    """
    w_acc_importance = float(np.clip(accuracy_weight, 0.0, 1.0))
    return np.asarray([1.0 - w_acc_importance, w_acc_importance], dtype=float)


def _normalized_loss_direction_from_accuracy_weight(accuracy_weight: float, eps: float = 1e-12) -> np.ndarray:
    direction = _pbi_loss_direction_from_accuracy_weight(accuracy_weight)
    return direction / (np.linalg.norm(direction) + eps)


def pbi_loss(
    points: np.ndarray,
    *,
    accuracy_weight: float = 0.5,
    ideal_point: tuple[float, float] = (1.0, 1.0),
    theta: float = 5.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Penalty Boundary Intersection scalarization for this thesis pipeline.

    Input points are in transformed higher-is-better objective space:
        points[:, 0] = transformed_accuracy
        points[:, 1] = transformed_runtime

    PBI is computed in minimization-loss space:
        loss = max(ideal_point - points, 0)

    The returned value is a lower-is-better scalar loss:
        PBI = d1 + theta * d2
    """
    pts = np.asarray(points, dtype=float)
    ideal = np.asarray(ideal_point, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pbi_loss expects points with shape (n_samples, 2).")

    losses = np.maximum(ideal[None, :] - pts, 0.0)

    direction_norm = _normalized_loss_direction_from_accuracy_weight(accuracy_weight, eps=eps)

    d1 = np.abs(losses @ direction_norm)
    projection = np.outer(d1, direction_norm)
    d2 = np.linalg.norm(losses - projection, axis=1)

    out = d1 + float(theta) * d2

    invalid = ~np.isfinite(losses).all(axis=1)
    if invalid.any():
        out[invalid] = np.nan

    return out


def _apd_loss_direction_from_accuracy_weight(accuracy_weight: float) -> np.ndarray:
    """
    Convert an accuracy-importance weight into an APD reference direction
    in minimization-loss space.

    Points are internally converted to:
        [accuracy_loss, runtime_loss]

    If accuracy_weight is high, accuracy_loss should be small while more
    runtime_loss is allowed, so the direction moves toward the runtime-loss axis.
    """
    w_acc_importance = float(np.clip(accuracy_weight, 0.0, 1.0))
    return np.asarray([1.0 - w_acc_importance, w_acc_importance], dtype=float)


def _angle_between_vectors(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = (np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)
    cos_value = float(np.dot(a, b) / denom)
    cos_value = float(np.clip(cos_value, -1.0, 1.0))
    return float(np.arccos(cos_value))


def apd_gamma_from_weight_set(
    accuracy_weights: list[float] | tuple[float, ...],
    *,
    eps: float = 1e-12,
) -> dict[float, float]:
    weights = [float(weight) for weight in accuracy_weights]
    if len(weights) < 2:
        raise ValueError("APD regional gamma requires at least two reference directions.")
    directions = {
        weight: _normalized_loss_direction_from_accuracy_weight(weight, eps=eps)
        for weight in weights
    }
    out: dict[float, float] = {}
    for weight, direction in directions.items():
        angles = [
            _angle_between_vectors(direction, other_direction, eps=eps)
            for other_weight, other_direction in directions.items()
            if not np.isclose(other_weight, weight, rtol=0.0, atol=eps)
        ]
        if not angles:
            raise ValueError("APD regional gamma could not find a neighboring reference direction.")
        out[weight] = float(max(min(angles), eps))
    return out


def apd_loss(
    points: np.ndarray,
    *,
    accuracy_weight: float = 0.5,
    ideal_point: tuple[float, float] = (1.0, 1.0),
    alpha: float = 2.0,
    eval_ratio: float = 1.0,
    gamma: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Angle Penalized Distance scalarization for this thesis pipeline.

    Input points are in transformed higher-is-better objective space:
        points[:, 0] = transformed_accuracy
        points[:, 1] = transformed_runtime

    APD is computed in minimization-loss space:
        loss = max(ideal_point - points, 0)

    Offline APD form:
        APD = (1 + M * (eval_ratio ** alpha) * angle / gamma) * ||loss||

    The returned value is a lower-is-better scalar loss.
    """
    pts = np.asarray(points, dtype=float)
    ideal = np.asarray(ideal_point, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("apd_loss expects points with shape (n_samples, 2).")

    losses = np.maximum(ideal[None, :] - pts, 0.0)

    direction_norm = _normalized_loss_direction_from_accuracy_weight(accuracy_weight, eps=eps)

    distances = np.linalg.norm(losses, axis=1)
    loss_norms = distances + eps

    cos_values = (losses @ direction_norm) / loss_norms
    cos_values = np.clip(cos_values, -1.0, 1.0)
    angles = np.arccos(cos_values)
    angles[distances <= eps] = 0.0

    if gamma is None:
        gamma = float(np.pi / 4.0)

    n_objectives = losses.shape[1]
    safe_eval_ratio = float(np.clip(eval_ratio, 0.0, 1.0))
    penalty = n_objectives * (safe_eval_ratio ** float(alpha)) * (angles / (float(gamma) + eps))

    out = (1.0 + penalty) * distances

    invalid = ~np.isfinite(losses).all(axis=1)
    if invalid.any():
        out[invalid] = np.nan

    return out



def _distance_with_contributions(
    points: np.ndarray,
    front_points: np.ndarray,
    *,
    distance_method: str,
    weights: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    front = np.asarray(front_points, dtype=float)
    n = len(pts)
    if len(front) == 0:
        nan = np.full(n, np.nan, dtype=float)
        return nan, nan.copy(), nan.copy()

    weight_vec = np.asarray(weights, dtype=float)
    all_dist = []
    all_contrib_acc = []
    all_contrib_rt = []

    for front_point in front:
        if distance_method == "euc_dist":
            diff = pts - front_point
        elif distance_method == "mod_dist":
            diff = np.maximum(front_point - pts, 0.0)
        else:
            raise ValueError("_distance_with_contributions does not support pareto_score.")
        contrib_sq = (diff * diff) * weight_vec
        dist = np.sqrt(np.sum(contrib_sq, axis=1))
        total = contrib_sq.sum(axis=1)
        share_acc = np.full(n, np.nan, dtype=float)
        share_rt = np.full(n, np.nan, dtype=float)
        valid = total > 0
        share_acc[valid] = contrib_sq[valid, 0] / total[valid]
        share_rt[valid] = contrib_sq[valid, 1] / total[valid]
        all_dist.append(dist)
        all_contrib_acc.append(share_acc)
        all_contrib_rt.append(share_rt)

    dist_mat = np.column_stack(all_dist)
    argmin = np.argmin(dist_mat, axis=1)
    row_idx = np.arange(n)
    min_dist = dist_mat[row_idx, argmin]
    contrib_acc = np.column_stack(all_contrib_acc)[row_idx, argmin]
    contrib_rt = np.column_stack(all_contrib_rt)[row_idx, argmin]
    return min_dist, contrib_acc, contrib_rt


def _renormalize_unit_interval(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(arr)
    if not mask.any():
        return arr
    lo = float(np.nanmin(arr[mask]))
    hi = float(np.nanmax(arr[mask]))
    den = hi - lo
    if den <= 0:
        arr[mask] = 0.0
    else:
        arr[mask] = (arr[mask] - lo) / den
    return arr


def ordered_preference_regions(
    region_names: tuple[str, ...] | list[str],
    region_accuracy_weights: dict[str, float],
) -> list[tuple[str, float, float]]:
    names = [str(name) for name in region_names]
    if len(names) != 5 or len(set(names)) != 5:
        raise ValueError("Exactly five unique preference-region names are required.")
    if set(names) != set(region_accuracy_weights):
        raise ValueError("Preference-region names and accuracy-weight keys must match exactly.")

    rows: list[tuple[str, float, float]] = []
    for name in names:
        weight = float(region_accuracy_weights[name])
        if not np.isfinite(weight) or weight <= 0.0 or weight >= 1.0:
            raise ValueError("Preference-region accuracy weights must be finite and strictly between 0 and 1.")
        rows.append((name, weight, 1.0 - weight))

    weights = [row[1] for row in rows]
    if len(set(weights)) != 5:
        raise ValueError("Preference-region accuracy weights must be exactly five unique values.")
    return sorted(rows, key=lambda row: (row[1], row[0]))


def compute_scalar_target_by_method(
    points: np.ndarray,
    front_points: np.ndarray,
    *,
    method: str,
    lambda_value: float,
    scalarization_accuracy_weight: float,
    scalarization_ideal_point: tuple[float, float],
    pbi_theta: float,
    apd_alpha: float,
    apd_eval_ratio: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    manual_weight_accuracy, manual_weight_runtime = _resolve_manual_weights(lambda_value)
    reward_factor = None

    if method == "pareto_score":
        target, reward_factor = novel_pareto_score(points, front_points)

    elif method in {"euc_dist", "mod_dist"}:
        target, _, _ = _distance_with_contributions(
            points,
            front_points,
            distance_method=method,
            weights=(manual_weight_accuracy, manual_weight_runtime),
        )

    elif method == "tchebycheff":
        target = tchebycheff_loss(
            points,
            accuracy_weight=scalarization_accuracy_weight,
            ideal_point=scalarization_ideal_point,
        )

    elif method == "pareto_loss":
        target = pareto_loss(points, front_points)

    elif method == "pareto_rank":
        target = pareto_layer_rank(points)

    elif method == "pbi":
        target = pbi_loss(
            points,
            accuracy_weight=scalarization_accuracy_weight,
            ideal_point=scalarization_ideal_point,
            theta=pbi_theta,
        )

    elif method == "apd":
        target = apd_loss(
            points,
            accuracy_weight=scalarization_accuracy_weight,
            ideal_point=scalarization_ideal_point,
            alpha=apd_alpha,
            eval_ratio=apd_eval_ratio,
        )

    else:
        raise ValueError(f"Unknown scalar target method: {method}")

    return _renormalize_unit_interval(target), reward_factor


def compute_dataset_targets(
    transformed_accuracy: np.ndarray,
    transformed_runtime: np.ndarray,
    *,
    distance_method: str,
    lambda_value: float,
    scalarization_accuracy_weight: float = 0.5,
    scalarization_ideal_point: tuple[float, float] = (1.0, 1.0),
    pbi_theta: float = 5.0,
    apd_alpha: float = 2.0,
    apd_eval_ratio: float = 1.0,
) -> DatasetTargetArtifacts:
    """
    Compute per-row target artifacts for one processed benchmark dataset.

    The active scalar target is built from transformed higher-is-better
    accuracy/runtime objectives without changing the underlying transformed
    objectives.
    """
    acc = np.asarray(transformed_accuracy, dtype=float)
    rt = np.asarray(transformed_runtime, dtype=float)
    points = np.column_stack([acc, rt])
    is_pareto = pareto_front_mask(points)
    front = points[is_pareto]
    active_distance, active_reward_factor = compute_scalar_target_by_method(
        points,
        front,
        method=distance_method,
        lambda_value=lambda_value,
        scalarization_accuracy_weight=scalarization_accuracy_weight,
        scalarization_ideal_point=scalarization_ideal_point,
        pbi_theta=pbi_theta,
        apd_alpha=apd_alpha,
        apd_eval_ratio=apd_eval_ratio,
    )

    return DatasetTargetArtifacts(
        transformed_accuracy=acc,
        transformed_runtime=rt,
        is_pareto=is_pareto.astype(int),
        active_distance=active_distance,
        front_points=front,
        active_reward_factor=active_reward_factor,
    )


def compute_preference_region_targets(
    transformed_accuracy: np.ndarray,
    transformed_runtime: np.ndarray,
    *,
    base_method: str,
    lambda_value: float,
    region_names: tuple[str, ...],
    region_accuracy_weights: dict[str, float],
    scalarization_ideal_point: tuple[float, float],
    pbi_theta: float,
    apd_alpha: float,
    apd_eval_ratio: float,
) -> dict[str, np.ndarray]:
    """
    Compute region-specific scalar targets for Tch, PBI, or APD.

    Each region uses its configured accuracy weight; runtime weight is derived
    as one minus the accuracy weight.
    """
    if base_method not in REGIONAL_TARGETS:
        raise ValueError(
            "compute_preference_region_targets supports only 'tchebycheff', 'pbi', and 'apd'."
        )

    acc = np.asarray(transformed_accuracy, dtype=float)
    rt = np.asarray(transformed_runtime, dtype=float)
    points = np.column_stack([acc, rt])

    out: dict[str, np.ndarray] = {}
    ordered_regions = ordered_preference_regions(region_names, region_accuracy_weights)
    apd_gammas = apd_gamma_from_weight_set([row[1] for row in ordered_regions]) if base_method == "apd" else {}

    for region_name, w_acc, _ in ordered_regions:

        if base_method == "tchebycheff":
            target = tchebycheff_loss(
                points,
                accuracy_weight=w_acc,
                ideal_point=scalarization_ideal_point,
            )

        elif base_method == "pbi":
            target = pbi_loss(
                points,
                accuracy_weight=w_acc,
                ideal_point=scalarization_ideal_point,
                theta=pbi_theta,
            )

        elif base_method == "apd":
            target = apd_loss(
                points,
                accuracy_weight=w_acc,
                ideal_point=scalarization_ideal_point,
                alpha=apd_alpha,
                eval_ratio=apd_eval_ratio,
                gamma=apd_gammas[w_acc],
            )
        else:
            raise AssertionError("Unreachable regional target branch.")

        out[f"{base_method}_{region_name}"] = _renormalize_unit_interval(target)

    return out
