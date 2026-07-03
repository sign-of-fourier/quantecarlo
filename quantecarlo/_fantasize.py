"""Sequential kriging (fantasize) batch suggestion.

Self-contained batch BO that requires no remote service.  Fits a simple
RBF GP on current observations, picks the highest-EI candidate, fantasizes
its outcome as the GP posterior mean, refits, and repeats q times.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm


def fantasize_suggest(
    X: list[list[float]],
    y: list[float],
    search_space: list,
    q: int,
    direction: str = "minimize",
    n_candidates: int = 512,
    noise: float = 1e-3,
    xi: float = 0.01,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Return q candidates via sequential kriging.

    Each candidate is chosen by Expected Improvement under a fresh GP fit.
    After each pick the selected point is added to the training set with its
    GP posterior mean as the fantasized observation, so subsequent picks
    account for it and spread across the space.

    Parameters
    ----------
    X:
        Completed-trial parameter vectors, shape (n, d).
    y:
        Raw objective values. Pass in the same convention as the Optuna study
        direction — lower is better for "minimize", higher is better for "maximize".
    search_space:
        DimSpec-like objects with .name, .type, .low, .high, .log, .step.
    q:
        Number of candidates to return.
    direction:
        "minimize" or "maximize" — must match the Optuna study direction.
    n_candidates:
        Random candidates evaluated per GP call.
    noise:
        GP observation noise variance.
    xi:
        EI exploration bonus.
    seed:
        Random seed for the candidate pool.
    """
    rng = np.random.default_rng(seed)
    d = len(search_space)

    def _to_unit(vals: np.ndarray, i: int) -> np.ndarray:
        dim = search_space[i]
        if dim.log:
            lo, hi = np.log(float(dim.low)), np.log(float(dim.high))
            return (np.log(vals) - lo) / (hi - lo)
        return (vals - float(dim.low)) / (float(dim.high) - float(dim.low))

    def _from_unit(u: float, i: int) -> float:
        dim = search_space[i]
        if dim.log:
            lo, hi = np.log(float(dim.low)), np.log(float(dim.high))
            return float(np.exp(lo + u * (hi - lo)))
        return float(dim.low) + u * (float(dim.high) - float(dim.low))

    # Transform observations to unit hypercube
    X_arr = np.array(X, dtype=float)
    X_unit = np.column_stack([_to_unit(X_arr[:, i], i) for i in range(d)])

    # GP always works higher-is-better internally; negate for minimize studies.
    y_arr = np.array(y, dtype=float)
    if direction == "minimize":
        y_arr = -y_arr
    y_std = y_arr.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y_arr - y_arr.mean()) / y_std

    # Random candidate pool in unit hypercube
    cands = rng.random((n_candidates, d))

    selected: list[dict[str, Any]] = []
    X_aug = X_unit.copy()
    y_aug = y_norm.copy()

    for _ in range(q):
        # Median heuristic for RBF length scale
        if len(X_aug) >= 2:
            sq_dists = np.sum((X_aug[:, None] - X_aug[None]) ** 2, axis=-1)
            pos = sq_dists[sq_dists > 0]
            ls = float(np.sqrt(np.median(pos) / 2.0)) if pos.size else 1.0
        else:
            ls = 1.0
        ls = max(ls, 1e-4)

        def _rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            diff = A[:, None, :] - B[None, :, :]
            return np.exp(-0.5 * np.sum(diff ** 2, axis=-1) / ls ** 2)

        # Cholesky-based GP posterior
        K = _rbf(X_aug, X_aug) + (noise + 1e-6) * np.eye(len(X_aug))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 0.1 * np.eye(len(K)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_aug))

        k_s = _rbf(cands, X_aug)       # (n_cands, n_obs)
        mu = k_s @ alpha
        v = np.linalg.solve(L, k_s.T)  # (n_obs, n_cands)
        var = np.maximum(1.0 - np.sum(v ** 2, axis=0), 1e-9)
        sigma = np.sqrt(var)

        # Expected Improvement (maximise — y_aug is already higher-is-better)
        best_y = y_aug.max()
        z = (mu - best_y - xi) / sigma
        ei = (mu - best_y - xi) * norm.cdf(z) + sigma * norm.pdf(z)

        idx = int(np.argmax(ei))
        best_unit = cands[idx]

        # Decode back to original parameter space
        params: dict[str, Any] = {}
        for i, dim in enumerate(search_space):
            val = _from_unit(float(best_unit[i]), i)
            if dim.type == "int":
                params[dim.name] = int(round(np.clip(val, dim.low, dim.high)))
            else:
                params[dim.name] = float(np.clip(val, dim.low, dim.high))
        selected.append(params)

        # Fantasize: extend training set with posterior mean as pseudo-observation
        X_aug = np.vstack([X_aug, best_unit])
        y_aug = np.append(y_aug, mu[idx])

        # Remove selected candidate to avoid re-selecting it
        cands = np.delete(cands, idx, axis=0)

    return selected
