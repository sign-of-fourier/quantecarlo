# quantecarlo/bo_sampler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quantecarlo._modal_api import call_modal_api


@dataclass
class DimSpec:
    """Describes one dimension of the search space."""
    name: str
    type: str          # "float" | "int"
    low: float
    high: float
    log: bool = False
    step: float | None = None


def modal_suggest(
    X: list[list[float]],
    y: list[float],
    search_space: list[DimSpec],
    q: int,
    *,
    direction: str = "minimize",
    api_url: str = "https://markshipman4273--bo-gp-service-gp-suggest.modal.run",
    n_candidates: int = 512,
    train_steps: int = 60,
    lr: float = 0.1,
    xi: float = 0.01,
    mode: str = "production",
    seed: int | None = None,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """suggest_fn for BatchSampler that delegates to a remote Modal GP endpoint.

    direction: "minimize" or "maximize" — must match the Optuna study direction.
    BatchSampler passes raw study values; this function converts to the Modal API's
    higher-is-better convention internally.

    Bind extra parameters with functools.partial before passing to BatchSampler:

        from functools import partial
        from quantecarlo import modal_suggest, DimSpec
        suggest = partial(modal_suggest, direction="minimize", api_url="https://...", n_candidates=1024)
        sampler = BatchSampler(search_space=dims, suggest_fn=suggest, q=4)
    """
    rng = np.random.default_rng(seed)
    candidates = np.array(_sample_candidates(search_space, n_candidates, rng), dtype=np.float32)
    # Modal API is higher-is-better; negate y for minimize studies.
    y_send = np.array([-v for v in y] if direction == "minimize" else list(y), dtype=np.float32)
    X_arr = np.array(X, dtype=np.float32)

    raw = call_modal_api(
        api_url, X_arr, y_send, candidates,
        q=q, n_batches=n_candidates, train_steps=train_steps,
        lr=lr, xi=xi, mode=mode, timeout=timeout,
    )

    results: list[dict[str, Any]] = []
    for item in raw:
        params: dict[str, Any] = {}
        for i, dim in enumerate(search_space):
            val = float(item["x"][i])
            if dim.type == "int":
                val = int(round(val))
            params[dim.name] = val
        results.append(params)
    return results


def _sample_candidates(
    dims: list[DimSpec], n: int, rng: np.random.Generator
) -> list[list[float]]:
    candidates = []
    for _ in range(n):
        point = []
        for dim in dims:
            if dim.type == "float":
                if dim.log:
                    val = float(np.exp(rng.uniform(np.log(dim.low), np.log(dim.high))))
                else:
                    val = float(rng.uniform(dim.low, dim.high))
            else:
                val = float(rng.integers(int(dim.low), int(dim.high) + 1))
            point.append(val)
        candidates.append(point)
    return candidates
