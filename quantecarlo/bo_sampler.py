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
    n_probe_points: int = 512,
    n_candidate_batches: int | None = None,
    train_steps: int = 60,
    lr: float = 0.1,
    xi: float = 0.01,
    mode: str = "production",
    seed: int | None = None,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """suggest_fn for BatchSampler over a CONTINUOUS search space.

    direction: "minimize" or "maximize" — must match the Optuna study direction.
    BatchSampler passes raw study values; this function converts to the Modal API's
    higher-is-better convention internally.

    Use this only when search_space genuinely has no enumerable set of valid points
    (e.g. a learning rate or hidden-layer-size range) — that's why it exists at all:
    with no finite pool to select from, some finite stand-in has to be invented.
    n_probe_points is the size of that invented stand-in, nothing more.

    If you already have a fixed, enumerable collection of real candidates (a product
    catalog, an ad pool, any discrete list you can materialize), do NOT use this
    function — call quantecarlo.call_modal_api directly with your real points as
    `candidates`. Routing a real enumerable pool through modal_suggest means the GP
    never sees your actual items, only random continuous stand-ins for them, which
    then have to be snapped back to the nearest real item after the fact — an
    approximation with no purpose when the real pool was available all along.

    n_probe_points and n_candidate_batches are two different knobs, not one:
      n_probe_points      — how many random continuous points are invented from the
                            search space bounds and sent to the server as the pool
                            the GP can choose from. Meaningless once you're passing
                            call_modal_api real points directly — there is nothing
                            to invent when the real, finite set is already known.
      n_candidate_batches — how many random size-q index-combinations of that pool
                            the server scores with joint q-EI before returning the
                            best one (see GPRequest.n_batches in modal_gp_api.py;
                            same knob, renamed here to not collide with "batch"
                            meaning a q-sized round of picks elsewhere in a caller's
                            ask-tell loop). Defaults to n_probe_points for backward
                            compatibility with callers that only ever set one knob;
                            pass both explicitly to decouple them.

    Bind extra parameters with functools.partial before passing to BatchSampler:

        from functools import partial
        from quantecarlo import modal_suggest, DimSpec
        suggest = partial(modal_suggest, direction="minimize", api_url="https://...",
                           n_probe_points=1024, n_candidate_batches=256)
        sampler = BatchSampler(search_space=dims, suggest_fn=suggest, q=4)
    """
    rng = np.random.default_rng(seed)
    candidates = np.array(_sample_candidates(search_space, n_probe_points, rng), dtype=np.float32)
    # Modal API is higher-is-better; negate y for minimize studies.
    y_send = np.array([-v for v in y] if direction == "minimize" else list(y), dtype=np.float32)
    X_arr = np.array(X, dtype=np.float32)

    raw = call_modal_api(
        api_url, X_arr, y_send, candidates,
        q=q, n_batches=n_candidate_batches if n_candidate_batches is not None else n_probe_points,
        train_steps=train_steps,
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
