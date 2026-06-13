# quantecarlo/bo_sampler.py
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import optuna
from optuna.distributions import (
    BaseDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler, RandomSampler
from optuna.trial import TrialState


@dataclass
class DimSpec:
    """Describes one dimension of the search space."""
    name: str
    type: str          # "float" | "int"  (categorical deferred)
    low: float
    high: float
    log: bool = False
    step: float | None = None  # grid step for int dims (default 1)


class qEISampler(BaseSampler):
    """Optuna sampler that outsources GP fitting and q-EI scoring to a remote endpoint.

    Fills a local cache with q suggestions on the first ask after the cache empties,
    then hands them out one at a time. Falls back to random sampling during startup
    and if the API call fails.

    Thread-safety: a single threading.Lock ensures only one API call fires per batch
    even when study.optimize(n_jobs=q) drives concurrent sample_relative calls.
    """

    def __init__(
        self,
        search_space: list[DimSpec],
        api_url: str = "https://info-29741--bo-gp-service-gp-suggest.modal.run",
        n_startup_trials: int = 8,
        q: int = 4,
        n_candidates: int = 512,
        train_steps: int = 60,
        lr: float = 0.1,
        xi: float = 0.01,
        mode: str = "production",
        seed: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._api_url = api_url
        self._search_space = search_space
        self._n_startup_trials = n_startup_trials
        self._q = q
        self._n_candidates = n_candidates
        self._train_steps = train_steps
        self._lr = lr
        self._xi = xi
        self._mode = mode
        self._timeout = timeout
        self._independent_sampler = RandomSampler(seed=seed)
        self._pending: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # BaseSampler interface
    # ------------------------------------------------------------------

    def infer_relative_search_space(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        # Use the explicit construction-time space rather than intersection_search_space
        # so the sampler works immediately without waiting for multiple param-overlapping
        # trials to accumulate.
        result: dict[str, BaseDistribution] = {}
        for dim in self._search_space:
            if dim.type == "float":
                result[dim.name] = FloatDistribution(
                    dim.low, dim.high, log=dim.log, step=dim.step
                )
            elif dim.type == "int":
                result[dim.name] = IntDistribution(
                    int(dim.low),
                    int(dim.high),
                    log=dim.log,
                    step=int(dim.step) if dim.step is not None else 1,
                )
        return result

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        with self._lock:
            # Serve from cache first — no API call needed.
            if self._pending:
                return self._pending.popleft()

            # Cache empty: check whether we have enough data to fit the GP.
            complete_trials = study.get_trials(
                deepcopy=False, states=(TrialState.COMPLETE,)
            )
            if len(complete_trials) < self._n_startup_trials:
                return {}  # → Optuna falls back to sample_independent (random)

            param_names = [dim.name for dim in self._search_space]
            usable = [
                t for t in complete_trials
                if all(n in t.params for n in param_names)
            ]
            if len(usable) < self._n_startup_trials:
                return {}

            # Build the observation matrices and call the remote API.
            X = [[float(t.params[n]) for n in param_names] for t in usable]
            # q-EI maximises, so negate trial values: minimisation targets become
            # maximisation targets (lower error → higher negated value → q-EI seeks it).
            y = [-float(t.value) for t in usable]

            payload = {
                "X": X,
                "y": y,
                "search_space": [asdict(dim) for dim in self._search_space],
                "q": self._q,
                "n_candidates": self._n_candidates,
                "train_steps": self._train_steps,
                "lr": self._lr,
                "xi": self._xi,
                "mode": self._mode,
            }

            try:
                data = self._post(payload)
                if self._mode == "debug" and data.get("ei_all") is not None:
                    ei_all = data["ei_all"]
                    display = [round(v, 6) if v is not None else "NaN" for v in ei_all]
                    print(f"\n[debug] ei_all ({len(ei_all)} batches): {display}")
                    valid = [v for v in ei_all if v is not None]
                    if valid:
                        print(f"[debug] max ei : {max(valid):.6f}  winning batch ei_score: {data.get('ei_scores')}")
            except Exception as exc:
                warnings.warn(
                    f"qEISampler: API call failed ({exc}), falling back to random."
                )
                return {}

            # Populate cache; snap int dims to Python int so suggest_int is happy.
            for candidate in data["candidates"]:
                params: dict[str, Any] = {}
                for i, dim in enumerate(self._search_space):
                    val = float(candidate["x"][i])
                    if dim.type == "int":
                        val = int(round(val))  # type: ignore[assignment]
                    params[dim.name] = val
                self._pending.append(params)

            return self._pending.popleft() if self._pending else {}

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
