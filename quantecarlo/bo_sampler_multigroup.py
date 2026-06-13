# quantecarlo/bo_sampler_multigroup.py
"""
Multi-group q-EI sampler.

Extends the single-group qEISampler to handle N groups, each backed by its
own Optuna study and its own DimSpec search space.

Why N groups instead of 1
--------------------------
Groups with different len(search_space) produce X matrices with different
column counts.  A single API call cannot merge them (the remote GP expects
a rectangular X).  The solution is one API call per group.  If you have
three groups you make three calls.  The ECDF (below) is what makes those
calls' results comparable.

Shared ECDF — the core idea
-----------------------------
Each group's raw trial values are collected, combined into one pool, and
passed through a single ECDF (Empirical CDF) before being sent to the
remote GP.  This rank-normalises every group's y relative to the *joint*
distribution rather than each group's private distribution.

The remote GP service also applies its own internal rank-normalisation
(_transform_y).  So y gets transformed twice.  That is intentional and
harmless: rank-transforming data that is already standard-normal is
approximately the identity (empirical CDF ≈ theoretical normal CDF, so
norm.ppf(empirical_CDF(x)) ≈ x).  Relative ordering is preserved exactly
because both transforms are monotone.

One API call per group, N groups
----------------------------------
There is no "detect whether to run once or twice" logic.  The loop is
always over all groups — two today, three or four tomorrow.  Whether the
loop body executes once or N times is purely a function of how many groups
are registered.

Global EI ranking
-----------------
The remote API in production mode returns candidates sorted by EI *within*
each group, but does not return raw EI scores.  Picks are therefore
interleaved round-robin across groups rather than globally ranked.

TODO: when the remote API is extended to return ei_scores in the standard
(non-debug) response, replace the round-robin interleave in _fill_all_caches
with a proper global sort.  All the other logic stays the same.
"""

from __future__ import annotations

import json
import threading
import urllib.request
import warnings
from collections import deque
from dataclasses import asdict
from typing import Any

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler, RandomSampler
from optuna.trial import TrialState
from scipy.stats import norm

from quantecarlo.bo_sampler import DimSpec


# ─────────────────────────────────────────────────────────────────────────────
# ECDF helper (self-contained; mirrors bo_pipeline/ecdf.py in meta-ads-demo)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_ecdf(all_scores: list[float] | np.ndarray):
    """
    Fit an ECDF on the combined pool of scores from all groups.

    Returns a callable transform(scores: array-like) -> np.ndarray that maps
    any score to its standard-normal quantile anchored to the combined pool.

    Formula (identical to the remote GP service's _transform_y):
        rank  = searchsorted(sorted_pool, score, side='right')   ∈ {1, …, n}
        u     = rank / (n + 1)
        u     = u * 0.9999 + 0.00005      # clamp away from 0 and 1
        output = norm.ppf(u)              # Gaussian targets for the GP
    """
    sorted_pool = np.sort(np.asarray(all_scores, dtype=np.float64))
    n = len(sorted_pool)

    def transform(scores: list[float] | np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        ranks = np.searchsorted(sorted_pool, s, side="right").astype(np.float64)
        u = ranks / (n + 1.0)
        u = u * 0.9999 + 0.00005
        return norm.ppf(u)

    return transform


# ─────────────────────────────────────────────────────────────────────────────
# MultiGroupqEISampler
# ─────────────────────────────────────────────────────────────────────────────

class MultiGroupqEISampler(BaseSampler):
    """
    Batch BO sampler for N groups with (potentially) different search spaces.

    This is a single BaseSampler instance shared across N Optuna studies (one
    per group).  It coordinates all groups in one place:

      • Fits one ECDF on the combined y pool so every group's GP targets are
        on the same scale and EI values are globally comparable.
      • Makes one API call per group (cannot merge groups with different
        len(search_space) into a single rectangular X matrix).
      • Fills per-group pending caches and serves suggestions round-robin.

    Setup pattern
    -------------
        sampler = MultiGroupqEISampler(
            groups=[("meta", meta_dims), ("google", google_dims)],
            api_url="https://...",
        )
        study_meta   = optuna.create_study(study_name="meta",   sampler=sampler)
        study_google = optuna.create_study(study_name="google", sampler=sampler)
        sampler.register_study("meta",   study_meta)
        sampler.register_study("google", study_google)

    The ``study_name`` must match the group name passed to ``groups=``.
    That is how ``sample_relative`` (which receives the calling study) knows
    which group it belongs to.

    Ask-tell loop
    -------------
    Ask from ALL studies before evaluating any of them.  The first ask on
    any study triggers a cross-group BO run (all groups, one API call each)
    and fills every group's cache.  Subsequent asks within the same batch
    pop from the already-filled cache without another API call.

        trials_meta   = [study_meta.ask()   for _ in range(q)]
        trials_google = [study_google.ask() for _ in range(q)]
        # evaluate all in parallel, then tell
        for t, v in zip(trials_meta,   meta_values):   study_meta.tell(t, v)
        for t, v in zip(trials_google, google_values): study_google.tell(t, v)

    Adding more groups
    ------------------
    Append to the ``groups`` list and register the new study.  Everything
    else is automatic — the inner loop in ``_fill_all_caches`` runs once per
    group, however many you have.

    Parameters
    ----------
    groups : list of (name, dims) pairs
        ``name`` must match the corresponding ``study.study_name``.
        ``dims`` is a list of DimSpec, one per search-space dimension.
        Different groups may have different len(dims).
    api_url : str
        Remote GP q-EI endpoint.
    n_startup_trials : int
        Minimum *total* complete trials (across all groups combined) before
        the GP kicks in.  Individual groups also need at least this many
        of their own trials; groups below the threshold fall back to random.
    q : int
        Suggestions requested per group per API call.
    """

    def __init__(
        self,
        groups: list[tuple[str, list[DimSpec]]],
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
        self._group_dims: dict[str, list[DimSpec]] = dict(groups)
        self._api_url = api_url
        self._n_startup_trials = n_startup_trials
        self._q = q
        self._n_candidates = n_candidates
        self._train_steps = train_steps
        self._lr = lr
        self._xi = xi
        self._mode = mode
        self._timeout = timeout
        self._independent_sampler = RandomSampler(seed=seed)

        # Per-group caches: filled in one cross-group BO run, drained one at a time.
        self._pending: dict[str, deque[dict[str, Any]]] = {
            name: deque() for name in self._group_dims
        }
        # Registered studies — needed so _fill_all_caches can read other groups'
        # completed trials when sample_relative is called for any one group.
        self._studies: dict[str, optuna.Study] = {}
        self._lock = threading.Lock()

    def register_study(self, group_name: str, study: optuna.Study) -> None:
        """
        Associate an Optuna study with a group name.

        Must be called for every group before the ask-tell loop begins.
        The study's study_name must equal group_name.
        """
        if group_name not in self._group_dims:
            raise ValueError(
                f"Unknown group {group_name!r}. "
                f"Registered groups: {list(self._group_dims)}"
            )
        if study.study_name != group_name:
            raise ValueError(
                f"study.study_name={study.study_name!r} must equal "
                f"group_name={group_name!r} so sample_relative can identify "
                f"which group it is serving."
            )
        self._studies[group_name] = study

    # ── BaseSampler interface ─────────────────────────────────────────────────

    def infer_relative_search_space(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        dims = self._group_dims.get(study.study_name, [])
        result: dict[str, BaseDistribution] = {}
        for dim in dims:
            if dim.type == "float":
                result[dim.name] = FloatDistribution(
                    dim.low, dim.high, log=dim.log, step=dim.step
                )
            elif dim.type == "int":
                result[dim.name] = IntDistribution(
                    int(dim.low), int(dim.high),
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
        group_name = study.study_name
        if group_name not in self._group_dims:
            return {}

        with self._lock:
            # Serve from this group's cache if available.
            if self._pending[group_name]:
                return self._pending[group_name].popleft()

            # Cache empty — check if we have enough total data to run the GP.
            total_complete = sum(
                len(s.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
                for s in self._studies.values()
            )
            if total_complete < self._n_startup_trials:
                return {}  # → Optuna calls sample_independent (random)

            # Trigger cross-group BO: fills caches for ALL groups at once.
            try:
                self._fill_all_caches()
            except Exception as exc:
                warnings.warn(
                    f"MultiGroupqEISampler: cross-group BO failed ({exc!r}), "
                    f"falling back to random."
                )
                return {}

            return self._pending[group_name].popleft() if self._pending[group_name] else {}

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

    # ── Cross-group BO ────────────────────────────────────────────────────────

    def _fill_all_caches(self) -> None:
        """
        Core cross-group BO step.

        Called once when any group's cache runs dry.  Fills ALL groups' caches
        in a single pass so subsequent asks on any group are served from cache.

        Steps
        -----
        1. Collect completed trials per group.
        2. Negate values (minimisation → maximisation).
        3. Fit one ECDF on the combined negated pool.
        4. For each group: transform y via shared ECDF, POST to remote GP.
        5. Interleave suggestions round-robin into per-group pending deques.

        Note on double-transformation
        ------------------------------
        The remote GP applies its own _transform_y (rank → standard-normal).
        The y we send is already standard-normal (step 3 above).  The second
        transform is approximately idempotent: empirical_CDF(standard-normal) ≈
        theoretical normal CDF, so norm.ppf(empirical_CDF(x)) ≈ x.  Signal
        is preserved; relative ordering is preserved exactly (both transforms
        are monotone).

        Note on round-robin ordering
        -----------------------------
        Production API responses don't include raw EI scores, only ranked
        candidate lists.  We interleave rather than globally sort.  To get
        true global EI ranking, extend the API to return ei_scores in the
        standard response and replace the interleave loop with a global sort.
        """
        # ── 1. Collect usable trials per group ────────────────────────────────
        trials_by_group: dict[str, list] = {}
        for name, study in self._studies.items():
            pnames = [d.name for d in self._group_dims[name]]
            complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            trials_by_group[name] = [
                t for t in complete if all(p in t.params for p in pnames)
            ]

        # ── 2 & 3. Shared ECDF on combined negated y pool ─────────────────────
        all_y_neg = [
            -float(t.value)
            for trials in trials_by_group.values()
            for t in trials
        ]
        if not all_y_neg:
            return
        ecdf = _fit_ecdf(all_y_neg)

        # ── 4. One API call per group ─────────────────────────────────────────
        suggestions_by_group: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self._group_dims
        }

        for name, trials in trials_by_group.items():
            # Skip groups that don't yet have enough of their own data.
            if len(trials) < self._n_startup_trials:
                continue

            dims = self._group_dims[name]
            pnames = [d.name for d in dims]

            X = [[float(t.params[p]) for p in pnames] for t in trials]
            y_ecdf = ecdf(
                np.array([-float(t.value) for t in trials])
            ).tolist()

            payload = {
                "X": X,
                "y": y_ecdf,
                "search_space": [asdict(d) for d in dims],
                "q": self._q,
                "n_candidates": self._n_candidates,
                "train_steps": self._train_steps,
                "lr": self._lr,
                "xi": self._xi,
                "mode": self._mode,
            }

            try:
                data = self._post(payload)
            except Exception as exc:
                warnings.warn(
                    f"MultiGroupqEISampler: API call for group {name!r} "
                    f"failed ({exc!r}). Group will fall back to random."
                )
                continue

            for candidate in data.get("candidates", []):
                params: dict[str, Any] = {}
                for i, dim in enumerate(dims):
                    val = float(candidate["x"][i])
                    if dim.type == "int":
                        val = int(round(val))  # type: ignore[assignment]
                    params[dim.name] = val
                suggestions_by_group[name].append(params)

        # ── 5. Round-robin interleave into per-group caches ───────────────────
        # Pattern: g0[0], g1[0], g2[0], g0[1], g1[1], g2[1], ...
        # Each group's top pick is served before any group's second pick.
        # TODO: replace with global EI sort once the API returns ei_scores.
        max_len = max(
            (len(s) for s in suggestions_by_group.values()), default=0
        )
        for i in range(max_len):
            for name, suggestions in suggestions_by_group.items():
                if i < len(suggestions):
                    self._pending[name].append(suggestions[i])

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
