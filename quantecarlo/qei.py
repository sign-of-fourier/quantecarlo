"""Direct q-EI interface — batch Bayesian optimization without Optuna.

QEIClient wraps the remote GP service for callers running their own
ask-tell loop. You hold the data; each call is self-contained:

    X           points you have already evaluated, shape (n_obs, n_dims)
    y           their scores, shape (n_obs,)
    candidates  the pool to choose from, shape (n_cands, n_dims)
    q           how many to pick

The server fits a GP to (X, y), scores size-q subsets of `candidates` by
joint Expected Improvement, and returns the best subset as indices into
`candidates`. Nothing is stored between calls; nothing is fitted client-side.

    from quantecarlo import QEIClient
    client = QEIClient(api_url)
    picks = client.suggest(X, y, candidates, q=4, direction="maximize")
    for p in picks:
        candidates[p["index"]]      # a real member of your pool

Continuous search space with no enumerable pool? Invent one:

    from quantecarlo import DimSpec, sample_candidates
    cands = sample_candidates([DimSpec("lr", "float", 1e-4, 1e-1, log=True)], n=512)
    picks = client.suggest(X, y, cands, q=4)

Already have a surrogate of your own? Skip the fit and use only the batch
selection -- send the posterior instead of the data:

    mu, cov = my_gp.predict(candidates, return_cov=True)
    picks = client.select(mu, cov, best_y=my_incumbent, q=4)
    picks["indices"]                # q indices into candidates

Every method is a thin pass-through to the call_modal_api* functions in
quantecarlo._modal_api (suggest*) or call_select_api in
quantecarlo._select_api (select), which remain the single source of truth
for the wire contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from quantecarlo._modal_api import (
    _TUNING_FIELDS,
    call_modal_api,
    call_modal_api_composite,
    call_modal_api_multioutput,
)
from quantecarlo._select_api import _SELECT_TUNING_FIELDS, call_select_api

from quantecarlo.bo_sampler import DEFAULT_API_URL

DEFAULT_SELECT_URL = "https://info-29741--bo-gp-service-qei-select.modal.run"


def _as_higher_is_better(y, direction: str) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float32)
    if direction == "maximize":
        return y_arr
    if direction == "minimize":
        return -y_arr
    raise ValueError(f'direction must be "maximize" or "minimize", got {direction!r}')


class QEIClient:
    """Bind the endpoint and per-call defaults once; call suggest() in your loop.

    api_url:     GP service endpoint (suggest, suggest_multioutput, suggest_composite).
    select_url:  acquisition-only endpoint (select).
    timeout:     HTTP timeout in seconds.
    train_steps: server-side GP fitting budget (iterations). suggest() only;
                 suggest_multioutput / suggest_composite ignore it.
    lr:          server-side GP fitting step size. suggest() only, as above.
    xi:          accepted for compatibility; the service ignores it.
    mode:        "production" (default) or "debug". In debug mode every call
                 fills `self.last_diagnostics` with the response's extra keys.
    dtype:       float width select() packs the posterior in, "float64"
                 (default) or "float32". See quantecarlo._select_api for why
                 the default is float64 and when to switch.
    **tuning:    server-side tuning fields, forwarded verbatim on every call
                 they apply to; unknown names raise TypeError. suggest* take
                 n_prefilter, ei_direct_max; select takes ei_budget,
                 mpi_bytes, pi_floor, seed; orthant_mode, order, gh_nodes,
                 gh_nodes_corr, dup_corr go to both. See quantecarlo._modal_api and
                 quantecarlo._select_api for what each one does.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        *,
        select_url: str = DEFAULT_SELECT_URL,
        timeout: float = 120.0,
        train_steps: int = 60,
        lr: float = 0.1,
        xi: float = 0.01,
        mode: str = "production",
        dtype: str = "float64",
        **tuning: Any,
    ):
        self.api_url = api_url
        self.select_url = select_url
        self.timeout = timeout
        self.train_steps = train_steps
        self.lr = lr
        self.xi = xi
        self.mode = mode
        self.dtype = dtype
        unknown = set(tuning) - (_TUNING_FIELDS | _SELECT_TUNING_FIELDS)
        if unknown:
            raise TypeError(
                f"unknown tuning field(s) {sorted(unknown)}; accepted: "
                f"{sorted(_TUNING_FIELDS | _SELECT_TUNING_FIELDS)}"
            )
        self.tuning = tuning
        self.last_diagnostics: dict[str, Any] = {}

    def _kw(self, n_batches: int) -> dict[str, Any]:
        return dict(
            n_batches=n_batches, train_steps=self.train_steps, lr=self.lr,
            xi=self.xi, mode=self.mode, timeout=self.timeout,
            diagnostics=self.last_diagnostics,
            **{k: v for k, v in self.tuning.items() if k in _TUNING_FIELDS},
        )

    def select(
        self,
        mu,
        cov,
        best_y: float,
        q: int,
        *,
        dtype: str | None = None,
        **tuning: Any,
    ) -> dict[str, Any]:
        """Pick q candidates by joint q-EI from a posterior you computed.

        mu:      posterior mean at each candidate, shape (n,). Higher = better.
        cov:     posterior covariance over the candidates, shape (n, n).
        best_y:  the incumbent, on mu's scale. Your rule; the service only
                 compares mu against it.  Scale: improvement is computed on
                 exp of the posterior (q-EI on a lognormal objective, as
                 suggest() does after rank-normalising y), so mu, cov and
                 best_y must be normal-scale -- rank-normal, PIT, or the log
                 of a lognormal target.  `quantecarlo.rank_normal` is the
                 transform suggest() uses.  |best_y| > 5 is answered with a
                 warning (logged, and in the returned "warnings").
        q:       how many to pick.
        dtype:   overrides the client's dtype for this call.
        **tuning: per-call select fields (ei_budget, mpi_bytes, pi_floor,
                 seed, orthant_*); override the client-level ones.

        Returns a dict: "indices" (q ints into mu), "qei", "regime" ("exact"
        | "screen" | "sample"), "n_cands", "n_sampled", "n_batches",
        "warnings" (list of str or None). Fewer
        than q indices only when mu has fewer than q entries. In debug mode
        `self.last_diagnostics` is filled as for suggest().
        """
        fields = {k: v for k, v in self.tuning.items() if k in _SELECT_TUNING_FIELDS}
        fields.update(tuning)
        return call_select_api(
            self.select_url, np.asarray(mu, dtype=np.float64), np.asarray(cov, dtype=np.float64),
            best_y, q=q, dtype=self.dtype if dtype is None else dtype,
            mode=self.mode, timeout=self.timeout, diagnostics=self.last_diagnostics,
            **fields,
        )

    def suggest(
        self,
        X,
        y,
        candidates,
        q: int,
        *,
        direction: str = "maximize",
        n_batches: int = 512,
    ) -> list[dict[str, Any]]:
        """Pick q points from `candidates` by joint q-EI.

        X:          evaluated points, shape (n_obs, n_dims). Same columns as candidates.
        y:          their scores, shape (n_obs,). Raw values; no scaling needed.
        candidates: the pool to choose from, shape (n_cands, n_dims).
        q:          number of points to return.
        direction:  "maximize" (default) or "minimize" — which way y is better.
        n_batches:  how many batches survive the server's screening pass and
                    are scored by joint q-EI. The screen only runs when
                    n_prefilter * q > ei_direct_max (q >= 5 with the server
                    defaults); below that every drawn batch is scored and
                    this is ignored. The width of the search is n_prefilter.

        Returns a list of q dicts: "index" (int, into candidates), "x" (the
        candidate vector), "mu" and "sigma" (GP posterior mean / std, on an
        internal scale -- comparable across candidates in one call, not to
        y). Fewer than q only when candidates has fewer than q rows.
        """
        return call_modal_api(
            self.api_url,
            np.asarray(X, dtype=np.float32),
            _as_higher_is_better(y, direction),
            np.asarray(candidates, dtype=np.float32),
            q=q, **self._kw(n_batches),
        )

    def suggest_multioutput(
        self,
        X,
        y,
        candidates,
        d_train,
        d_cands,
        q: int,
        *,
        rho: float = 0.5,
        direction: str = "maximize",
        n_batches: int = 512,
    ) -> list[dict[str, Any]]:
        """suggest() for candidates from two related sources sharing one model.

        d_train / d_cands: int array, output index (0 or 1) per row of X / candidates.
                           Only two outputs are supported.
        rho:               correlation between the two outputs, in (-1, 1).
        Everything else as suggest(), except train_steps / lr have no effect here.
        """
        return call_modal_api_multioutput(
            self.api_url,
            np.asarray(X, dtype=np.float32),
            _as_higher_is_better(y, direction),
            np.asarray(candidates, dtype=np.float32),
            np.asarray(d_train), np.asarray(d_cands), rho=rho,
            q=q, **self._kw(n_batches),
        )

    def suggest_composite(
        self,
        text,
        image,
        has_image,
        y,
        text_candidates,
        image_candidates,
        has_image_candidates,
        d_train,
        d_cands,
        q: int,
        *,
        rho: float = 0.5,
        direction: str = "maximize",
        n_batches: int = 512,
    ) -> list[dict[str, Any]]:
        """suggest_multioutput() for text + optional-image rows.

        text / image / has_image: per-row text vector, image vector (any
        placeholder when absent), and 0/1 flag. *_candidates: the same for
        the pool. Rows with has_image=0 have their image vector ignored.
        Everything else as suggest_multioutput().
        """
        return call_modal_api_composite(
            self.api_url,
            np.asarray(text, dtype=np.float32),
            np.asarray(image, dtype=np.float32),
            np.asarray(has_image, dtype=np.float32),
            _as_higher_is_better(y, direction),
            np.asarray(text_candidates, dtype=np.float32),
            np.asarray(image_candidates, dtype=np.float32),
            np.asarray(has_image_candidates, dtype=np.float32),
            np.asarray(d_train), np.asarray(d_cands), rho=rho,
            q=q, **self._kw(n_batches),
        )
