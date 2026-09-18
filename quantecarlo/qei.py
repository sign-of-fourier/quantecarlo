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

Every method is a thin pass-through to the call_modal_api* functions in
quantecarlo._modal_api, which remain the single source of truth for the
wire contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from quantecarlo._modal_api import (
    call_modal_api,
    call_modal_api_composite,
    call_modal_api_multioutput,
)

from quantecarlo.bo_sampler import DEFAULT_API_URL


def _as_higher_is_better(y, direction: str) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float32)
    if direction == "maximize":
        return y_arr
    if direction == "minimize":
        return -y_arr
    raise ValueError(f'direction must be "maximize" or "minimize", got {direction!r}')


class QEIClient:
    """Bind the endpoint and per-call defaults once; call suggest() in your loop.

    api_url:     GP service endpoint.
    timeout:     HTTP timeout in seconds.
    train_steps: server-side GP fitting budget (iterations).
    lr:          server-side GP fitting step size.
    xi:          EI exploration bonus — larger favours uncertain regions.
    mode:        "production" (default) or "debug".
    **tuning:    server-side tuning fields (n_prefilter, ei_direct_max,
                 orthant_mode, order, gh_nodes, gh_nodes_corr). Forwarded
                 verbatim on every call; unknown names raise TypeError.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        *,
        timeout: float = 120.0,
        train_steps: int = 60,
        lr: float = 0.1,
        xi: float = 0.01,
        mode: str = "production",
        **tuning: Any,
    ):
        self.api_url = api_url
        self.timeout = timeout
        self.train_steps = train_steps
        self.lr = lr
        self.xi = xi
        self.mode = mode
        self.tuning = tuning

    def _kw(self, n_batches: int) -> dict[str, Any]:
        return dict(
            n_batches=n_batches, train_steps=self.train_steps, lr=self.lr,
            xi=self.xi, mode=self.mode, timeout=self.timeout, **self.tuning,
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
        n_batches:  size-q batches scored by joint q-EI before the best is
                    returned. More = better batch, slower call.

        Returns a list of q dicts: "index" (int, into candidates), "x" (the
        candidate vector), "mu" and "sigma" (GP posterior mean / std).
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
        rho:               correlation between the two outputs, in (-1, 1).
        Everything else as suggest().
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
