"""Raw HTTP client for the acquisition-only endpoint (qei_select).

You fitted your own surrogate; the service only picks the batch. Send the
posterior mean over your candidates, its covariance, and the incumbent, and
get back q indices chosen by joint q-EI. Nothing about your model, your
observations, or how you scaled y leaves the machine -- only (mu, cov,
best_y), which are scale-free as far as the service is concerned (it never
compares them to anything of its own).

Wire format: an npz body (zip-compressed) carrying `mu` (n,), `cov_tril`
(n(n+1)/2,) -- the row-major lower triangle of cov, diagonal included -- and
the scalar fields as JSON under `params`. Symmetry is why half the matrix is
enough; the server rebuilds the full matrix.

dtype (default "float64"): the float width `mu` and `cov_tril` are packed in.
    float64 is the default because it is what your GP produced and the wire
    should not be where precision is decided. Above ~1000 candidates the
    triangle is the payload: n=5000 is 100 MB in float64 before compression
    (a dense posterior compresses poorly, expect 10-20% off), 50 MB in
    float32. The service casts to float32 on the GPU for the screening pass
    anyway, so for large n switch to dtype="float32" -- call_select_api logs
    a reminder when n is large and you have not. It is a 2x, not a fix; if
    the payload is still a problem the answer is fewer candidates (see
    pi_floor), not narrower floats.

Batch-search knobs (server-defaulted unless given; all optional):
    ei_budget     work the joint q-EI stage may spend, in units that scale
                  with q (default 40000). The number of batches it scores is
                  ei_budget // q.
    mpi_bytes     memory the cheaper screening pass may use (default
                  512 MB). The number of subsets it screens is
                  mpi_bytes // (8 q^2) -- about 1M at q=8, 65K at q=32 --
                  so the width of the search is solved from the budget and
                  q, not chosen.
    pi_floor      drop candidates whose single-point probability of beating
                  best_y is below this, before batches are formed. Off by
                  default. This is the lever that shrinks a large pool.
    seed          makes the sampling regime reproducible.
    orthant_mode, order, gh_nodes, gh_nodes_corr
                  numerical-accuracy settings; leave unset unless instructed.

The server chooses between three regimes from the pool size and q, and the
response says which ran:
    "exact"   every size-q subset was scored by joint q-EI
    "screen"  every subset was screened, the top ei_budget // q scored
    "sample"  as many subsets as mpi_bytes allows were drawn, screened,
              the top scored
"""
from __future__ import annotations

import io
import json
import logging
import urllib.request
from typing import Any

import numpy as np

from quantecarlo._modal_api import _http_error

logger = logging.getLogger(__name__)

_SELECT_TUNING_FIELDS = frozenset({
    "ei_budget", "mpi_bytes", "pi_floor", "seed",
    "orthant_mode", "order", "gh_nodes", "gh_nodes_corr",
})

_DTYPES = {"float64": np.float64, "float32": np.float32}

# Above this many candidates call_select_api reminds you about dtype.
_LARGE_N = 1000


def pack_tril(cov: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Row-major lower triangle of a symmetric (n, n) matrix, diagonal included."""
    cov = np.asarray(cov)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square (n, n); got shape {cov.shape}")
    n = cov.shape[0]
    return np.ascontiguousarray(cov[np.tril_indices(n)], dtype=dtype)


def build_select_body(mu, cov, params: dict, dtype: str = "float64") -> bytes:
    """The npz body call_select_api posts. Exposed for tests and for callers
    that want to inspect or store what would be sent."""
    if dtype not in _DTYPES:
        raise ValueError(f'dtype must be "float64" or "float32"; got {dtype!r}')
    np_dtype = _DTYPES[dtype]
    mu = np.ascontiguousarray(np.asarray(mu).ravel(), dtype=np_dtype)
    tril = pack_tril(cov, np_dtype)
    if tril.shape[0] != mu.shape[0] * (mu.shape[0] + 1) // 2:
        raise ValueError(f"mu has {mu.shape[0]} entries but cov is {np.asarray(cov).shape}")
    buf = io.BytesIO()
    np.savez_compressed(buf, mu=mu, cov_tril=tril, params=json.dumps(params))
    return buf.getvalue()


def call_select_api(
    api_url: str,
    mu: np.ndarray,
    cov: np.ndarray,
    best_y: float,
    q: int = 2,
    *,
    dtype: str = "float64",
    mode: str = "production",
    timeout: float = 120.0,
    diagnostics: dict | None = None,
    **tuning: Any,
) -> dict[str, Any]:
    """POST a posterior to the acquisition-only endpoint; return the chosen batch.

    mu:       posterior mean at each candidate, shape (n,). Higher = better.
    cov:      posterior covariance over the candidates, shape (n, n). Full
              matrix; the client packs the lower triangle.
    best_y:   the incumbent value, on the same scale as mu. Your rule (max
              observed y, best posterior mean at the training inputs, ...).
    q:        how many candidates to pick.
    dtype:    "float64" (default) or "float32" -- see the module docstring.
    mode:     "debug" adds diagnostics to the response; pass a dict as
              `diagnostics` to receive them.
    **tuning: any of the fields listed in the module docstring.

    Returns a dict: "indices" (list of q ints into mu), "qei" (the batch's
    joint q-EI), "regime" ("exact" | "screen" | "sample"), "n_cands",
    "n_sampled", "n_batches".
    """
    unknown = set(tuning) - _SELECT_TUNING_FIELDS
    if unknown:
        raise TypeError(
            f"unknown select field(s) {sorted(unknown)}; "
            f"accepted tuning fields: {sorted(_SELECT_TUNING_FIELDS)}"
        )
    n = int(np.asarray(mu).size)
    if dtype == "float64" and n > _LARGE_N:
        logger.warning(
            "call_select_api: %d candidates -> %.0f MB of covariance in float64 "
            "before compression; consider dtype=\"float32\" (halves it) or pi_floor "
            "(shrinks the pool).", n, 8 * n * (n + 1) / 2 / 1e6,
        )
    params = {"q": int(q), "best_y": float(best_y), "mode": mode, **tuning}
    body = build_select_body(mu, cov, params, dtype)
    req = urllib.request.Request(
        api_url, data=body,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise _http_error(exc) from None
    return _parse_select(data, diagnostics)


_RESULT_KEYS = ("indices", "qei", "regime", "n_cands", "n_sampled", "n_batches")


def _parse_select(data: dict, diagnostics: dict | None = None) -> dict[str, Any]:
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({k: v for k, v in data.items() if k not in _RESULT_KEYS and v is not None})
    out = {k: data[k] for k in _RESULT_KEYS}
    out["indices"] = [int(i) for i in out["indices"]]
    return out
