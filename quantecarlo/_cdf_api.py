"""Raw HTTP client for the orthant-probability endpoint (orthant_cdf).

Many multivariate normal CDFs under one covariance:

    p[i] = P(Z_1 <= upper[i, 1], ..., Z_d <= upper[i, d]),   Z ~ N(0, cov)

The covariance travels once; only the rows of upper scale with N. That is
the shape of scoring a fitted multivariate-normal model (a multivariate
probit, a Gaussian copula) on a test set: one fitted matrix, one row of
limits per observation.

signs (d,) of +1/-1 turns the event into {s_j Z_j <= s_j upper[i, j]}: a
fixed mix of "below" (+1) and "above" (-1) constraints, the same for every
row. For a multivariate probit with latent index eta and Y_j = 1 iff
Z_j <= eta_j:

    P(all Y = 1)      upper=eta
    P(all Y = 0)      upper=-eta            (P(any Y = 1) = 1 - this)
    P(Y = y)          upper=eta, signs=2*y - 1

A query whose pattern varies by row (the observed y of each test row, for a
log score) is one call per distinct pattern, rows grouped by pattern.

cov may instead be (N, d, d), one covariance per row, when the rows are not
draws from one fitted model: e.g. choice probabilities across many choice
sets, where each set's utility-difference covariance depends on its design.
Every row's matrix then travels (in float64), so the upload grows with
N * d^2 rather than N * d; send a shared (d, d) whenever the rows have one.

resolution: "high" (default) or "low". "low" is coarser and faster; the gap
    is small for 2-4 variables and grows with dimension.

dup_corr: variables whose correlation is >= 1 - dup_corr are merged into one
    before integrating (keeping the tighter limit). Near-perfect correlation
    is where the approximation is weakest, and merging is exact at 1. None
    (default) uses the service's value, 0.03 (merge at 0.97); 0 merges only
    exact duplicates. Accepted range [0, 0.5].

Approximate, not exact. Results are clipped to [0, 1]; a value of exactly
0.0 or 1.0 means the error is large there, which matters for a log score.
Accuracy depends on the covariance and the limits, and no argument buys it
back on a problem the method is weak on. Measure on your own inputs: take a
random subsample of rows and compare against
scipy.stats.multivariate_normal.cdf.

Wire format: an uncompressed npz body carrying `upper` (N, d), `cov_tril`
(d(d+1)/2,) -- the row-major lower triangle of cov, diagonal included; or
(N, d(d+1)/2) for a per-row cov --
optional `signs`, and the scalar fields as JSON under `params`. The response
is an npz with `p` (N,). Uncompressed because the payload is upper, and
floats barely compress: at N = 1M zip saves ~4% and costs seconds.

dtype (default "float32"): the float width `upper` is sent in. At large N
the upload is the whole cost of a call -- the service computes a million
rows in about a second -- and float32 halves it. Rounding the limits to
float32 moves p by ~1e-7, far below the method's own error. The covariance
always travels in float64: it is small, and a near-singular one is where
rounding would matter. The service computes in float64 either way.
"""
from __future__ import annotations

import io
import json
import urllib.request

import numpy as np

from quantecarlo._modal_api import _http_error
from quantecarlo._select_api import _DTYPES, pack_tril

DEFAULT_CDF_URL = "https://info-29741--bo-gp-service-orthant-cdf.modal.run"

_RESOLUTIONS = ("high", "low")


def build_cdf_body(upper, cov, signs=None, resolution: str = "high",
                   dtype: str = "float32", dup_corr: float | None = None) -> bytes:
    """The npz body orthant_cdf posts. Exposed for tests and for callers that
    want to inspect or store what would be sent."""
    if dtype not in _DTYPES:
        raise ValueError(f'dtype must be "float64" or "float32"; got {dtype!r}')
    if resolution not in _RESOLUTIONS:
        raise ValueError(f'resolution must be "high" or "low"; got {resolution!r}')
    np_dtype = _DTYPES[dtype]
    upper = np.ascontiguousarray(np.atleast_2d(np.asarray(upper)), dtype=np_dtype)
    if upper.ndim != 2:
        raise ValueError(f"upper must be (N, d) or (d,); got shape {upper.shape}")
    N, d = upper.shape
    cov = np.asarray(cov)
    if cov.ndim == 3:
        if cov.shape != (N, d, d):
            raise ValueError(f"per-row cov must be ({N}, {d}, {d}) to match upper; got {cov.shape}")
        i, j = np.tril_indices(d)
        tril = np.ascontiguousarray(cov[:, i, j], dtype=np.float64)
    else:
        tril = pack_tril(cov, np.float64)
        if tril.shape[0] != d * (d + 1) // 2:
            raise ValueError(f"upper has d={d} columns but cov is {cov.shape}")
    params = {"resolution": resolution}
    if dup_corr is not None:
        params["dup_corr"] = float(dup_corr)
    arrays = {"upper": upper, "cov_tril": tril, "params": json.dumps(params)}
    if signs is not None:
        signs = np.asarray(signs, dtype=np.float64).ravel()
        if signs.shape[0] != d or not np.all(np.abs(signs) == 1.0):
            raise ValueError(f"signs must be {d} entries of +1/-1")
        arrays["signs"] = signs
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def orthant_cdf(
    upper,
    cov,
    *,
    signs=None,
    resolution: str = "high",
    api_url: str = DEFAULT_CDF_URL,
    dtype: str = "float32",
    dup_corr: float | None = None,
    timeout: float = 300.0,
) -> np.ndarray:
    """P(Z <= upper[i]) for each row i, Z ~ N(0, cov); see the module docstring.

    upper:      (N, d) upper limits, one problem per row, or (d,) for one.
    cov:        (d, d) covariance shared by every row, or (N, d, d) one per
                row. A correlation matrix is the usual case; any positive
                diagonal works, the service standardises. Singular (PSD)
                matrices are accepted.
    signs:      optional (d,) of +1/-1 -- a fixed "below"/"above" pattern.
    resolution: "high" (default) or "low".
    dtype:      "float32" (default) or "float64": the wire width of upper
                only; see the module docstring.
    dup_corr:   merge variables correlated >= 1 - dup_corr; None (default)
                uses the service's 0.03. See the module docstring.

    Returns p, a float64 array of shape (N,).
    """
    body = build_cdf_body(upper, cov, signs, resolution, dtype, dup_corr)
    req = urllib.request.Request(
        api_url, data=body,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise _http_error(exc) from None
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        return np.asarray(npz["p"], dtype=np.float64)
