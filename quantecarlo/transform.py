"""The target transform the service applies inside `suggest`, for `select` users.

`select` scores improvement on exp of the posterior it is sent (see
`QEIClient.select`), so the posterior has to be on a normal scale: N(0, 1)
marginals, the way `suggest` gets them by rank-normalising y before the fit.
If your surrogate does not already do that (a PIT, a log of a lognormal
objective), fit it on `rank_normal(y)` and pass `best_y = rank_normal(y).max()`
or your own rule on that scale.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["rank_normal"]


def rank_normal(y) -> np.ndarray:
    """Map y (n,) to its normal scores: N(0, 1) marginals, ranks preserved.

    Identical to what `suggest` does server-side: rank / (n + 1), squeezed to
    [0.00005, 0.99995] so the extremes stay finite (|f| <= 3.9), then Phi^-1.
    Ties keep argsort order.  Returns float64.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n == 0:
        return y
    ranks = np.argsort(np.argsort(y)).astype(np.float64)
    u = (ranks + 1.0) / (n + 1.0)
    u = u * 0.9999 + 0.00005
    return norm.ppf(u)
