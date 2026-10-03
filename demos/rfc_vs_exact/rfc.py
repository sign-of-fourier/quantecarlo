"""Randomized First Choice (RFC) against exact multinomial-probit shares.

Logic for rfc_vs_exact.ipynb. Synthetic data only: orthant_cdf sends its
inputs to a hosted service.

Model (variant a): U = X (beta + E_A) + E_P, E_A ~ N(0, Sigma_A) shared by the
alternatives of one draw, E_P ~ N(0, sigma_P^2 I) per alternative. Then
U ~ N(X beta, X Sigma_A X' + sigma_P^2 I) and

    P(choose k) = P(U_k - U_j >= 0 for all j != k),

a (K-1)-dimensional orthant of the difference vector A_k U, with mean A_k X beta
and covariance A_k Sigma_U A_k'. Variant (b) replaces E_P with Gumbel noise of
the same variance; it has no closed form here and is simulated only.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # this checkout
from quantecarlo import orthant_cdf as _orthant_cdf  # noqa: E402


def orthant_cdf(*args, retries=3, **kw):
    """quantecarlo.orthant_cdf with retries: a fit makes hundreds of calls, and
    one transient HTTP error (a timeout, a worker restart) should not end it."""
    import urllib.error
    for attempt in range(retries + 1):
        try:
            return _orthant_cdf(*args, **kw)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            code = getattr(exc, "code", None)
            if attempt == retries or (code is not None and code < 500 and code != 408):
                raise
            time.sleep(2 ** attempt)

# ---------------------------------------------------------------------------
# Design: 6 attributes, dummy-coded (first level is the reference), price linear.
# ---------------------------------------------------------------------------
ATTRS = [  # (name, levels, part-worths of levels 2.., sd of each part-worth in Sigma_A)
    ("brand", 4, [0.8, 0.4, -0.2], 0.6),
    ("A2",    3, [0.6, 0.3],       0.4),
    ("A3",    3, [0.5, -0.3],      0.3),
    ("A4",    2, [0.15],           0.1),   # the two least important attributes:
    ("A5",    2, [0.10],           0.1),   # near-duplicates differ only on these
]
PRICES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
BETA_PRICE, SD_PRICE = -1.2, 0.5
SIGMA_P = 0.5
K = 5

BETA = np.concatenate([np.r_[b] for _, _, b, _ in ATTRS] + [[BETA_PRICE]])
SD_A = np.concatenate([np.full(len(b), s) for _, _, b, s in ATTRS] + [[SD_PRICE]])
P = len(BETA)  # 10 coefficients
# which coefficient belongs to which attribute (for reporting fitted sds)
COEF_ATTR = sum(([name] * len(b) for name, _, b, _ in ATTRS), []) + ["price"]


def encode(levels, price):
    """(n, 5) level indices + (n,) price -> (n, P) design rows."""
    cols = []
    for a, (_, n_lev, _, _) in enumerate(ATTRS):
        cols.append(np.eye(n_lev)[levels[:, a]][:, 1:])
    return np.hstack(cols + [price[:, None]])


def random_alts(n, rng):
    levels = np.column_stack([rng.integers(0, n_lev, n) for _, n_lev, _, _ in ATTRS])
    return levels, rng.choice(PRICES, n)


def make_sets(n_sets, rng, kinds=("plain", "twin", "near")):
    """(n_sets, K, P) designs and each set's kind.

    twin: alternative 1 is an exact copy of alternative 0.
    near: alternative 1 is alternative 0 with A4 and A5 flipped.
    """
    X = np.empty((n_sets, K, P))
    kind = np.array([kinds[i % len(kinds)] for i in range(n_sets)])
    for s in range(n_sets):
        lev, pr = random_alts(K, rng)
        if kind[s] == "twin":
            lev[1], pr[1] = lev[0], pr[0]
        elif kind[s] == "near":
            lev[1], pr[1] = lev[0].copy(), pr[0]
            lev[1, 3] = 1 - lev[0, 3]
            lev[1, 4] = 1 - lev[0, 4]
        X[s] = encode(lev, pr)
    return X, kind


# ---------------------------------------------------------------------------
# Exact shares (variant a)
# ---------------------------------------------------------------------------
def diff_ops(k_alts):
    """A[k] (K-1, K): rows e_k - e_j for j != k."""
    ops = []
    for k in range(k_alts):
        A = np.zeros((k_alts - 1, k_alts))
        for r, j in enumerate([j for j in range(k_alts) if j != k]):
            A[r, k], A[r, j] = 1.0, -1.0
        ops.append(A)
    return ops


def utility_moments(X, beta=BETA, sd_a=SD_A, sigma_p=SIGMA_P):
    """V (M, K) and Sigma_U (M, K, K) for designs X (M, K, P)."""
    V = X @ beta
    S = np.einsum("mkp,p,mjp->mkj", X, sd_a ** 2, X) + sigma_p ** 2 * np.eye(X.shape[1])
    return V, S


def orthant_problems(V, S):
    """upper (M*K, K-1), cov (M*K, K-1, K-1); row m*K + k is P(choose k in set m)."""
    M, k_alts = V.shape
    ops = diff_ops(k_alts)
    upper = np.stack([V @ A.T for A in ops], axis=1)                       # (M, K, K-1)
    cov = np.stack([A @ S @ A.T for A in ops], axis=1)                     # (M, K, K-1, K-1)
    return upper.reshape(M * k_alts, k_alts - 1), cov.reshape(M * k_alts, k_alts - 1, k_alts - 1)


def shares_scipy(X, tol=1e-6, **kw):
    """Reference: SciPy's adaptive integration, one orthant at a time. (M, K)."""
    V, S = utility_moments(X, **kw)
    up, cov = orthant_problems(V, S)
    p = np.array([multivariate_normal.cdf(u, mean=np.zeros(len(u)), cov=c, abseps=tol, releps=tol)
                  for u, c in zip(up, cov)])
    return p.reshape(V.shape)


def shares_orthant(X, resolution="high", dtype="float32", **kw):
    """All sets and alternatives in one orthant_cdf call (per-row covariance). (M, K)."""
    V, S = utility_moments(X, **kw)
    up, cov = orthant_problems(V, S)
    return orthant_cdf(up, cov, resolution=resolution, dtype=dtype).reshape(V.shape)


# One design, many respondents: beta_i differs, X (K, P) is shared, so
# Sigma_U = X Sigma_A X' + sigma_P^2 I is one matrix and each alternative k is
# one orthant_cdf call with a shared covariance.
def respondent_betas(n, rng, spread=0.3):
    return BETA + spread * rng.standard_normal((n, P))


def shared_problems(X1, B, sd_a=SD_A, sigma_p=SIGMA_P):
    """[(upper (N, K-1), cov (K-1, K-1)) for k in range(K)] for design X1 (K, P), betas B (N, P)."""
    V = B @ X1.T
    S = X1 @ np.diag(sd_a ** 2) @ X1.T + sigma_p ** 2 * np.eye(len(X1))
    return [(V @ A.T, A @ S @ A.T) for A in diff_ops(len(X1))]


def respondent_shares_orthant(X1, B, **kw):
    return np.column_stack([orthant_cdf(u, c) for u, c in shared_problems(X1, B, **kw)])


def respondent_shares_scipy(X1, B, tol=1e-6, **kw):
    return np.column_stack([[multivariate_normal.cdf(r, mean=np.zeros(len(r)), cov=c, abseps=tol, releps=tol)
                             for r in u] for u, c in shared_problems(X1, B, **kw)])


# ---------------------------------------------------------------------------
# RFC (Monte Carlo)
# ---------------------------------------------------------------------------
def gumbel_scale(sigma_p):
    """Scale b of a Gumbel with variance sigma_p^2 (Var = pi^2 b^2 / 6)."""
    return sigma_p * np.sqrt(6.0) / np.pi


def rfc_shares(X, R, rng, variant="a", beta=BETA, sd_a=SD_A, sigma_p=SIGMA_P, chunk=20_000, draws=None, V=None):
    """RFC shares (M, K): frequency of the argmax over R draws.

    E_A is drawn once per draw and shared by the K alternatives of a set; E_P
    is drawn per alternative. Every set in X sees the same draws (common random
    numbers across sets, as a simulator with a fixed seed does); call once per
    set for independent draws. variant "a": normal E_P; "b": Gumbel E_P scaled
    to the same variance. `draws` = (Z_A (R, P), Z_P (R, K)) standard draws to
    reuse across parameter values (common random numbers, for a grid search).
    V (M, K) overrides X @ beta (respondent-level part-worths).
    """
    M, k_alts, _ = X.shape
    counts = np.zeros((M, k_alts))
    V = X @ beta if V is None else V
    for a in range(0, R, chunk):
        b = min(a + chunk, R)
        if draws is None:
            za = rng.standard_normal((b - a, P))
            zp = rng.standard_normal((b - a, k_alts)) if variant == "a" else rng.gumbel(size=(b - a, k_alts))
        else:
            za, zp = draws[0][a:b], draws[1][a:b]
        ep = sigma_p * zp if variant == "a" else gumbel_scale(sigma_p) * zp
        for m in range(M):
            U = V[m] + (za * sd_a) @ X[m].T + ep                          # (r, K)
            counts[m] += np.bincount(U.argmax(axis=1), minlength=k_alts)
    return counts / R


def respondent_shares_rfc(X1, B, R, rng, variant="a", **kw):
    Xs = np.broadcast_to(X1, (len(B),) + X1.shape)
    return rfc_shares(Xs, R, rng, variant, V=B @ X1.T, **kw)


def timed(f):
    t = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t


# ---------------------------------------------------------------------------
# Simulated respondents (holdout choices)
# ---------------------------------------------------------------------------
def simulate_choices(X, n_resp, rng, dgp="a"):
    """Counts (M, K): n_resp respondents per set, each one draw of the model."""
    return np.round(rfc_shares(X, n_resp, rng, dgp) * n_resp).astype(int)


# ---------------------------------------------------------------------------
# Fitting the variances by holdout log-likelihood (exact probabilities)
# ---------------------------------------------------------------------------
def unpack(theta, structure):
    """log-sds -> (sd_a (P,), sigma_p). structure "2": one common sd for every
    part-worth plus sigma_P (what an RFC grid search tunes). "diag": one sd per
    part-worth plus sigma_P."""
    s = np.exp(theta)
    return (np.full(P, s[0]) if structure == "2" else s[:P]), s[-1]


def fit_exact(X, counts, structure="diag", floor=1e-12):
    """Maximize sum counts * log p over the variances; p from one orthant_cdf call
    per evaluation. Returns (sd_a, sigma_p, result, n_calls)."""
    from scipy.optimize import minimize

    n = 2 if structure == "2" else P + 1
    calls = [0]

    def nll(theta):
        sd_a, sp = unpack(theta, structure)
        p = shares_orthant(X, dtype="float64", sd_a=sd_a, sigma_p=sp)
        calls[0] += 1
        return -np.sum(counts * np.log(np.maximum(p, floor)))

    res = minimize(nll, np.full(n, np.log(0.3)), method="L-BFGS-B",
                   bounds=[(np.log(1e-3), np.log(5.0))] * n, options=dict(eps=1e-3))
    sd_a, sp = unpack(res.x, structure)
    return sd_a, sp, res, calls[0]


def grid_rfc(X, counts, grid_a, grid_p, R, rng, variant="a"):
    """RFC with one common attribute sd and sigma_P, chosen by MAE against the
    observed shares on X. Common random numbers across the grid."""
    za = rng.standard_normal((R, P))
    zp = rng.standard_normal((R, K)) if variant == "a" else rng.gumbel(size=(R, K))
    obs = counts / counts.sum(axis=1, keepdims=True)
    best = None
    table = np.empty((len(grid_a), len(grid_p)))
    for i, a in enumerate(grid_a):
        for j, sp in enumerate(grid_p):
            sh = rfc_shares(X, R, rng, variant, sd_a=np.full(P, a), sigma_p=sp, draws=(za, zp))
            table[i, j] = np.mean(np.abs(sh - obs))
            if best is None or table[i, j] < best[0]:
                best = (table[i, j], a, sp)
    return best[1], best[2], table


# ---------------------------------------------------------------------------
# Section 6: synthetic stand-in for HB posterior draws
# ---------------------------------------------------------------------------
def hb_draws(n, S, rng, omega_sd=0.4, post_sd=0.25, mu_sd=0.08):
    """SYNTHETIC stand-in for HB output, not a fitted posterior.

    beta_n ~ N(BETA, omega_sd^2 I) for n respondents; S pseudo-posterior draws
    per respondent: beta_n + delta_s + noise. delta_s ~ N(0, mu_sd^2 I) is
    shared by every respondent in draw s, standing in for the posterior
    uncertainty of the population mean that real HB draws carry (without it,
    market-share uncertainty averages away over respondents). The noise has a
    respondent-specific covariance (random scales and one random correlation
    factor per respondent).
    Returns beta_n (n, P) and draws (n, S, P).
    TODO: replace with draws from a real HB fit (e.g. bayesm rhierMnlRwMixture).
    """
    beta_n = BETA + omega_sd * rng.standard_normal((n, P))
    scale = post_sd * np.exp(0.3 * rng.standard_normal((n, P)))
    load = 0.7 * rng.standard_normal((n, P))
    z = rng.standard_normal((n, S, P))
    u = rng.standard_normal((n, S, 1))
    noise = (z + u * load[:, None, :]) / np.sqrt(1.0 + load[:, None, :] ** 2)   # unit variance, correlated
    delta = mu_sd * rng.standard_normal((1, S, P))
    return beta_n, beta_n[:, None, :] + delta + noise * scale[:, None, :]


def draw_shares_orthant(X1, draws, **kw):
    """Exact shares for every (respondent, draw): (n, S, K). One orthant_cdf call per
    alternative, all n*S rows under the scenario's shared covariance."""
    n, S, _ = draws.shape
    return respondent_shares_orthant(X1, draws.reshape(n * S, P), **kw).reshape(n, S, len(X1))


def rfc_shared_design(V, X1, R, rng, sd_a=SD_A, sigma_p=SIGMA_P, rows_per_chunk=2000):
    """RFC variant a for many rows sharing one design X1 (K, P), row means V (M, K).

    Independent draws per row. With X1 shared, X1 E_A is a K-vector with
    covariance X1 Sigma_A X1', so it is drawn in K dimensions (same distribution
    as drawing the P-vector E_A and multiplying, fewer random numbers).
    """
    M, k_alts = V.shape
    L = np.linalg.cholesky(X1 @ np.diag(sd_a ** 2) @ X1.T + 1e-12 * np.eye(k_alts))
    out = np.empty((M, k_alts))
    for a in range(0, M, rows_per_chunk):
        b = min(a + rows_per_chunk, M)
        U = V[a:b, None, :] + rng.standard_normal((b - a, R, k_alts)) @ L.T \
            + sigma_p * rng.standard_normal((b - a, R, k_alts))
        win = U.argmax(axis=2)
        out[a:b] = np.stack([(win == k).mean(axis=1) for k in range(k_alts)], axis=1)
    return out


# ---------------------------------------------------------------------------
# Section 7: product-line optimization
# ---------------------------------------------------------------------------
def all_products():
    """Every attribute-level combination: levels (720, 5), price codes (720,), X (720, P)."""
    grids = np.meshgrid(*[np.arange(n_lev) for _, n_lev, _, _ in ATTRS], np.arange(len(PRICES)), indexing="ij")
    flat = np.column_stack([g.ravel() for g in grids])
    levels, price = flat[:, :-1], PRICES[flat[:, -1]]
    return levels, price, encode(levels, price)


def dollars(X):
    """Illustrative economics: price $10 +/- $4 across the price codes; unit cost
    $6 plus $2 per unit of non-price part-worth (better levels cost more)."""
    price = 10.0 + 4.0 * X[..., -1]
    cost = 6.0 + 2.0 * (X[..., :-1] @ BETA[:-1])
    return price - cost


def line_problems(X_sets, n_line):
    """Orthant problems for the first n_line alternatives of each set only:
    upper (L*n_line, K-1), cov (L*n_line, K-1, K-1)."""
    V, S = utility_moments(X_sets)
    ops = diff_ops(X_sets.shape[1])[:n_line]
    up = np.stack([V @ A.T for A in ops], axis=1)
    cov = np.stack([A @ S @ A.T for A in ops], axis=1)
    k1 = X_sets.shape[1] - 1
    return up.reshape(-1, k1), cov.reshape(-1, k1, k1)


def line_profit(X_line, X_comp, chunk=100_000, method="orthant"):
    """Expected margin per buyer, sum_k share_k * margin_k, for lines X_line (L, m, P)
    against fixed competitors X_comp (c, P). Returns (profit (L,), shares (L, m))."""
    L, m, _ = X_line.shape
    shares = np.empty((L, m))
    for a in range(0, L, chunk):
        b = min(a + chunk, L)
        Xs = np.concatenate([X_line[a:b], np.broadcast_to(X_comp, (b - a,) + X_comp.shape)], axis=1)
        up, cov = line_problems(Xs, m)
        if method == "orthant":
            p = orthant_cdf(up, cov)
        else:
            p = np.array([multivariate_normal.cdf(u, mean=np.zeros(len(u)), cov=c, abseps=1e-6, releps=1e-6)
                          for u, c in zip(up, cov)])
        shares[a:b] = p.reshape(b - a, m)
    return (shares * dollars(X_line)).sum(axis=1), shares


def random_lines(n_lines, n_products, m, rng):
    """n_lines distinct m-subsets (as sorted index rows) of range(n_products)."""
    idx = np.sort(rng.integers(0, n_products, (int(n_lines * 1.05) + 100, m)), axis=1)
    ok = np.all(np.diff(idx, axis=1) > 0, axis=1)
    return np.unique(idx[ok], axis=0)[:n_lines]
