"""Raw HTTP client for the Modal GP endpoint.

This is the single source of truth for the Modal API contract. Both
modal_suggest (Optuna/BatchSampler integration) and direct callers such as
meta-ads-demo import from here. When the API payload changes, update this
file and every consumer gets the change.

y convention: higher is better. Callers that use a minimise objective must
negate y before calling these functions.

Server-side tuning fields (all optional, all server-defaulted):
    n_batches      index-batches scored with joint q-EI — the survivors of the
                   MPI prefilter, or everything sampled when it is skipped
    n_prefilter    distinct size-q index-batches drawn from the pool before
                   scoring (every combination when C(n_cands, q) is smaller)
    ei_direct_max  when n_sampled * q exceeds this, batches are ranked by q-PI
                   first and only the top n_batches go on to q-EI
    orthant_mode   "auto" (default) or "legacy"
    order          0 or 1 — Plackett correction on/off in the "auto" path
    gh_nodes, gh_nodes_corr   Gauss-Hermite node counts in the "auto" path
Any of these can be passed to the call_modal_api* functions as keyword
arguments; anything else is rejected client-side rather than 422'd by the
server.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Server-side knobs forwarded verbatim when given.  Server defaults apply when
# omitted, so a payload never carries a value the caller didn't set.
_TUNING_FIELDS = frozenset({
    "n_prefilter", "ei_direct_max",
    "orthant_mode", "order", "gh_nodes", "gh_nodes_corr",
})


def _post(api_url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise urllib.error.HTTPError(
            exc.url, exc.code, f"{exc.reason} — {detail}", exc.headers, None
        ) from None


def _common_fields(q, n_batches, train_steps, lr, xi, mode, tuning: dict) -> dict:
    unknown = set(tuning) - _TUNING_FIELDS
    if unknown:
        raise TypeError(
            f"unknown Modal API field(s) {sorted(unknown)}; "
            f"accepted tuning fields: {sorted(_TUNING_FIELDS)}"
        )
    fields = {
        "q": q,
        "n_batches": n_batches,
        "train_steps": train_steps,
        "lr": lr,
        "xi": xi,
        "mode": mode,
    }
    fields.update(tuning)
    return fields


def _parse(data: dict) -> list[dict[str, Any]]:
    return [
        {
            "index": int(c["index"]),
            "x": np.array(c["x"], dtype=np.float32),
            "mu": c.get("mu"),
            "sigma": c.get("sigma"),
        }
        for c in data["candidates"]
    ]


def call_modal_api(
    api_url: str,
    X: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
    q: int = 2,
    n_batches: int = 512,
    train_steps: int = 60,
    lr: float = 0.1,
    xi: float = 0.01,
    mode: str = "production",
    timeout: float = 120.0,
    **tuning: Any,
) -> list[dict[str, Any]]:
    """POST to the Modal GP endpoint and return q candidate dicts.

    y convention: higher is better. Pass scores directly for maximisation
    objectives; negate first for minimisation.

    X:          observed points, shape (n_obs, n_dims)
    y:          observed scores, shape (n_obs,)
    candidates: discrete candidate pool to select from, shape (n_cands, n_dims)
    **tuning:   any of the server-side tuning fields listed in the module
                docstring (n_prefilter, ei_direct_max, orthant_mode, order,
                gh_nodes, gh_nodes_corr)

    Each returned dict has:
        "index"  — int, index into candidates
        "x"      — np.ndarray shape (n_dims,), the selected candidate vector
        "mu"     — float | None, GP posterior mean
        "sigma"  — float | None, GP posterior std
    """
    payload: dict[str, Any] = {
        "X": X.tolist(),
        "y": y.tolist(),
        "candidates": candidates.tolist(),
        **_common_fields(q, n_batches, train_steps, lr, xi, mode, tuning),
    }
    logger.debug(
        "call_modal_api: POST %s (n_obs=%d, n_cands=%d, n_dims=%d, q=%d)",
        api_url, len(y), len(candidates), candidates.shape[1], q,
    )
    return _parse(_post(api_url, payload, timeout))


def call_modal_api_multioutput(
    api_url: str,
    X: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
    d_train: np.ndarray,
    d_cands: np.ndarray,
    rho: float = 0.5,
    q: int = 2,
    n_batches: int = 512,
    train_steps: int = 60,
    lr: float = 0.1,
    xi: float = 0.01,
    mode: str = "production",
    timeout: float = 120.0,
    **tuning: Any,
) -> list[dict[str, Any]]:
    """POST to the Modal GP multioutput endpoint and return q candidate dicts.

    Identical to call_modal_api but additionally sends d_train, d_cands, and
    rho. The server branches to the multioutput GP when d and d_candidates are
    present in the payload.

    d_train: int array shape (n_obs,)   — output index per training point (e.g. 0=Meta, 1=Google)
    d_cands: int array shape (n_cands,) — output index per candidate
    rho:     cross-platform correlation in (-1, 1)

    Return format is identical to call_modal_api.
    """
    payload: dict[str, Any] = {
        "X": X.tolist(),
        "y": y.tolist(),
        "candidates": candidates.tolist(),
        "d": d_train.tolist(),
        "d_candidates": d_cands.tolist(),
        "rho": float(rho),
        **_common_fields(q, n_batches, train_steps, lr, xi, mode, tuning),
    }
    logger.debug(
        "call_modal_api_multioutput: POST %s (n_obs=%d, n_cands=%d, n_dims=%d, q=%d, rho=%.2f)",
        api_url, len(y), len(candidates), candidates.shape[1], q, rho,
    )
    return _parse(_post(api_url, payload, timeout))


def call_modal_api_composite(
    api_url: str,
    text: np.ndarray,
    image: np.ndarray,
    has_image: np.ndarray,
    y: np.ndarray,
    text_candidates: np.ndarray,
    image_candidates: np.ndarray,
    has_image_candidates: np.ndarray,
    d_train: np.ndarray,
    d_cands: np.ndarray,
    rho: float = 0.5,
    q: int = 2,
    n_batches: int = 512,
    train_steps: int = 60,
    lr: float = 0.1,
    xi: float = 0.01,
    mode: str = "production",
    timeout: float = 120.0,
    **tuning: Any,
) -> list[dict[str, Any]]:
    """POST to the Modal GP composite-kernel endpoint and return q candidate dicts.

    One row per ad (not one row per platform-PCA'd combination). Each row
    carries its own text vector, image vector (placeholder allowed when
    absent), and a has_image flag. The server sums a shared text RBF kernel
    with a masked shared image RBF kernel, then applies the same platform-level
    B/rho coregionalization as call_modal_api_multioutput -- this is an
    additive-kernel alternative to that function, not a replacement; both are
    supported server-side (kernel_mode="composite" vs the default path).

    text/image/has_image: shape (n_obs, ...) — training rows.
    text_candidates/image_candidates/has_image_candidates: shape (n_cands, ...).
    d_train/d_cands: int array, output index per row (e.g. 0=Meta, 1=Google) —
        same meaning and same 2-output constraint as call_modal_api_multioutput.
    rho: platform-level correlation in (-1, 1), same B = [[1,rho],[rho,1]].

    Return format is identical to call_modal_api / call_modal_api_multioutput.
    """
    payload: dict[str, Any] = {
        "X": [],            # required by the schema, unused in composite mode
        "candidates": [],
        "y": y.tolist(),
        "kernel_mode": "composite",
        "text": text.tolist(),
        "image": image.tolist(),
        "has_image": has_image.tolist(),
        "text_candidates": text_candidates.tolist(),
        "image_candidates": image_candidates.tolist(),
        "has_image_candidates": has_image_candidates.tolist(),
        "d": d_train.tolist(),
        "d_candidates": d_cands.tolist(),
        "rho": float(rho),
        **_common_fields(q, n_batches, train_steps, lr, xi, mode, tuning),
    }
    logger.debug(
        "call_modal_api_composite: POST %s (n_obs=%d, n_cands=%d, q=%d, rho=%.2f)",
        api_url, len(y), len(text_candidates), q, rho,
    )
    return _parse(_post(api_url, payload, timeout))
