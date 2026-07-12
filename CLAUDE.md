# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cleanup plan

See `TECH_DEBT.md` in this repo for the staged cleanup plan, repo map, and
critical conventions. Read that before starting any new session on this codebase.

## Setup and Development

```bash
pip install -e .          # install in editable mode (hard deps: optuna>=3.0, numpy, scipy)
pip install optunahub     # needed to run demos
python demos/demo.py      # ask-tell demo using BatchSampler + modal_suggest (requires Modal endpoint)
```

Run tests:
```bash
python -m pytest tests/ -v -k "not Live"          # unit tests only
python -m pytest tests/test_modal_live.py -v -s   # requires Modal endpoint
```

No linter configuration in this repo.

## Architecture

`quantecarlo` provides two `suggest_fn` implementations for the optunahub `BatchSampler`,
plus the raw HTTP client both of them sit on top of:

1. **`fantasize_suggest`** (`quantecarlo/_fantasize.py`) — self-contained in-process GP.
   Fits an RBF GP using numpy/scipy, selects a batch by Expected Improvement with
   fantasization. No remote service required. This is the primary example in the
   optunahub `batch_sampler` package.

2. **`modal_suggest`** (`quantecarlo/bo_sampler.py`) — delegates to the remote Modal GP
   service (`~/projects/boaz/modal/modal_gp_api.py`). Generates `n_probe_points` random
   points client-side, POSTs them with observations to the endpoint, returns the
   highest q-EI batch. Use via `functools.partial` to bind `api_url` and other params.

Both follow the `suggest_fn` contract of the optunahub `BatchSampler`:

```
suggest_fn(X: list[list[float]], y: list[float], search_space: list[DimSpec], q: int)
    -> list[dict[str, Any]]   # exactly q parameter dicts
```

`BatchSampler` owns the lock, cache, startup fallback, and threading. The `suggest_fn`
just does the math / HTTP call.

**Both of the above exist only to handle a genuinely continuous search space** — one
with no enumerable set of valid points (a learning rate, a hidden-layer-size range).
That's the entire reason `n_probe_points`/`n_candidates` exists: with nothing to
enumerate, some finite stand-in has to be invented before a GP can be asked anything.

**If you already have a real, finite, enumerable set of candidates** (a product
catalog, an ad pool, any materializable list), do not route it through `modal_suggest`
or `DimSpec` — call `call_modal_api` directly with your real vectors as `candidates`.
Going through the continuous path in that case means inventing random points, asking
the GP to choose among the fakes, then snapping the result back to the nearest real
item — an approximation of an approximation with no purpose when the real candidates
were available to hand the GP directly. See `demos/demo9.py`'s `run_arm_qei` for a
self-contained worked example of the direct-pool pattern (and its module docstring
for the full reasoning). The `demo7*`/`demo8` family in `~/projects/chi_bad_ads/demos/`
runs the same pattern against real ad embeddings instead of `demo9.py`'s synthetic
pool, and still contrasts it against continuous-relaxation `modal_suggest` usage
(`demo7.py`'s Experiment A).

3. **`call_modal_api` / `call_modal_api_multioutput` / `call_modal_api_composite`**
   (`quantecarlo/_modal_api.py`) — the raw Modal HTTP client and the **recommended
   direct entry point whenever you have real, materialized candidate points**. Takes
   numpy arrays directly (higher = better, no direction param). Used internally by
   `modal_suggest` and re-exported by `meta-ads-demo`. This is the **single source of
   truth** for the Modal API contract — update here when the API payload changes.
   `call_modal_api_composite` (added 2026-07-06) is a one-row-per-ad alternative to
   `call_modal_api_multioutput` — server sums a shared text kernel with a masked shared
   image kernel (`kernel_mode="composite"`) instead of platform-PCA'd concatenation;
   both are supported server-side, neither replaces the other.

### `DimSpec` (`quantecarlo/bo_sampler.py`)

`@dataclass` describing one search-space dimension. Fields: `name`, `type` (`"float"` |
`"int"`), `low`, `high`, `log`, `step`. Shared between both suggest functions and the
optunahub `BatchSampler`.

### `modal_suggest` (`quantecarlo/bo_sampler.py`)

```python
modal_suggest(X, y, search_space, q, *, direction="minimize", api_url, n_probe_points=512,
              n_candidate_batches=None, train_steps=60, lr=0.1, xi=0.01, mode="production",
              seed=None, timeout=120.0)
```

**y convention**: `BatchSampler` passes raw study values in the study's direction
convention (lower = better for minimize, higher = better for maximize). Pass
`direction` matching your Optuna study direction — `modal_suggest` handles conversion
to the Modal API's higher-is-better convention internally. Bind via `functools.partial`.

**`n_probe_points` vs `n_candidate_batches` — two different knobs, not one**:
- `n_probe_points` — how many random continuous points are invented from `search_space`
  bounds and sent as the pool the GP can choose from. Only meaningful when there is no
  real enumerable pool (see the continuous-vs-enumerable note above).
- `n_candidate_batches` — how many random size-`q` index-combinations of that pool the
  server scores with joint q-EI before returning the best one (server's
  `GPRequest.n_batches` — renamed here so "batch" doesn't also mean "a q-sized round of
  picks" the way it does in a typical ask-tell loop). Defaults to `n_probe_points` if
  omitted, for backward compatibility with callers that only ever set one knob.

**Payload sent**: `{X, y_higher_is_better, candidates, q, n_batches, train_steps, lr, xi, mode}`
(`n_batches` here is `n_candidate_batches`, or `n_probe_points` if that wasn't set —
the wire field name matches the server's `GPRequest.n_batches`, unchanged by the
client-side rename.)

**Response parsed**: `{"candidates": [{"index": int, "x": [...], "mu": float, "sigma": float}, ...]}`

### `fantasize_suggest` (`quantecarlo/_fantasize.py`)

```
fantasize_suggest(X, y, search_space, q, direction="minimize", n_candidates=512, noise=1e-3, xi=0.01, seed=None)
```

Takes `y` in the study's direction convention (same as `modal_suggest`). Negates internally
for minimize studies so the GP always works higher-is-better. EI formula is maximize convention throughout.

### `_modal_api.py` (`quantecarlo/_modal_api.py`)

```python
call_modal_api(api_url, X, y, candidates, q, n_batches, train_steps, lr, xi, mode, timeout)
call_modal_api_multioutput(api_url, X, y, candidates, d_train, d_cands, rho, q, ...)
call_modal_api_composite(api_url, text, image, has_image, y, text_candidates,
                          image_candidates, has_image_candidates, d_train, d_cands, rho, q, ...)
```

All three take numpy arrays, return `list[dict]` with `index`, `x` (np.ndarray), `mu`, `sigma`.
`call_modal_api_multioutput` adds `d_train`, `d_cands`, `rho` for the cross-platform GP path
(one row per platform-PCA'd combination). `call_modal_api_composite` is the one-row-per-ad
alternative — no PCA, no concatenation; the server sums a shared text RBF kernel with a
masked shared image RBF kernel and applies the same `d`/`rho` coregionalization on top.
`has_image`/`has_image_candidates` are 0/1 float arrays; rows without an image still need a
placeholder `image`/`image_candidates` vector (zeros is fine) — the mask zeroes its
contribution.

**`candidates` here is whatever you pass** — it does not have to be `modal_suggest`'s
invented continuous points. If you already have real candidate vectors (an embedded
product/ad pool, any materializable finite set), pass them here directly: `X` = your
observed points, `y` = their scores (higher = better, negate first if minimizing),
`candidates` = the real remaining pool. The returned `index` is a real index into that
pool, immediately usable — no synthesis, no snapping to the nearest real item
afterward, because there was no invented item to begin with. This is the right layer
to call for any problem with an enumerable candidate set, `BatchSampler`/`DimSpec`
involved or not.

### Related repos

| Repo | Role |
|---|---|
| `~/projects/boaz/modal/modal_gp_api.py` | The Modal GP server — `modal_suggest` POSTs here |
| `~/projects/optunahub-registry/package/samplers/batch_sampler/` | `BatchSampler` — wraps any `suggest_fn` with lock + cache |
| `~/projects/meta-ads-demo/backend/bo_pipeline/modal_bo.py` | Re-exports `call_modal_api` / `call_modal_api_multioutput` from quantecarlo; owns PCA helpers and env-var config |

### demos/

- **`demo.py`** — NAS on the breast-cancer dataset using an MLP. Uses `BatchSampler` +
  `modal_suggest` with an explicit ask-tell loop so the batching contract is visible.
  Requires a running Modal endpoint and `optunahub`.
- **`demo9.py`** — Self-contained version of the `demo7_categorical_vs_embedding.py`
  hypothesis (does exploiting embedding structure via qEI beat embedding-blind
  categorical TPE on a fixed pool?) with no external data: pool = a random subset of
  sklearn's bundled `load_digits()` pixel vectors (64-dim, real feature structure),
  score = synthetic distance-to-archetype-centroid + noise (not the digit label —
  the model never sees it, only pixel vectors as candidates). Same two arms,
  `call_modal_api` direct on the real pool for qEI. Requires only a running Modal
  endpoint — no `~/projects/chi_bad_ads` data.
- **`demo_latency_test.py`** — One-off latency probe for `modal_suggest` at a much
  larger candidate pool (10000 vs `demo9.py`'s hundreds). Synthetic random points,
  not part of `pytest`. Requires a running Modal endpoint.

The `demo7.py` / `demo7_pca_vs_pls.py` / `demo7_categorical_vs_embedding.py` /
`demo8.py` family that originally lived here has moved to
`~/projects/chi_bad_ads/demos/`, since all four hardcode paths into that repo's data
files (`ads_all_labels.json`, `embedding_cache/`) and don't belong in this repo.
They cover the same ground as `demo9.py` above, plus a PCA-vs-PLS comparison and a
PCA-vs-full-embedding-dims comparison, using real ad embeddings and human ratings
instead of `demo9.py`'s synthetic pool. See that repo's copies for the full
docstrings — the reasoning is unchanged, only the data paths and (for `demo7.py` /
`demo7_categorical_vs_embedding.py`) the import style (plain `import quantecarlo`,
no `sys.path` hack, since `quantecarlo` is `pip install -e`'d there) were updated.

### API contract (Modal endpoint)

Endpoint: `POST https://markshipman4273--bo-gp-service-gp-suggest.modal.run`

Request fields: `X`, `y` (higher = better), `candidates` (actual pool vectors),
`q`, `n_batches`, `train_steps`, `lr`, `xi`, `mode`.

Response: `{"candidates": [{"index": int, "x": [...], "mu": float, "sigma": float}]}`

Full spec: `~/projects/boaz/modal/API.md`
