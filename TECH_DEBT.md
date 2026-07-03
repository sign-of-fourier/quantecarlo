# TECH_DEBT.md — Quantecarlo Cleanup Plan

This file is a self-standing briefing for picking up work across the quantecarlo
ecosystem. Read this first. It tells you exactly which files to read in which
repos without having to survey everything.

---

## The Ecosystem

| Repo | Role | Key files |
|---|---|---|
| `~/projects/quantecarlo` | pip package — suggest_fns + shared Modal HTTP client | `CLAUDE.md`, `quantecarlo/_modal_api.py`, `quantecarlo/bo_sampler.py`, `quantecarlo/_fantasize.py` |
| `~/projects/boaz/modal` | Modal GP **server only** — no client code lives here | `modal_gp_api.py`, `API.md` |
| `~/projects/optunahub-registry` | Community BatchSampler package — PR #376 submitted, do not modify | `package/samplers/batch_sampler/_bo_sampler.py` (read-only reference) |
| `~/projects/meta-ads-demo` | Production ads app — imports Modal client from quantecarlo | `backend/bo_pipeline/modal_bo.py`, `backend/bo_pipeline/cross_platform.py`, `backend/bo_pipeline/pipeline.py` |

---

## Architecture

`BatchSampler` (optunahub-registry) owns threading/lock/cache. It calls a
user-supplied `suggest_fn(X, y, search_space, q) -> list[dict]`. quantecarlo
provides two: `fantasize_suggest` (in-process numpy/scipy GP) and `modal_suggest`
(POSTs to the Modal endpoint).

`quantecarlo/_modal_api.py` is the **single source of truth** for the Modal HTTP
contract. Both `modal_suggest` and `meta-ads-demo` import from it. When the API
payload changes, update `_modal_api.py` only.

`meta-ads-demo` uses `call_modal_api` / `call_modal_api_multioutput` imported from
quantecarlo. The re-export lives in `bo_pipeline/modal_bo.py` alongside the
app-specific PCA helpers and env-var config that stay there permanently.

---

## Critical conventions — do not get these wrong

**y convention — higher is better throughout**: the Modal API is a maximisation
service. `call_modal_api` and `call_modal_api_multioutput` expect higher = better
and pass y straight through. `modal_suggest` and `fantasize_suggest` each take a
`direction` parameter (`"minimize"` or `"maximize"`, matching the Optuna study
direction) and handle the negation internally. Pass `direction` via
`functools.partial` when building the `suggest_fn` for `BatchSampler`.

**meta-ads-demo controls sign itself**: `call_modal_api` has no `direction` param.
The app negates (or doesn't) before calling. Do not add auto-negation there.

**Two Modal workspace URLs** — both are valid deployments of the same server code:
- `https://markshipman4273--bo-gp-service-gp-suggest.modal.run`
- `https://info-29741--bo-gp-service-gp-suggest.modal.run`

**optunahub-registry**: PR #376 is pending. Until merged, load locally:
```python
optunahub.load_local_module("samplers/batch_sampler",
    registry_root=os.path.expanduser("~/projects/optunahub-registry/package"))
```

**quantecarlo in meta-ads-demo**: requirements.txt pins `quantecarlo>=0.1.2`.
Currently satisfied by the local editable install (`pip install -e .` in
`~/projects/quantecarlo`). After Stage 5 (PyPI publish), this will resolve from
PyPI instead. Until then, the editable install must be present.

---

## What was completed (session 2026-06-14)

### Direction convention (pre-Stage 1)
- `modal_suggest` and `fantasize_suggest` now take `direction="minimize"|"maximize"`
  instead of always negating y. Fixes silent bug where maximize studies would have
  been wrongly inverted.
- `demos/demo.py` updated to pass `direction="minimize"` explicitly.

### Stage 1 — broken imports fixed ✅
- `demos/demo7.py` — replaced `qEISampler` import with `BatchSampler + modal_suggest`.
  Note: demo7 requires external data files (`ads_all_labels.json`, `embedding_cache/`)
  not included in this repo. It will import cleanly but cannot run without those files.
- `boaz/modal/CLAUDE.md` — removed stale `bo_sampler.py` STALE entry and its
  architecture section. Updated data-flow diagram.
- `quantecarlo/README.md` — fully rewritten to document `modal_suggest`,
  `fantasize_suggest`, `DimSpec`, and the ask-tell pattern. `qEISampler` gone.

### Stage 2 — tests ✅
- `tests/test_bo_sampler.py` — 13 unit tests: `_sample_candidates` shapes/bounds,
  `modal_suggest` payload keys, y-negation for both directions, n_candidates,
  return format, int rounding, HTTPError surfacing. All pass.
- `tests/test_fantasize.py` — 10 unit tests: return shape, param names, bounds,
  log dims, int dims, single-obs edge case, direction divergence, constant-y
  stability. All pass.
- `tests/test_modal_live.py` — 5 live tests against the Modal endpoint: q dicts,
  param names, float bounds, int dims, maximize direction. All pass (endpoint live).
- **Patch path**: `quantecarlo._modal_api.urllib.request.urlopen` (not bo_sampler).

### Stage 3 — MultiGroupqEISampler deleted ✅
- `quantecarlo/bo_sampler_multigroup.py` deleted. It was an untested prototype with
  no consumers that duplicated `_post` and `_sample_candidates` from `bo_sampler.py`
  and used the old always-negate convention.
- Removed from `__init__.py` `__all__`.

### Stage 6 — Modal HTTP client unified ✅
This was done ahead of Stages 4 and 5 because it was the highest-value change.

**New file**: `quantecarlo/_modal_api.py`
- `call_modal_api(api_url, X, y, candidates, q, n_batches, train_steps, lr, xi, mode, timeout)`
- `call_modal_api_multioutput(...)` — same plus `d_train, d_cands, rho`
- `_post(api_url, payload, timeout)` — shared HTTP primitive with HTTPError body surfacing
- Both functions take numpy arrays, return `list[dict]` with `index`, `x`, `mu`, `sigma`

**`quantecarlo/bo_sampler.py`** — `modal_suggest` refactored to call `call_modal_api`
internally. Its own `_post` removed.

**`meta-ads-demo/backend/bo_pipeline/modal_bo.py`** — ~100 lines of duplicate HTTP
code removed. Now imports and re-exports `call_modal_api`, `call_modal_api_multioutput`
from quantecarlo. PCA helpers, env-var config, and `modal_bo_enabled()` remain here.

**Callsite param renames** (4 sites across 2 files):
- `cross_platform.py` lines ~329, ~484, ~840: `X_train_pca=` → `X=`, `X_cands_pca=` → `candidates=`
- `pipeline.py` line ~175: same rename

**`meta-ads-demo/backend/tests/test_modal_bo.py`** — two unit tests updated:
- Patch path changed from `urllib.request.urlopen` → `quantecarlo._modal_api.urllib.request.urlopen`
- Param names updated to match new interface

**`meta-ads-demo/backend/requirements.txt`** — added `quantecarlo>=0.1.2`

---

## What remains

### Stage 4 — Documentation cleanup ✅
- `~/projects/boaz/modal/old/` deleted.
- `CLAUDE.md` had no stale `MultiGroupqEISampler` references (already clean).
- `qEISampler` references in quantecarlo repo are only in TECH_DEBT.md historical
  notes and the auto-generated `egg-info/PKG-INFO` (regenerated on every build).

### Stage 5 — Version bump and publish to PyPI ✅
- Bumped to `0.2.0` in `pyproject.toml`.
- Built and uploaded to PyPI (`twine upload dist/*`).
- `optunahub-registry/example.py` already used `load_module` (no change needed).
- `meta-ads-demo/requirements.txt` pin updated to `quantecarlo>=0.2.0`.

### Stage 7 — Republish after BatchSampler migration ✅
The BatchSampler/multi-output/fantasize_suggest work above (Stages 4–6) was committed
(`3d73aa0`) and pushed to `main` while `pyproject.toml` still said `0.2.0` — but `0.2.0`
was already published on PyPI from Stage 5, with the old `qEISampler` API. That left
git `main` and the published `0.2.0` disagreeing under the same version string.
- Bumped to `0.3.0` (`6c0430a`) — minor, not patch, since `qEISampler` removal breaks
  existing callers.
- Built and uploaded to PyPI: https://pypi.org/project/quantecarlo/0.3.0/
- `meta-ads-demo/requirements.txt` pin (`quantecarlo>=0.2.0`) not bumped — resolves to
  `0.3.0` fine as a floor, and meta-ads-demo never used `qEISampler`/`BatchSampler` so
  the breaking change doesn't affect it.

---

## Manual testing checklist

See the section below for what to verify by hand before Stage 5.

---

## What to test manually before publishing

The automated tests cover the contract of individual functions. They do not cover
end-to-end flows or catch subtle integration bugs. Here is what to verify by hand.

### 1. `demos/demo.py` — the primary end-to-end smoke test

```bash
cd ~/projects/quantecarlo
python demos/demo.py
```

**What it does**: runs a NAS study on the breast-cancer dataset using
`BatchSampler + modal_suggest` with `direction="minimize"`. Runs 15 iterations × q=4.

**What to look for**:
- No import errors or crashes on startup
- First 8 iterations (startup): suggestions look random (they are — fallback to random sampler)
- After iteration 8+: `best=` value should trend downward (GP is now guiding search)
- No HTTP errors from the Modal endpoint
- Final best value should be somewhere around 0.02–0.06 (1 − accuracy on breast-cancer)

**What changed that could break this**:
- `modal_suggest` now builds numpy arrays and calls `call_modal_api` internally instead
  of building a dict payload directly. The payload sent to Modal is identical but the
  code path changed. If you see HTTP errors or wrong results, compare the printed
  `best=` values to a run before this session.

### 2. `modal_suggest` direction sanity check

Run this in a Python shell:

```python
from functools import partial
from quantecarlo import DimSpec, modal_suggest

dims = [DimSpec("x", "float", -5.0, 5.0), DimSpec("y", "float", -5.0, 5.0)]
X = [[-4.0, -4.0], [0.0, 0.0], [4.0, 4.0]]

# Minimize: best observed is X[0] (y=0.1, lowest). GP should explore away from it.
result_min = modal_suggest(X, [0.1, 1.0, 2.0], dims, q=2, direction="minimize",
    api_url="https://markshipman4273--bo-gp-service-gp-suggest.modal.run",
    n_candidates=64, train_steps=20)

# Maximize: best observed is X[2] (y=2.0, highest). GP should explore away from it.
result_max = modal_suggest(X, [0.1, 1.0, 2.0], dims, q=2, direction="maximize",
    api_url="https://markshipman4273--bo-gp-service-gp-suggest.modal.run",
    n_candidates=64, train_steps=20)

print("minimize suggestions:", result_min)
print("maximize suggestions:", result_max)
```

**What to look for**:
- Both calls return 2 dicts with keys `"x"` and `"y"`
- The two result sets are meaningfully different (minimize steers one way, maximize another)
- No crash or HTTP error

**What changed that could break this**: `direction` param is new. Before this session,
`modal_suggest` always negated y (equivalent to `direction="minimize"` only). The new
code path for `direction="maximize"` is only covered by the live test — it has not
been run against a real multi-iteration study.

### 3. meta-ads-demo BO pipeline

The BO pipeline in meta-ads-demo now imports `call_modal_api` and
`call_modal_api_multioutput` from quantecarlo instead of defining them locally.
The interface is the same but the code path changed.

**Quick import check**:
```bash
cd ~/projects/meta-ads-demo/backend
python -c "from bo_pipeline.modal_bo import call_modal_api, call_modal_api_multioutput, fit_pca, modal_bo_enabled; print('OK')"
```

**What to look for**: prints `OK` with no errors. If you see `ModuleNotFoundError:
quantecarlo`, the editable install is missing — run `pip install -e ~/projects/quantecarlo`.

**Full pipeline test**: if you have the meta-ads-demo backend running with
`MODAL_BO_API_URL` set, trigger a BO round and check that suggestions come back.
The existing `tests/test_modal_bo.py` unit tests all pass (verified), but a live
round through the full `cross_platform.py` → `call_modal_api_multioutput` path
has not been exercised since the rename.

**What changed that could break this**:
- The 4 callsites in `cross_platform.py` and `pipeline.py` had keyword argument
  renames: `X_train_pca=` → `X=`, `X_cands_pca=` → `candidates=`. If any call
  was missed, you'll get `TypeError: unexpected keyword argument` at runtime.
  The grep at the time of editing showed exactly 4 sites; all were updated.

### 4. What the automated tests do NOT cover

- **`demo7.py` actually running**: it imports cleanly but requires `ads_all_labels.json`
  and `embedding_cache/` data files not in this repo. The import is verified; a full
  run is not.
- **`fantasize_suggest` in a real Optuna study**: unit tests verify the math and
  output format, but it has not been run inside a `BatchSampler` end-to-end since
  the `direction` parameter was added.
- **`MultiGroupqEISampler` consumers**: there were none, but if you find something
  that imported it, it will now fail with `ImportError`. Grep: `grep -r "MultiGroupqEISampler" ~/projects/`.

---

## Order of operations (remaining)

```
Stage 4  →  Stage 5  →  (meta-ads-demo live test)
  docs       publish      confirm pipeline works
  cleanup    to PyPI      with published package
```

Stage 4 has no external dependencies.
Stage 5 requires Modal endpoint running for the post-publish optunahub update.
