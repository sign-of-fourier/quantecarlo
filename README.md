# quantecarlo

Batch Bayesian optimization for [Optuna](https://optuna.org) using **q-Expected Improvement (q-EI)**. Two drop-in `suggest_fn` implementations for the optunahub [`BatchSampler`](https://hub.optuna.org/samplers/batch_sampler/):

| Function | Description |
|---|---|
| `fantasize_suggest` | Self-contained in-process GP (numpy/scipy). No server required. |
| `modal_suggest` | Delegates to a hosted GPU GP endpoint (Modal). Higher quality, requires deployment. |

**Both of these are for continuous search spaces only** — spaces with no enumerable set
of valid points (a learning rate, a hidden-layer-size range). That's their whole reason
for existing: with nothing to enumerate, some finite set of points has to be invented
before a GP can be asked anything, and `n_probe_points` controls how many.

**If you already have a real, finite, enumerable set of candidates** — a product
catalog, an ad pool, embeddings for a fixed set of items — don't route it through
either `suggest_fn`. Call [`call_modal_api`](#call_modal_api) directly with your real
vectors as `candidates`. Going through `modal_suggest`/`DimSpec` in that case means
inventing random points, asking the GP to pick among the fakes, then snapping the
result back to the nearest real item afterward — an approximation of an approximation
that buys you nothing when the real candidates were available to hand the GP directly.

---

## Quickstart

```bash
pip install quantecarlo
pip install optunahub
```

### In-process GP — no server required

```python
import optuna
import optunahub
from functools import partial
from quantecarlo import DimSpec, fantasize_suggest

search_space = [
    DimSpec(name="x", type="float", low=-5.0, high=5.0),
    DimSpec(name="y", type="float", low=-5.0, high=5.0),
]

module = optunahub.load_module("package/samplers/batch_sampler")
BatchSampler = module.BatchSampler

sampler = BatchSampler(
    search_space=search_space,
    suggest_fn=partial(fantasize_suggest, direction="minimize"),
    q=4,
    n_startup_trials=8,
)

def objective(trial):
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return (x - 1.3) ** 2 + (y + 0.7) ** 2

study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=40)
print(study.best_params)
```

### Remote GP — Modal endpoint

```python
from quantecarlo import DimSpec, modal_suggest

sampler = BatchSampler(
    search_space=search_space,
    suggest_fn=partial(modal_suggest, direction="minimize",
                       api_url="https://markshipman4273--bo-gp-service-gp-suggest.modal.run"),
    q=4,
    n_startup_trials=8,
)
```

### Why ask-tell instead of `study.optimize`?

The ask-tell loop makes batching explicit and correct. With `study.optimize(n_jobs=q)`, each worker calls the sampler independently — no worker knows what the other `q-1` workers are about to try. Suggestions cluster.

The ask-tell pattern fixes this: all `q` asks happen before any evaluation. The first ask fires one API call that selects `q` jointly diverse candidates; asks 2 through `q` pop from a local cache. This is what makes joint q-EI meaningful in practice.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=Q) as executor:
    for _ in range(N_ITERATIONS):
        trials = [study.ask() for _ in range(Q)]          # fills cache on ask #1
        futures = {executor.submit(objective, t): t for t in trials}
        for future, trial in futures.items():
            study.tell(trial, future.result())
```

See `demos/demo.py` for the full working example.

---

## Reference

### `DimSpec`

Describes one dimension of the search space.

| Field  | Type                  | Description |
|--------|-----------------------|-------------|
| `name` | `str`                 | Must match the `suggest_*` call in your objective. |
| `type` | `"float"` \| `"int"` | Continuous float or integer (snapped on decode). |
| `low`  | `float`               | Lower bound (inclusive). |
| `high` | `float`               | Upper bound (inclusive). |
| `log`  | `bool`                | Log-uniform sampling. Default `False`. |
| `step` | `float \| None`       | Grid step for `int` dims. Default `1`. |

### `modal_suggest`

```python
modal_suggest(X, y, search_space, q, *, direction="minimize", api_url, n_probe_points=512,
              n_candidate_batches=None, train_steps=60, lr=0.1, xi=0.01,
              mode="production", seed=None, timeout=120.0)
```

Invents `n_probe_points` random continuous points from `search_space`'s bounds, sends them with `X`/`y` to the Modal GP endpoint, returns the highest q-EI batch. Bind parameters with `functools.partial` before passing to `BatchSampler`.

Only use this for a genuinely continuous `search_space`. If you have a real enumerable candidate pool, skip this function and call [`call_modal_api`](#call_modal_api) directly — see the note at the top of this README.

| Parameter             | Default          | Description |
|-----------------------|------------------|-------------|
| `direction`           | `"minimize"`     | Must match the Optuna study direction. |
| `api_url`             | *(hosted)*       | Modal GP endpoint URL. |
| `n_probe_points`      | `512`            | Random continuous points invented per call and sent as the GP's candidate pool. Meaningless once you're calling `call_modal_api` with real points — there's nothing left to invent. |
| `n_candidate_batches` | `n_probe_points` | How many random size-`q` index-combinations of that pool the server scores with joint q-EI. Independent of `n_probe_points` — pass both explicitly to decouple them. |
| `train_steps`         | `60`             | Adam steps for GP kernel optimisation. |
| `lr`                  | `0.1`            | Adam learning rate. |
| `xi`                  | `0.01`           | EI exploration bonus. |
| `mode`                | `"production"`   | `"debug"` returns full posterior arrays. |
| `seed`                | `None`           | Random seed for the invented candidate pool. |
| `timeout`             | `120.0`          | HTTP timeout in seconds. |

### `call_modal_api`

```python
call_modal_api(api_url, X, y, candidates, q=2, n_batches=512, train_steps=100,
                lr=0.1, xi=0.01, mode="production", timeout=120.0)
```

The raw Modal HTTP client `modal_suggest` is built on — and the function to call directly whenever you have a real, materialized set of candidates (an embedded product catalog, an ad pool, any finite list you can turn into vectors). No `DimSpec`, no `BatchSampler`, no invented points: `candidates` is exactly the array you pass, and each returned `index` is a real index into it.

```python
import numpy as np
from quantecarlo import call_modal_api

# X/y: your observed points and their scores so far (higher = better; negate first if minimizing)
# candidates: the real, unclaimed pool — e.g. embeddings for products/ads not yet tried
picks = call_modal_api(api_url, X, y, candidates, q=4)
for p in picks:
    real_item = candidates[p["index"]]   # already a real candidate — no snapping needed
```

| Parameter     | Default        | Description |
|---------------|----------------|-------------|
| `api_url`     | *(required)*   | Modal GP endpoint URL. |
| `X`           | *(required)*   | Observed points, shape `(n_obs, n_dims)`. |
| `y`           | *(required)*   | Observed scores, higher = better. Negate first for minimisation. |
| `candidates`  | *(required)*   | Discrete candidate pool to select from, shape `(n_cands, n_dims)` — your real items. |
| `q`           | `2`            | Number of candidates to return. |
| `n_batches`   | `512`          | Random size-`q` index-combinations of `candidates` scored with joint q-EI before the best one is returned. |
| `train_steps` | `100`          | Adam steps for GP kernel optimisation. |
| `lr`          | `0.1`          | Adam learning rate. |
| `xi`          | `0.01`         | EI exploration bonus. |
| `mode`        | `"production"` | `"debug"` returns full posterior arrays. |
| `timeout`     | `120.0`        | HTTP timeout in seconds. |

Returns `list[dict]`, each with `index` (int, into `candidates`), `x` (the candidate vector), `mu` (GP posterior mean), `sigma` (GP posterior std).

See `demos/demo7_pca_vs_pls.py`'s `run_arm_qei` for a worked ask-tell loop built directly on this function, with no Optuna/`BatchSampler` involved at all.

### `fantasize_suggest`

```python
fantasize_suggest(X, y, search_space, q, direction="minimize", n_candidates=512,
                  noise=1e-3, xi=0.01, seed=None)
```

In-process RBF GP with sequential kriging (fantasization). Picks one candidate per GP fit, then fantasizes its outcome as the posterior mean before the next pick — so the batch spreads across the space without a remote call.

| Parameter      | Default      | Description |
|----------------|--------------|-------------|
| `direction`    | `"minimize"` | Must match the Optuna study direction. |
| `n_candidates` | `512`        | Random candidates evaluated per GP call. |
| `noise`        | `1e-3`       | GP observation noise variance. |
| `xi`           | `0.01`       | EI exploration bonus. |
| `seed`         | `None`       | Random seed for the candidate pool. |

---

## Why q-EI instead of just adding more threads?

Running `study.optimize(n_jobs=q)` with a standard sampler (TPE, random) parallelises evaluation but each worker samples **independently** — it has no visibility into what the other `q-1` workers are about to try. Candidates often cluster near the same local optimum.

**q-EI scores the whole batch jointly.** It computes the expected improvement of the *best point in the batch* over the current best, accounting for the full joint posterior covariance across all `q` candidates. The algorithm naturally diversifies: a second candidate near an already-selected point contributes little to the joint maximum, so the batch spreads across promising but distinct regions.

Each batch of `q` trials carries more information than `q` independently-drawn trials. You reach good solutions in fewer total evaluations — which matters when each evaluation is expensive (a training run, an experiment, a simulation).
