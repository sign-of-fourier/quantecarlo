# quantecarlo

Batch Bayesian optimization for [Optuna](https://optuna.org) using **q-Expected Improvement (q-EI)**. Two drop-in `suggest_fn` implementations for the optunahub [`BatchSampler`](https://hub.optuna.org/samplers/batch_sampler/):

| Function | Description |
|---|---|
| `fantasize_suggest` | Self-contained in-process GP (numpy/scipy). No server required. |
| `modal_suggest` | Delegates to a hosted GPU GP endpoint (Modal). Higher quality, requires deployment. |

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
modal_suggest(X, y, search_space, q, *, direction="minimize", api_url, n_candidates=512,
              train_steps=60, lr=0.1, xi=0.01, mode="production", seed=None, timeout=120.0)
```

Sends `X`, `y`, and a random candidate pool to the Modal GP endpoint; returns the highest q-EI batch. Bind parameters with `functools.partial` before passing to `BatchSampler`.

| Parameter      | Default          | Description |
|----------------|------------------|-------------|
| `direction`    | `"minimize"`     | Must match the Optuna study direction. |
| `api_url`      | *(hosted)*       | Modal GP endpoint URL. |
| `n_candidates` | `512`            | Random candidates scored per call. |
| `train_steps`  | `60`             | Adam steps for GP kernel optimisation. |
| `lr`           | `0.1`            | Adam learning rate. |
| `xi`           | `0.01`           | EI exploration bonus. |
| `mode`         | `"production"`   | `"debug"` returns full posterior arrays. |
| `seed`         | `None`           | Random seed for the candidate pool. |
| `timeout`      | `120.0`          | HTTP timeout in seconds. |

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
