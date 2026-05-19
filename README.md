# quantecarlo

Batch Bayesian optimization for [Optuna](https://optuna.org) using **q-Expected Improvement (q-EI)**. Drop in one sampler, point it at a hosted GP endpoint, and get a batch of `q` well-chosen candidates back per iteration instead of one at a time.

---

## Quickstart

```bash
pip install quantecarlo
```

```python
from quantecarlo import DimSpec, ModalBOSampler
import optuna

SEARCH_SPACE = [
    DimSpec(name="lr",      type="float", low=1e-4, high=1e-1, log=True),
    DimSpec(name="depth",   type="int",   low=1,    high=20),
    DimSpec(name="dropout", type="float", low=0.0,  high=0.5),
]

sampler = ModalBOSampler(
    api_url="https://<your-endpoint>/gp_suggest",
    search_space=SEARCH_SPACE,
    n_startup_trials=8,
    q=4,
)

study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(your_objective, n_trials=60, n_jobs=4)
```

The `suggest_*` calls inside your objective must use the same names, bounds, and `log` flags as the `DimSpec` list.

See `demo.py` for a complete ask-tell example against a breast-cancer MLP objective.

---

## What's happening under the hood (you don't need to touch any of this)

Each time the local suggestion cache runs dry, `ModalBOSampler` POSTs your observed `(X, y)` pairs to a remote GP service. That service:

1. **Normalises** each parameter to [0, 1] (log-scale for `log=True` dims).
2. **Rank-transforms** `y` to standard-normal via the Probability Integral Transform — so the GP always sees well-behaved Gaussian targets regardless of the shape of your objective's distribution.
3. **Fits an ExactGP** (Matérn-5/2 ARD kernel) on a GPU via Adam on the marginal log-likelihood.
4. **Draws `n_candidates` random candidate batches** of size `q` and scores each batch jointly with q-EI.
5. **Returns the highest-scoring batch** decoded back to your original parameter scale. Int dims are snapped to the nearest integer.

The sampler then hands out one candidate per `study.ask()` call from the local cache. The next API call doesn't fire until the cache is exhausted — so `q` threads share a single round-trip.

---

## Parameters

### `DimSpec`

Describes one dimension of your search space.

| Field  | Type                    | Description |
|--------|-------------------------|-------------|
| `name` | `str`                   | Must match the corresponding `suggest_*` call in your objective. |
| `type` | `"float"` \| `"int"`   | Continuous float or integer. Int dims are snapped on decode. |
| `low`  | `float`                 | Lower bound (inclusive). |
| `high` | `float`                 | Upper bound (inclusive). |
| `log`  | `bool`                  | Log-uniform sampling. Use for parameters that span orders of magnitude (learning rates, weight decay). Default `False`. |
| `step` | `float \| None`         | Grid step for `int` dims. Default `1`. |

Categorical dimensions are not yet supported.

### `ModalBOSampler`

| Parameter         | Default        | Description |
|-------------------|----------------|-------------|
| `api_url`         | —              | URL of the hosted GP endpoint. |
| `search_space`    | —              | List of `DimSpec`, one per hyperparameter. |
| `n_startup_trials`| `8`            | Number of random trials before the GP is used. Too few observations make GP fitting unreliable. |
| `q`               | `4`            | Batch size. Set `n_jobs=q` in `study.optimize` to evaluate the batch in parallel. |
| `n_candidates`    | `512`          | Random candidate batches scored per API call. Larger = better coverage; diminishing returns above ~1024 for most spaces. |
| `train_steps`     | `60`           | Adam steps for GP kernel hyperparameter optimisation. Increase for tighter fits on noisy objectives. |
| `lr`              | `0.1`          | Adam learning rate for GP training. |
| `xi`              | `0.01`         | EI exploration bonus. Larger values bias toward uncertain regions; smaller values exploit the current best. |
| `mode`            | `"production"` | `"debug"` returns the full GP posterior surface in the API response — useful for diagnostics. |
| `seed`            | `None`         | Random seed for the fallback random sampler. |
| `timeout`         | `120.0`        | HTTP timeout in seconds for the API call. |

---

## Why q-EI instead of just adding more threads?

Running `study.optimize(n_jobs=q)` with a standard sampler (TPE, random) does parallelize objective evaluation, but each worker samples **independently** — it has no idea what the other `q-1` workers are about to try. You often end up with a batch where several candidates cluster near the same local optimum.

**q-EI scores the whole batch jointly.** It computes the expected improvement of the *best point in the batch* over the current best, taking into account the full joint posterior covariance across the `q` candidates. The optimizer naturally diversifies: a second candidate near an already-selected point contributes little to the joint maximum, so the algorithm spreads the batch across promising but distinct regions.

In practice this means each batch of `q` trials carries more information than `q` independently-drawn trials. You cover the space more efficiently and tend to reach good solutions in fewer total function evaluations — which matters when each evaluation is expensive (a training run, an experiment, a simulation).

The cost is one API call per batch (a few seconds for a warm GP endpoint) in exchange for a smarter set of `q` candidates. That tradeoff is almost always worth it when objective evaluations take more than a minute.
