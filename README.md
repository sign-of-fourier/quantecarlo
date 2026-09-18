# quantecarlo

Batch Bayesian optimization with **q-Expected Improvement (q-EI)**, backed by a hosted GP
service. Use it two ways:

| Entry point | When |
|---|---|
| [`QEIClient`](#direct-q-ei-no-optuna) | You run your own ask-tell loop. No Optuna required. |
| [`modal_suggest` / `fantasize_suggest`](#with-optuna-batchsampler) | You use Optuna via the optunahub [`BatchSampler`](https://hub.optuna.org/samplers/batch_sampler/). |

Both call the same service with the same contract. Each request sends your evaluated
points, their scores, and a candidate pool; the service fits a GP and returns the `q`
candidates with the highest **joint** expected improvement. Nothing is stored between
calls and nothing is fitted client-side.

```bash
pip install quantecarlo            # QEIClient only — numpy + scipy
pip install "quantecarlo[optuna]"  # adds optuna + optunahub for the BatchSampler path
```

---

## Direct q-EI (no Optuna)

You have a finite pool of candidates as vectors (products, ads, configurations, anything
embeddable) and an expensive score function. Each round, ask for the `q` most promising
untried items, evaluate them, append the results, repeat.

```python
import numpy as np
from quantecarlo import QEIClient

client = QEIClient()                       # or QEIClient(api_url="https://...")

observed  = [...]                          # indices into pool you have evaluated
scores    = [...]                          # their scores
remaining = [...]                          # indices not yet evaluated

for _ in range(N_ROUNDS):
    picks = client.suggest(
        X=pool[observed], y=scores, candidates=pool[remaining],
        q=4, direction="maximize",
    )
    chosen = [remaining[p["index"]] for p in picks]   # real members of your pool
    for i in chosen:
        observed.append(i)
        scores.append(evaluate(pool[i]))
    remaining = [i for i in remaining if i not in chosen]
```

`demos/demo_no_optuna.py` is this loop, complete and runnable.

**Continuous search space, no pool to enumerate?** Invent one with
[`sample_candidates`](#sample_candidates) and pass it as `candidates`:

```python
from quantecarlo import DimSpec, sample_candidates

dims = [DimSpec("lr", "float", 1e-4, 1e-1, log=True), DimSpec("hidden", "int", 16, 256)]
cands = sample_candidates(dims, n=512, seed=0)         # shape (512, 2)
picks = client.suggest(X, y, cands, q=4, direction="minimize")
```

### `QEIClient`

```python
QEIClient(api_url=DEFAULT_API_URL, *, timeout=120.0, train_steps=60, lr=0.1,
          xi=0.01, mode="production", **tuning)
```

Binds the endpoint and per-call defaults once. Construct one and reuse it.

| Parameter     | Default        | Description |
|---------------|----------------|-------------|
| `api_url`     | hosted service | GP service endpoint URL. |
| `timeout`     | `120.0`        | HTTP timeout in seconds. |
| `train_steps` | `60`           | Server-side GP fitting budget (iterations). More = better fit, slower call. `suggest` only; the multioutput / composite paths ignore it. |
| `lr`          | `0.1`          | Server-side GP fitting step size. `suggest` only, as above. |
| `xi`          | `0.01`         | Accepted for compatibility; the service ignores it. |
| `mode`        | `"production"` | `"debug"` fills `client.last_diagnostics` after each call (see [Debug diagnostics](#debug-diagnostics)). |
| `**tuning`    |                | Any [server-side tuning](#server-side-tuning) field, forwarded on every call. |

### `QEIClient.suggest`

```python
client.suggest(X, y, candidates, q, *, direction="maximize", n_batches=512) -> list[dict]
```

| Parameter    | Default      | Description |
|--------------|--------------|-------------|
| `X`          | *(required)* | Points you have evaluated, shape `(n_obs, n_dims)`. Same columns as `candidates`. |
| `y`          | *(required)* | Their scores, shape `(n_obs,)`. Raw values — no scaling or transform needed. |
| `candidates` | *(required)* | The pool to choose from, shape `(n_cands, n_dims)`. |
| `q`          | *(required)* | How many candidates to return. |
| `direction`  | `"maximize"` | `"maximize"` or `"minimize"` — which way `y` is better. |
| `n_batches`  | `512`        | Batches that survive the server's screening pass and are scored by joint q-EI. Only takes effect when the screen runs — `n_prefilter × q > ei_direct_max`, i.e. `q ≥ 5` with the server defaults. Below that every drawn batch is scored and this is ignored. See [Server-side tuning](#server-side-tuning). |

Returns a list of `q` dicts:

| Key     | Type         | Meaning |
|---------|--------------|---------|
| `index` | `int`        | Index into `candidates`. |
| `x`     | `np.ndarray` | The selected candidate vector. |
| `mu`    | `float`      | GP posterior mean at that point. Internal scale, higher-is-better: comparable across candidates within one call, not to your `y`. |
| `sigma` | `float`      | GP posterior standard deviation, same scale. |

Arrays or plain nested lists are both accepted. Fewer than `q` dicts come back only
when `candidates` has fewer than `q` rows.

### `QEIClient.suggest_multioutput` / `QEIClient.suggest_composite`

Same contract, for candidates from two related sources that should share one model
(e.g. two ad platforms). `suggest_multioutput(X, y, candidates, d_train, d_cands, q, *,
rho=0.5, direction, n_batches)` adds `d_train` / `d_cands` (output index `0` or `1` per
row — only two outputs are supported) and `rho` (cross-output correlation in `(-1, 1)`).
`train_steps` / `lr` have no effect on either. `suggest_composite(text, image,
has_image, y, text_candidates, image_candidates, has_image_candidates, d_train, d_cands,
q, *, rho, direction, n_batches)` is the one-row-per-item form for text + optional-image
inputs; rows with `has_image=0` send any placeholder image vector. Both return the same
`list[dict]` as `suggest`.

### `sample_candidates`

```python
sample_candidates(dims: list[DimSpec], n: int, seed=None) -> np.ndarray   # shape (n, len(dims))
```

Draws `n` random points inside the bounds of `dims` (log dims log-uniformly, int dims as
integers). For continuous search spaces only — if you have a real pool, pass that instead.

### Rules for `X` / `y` / `candidates`

1. `X` and `candidates` must be in the **same coordinate space** (same columns,
   same meaning). Any numeric embedding works; the service rescales internally.
2. Higher `y` is better by default. For a loss / error / cost, pass
   `direction="minimize"` (`QEIClient`, `modal_suggest`) or negate `y` yourself
   (`call_modal_api`).
3. `y` needs no scaling or transform of any kind. Send raw values.
4. `candidates` is the complete menu: the response only ever contains members
   of it. Include already-evaluated points only if re-evaluating them is
   acceptable.
5. Around 5 rows of `X` is the practical minimum for useful suggestions; fewer
   is allowed.

One call per round, not one per candidate. Nothing is stored server-side; every
request is self-contained.

### Preprocessing: reduce the dimensionality first

The service's model complexity grows with the number of columns in `X`. It needs
many more rows than columns to fit well. Sending 256- or 1536-dim embeddings
with 20 evaluated rows is under-determined and gives poor suggestions —
measured, not theoretical: raw 1536-dim embeddings hit the curse of
dimensionality in calibration runs.

So: project client-side before the call, and send the projected vectors.

- Fit PCA on the **pooled** rows — evaluated points and candidates together —
  and project both with the same fit. Refit every call; the basis should follow
  the current pool. Keep the full embeddings in your own storage.
- Pick `k` from your row count, not from the embedding size. With ~20 evaluated
  rows use **8–16 components**; go higher only as rows accumulate. PCA cannot
  return more components than pooled rows anyway.
- PCA is unsupervised: it keeps the high-variance axes, which are not
  necessarily the ones that predict `y`. With very few rows a supervised
  projection (PLS on `y`) may do better. Untested; flagged, not recommended.

```python
from sklearn.decomposition import PCA
import numpy as np

pool = np.vstack([X_full, cand_full])                # all rows, full dims
k = min(16, len(pool))
pca = PCA(n_components=k, random_state=0).fit(pool)
X, candidates = pca.transform(X_full), pca.transform(cand_full)
```

### Server-side tuning

`QEIClient`, `modal_suggest`, and the `call_modal_api*` functions accept these as extra
keyword arguments and forward them verbatim. Leave them out unless told
otherwise — the server defaults apply and the payload never carries a value
you didn't set. Unknown names raise `TypeError` client-side.

The server draws `n_prefilter` distinct size-`q` batches from the pool (every
combination when there are fewer). If `n_prefilter × q ≤ ei_direct_max` it scores
all of them by joint q-EI. Otherwise it ranks them with a cheaper screen and
scores only the top `n_batches`. With the defaults the screen kicks in at `q ≥ 5`.

| Field           | Default | Meaning |
|-----------------|---------|---------|
| `n_prefilter`   | `10000` | Batches drawn from the pool. This is the width of the search: more = wider, slower. |
| `ei_direct_max` | `40000` | Work budget in units of `n_prefilter × q`. Above it the screen runs. |
| `n_batches`     | `512`   | Batches that survive the screen. Ignored when the screen does not run. (Passed positionally, not via `**tuning`.) |
| `orthant_mode`, `order`, `gh_nodes`, `gh_nodes_corr` | | Numerical-accuracy settings for the q-EI computation. Leave unset unless instructed. |

### Debug diagnostics

With `mode="debug"` the response carries extra keys alongside the picks. `QEIClient`
stores them in `client.last_diagnostics` after every call; the `call_modal_api*`
functions fill a dict you pass as `diagnostics=`. They are for inspection, not
for the loop: posterior `mu`/`sigma` for every candidate, the winning batch's
score, the score of every batch that reached q-EI, wall-clock per stage, and the
screening pass's statistics when it ran. Key names follow the service's response
and may change.

```python
client = QEIClient(mode="debug")
picks = client.suggest(X, y, candidates, q=4)
print(client.last_diagnostics["timing_s"])
```

---

## With Optuna (`BatchSampler`)

Two drop-in `suggest_fn` implementations for the optunahub
[`BatchSampler`](https://hub.optuna.org/samplers/batch_sampler/). Both are for
**continuous search spaces** — a learning rate, a hidden-layer-size range — where there is
no pool to enumerate, so `n_probe_points` random points are invented per call. If you
already have a real pool, use [`QEIClient`](#direct-q-ei-no-optuna) instead.

| Function | Description |
|---|---|
| `fantasize_suggest` | Self-contained in-process GP (numpy/scipy). No server required. |
| `modal_suggest` | Delegates to the hosted GP service. Higher quality. |

```bash
pip install "quantecarlo[optuna]"
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
    suggest_fn=partial(modal_suggest, direction="minimize"),   # api_url defaults to the hosted service
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

### Optuna-path reference

#### `DimSpec`

Describes one dimension of the search space.

| Field  | Type                  | Description |
|--------|-----------------------|-------------|
| `name` | `str`                 | Must match the `suggest_*` call in your objective. |
| `type` | `"float"` \| `"int"` | Continuous float or integer (snapped on decode). |
| `low`  | `float`               | Lower bound (inclusive). |
| `high` | `float`               | Upper bound (inclusive). |
| `log`  | `bool`                | Log-uniform sampling. Default `False`. |
| `step` | `float \| None`       | Grid step for `int` dims. Default `1`. |

#### `modal_suggest`

```python
modal_suggest(X, y, search_space, q, *, direction="minimize", api_url, n_probe_points=512,
              n_candidate_batches=None, train_steps=60, lr=0.1, xi=0.01,
              mode="production", seed=None, timeout=120.0, **tuning)
```

Invents `n_probe_points` random continuous points from `search_space`'s bounds, sends them with `X`/`y` to the Modal GP endpoint, returns the highest q-EI batch. Bind parameters with `functools.partial` before passing to `BatchSampler`.

Only use this for a genuinely continuous `search_space`. If you have a real enumerable candidate pool, use [`QEIClient`](#direct-q-ei-no-optuna) directly.

| Parameter             | Default          | Description |
|-----------------------|------------------|-------------|
| `direction`           | `"minimize"`     | Must match the Optuna study direction. |
| `api_url`             | *(hosted)*       | Modal GP endpoint URL. |
| `n_probe_points`      | `512`            | Random continuous points invented per call and sent as the GP's candidate pool. Meaningless once you're calling `QEIClient` with real points — there's nothing left to invent. |
| `n_candidate_batches` | `n_probe_points` | Server's `n_batches`: batches that survive the screening pass and reach joint q-EI. Ignored unless the screen runs (see [Server-side tuning](#server-side-tuning)). Independent of `n_probe_points` — pass both explicitly to decouple them. |
| `train_steps`         | `60`             | Server-side GP fitting budget (iterations). |
| `lr`                  | `0.1`            | Server-side GP fitting step size. |
| `xi`                  | `0.01`           | Accepted for compatibility; the service ignores it. |
| `mode`                | `"production"`   | `"debug"` — diagnostics are discarded on this path; use `QEIClient` or `call_modal_api(diagnostics=...)` to receive them. |
| `seed`                | `None`           | Random seed for the invented candidate pool. |
| `timeout`             | `120.0`          | HTTP timeout in seconds. |
| `**tuning`            |                  | Any [server-side tuning](#server-side-tuning) field, forwarded verbatim. |

#### `fantasize_suggest`

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

### Low-level: `call_modal_api`

```python
call_modal_api(api_url, X, y, candidates, q=2, n_batches=512, train_steps=60,
               lr=0.1, xi=0.01, mode="production", timeout=120.0, diagnostics=None, **tuning)
```

The raw HTTP function everything above is built on. Same arguments and return value as
`QEIClient.suggest`, except `y` must already be higher-is-better (no `direction`),
inputs must be numpy arrays, and debug output goes into a dict you pass as
`diagnostics=`. `call_modal_api_multioutput` and `call_modal_api_composite`
correspond to the `QEIClient` methods of the same names. Prefer `QEIClient` unless you
need a plain function.

---

## Why q-EI instead of just adding more threads?

Running `study.optimize(n_jobs=q)` with a standard sampler (TPE, random) parallelises evaluation but each worker samples **independently** — it has no visibility into what the other `q-1` workers are about to try. Candidates often cluster near the same local optimum.

**q-EI scores the whole batch jointly.** It computes the expected improvement of the *best point in the batch* over the current best, accounting for the full joint posterior covariance across all `q` candidates. The algorithm naturally diversifies: a second candidate near an already-selected point contributes little to the joint maximum, so the batch spreads across promising but distinct regions.

Each batch of `q` trials carries more information than `q` independently-drawn trials. You reach good solutions in fewer total evaluations — which matters when each evaluation is expensive (a training run, an experiment, a simulation).
