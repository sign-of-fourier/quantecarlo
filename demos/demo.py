# demo.py — ask-tell tutorial for BatchSampler + modal_suggest
#
# Usage:
#   1. Deploy the GP service (see ~/projects/boaz/modal — modal deploy modal_gp_api.py)
#   2. pip install -e .   (from repo root)
#   3. pip install optunahub
#   4. python demos/demo.py
#
# The ask-tell loop is explicit rather than study.optimize() so the batching
# contract is visible: q sequential asks fill the cache once, then q parallel
# objective evaluations run, then q tells report back before the next batch.

import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import optuna
import optunahub
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from optuna.trial import TrialState

from quantecarlo import DimSpec, modal_suggest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "https://markshipman4273--bo-gp-service-gp-suggest.modal.run"

Q = 4
N_STARTUP = 8
N_ITERATIONS = 15

SEARCH_SPACE = [
    DimSpec(name="lr",       type="float", low=1e-4, high=1e-1, log=True),
    DimSpec(name="n_hidden", type="int",   low=16,   high=256),
    DimSpec(name="alpha",    type="float", low=1e-5, high=1e-2, log=True),
]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_X, _y = load_breast_cancer(return_X_y=True)
_X_train, _X_val, _y_train, _y_val = train_test_split(
    _X, _y, test_size=0.2, random_state=42, stratify=_y
)

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """1 − 3-fold CV accuracy on breast-cancer. Minimisation target."""
    lr       = trial.suggest_float("lr",       1e-4, 1e-1, log=True)
    n_hidden = trial.suggest_int(  "n_hidden", 16,   256)
    alpha    = trial.suggest_float("alpha",    1e-5, 1e-2, log=True)

    clf = MLPClassifier(
        hidden_layer_sizes=(n_hidden,),
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=300,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = cross_val_score(clf, _X_train, _y_train, cv=3, scoring="accuracy")
    return 1.0 - float(scores.mean())

# ---------------------------------------------------------------------------
# Ask-tell helpers
# ---------------------------------------------------------------------------

def run_batch(study: optuna.Study, executor: ThreadPoolExecutor) -> None:
    # Ask: Q sequential calls.
    #   - Call 1 acquires the lock; if cache is empty, blocks on the API, fills cache.
    #   - Calls 2..Q acquire the lock, find the cache populated, pop immediately.
    # During startup (< n_startup complete trials) all calls fall back to random.
    trials = [study.ask() for _ in range(Q)]

    future_to_trial = {executor.submit(objective, t): t for t in trials}
    for future, trial in future_to_trial.items():
        try:
            study.tell(trial, future.result())
        except Exception as exc:
            warnings.warn(f"Trial {trial.number} raised {exc!r}; marking FAIL.")
            study.tell(trial, state=TrialState.FAIL)


def summarize_study_by_iteration(study: optuna.Study, batch_size: int) -> pd.DataFrame:
    completed = sorted(
        [t for t in study.trials if t.state == TrialState.COMPLETE],
        key=lambda t: t.number,
    )
    rows = []
    for i in range(0, len(completed), batch_size):
        batch = completed[i : i + batch_size]
        so_far = completed[: i + batch_size]
        rows.append({
            "iteration":       i // batch_size + 1,
            "trials":          f"{batch[0].number}–{batch[-1].number}",
            "batch_best":      min(t.value for t in batch),
            "cumulative_best": min(t.value for t in so_far),
            "n_complete":      len(so_far),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    module = optunahub.load_module(package="samplers/batch_sampler")
    BatchSampler = module.BatchSampler

    sampler = BatchSampler(
        search_space=SEARCH_SPACE,
        suggest_fn=partial(modal_suggest, direction="minimize", api_url=API_URL, n_probe_points=512, train_steps=75),
        n_startup_trials=N_STARTUP,
        q=Q,
        seed=42,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"Running {N_ITERATIONS} iterations × q={Q} = {N_ITERATIONS * Q} trials\n")

    with ThreadPoolExecutor(max_workers=Q) as executor:
        for iteration in range(1, N_ITERATIONS + 1):
            run_batch(study, executor)
            print(
                f"  iter {iteration:3d}/{N_ITERATIONS}"
                f"  best={study.best_value:.4f}"
                f"  {study.best_trial.params}"
            )

    print("\nPer-iteration summary:")
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summarize_study_by_iteration(study, batch_size=Q).to_string(index=False))

    print("\nBest trial:")
    print(f"  value : {study.best_value:.4f}")
    print(f"  params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
