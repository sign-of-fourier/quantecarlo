# demo7.py — Experiment A (qEISampler ask-tell) vs Experiment B (Optuna TPE, n_jobs=4)
#
# Both experiments use Optuna on a pool of ads with pre-computed image embeddings
# and ground-truth ratings (1–7).
#
# Experiment A: ask-tell with qEISampler
#   study.ask() × Q  →  qEISampler (GP + q-EI)  →  nearest-pool-member  →  study.tell() × Q
#   Sampling is sequential and batch-aware: all Q asks precede any evaluation, so
#   the GP sees an empty pending cache on ask #1 and selects Q jointly diverse
#   candidates in one API call.  Requires a running GP endpoint.
#
# Experiment B: study.optimize(n_jobs=4) with default TPE
#   Four threads call TPE concurrently.  Each thread samples from the same
#   completed-trial snapshot, so all four suggestions come from the same KDE
#   "good-region" model.  This is the cloning effect: suggestions cluster in
#   one corner of the space rather than spreading across it.
#
# Both arms share the same 16-ad random warm-up per seed (injected via
# study.add_trial so the GP and TPE start with identical history).
#
# Data files expected in the current working directory:
#   ads_all_labels.json  — JSON list of ad objects; each has an "id" field (string)
#                          and a "ratings" field (list of ints, 1–7).
#   embedding_cache/     — directory of .npy files; each file is named <ad_id>.npy,
#                          matching the ad's "id" field.  Each file stores a 1024-dim
#                          float32 vector — the image embedding for that ad.
#
# This demo is not self-contained: the data files are part of a separate experiment
# repo and are not included here.  It is provided as a reference implementation of
# the qEISampler ask-tell pattern on a real pool-based search task.
#
# Usage:
#   1. Deploy the GP service (see README for the backend repo / hosted endpoint).
#   2. Paste the endpoint URL into API_URL below.
#   3. python demos/demo7.py

import sys
import os
import threading
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

import optuna
from optuna.distributions import FloatDistribution
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState

from quantecarlo import DimSpec, qEISampler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "https://your-gp-endpoint.example.com/suggest"

CHI_JSON  = Path("ads_all_labels.json")
EMB_CACHE = Path("embedding_cache")

N_REPS       = 12
N_ITERATIONS = 10
BATCH_SIZE   = 4
PCA_DIMS     = 64
WARM_UP      = 16

# "maximize" → find most problematic ads (highest rating) first
# "minimize" → find least problematic ads (lowest rating) first
DIRECTION = "maximize"

# ---------------------------------------------------------------------------
# Data loading and PCA
# ---------------------------------------------------------------------------

def load_data():
    with open(CHI_JSON) as f:
        ads = json.load(f)
    ad_ids, ratings, embs = [], [], []
    for ad in ads:
        p = EMB_CACHE / f"{ad['id']}.npy"
        if p.exists():
            ad_ids.append(ad["id"])
            ratings.append(float(np.mean(ad["ratings"])))
            embs.append(np.load(p))
    ratings = np.array(ratings, dtype=np.float32)
    embs    = np.stack(embs)
    print(f"Loaded {len(ad_ids)} ads | rating {ratings.min():.2f}–{ratings.max():.2f}")
    return np.array(ad_ids), ratings, embs


def fit_pca(embs: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=PCA_DIMS, random_state=42)
    X = pca.fit_transform(embs).astype(np.float32)
    print(f"PCA: {embs.shape[1]}→{PCA_DIMS} dims | explained var: {pca.explained_variance_ratio_.sum():.3f}")
    return X

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dim_bounds(X_pca: np.ndarray) -> list[tuple[float, float]]:
    return [(float(X_pca[:, i].min()), float(X_pca[:, i].max())) for i in range(PCA_DIMS)]


def _study_value(rating: float) -> float:
    # qEISampler negates trial values before passing to q-EI (which maximises).
    # Passing -rating for a maximisation objective means the sampler computes
    # y = -(-rating) = +rating, and q-EI then seeks the highest rating.
    # For minimisation, passing +rating means y = -rating, so q-EI maximises
    # -rating, i.e. minimises rating.  The same convention is used for TPE so
    # both experiments see identical stored values.
    return -float(rating) if DIRECTION == "maximize" else float(rating)


def _original_rating(study_value: float) -> float:
    return -study_value if DIRECTION == "maximize" else study_value


def _inject_warmup(
    study: optuna.Study,
    X_pca: np.ndarray,
    observed: list[int],
    ratings: np.ndarray,
    dists_map: dict,
) -> None:
    """Add warm-up pool members as completed trials (bypasses sampling)."""
    now = datetime.now()
    for idx in observed:
        params = {f"pca_{i}": float(X_pca[idx, i]) for i in range(PCA_DIMS)}
        study.add_trial(FrozenTrial(
            number=-1, trial_id=-1,
            state=TrialState.COMPLETE,
            value=_study_value(ratings[idx]),
            datetime_start=now, datetime_complete=now,
            params=params, distributions=dists_map,
            user_attrs={}, system_attrs={}, intermediate_values={},
        ))

# ---------------------------------------------------------------------------
# Experiment A: qEISampler + ask-tell
# ---------------------------------------------------------------------------

def run_arm_qei(
    X_pca: np.ndarray,
    ratings: np.ndarray,
    bounds: list[tuple[float, float]],
    seed: int,
) -> np.ndarray:
    rng      = np.random.default_rng(seed)
    perm     = rng.permutation(len(ratings))
    observed = list(perm[:WARM_UP].tolist())
    remaining = list(perm[WARM_UP:].tolist())

    search_space = [
        DimSpec(name=f"pca_{i}", type="float", low=bounds[i][0], high=bounds[i][1])
        for i in range(PCA_DIMS)
    ]
    fixed_dists = {f"pca_{i}": FloatDistribution(bounds[i][0], bounds[i][1])
                   for i in range(PCA_DIMS)}

    sampler = qEISampler(
        api_url=API_URL,
        search_space=search_space,
        n_startup_trials=WARM_UP,
        q=BATCH_SIZE,
        n_candidates=512,
        train_steps=100,
        seed=seed,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    _inject_warmup(study, X_pca, observed, ratings, fixed_dists)

    cumulative_best = []

    for _ in range(N_ITERATIONS):
        # All Q asks before any tell.  qEISampler fires one API call on ask #1
        # (cache empty), fills its deque with Q candidates, then pops for asks
        # #2..Q without a second API call.
        trials_batch = [study.ask(fixed_distributions=fixed_dists) for _ in range(BATCH_SIZE)]

        # Map each GP-suggested PCA point to the nearest unvisited pool member.
        # Dedup within the batch so no two asks claim the same ad.
        used_in_batch: set[int] = set()
        for trial in trials_batch:
            coords = np.array([trial.params[f"pca_{i}"] for i in range(PCA_DIMS)],
                               dtype=np.float32)
            avail  = [r for r in remaining if r not in used_in_batch]
            dists  = np.linalg.norm(X_pca[avail] - coords, axis=1)
            pick   = avail[int(np.argmin(dists))]
            used_in_batch.add(pick)
            observed.append(pick)
            study.tell(trial, _study_value(ratings[pick]))

        for idx in used_in_batch:
            remaining.remove(idx)

        best = (float(np.max(ratings[observed])) if DIRECTION == "maximize"
                else float(np.min(ratings[observed])))
        cumulative_best.append(best)

    return np.array(cumulative_best)

# ---------------------------------------------------------------------------
# Experiment B: TPE, n_jobs=4
# ---------------------------------------------------------------------------

def run_arm_tpe(
    X_pca: np.ndarray,
    ratings: np.ndarray,
    bounds: list[tuple[float, float]],
    seed: int,
) -> np.ndarray:
    rng      = np.random.default_rng(seed)
    perm     = rng.permutation(len(ratings))
    observed = list(perm[:WARM_UP].tolist())
    remaining = list(perm[WARM_UP:].tolist())
    lock     = threading.Lock()

    fixed_dists = {f"pca_{i}": FloatDistribution(bounds[i][0], bounds[i][1])
                   for i in range(PCA_DIMS)}

    # n_startup_trials=0: warm-up data is already injected, start TPE immediately.
    sampler = TPESampler(seed=seed, n_startup_trials=0)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    _inject_warmup(study, X_pca, observed, ratings, fixed_dists)

    def objective(trial: optuna.Trial) -> float:
        # Each of the 4 threads independently calls TPE with the same completed-trial
        # snapshot.  TPE fits the same "good-region" KDE for all 4 threads, so all
        # 4 suggestions cluster near the same point — the cloning effect.
        coords = np.array(
            [trial.suggest_float(f"pca_{i}", bounds[i][0], bounds[i][1])
             for i in range(PCA_DIMS)],
            dtype=np.float32,
        )
        with lock:
            dists      = np.linalg.norm(X_pca[remaining] - coords, axis=1)
            local_idx  = int(np.argmin(dists))
            pool_idx   = remaining.pop(local_idx)
        return _study_value(ratings[pool_idx])

    study.optimize(objective, n_trials=N_ITERATIONS * BATCH_SIZE, n_jobs=BATCH_SIZE,
                   show_progress_bar=False)

    # Recover per-round cumulative_best from completed trials sorted by trial number.
    # Trials 0..WARM_UP-1 are the injected warm-up; WARM_UP.. are post-warmup.
    completed = sorted(
        [t for t in study.trials if t.state == TrialState.COMPLETE],
        key=lambda t: t.number,
    )
    warm_ratings = [_original_rating(t.value) for t in completed[:WARM_UP]]
    post_ratings = [_original_rating(t.value) for t in completed[WARM_UP:]]

    cumulative_best = []
    pool = warm_ratings[:]
    for i in range(0, len(post_ratings), BATCH_SIZE):
        pool.extend(post_ratings[i : i + BATCH_SIZE])
        cumulative_best.append(
            float(np.max(pool)) if DIRECTION == "maximize" else float(np.min(pool))
        )

    return np.array(cumulative_best[:N_ITERATIONS])

# ---------------------------------------------------------------------------
# Outer comparison loop
# ---------------------------------------------------------------------------

def run_all_seeds(
    X_pca: np.ndarray,
    ratings: np.ndarray,
    bounds: list[tuple[float, float]],
    use_qei: bool,
) -> dict:
    label  = "qEISampler (ask-tell)" if use_qei else "Optuna TPE (n_jobs=4)"
    run_fn = run_arm_qei if use_qei else run_arm_tpe

    print(f"\n{'='*60}")
    print(f"Arm: {label}  ({N_REPS} seeds × {N_ITERATIONS} rounds × q={BATCH_SIZE})")
    print(f"{'='*60}")

    times, bests = [], []
    for rep in range(N_REPS):
        print(f"  rep {rep+1:2d}/{N_REPS}  seed={rep} ...", end=" ", flush=True)
        t0      = time.perf_counter()
        cb      = run_fn(X_pca, ratings, bounds, seed=rep)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        bests.append(cb)
        print(f"best={cb[-1]:.3f}  ({elapsed:.1f}s)")

    arr = np.array(bests)
    return dict(
        label      = label,
        avg_bests  = arr.mean(axis=0),
        std_bests  = arr.std(axis=0),
        avg_time_s = float(np.mean(times)),
        std_time_s = float(np.std(times)),
    )

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(qei: dict, tpe: dict) -> None:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    if DIRECTION == "maximize":
        goal      = "higher = most problematic ad found first"
        advantage = qei["avg_bests"][-1] - tpe["avg_bests"][-1]
    else:
        goal      = "lower = least problematic ad found first"
        advantage = tpe["avg_bests"][-1] - qei["avg_bests"][-1]

    print(f"\n\n{'='*65}")
    print(f"RESULTS — qEISampler (ask-tell) vs Optuna TPE (n_jobs=4)")
    print(f"Ads pool, mean rating 1–7  |  {goal}")
    print(f"{'='*65}\n")

    rows = []
    for i, (ma, ms, ta, ts) in enumerate(
        zip(qei["avg_bests"], qei["std_bests"], tpe["avg_bests"], tpe["std_bests"]),
        start=1,
    ):
        rows.append({
            "round":    i,
            "trials":   WARM_UP + i * BATCH_SIZE,
            "qEI avg":  ma,
            "qEI std":  ms,
            "TPE avg":  ta,
            "TPE std":  ts,
            "qEI-TPE":  ma - ta,
        })

    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nFinal cumulative-best (mean ± std, {N_REPS} seeds):")
    print(f"  qEISampler : {qei['avg_bests'][-1]:.4f} ± {qei['std_bests'][-1]:.4f}")
    print(f"  Optuna TPE : {tpe['avg_bests'][-1]:.4f} ± {tpe['std_bests'][-1]:.4f}")
    print(f"  qEISampler advantage: {advantage:+.4f}  "
          f"(positive = qEI found a {'higher' if DIRECTION == 'maximize' else 'lower'} rating)")
    print(f"\nAvg wall-clock time per run:")
    print(f"  qEISampler : {qei['avg_time_s']:.1f}s ± {qei['std_time_s']:.1f}s")
    print(f"  Optuna TPE : {tpe['avg_time_s']:.1f}s ± {tpe['std_time_s']:.1f}s")


def main() -> None:
    ad_ids, ratings, embs = load_data()
    X_pca  = fit_pca(embs)
    bounds = _dim_bounds(X_pca)

    qei_results = run_all_seeds(X_pca, ratings, bounds, use_qei=True)
    tpe_results = run_all_seeds(X_pca, ratings, bounds, use_qei=False)
    print_report(qei_results, tpe_results)


if __name__ == "__main__":
    main()
