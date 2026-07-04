# demo7_categorical_vs_embedding.py — The actual question this whole exercise was
# testing: does exploiting embedding structure via qEI beat the normal, embedding-
# blind way anyone would reach for Optuna on a fixed pool of discrete items?
#
# Arm A: trial.suggest_categorical("ad_idx", all_ids) — the idiomatic Optuna answer
#   to "pick the best of N discrete things." No embeddings, no PCA, no continuous
#   relaxation, no BatchSampler. TPE buckets trials into "good"/"bad" per category
#   and weights future suggestions accordingly. This is what a normal user, who has
#   never heard of any of the GP/embedding machinery elsewhere in this repo, would
#   write. It is a legitimate, correct use of Optuna.
#
#   Its limitation is structural, not a bug: TPE can only learn "ad #137 scored
#   well" by having actually tried ad #137. It has no way to know ad #138's
#   embedding is nearly identical to #137's, because it never sees the embedding at
#   all — every ad is an opaque, unrelated label. It cannot generalize from tried
#   ads to similar untried ones. It's a pure multi-armed bandit over |pool| arms.
#
# Arm B: qEI via call_modal_api directly against the real remaining pool, using
#   PCA-reduced embeddings as features. The GP models score as a function of
#   embedding position, so an untried ad near a high-scoring one gets a high
#   posterior mean even with zero direct observations of it — the thing Arm A
#   structurally cannot do.
#
# Both arms explore the exact same fully-known, finite pool, starting from the same
# random warm-up ads per rep (same seed -> same np.random.default_rng draw). Arm A
# does NOT exclude already-tried ads from its choices — that's deliberate: forcing
# a shrinking categorical choice set on every trial would make its per-trial
# CategoricalDistribution inconsistent across trials, which breaks Optuna's TPE
# model for that parameter (it needs an identical distribution across trials to
# treat them as the same search dimension). Letting duplicates happen is also more
# faithful to what a normal user's naive loop actually does — "ask about ad #42
# twice" is a real, unremarkable cost of the embedding-blind approach, not
# something to paper over with bookkeeping the naive approach wouldn't have.
#
# Unlike demo7_pca_vs_pls.py, there's no train/test split here: PCA is unsupervised
# (never touches ratings), and this comparison isn't testing generalization of a
# fitted transform — it's testing whether embedding-aware search beats
# embedding-blind search on one fixed, fully-known pool. Fitting PCA on the whole
# pool doesn't leak anything a categorical-only baseline could exploit anyway.
#
# Data files: same as demo7.py / demo7_pca_vs_pls.py — ads_all_labels.json and
# embedding_cache/ from ~/projects/chi_bad_ads (paths hardcoded below).
#
# Usage:
#   python demos/demo7_categorical_vs_embedding.py

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState

from quantecarlo import call_modal_api

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "https://markshipman4273--bo-gp-service-gp-suggest.modal.run"

_CHI_BAD_ADS_DIR = Path.home() / "projects" / "chi_bad_ads"
CHI_JSON  = _CHI_BAD_ADS_DIR / "chi-bad-ads-data" / "data" / "ads_all_labels.json"
EMB_CACHE = _CHI_BAD_ADS_DIR / "embedding_cache"

N_REPS       = 12
N_ITERATIONS = 10
BATCH_SIZE   = 4
WARM_UP      = 16
REDUCED_DIMS = 64   # PCA output dims for the qEI arm

# How many random size-BATCH_SIZE index-combinations of the real remaining pool the
# server scores with joint q-EI before returning the best one. Easy to change here —
# this is the one knob the qEI arm exposes; there is nothing analogous for Arm A,
# since Optuna's categorical TPE has no candidate-batch concept at all.
N_GP_CANDIDATE_BATCHES = 4096

# "maximize" → find most problematic ads (highest rating) first
# "minimize" → find least problematic ads (lowest rating) first
DIRECTION = "maximize"

# ---------------------------------------------------------------------------
# Data loading and dimensionality reduction
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
    """Unsupervised — fit on the whole pool, no split needed (see module docstring)."""
    n_components = min(REDUCED_DIMS, embs.shape[0], embs.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X = pca.fit_transform(embs).astype(np.float32)
    print(f"PCA: {embs.shape[1]}→{n_components} dims | explained var: {pca.explained_variance_ratio_.sum():.3f}")
    return X

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _study_value(rating: float) -> float:
    # Optuna Study is always direction="minimize". For a maximize objective we
    # store -rating so Optuna minimises it.
    return -float(rating) if DIRECTION == "maximize" else float(rating)


def _original_rating(study_value: float) -> float:
    return -study_value if DIRECTION == "maximize" else study_value

# ---------------------------------------------------------------------------
# Arm A: categorical Optuna — the normal, embedding-blind way to do this
# ---------------------------------------------------------------------------

def run_arm_categorical(ratings: np.ndarray, seed: int) -> np.ndarray:
    n = len(ratings)
    all_ids = tuple(range(n))
    full_dist = CategoricalDistribution(choices=all_ids)

    rng      = np.random.default_rng(seed)
    perm     = rng.permutation(n)
    observed = perm[:WARM_UP].tolist()

    sampler = TPESampler(seed=seed, n_startup_trials=0)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Same warm-up ads (by index) as the qEI arm gets for this seed — the only
    # thing that should differ between arms is what happens after warm-up.
    now = datetime.now()
    for idx in observed:
        study.add_trial(FrozenTrial(
            number=-1, trial_id=-1, state=TrialState.COMPLETE,
            value=_study_value(ratings[idx]),
            datetime_start=now, datetime_complete=now,
            params={"ad_idx": int(idx)}, distributions={"ad_idx": full_dist},
            user_attrs={}, system_attrs={}, intermediate_values={},
        ))

    def objective(trial: optuna.Trial) -> float:
        # Fixed, full-universe choices every call — required for TPE to treat every
        # trial as the same search dimension. No exclusion of already-tried ads;
        # see module docstring for why that's deliberate, not an oversight.
        ad_idx = trial.suggest_categorical("ad_idx", all_ids)
        return _study_value(ratings[ad_idx])

    study.optimize(objective, n_trials=N_ITERATIONS * BATCH_SIZE, n_jobs=BATCH_SIZE,
                    show_progress_bar=False)

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
# Arm B: qEI directly against the real pool, using PCA embeddings as features
# ---------------------------------------------------------------------------

def run_arm_qei(X: np.ndarray, ratings: np.ndarray, seed: int) -> np.ndarray:
    sign = 1.0 if DIRECTION == "maximize" else -1.0  # call_modal_api convention: higher = better

    rng       = np.random.default_rng(seed)
    perm      = rng.permutation(len(ratings))
    observed  = list(perm[:WARM_UP].tolist())
    remaining = list(perm[WARM_UP:].tolist())

    cumulative_best = []

    for _ in range(N_ITERATIONS):
        q = min(BATCH_SIZE, len(remaining))
        X_train = X[observed].astype(np.float32)
        y_train = (sign * ratings[observed]).astype(np.float32)
        candidates = X[remaining].astype(np.float32)   # the real, unclaimed ads — nothing invented

        raw = call_modal_api(
            API_URL, X_train, y_train, candidates,
            q=q, n_batches=N_GP_CANDIDATE_BATCHES, train_steps=100,
        )

        seen: set[int] = set()
        local_indices: list[int] = []
        for item in raw:
            i = int(item["index"])
            if i not in seen:
                seen.add(i)
                local_indices.append(i)

        picked = [remaining[i] for i in local_indices]
        observed.extend(picked)
        remaining = [r for r in remaining if r not in set(picked)]

        best = (float(np.max(ratings[observed])) if DIRECTION == "maximize"
                else float(np.min(ratings[observed])))
        cumulative_best.append(best)

    return np.array(cumulative_best)

# ---------------------------------------------------------------------------
# Outer comparison loop + reporting
# ---------------------------------------------------------------------------

def run_all_seeds(run_fn, label: str, *args) -> dict:
    print(f"\n{'='*60}")
    print(f"Arm: {label}  ({N_REPS} seeds × {N_ITERATIONS} rounds × q={BATCH_SIZE})")
    print(f"{'='*60}")

    times, bests = [], []
    for rep in range(N_REPS):
        print(f"  rep {rep+1:2d}/{N_REPS}  seed={rep} ...", end=" ", flush=True)
        t0      = time.perf_counter()
        cb      = run_fn(*args, seed=rep)
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


def print_report(cat: dict, qei: dict) -> None:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    sign = 1.0 if DIRECTION == "maximize" else -1.0
    goal = ("higher = most problematic ad found first" if DIRECTION == "maximize"
            else "lower = least problematic ad found first")

    print(f"\n\n{'='*70}")
    print("RESULTS — categorical Optuna (no embeddings) vs qEI (PCA embeddings, direct pool)")
    print(f"Same ~500-ad pool, same warm-up ads per seed  |  {goal}")
    print(f"{'='*70}\n")

    rows = []
    for i in range(len(cat["avg_bests"])):
        rows.append({
            "round":            i + 1,
            "trials":           WARM_UP + (i + 1) * BATCH_SIZE,
            "Categorical avg":  cat["avg_bests"][i],
            "qEI avg":          qei["avg_bests"][i],
            "qEI-Categorical":  sign * (qei["avg_bests"][i] - cat["avg_bests"][i]),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    better = "higher" if DIRECTION == "maximize" else "lower"
    print(f"\nFinal cumulative-best (mean ± std, {N_REPS} seeds):")
    print(f"  Categorical (no embeddings) : {cat['avg_bests'][-1]:.4f} ± {cat['std_bests'][-1]:.4f}")
    print(f"  qEI (PCA embeddings)        : {qei['avg_bests'][-1]:.4f} ± {qei['std_bests'][-1]:.4f}")
    print(f"  Advantage: {sign * (qei['avg_bests'][-1] - cat['avg_bests'][-1]):+.4f}  "
          f"(positive = qEI found a {better} rating than categorical TPE)")

    print("\nAvg wall-clock time per run:")
    print(f"  Categorical : {cat['avg_time_s']:.1f}s ± {cat['std_time_s']:.1f}s")
    print(f"  qEI         : {qei['avg_time_s']:.1f}s ± {qei['std_time_s']:.1f}s")


def main() -> None:
    ad_ids, ratings, embs = load_data()

    required = WARM_UP + N_ITERATIONS * BATCH_SIZE
    if len(ratings) < required:
        raise ValueError(f"Pool too small: need >= {required} ads, got {len(ratings)}.")

    X = fit_pca(embs)

    cat = run_all_seeds(run_arm_categorical, "Categorical Optuna (no embeddings)", ratings)
    qei = run_all_seeds(run_arm_qei, "qEI (PCA embeddings, direct pool)", X, ratings)

    print_report(cat, qei)


if __name__ == "__main__":
    main()
