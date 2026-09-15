# demo9.py — Self-contained version of demo7_categorical_vs_embedding.py's actual
# question: does exploiting embedding structure via qEI beat the normal, embedding-
# blind way anyone would reach for Optuna on a fixed pool of discrete items?
#
# demo7_categorical_vs_embedding.py (and the rest of the demo7*/demo8 family) answer
# this using real chi_bad_ads image embeddings and human ad ratings — data that
# lives in a separate repo (~/projects/chi_bad_ads) and isn't included here. This
# demo asks the identical question with no external data at all:
#
#   "Embeddings" — sklearn's bundled digits dataset (load_digits): 1797 8x8
#   grayscale digit images, each already a 64-dim pixel-intensity vector. A random
#   POOL_SIZE-sized subset stands in for the ad pool.
#
#   "Ratings" — synthetic, not the digit label itself. We pick one archetype digit
#   (TARGET_DIGIT), compute the centroid of all its images in pixel space, and score
#   every pool item by negative Euclidean distance to that centroid (rescaled to a
#   1-7 range, plus Gaussian noise to mimic the disagreement-among-raters noise in
#   the original ratings). This gives a pool where nearby feature vectors have
#   similar scores — real embedding structure the qEI arm can exploit and the
#   categorical arm structurally cannot, without depending on the digit *label*
#   (which the model never sees; only the pixel vectors are passed as candidates).
#
# The two arms and the reporting are otherwise unchanged from
# demo7_categorical_vs_embedding.py:
#
# Arm A: trial.suggest_categorical("item_idx", all_ids) — the idiomatic Optuna
#   answer to "pick the best of N discrete things." No embeddings, no PCA, no
#   continuous relaxation, no BatchSampler. Every item is an opaque, unrelated
#   label to TPE; it cannot generalize from tried items to similar untried ones.
#
# Arm B: qEI via call_modal_api directly against the real remaining pool, using
#   the raw 64-dim pixel vectors as features (no PCA — 64 dims is already small
#   enough to hand the GP directly; see CLAUDE.md's direct-pool-candidates note).
#   The GP models score as a function of feature position, so an untried item near
#   a high-scoring one gets a high posterior mean with zero direct observations.
#
# Both arms explore the exact same fully-known, finite pool, starting from the same
# random warm-up items per rep. Arm A does not exclude already-tried items from its
# choices (see demo7_categorical_vs_embedding.py's docstring for why that's
# deliberate). No train/test split: the synthetic score doesn't depend on any fitted
# transform, so there's nothing to leak.
#
# Usage:
#   python demos/demo9.py

from datetime import datetime
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState

from quantecarlo import call_modal_api

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "https://markshipman4273--bo-gp-service-gp-suggest.modal.run"

POOL_SIZE     = 400   # random subset of the 1797 digit images
TARGET_DIGIT  = 8     # archetype whose centroid defines "high score" — arbitrary choice
NOISE_STD     = 0.4   # rating noise (1-7 scale), mimicking disagreement among raters
POOL_SEED     = 123   # fixes the pool + synthetic ratings across all reps below

N_REPS       = 12
N_ITERATIONS = 10
BATCH_SIZE   = 4
WARM_UP      = 16

# How many random size-BATCH_SIZE index-combinations of the real remaining pool the
# server scores with joint q-EI before returning the best one. Nothing analogous
# exists for the categorical arm, since Optuna's categorical TPE has no
# candidate-batch concept at all.
N_GP_CANDIDATE_BATCHES = 4096

# "maximize" → find items most similar to the archetype first
# "minimize" → find items least similar to the archetype first
DIRECTION = "maximize"

# ---------------------------------------------------------------------------
# Synthetic data: real feature vectors (digit pixels), synthetic score
# ---------------------------------------------------------------------------

def load_data():
    digits = load_digits()
    X_all = digits.data.astype(np.float32)          # (1797, 64), pixel intensities 0-16
    classes_all = digits.target

    rng = np.random.default_rng(POOL_SEED)
    idx = rng.choice(len(X_all), size=POOL_SIZE, replace=False)
    X = X_all[idx]

    # Centroid computed from the *full* dataset's archetype images, not just the
    # pool subset — an arbitrary but fixed reference point in pixel space.
    centroid = X_all[classes_all == TARGET_DIGIT].mean(axis=0)
    dist = np.linalg.norm(X - centroid, axis=1)
    sim_norm = (-dist - (-dist).min()) / ((-dist).max() - (-dist).min())

    ratings = 1.0 + 6.0 * sim_norm
    ratings += rng.normal(0.0, NOISE_STD, size=POOL_SIZE)
    ratings = np.clip(ratings, 1.0, 7.0).astype(np.float32)

    print(f"Pool: {POOL_SIZE} digit images (64-dim pixel vectors) | "
          f"synthetic rating {ratings.min():.2f}–{ratings.max():.2f} "
          f"(archetype: digit {TARGET_DIGIT})")
    return X, ratings

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

    # Same warm-up items (by index) as the qEI arm gets for this seed — the only
    # thing that should differ between arms is what happens after warm-up.
    now = datetime.now()
    for idx in observed:
        study.add_trial(FrozenTrial(
            number=-1, trial_id=-1, state=TrialState.COMPLETE,
            value=_study_value(ratings[idx]),
            datetime_start=now, datetime_complete=now,
            params={"item_idx": int(idx)}, distributions={"item_idx": full_dist},
            user_attrs={}, system_attrs={}, intermediate_values={},
        ))

    def objective(trial: optuna.Trial) -> float:
        # Fixed, full-universe choices every call — required for TPE to treat every
        # trial as the same search dimension. No exclusion of already-tried items;
        # see demo7_categorical_vs_embedding.py's docstring for why that's
        # deliberate, not an oversight.
        item_idx = trial.suggest_categorical("item_idx", all_ids)
        return _study_value(ratings[item_idx])

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
# Arm B: qEI directly against the real pool, using raw pixel vectors as features
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
        X_train = X[observed]
        y_train = (sign * ratings[observed]).astype(np.float32)
        candidates = X[remaining]   # the real, unclaimed items — nothing invented

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
    goal = ("higher = item most similar to the archetype found first" if DIRECTION == "maximize"
            else "lower = item least similar to the archetype found first")

    print(f"\n\n{'='*70}")
    print("RESULTS — categorical Optuna (no embeddings) vs qEI (pixel vectors, direct pool)")
    print(f"Same {POOL_SIZE}-item pool, same warm-up items per seed  |  {goal}")
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
    print(f"  qEI (pixel vectors)         : {qei['avg_bests'][-1]:.4f} ± {qei['std_bests'][-1]:.4f}")
    print(f"  Advantage: {sign * (qei['avg_bests'][-1] - cat['avg_bests'][-1]):+.4f}  "
          f"(positive = qEI found a {better} score than categorical TPE)")

    print("\nAvg wall-clock time per run:")
    print(f"  Categorical : {cat['avg_time_s']:.1f}s ± {cat['std_time_s']:.1f}s")
    print(f"  qEI         : {qei['avg_time_s']:.1f}s ± {qei['std_time_s']:.1f}s")


def main() -> None:
    X, ratings = load_data()

    required = WARM_UP + N_ITERATIONS * BATCH_SIZE
    if len(ratings) < required:
        raise ValueError(f"Pool too small: need >= {required} items, got {len(ratings)}.")

    cat = run_all_seeds(run_arm_categorical, "Categorical Optuna (no embeddings)", ratings)
    qei = run_all_seeds(run_arm_qei, "qEI (pixel vectors, direct pool)", X, ratings)

    print_report(cat, qei)


if __name__ == "__main__":
    main()
