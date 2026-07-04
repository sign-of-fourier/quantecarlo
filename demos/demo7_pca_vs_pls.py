# demo7_pca_vs_pls.py — Same experiment as demo7.py (qEI ask-tell vs Optuna TPE,
# n_jobs=4) but run twice in one script: once with PCA and once with Partial Least
# Squares (PLS) as the dimensionality-reduction step ahead of the GP / TPE search,
# both evaluated over the *same* held-out test pool.
#
# One deliberate departure from demo7.py: the qEI arm here calls quantecarlo's
# call_modal_api directly instead of going through modal_suggest/BatchSampler/
# DimSpec. Those exist to handle a genuinely CONTINUOUS search space (see demo.py,
# tuning a learning rate) where there is no enumerable set of valid points, so some
# finite stand-in has to be invented before the GP can be asked anything. That's not
# our situation: the ad pool is already a small, fully known, finite, enumerable set
# (the held-out test split). Routing it through modal_suggest would mean inventing
# random continuous points, asking the GP to choose among those, and then snapping
# the result back to the nearest *real* ad — an approximation of an approximation
# with no purpose when the real candidates were available to hand the GP directly.
# Calling call_modal_api(candidates=X[remaining], ...) gives the GP the actual ads
# and gets back a real index directly — no inventing, no snapping.
#
# Why compare them here instead of running demo7.py separately: PCA is unsupervised
# — it keeps the directions of highest variance in the raw embedding, with no regard
# for whether that variance predicts the rating. PLS is supervised — it factors the
# embedding against the rating and keeps the directions that covary most with it. In
# principle a GP built on PLS components should need fewer dimensions to represent
# the same predictive signal, which could mean tighter posteriors / better EI ranking
# for the same output-dimension budget. But PLS's fit uses ratings, so fitting it on
# the same ads the search arms later "discover" would leak label information into the
# projection and make PLS look artificially better. So both methods are fit on the
# same held-out training split and evaluated by searching the same held-out test
# split — the only thing that differs between the PCA run and the PLS run is the
# reduction method itself, not which ads are available to find or how many labels
# either one got to look at.
#
# PCA doesn't need the split for leakage reasons (it never touches ratings), but it's
# fit on the same train split anyway, purely so neither method sees more data than
# the other — otherwise a difference in results could be explained by data budget
# rather than by PCA vs PLS.
#
# In a real cold-start deployment, the initial labels PLS is fit on would come from a
# fine-tuned quality model — e.g. the Qwen2-VL scorer used elsewhere in this project's
# warm-start path — rather than ground-truth human ratings. This demo uses the pool's
# ground-truth ratings as a stand-in for whatever labels PLS would actually be fit on.
#
# Everything else — warm-up, batch size, iteration count, both search arms — is
# identical to demo7.py, just run once per reduction method.
#
# Data files expected in the current working directory (same as demo7.py):
#   ads_all_labels.json  — JSON list of ad objects; each has an "id" field (string)
#                          and a "ratings" field (list of ints, 1–7).
#   embedding_cache/     — directory of .npy files; each file is named <ad_id>.npy,
#                          matching the ad's "id" field.  Each file stores a 1024-dim
#                          float32 vector — the image embedding for that ad.
#
# This demo is not self-contained: the data files are part of a separate experiment
# repo (~/projects/chi_bad_ads) and are not included here. It is a reference
# implementation only.
#
# Usage:
#   1. Deploy the GP service (see README for the backend repo / hosted endpoint).
#   2. Paste the endpoint URL into API_URL below.
#   3. python demos/demo7_pca_vs_pls.py   (CHI_JSON/EMB_CACHE below are absolute
#      paths into ~/projects/chi_bad_ads, so this can run from any cwd; update
#      them if that repo lives somewhere else on your machine.)

import os
import sys
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split

import optuna
from optuna.distributions import FloatDistribution
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState

from quantecarlo import call_modal_api

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "https://markshipman4273--bo-gp-service-gp-suggest.modal.run"

# The chi_bad_ads repo splits these two across different directories:
#   ads_all_labels.json under chi-bad-ads-data/data/, embedding_cache/ at the repo root.
# Absolute paths so this demo runs regardless of cwd.
_CHI_BAD_ADS_DIR = Path.home() / "projects" / "chi_bad_ads"
CHI_JSON  = _CHI_BAD_ADS_DIR / "chi-bad-ads-data" / "data" / "ads_all_labels.json"
EMB_CACHE = _CHI_BAD_ADS_DIR / "embedding_cache"

N_REPS       = 12
N_ITERATIONS = 10
BATCH_SIZE   = 4
REDUCED_DIMS = 64   # output dims for both PCA and PLS, so the two are apples-to-apples
WARM_UP      = 16

# GP acquisition knob for the qEI arm — independent of the real search pool (which
# is fixed by TEST_FRACTION below, not a choice made here). candidates passed to
# call_modal_api are the real remaining ads themselves, so there's no "how many
# points to invent" knob (nothing is invented). The only free parameter is how many
# random size-BATCH_SIZE index-combinations of that real pool the server scores
# with joint q-EI before returning the best one (server's GPRequest.n_batches).
N_GP_CANDIDATE_BATCHES = 4096

# Passed as test_size to train_test_split. This fraction becomes the shared search
# pool (both reduction methods and both search arms draw from it); the rest is used
# only to fit PCA/PLS. The search pool must be large enough to cover WARM_UP plus
# N_ITERATIONS * BATCH_SIZE draws per rep (checked at startup in main()).
TEST_FRACTION = 0.5

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


def fit_pca(train_embs: np.ndarray) -> PCA:
    """Unsupervised baseline. Fit on the train split only, for parity with fit_pls."""
    n_components = min(REDUCED_DIMS, train_embs.shape[0], train_embs.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(train_embs)
    print(f"PCA: fit on {len(train_embs)} training ads | {train_embs.shape[1]}→{n_components} dims "
          f"| explained var: {pca.explained_variance_ratio_.sum():.3f}")
    return pca


def fit_pls(train_embs: np.ndarray, train_ratings: np.ndarray) -> PLSRegression:
    """
    Supervised alternative to fit_pca(): factors embs against ratings and keeps
    the REDUCED_DIMS components that covary most with the target, instead of the
    components with highest unsupervised variance.

    Fit on the training split only — see module docstring for why (leakage).
    Returns the fitted transformer; callers project the test split via .transform().
    """
    n_components = min(REDUCED_DIMS, train_embs.shape[0] - 1, train_embs.shape[1])
    pls = PLSRegression(n_components=n_components, scale=True)
    pls.fit(train_embs, train_ratings)
    r2 = pls.score(train_embs, train_ratings)
    print(f"PLS: fit on {len(train_ratings)} training ads | {train_embs.shape[1]}→{n_components} dims "
          f"| train R²: {r2:.3f}")
    return pls

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dim_bounds(X: np.ndarray) -> list[tuple[float, float]]:
    return [(float(X[:, i].min()), float(X[:, i].max())) for i in range(X.shape[1])]


def _study_value(rating: float) -> float:
    # Used only by the TPE arm (Optuna Study is always direction="minimize"). For a
    # maximize objective (find highest rating) we store -rating so Optuna minimises
    # it. The qEI arm doesn't use Optuna at all, so it applies its own sign flip
    # directly against call_modal_api's higher-is-better convention — see run_arm_qei.
    return -float(rating) if DIRECTION == "maximize" else float(rating)


def _original_rating(study_value: float) -> float:
    return -study_value if DIRECTION == "maximize" else study_value


def _inject_warmup(
    study: optuna.Study,
    X: np.ndarray,
    observed: list[int],
    ratings: np.ndarray,
    dists_map: dict,
) -> None:
    """Add warm-up pool members as completed trials (bypasses sampling)."""
    now = datetime.now()
    n_dims = X.shape[1]
    for idx in observed:
        params = {f"dim_{i}": float(X[idx, i]) for i in range(n_dims)}
        study.add_trial(FrozenTrial(
            number=-1, trial_id=-1,
            state=TrialState.COMPLETE,
            value=_study_value(ratings[idx]),
            datetime_start=now, datetime_complete=now,
            params=params, distributions=dists_map,
            user_attrs={}, system_attrs={}, intermediate_values={},
        ))

# ---------------------------------------------------------------------------
# Experiment A: qEI ask-tell directly against the real pool (no Optuna needed —
# see module docstring for why this arm doesn't go through modal_suggest/BatchSampler)
# ---------------------------------------------------------------------------

def run_arm_qei(
    X: np.ndarray,
    ratings: np.ndarray,
    seed: int,
) -> np.ndarray:
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

        # Server can return duplicate indices if its random index-combinations
        # collide; dedup while preserving its EI-rank order, same as production.
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
# Experiment B: TPE, n_jobs=4
# ---------------------------------------------------------------------------

def run_arm_tpe(
    X: np.ndarray,
    ratings: np.ndarray,
    seed: int,
) -> np.ndarray:
    # TPE is a genuinely continuous-relaxation sampler (Optuna's own KDE-based
    # suggest), so it needs a bounded box regardless of the pool being discrete —
    # unlike the qEI arm above, there's no way to hand TPE "the real pool" directly.
    n_dims = X.shape[1]
    bounds   = _dim_bounds(X)
    rng      = np.random.default_rng(seed)
    perm     = rng.permutation(len(ratings))
    observed = list(perm[:WARM_UP].tolist())
    remaining = list(perm[WARM_UP:].tolist())
    lock     = threading.Lock()

    fixed_dists = {f"dim_{i}": FloatDistribution(bounds[i][0], bounds[i][1])
                   for i in range(n_dims)}

    # n_startup_trials=0: warm-up data is already injected, start TPE immediately.
    sampler = TPESampler(seed=seed, n_startup_trials=0)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    _inject_warmup(study, X, observed, ratings, fixed_dists)

    def objective(trial: optuna.Trial) -> float:
        # Each of the 4 threads independently calls TPE with the same completed-trial
        # snapshot.  TPE fits the same "good-region" KDE for all 4 threads, so all
        # 4 suggestions cluster near the same point — the cloning effect.
        coords = np.array(
            [trial.suggest_float(f"dim_{i}", bounds[i][0], bounds[i][1])
             for i in range(n_dims)],
            dtype=np.float32,
        )
        with lock:
            dists      = np.linalg.norm(X[remaining] - coords, axis=1)
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
    X: np.ndarray,
    ratings: np.ndarray,
    use_qei: bool,
    reduction_label: str,
) -> dict:
    arm_label = "qEI ask-tell (call_modal_api direct)" if use_qei else "Optuna TPE (n_jobs=4)"
    run_fn = run_arm_qei if use_qei else run_arm_tpe

    print(f"\n{'='*60}")
    print(f"Arm: [{reduction_label}] {arm_label}  ({N_REPS} seeds × {N_ITERATIONS} rounds × q={BATCH_SIZE})")
    print(f"{'='*60}")

    times, bests = [], []
    for rep in range(N_REPS):
        print(f"  rep {rep+1:2d}/{N_REPS}  seed={rep} ...", end=" ", flush=True)
        t0      = time.perf_counter()
        cb      = run_fn(X, ratings, seed=rep)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        bests.append(cb)
        print(f"best={cb[-1]:.3f}  ({elapsed:.1f}s)")

    arr = np.array(bests)
    return dict(
        label      = f"{reduction_label} + {arm_label}",
        avg_bests  = arr.mean(axis=0),
        std_bests  = arr.std(axis=0),
        avg_time_s = float(np.mean(times)),
        std_time_s = float(np.std(times)),
    )

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(pca_qei: dict, pca_tpe: dict, pls_qei: dict, pls_tpe: dict) -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    sign = 1.0 if DIRECTION == "maximize" else -1.0
    goal = ("higher = most problematic ad found first" if DIRECTION == "maximize"
            else "lower = least problematic ad found first")

    def adv(a: float, b: float) -> float:
        """positive = a found a better rating than b, respecting DIRECTION."""
        return sign * (a - b)

    print(f"\n\n{'='*70}")
    print("RESULTS — PCA vs PLS dimensionality reduction, each run with both search arms")
    print("Same held-out test pool for all four arms (see 'Split' line above)")
    print(f"Ads pool, mean rating 1–7  |  {goal}")
    print(f"{'='*70}\n")

    rows = []
    for i in range(len(pca_qei["avg_bests"])):
        rows.append({
            "round":         i + 1,
            "trials":        WARM_UP + (i + 1) * BATCH_SIZE,
            "PCA qEI":       pca_qei["avg_bests"][i],
            "PCA TPE":       pca_tpe["avg_bests"][i],
            "PLS qEI":       pls_qei["avg_bests"][i],
            "PLS TPE":       pls_tpe["avg_bests"][i],
            "PLS-PCA(qEI)":  adv(pls_qei["avg_bests"][i], pca_qei["avg_bests"][i]),
            "PLS-PCA(TPE)":  adv(pls_tpe["avg_bests"][i], pca_tpe["avg_bests"][i]),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nFinal cumulative-best (mean ± std, {N_REPS} seeds):")
    print(f"  PCA + qEI : {pca_qei['avg_bests'][-1]:.4f} ± {pca_qei['std_bests'][-1]:.4f}")
    print(f"  PCA + Optuna TPE       : {pca_tpe['avg_bests'][-1]:.4f} ± {pca_tpe['std_bests'][-1]:.4f}")
    print(f"  PLS + qEI : {pls_qei['avg_bests'][-1]:.4f} ± {pls_qei['std_bests'][-1]:.4f}")
    print(f"  PLS + Optuna TPE       : {pls_tpe['avg_bests'][-1]:.4f} ± {pls_tpe['std_bests'][-1]:.4f}")

    better = "higher" if DIRECTION == "maximize" else "lower"
    print(f"\nPLS vs PCA advantage (same search arm, same test pool; positive = PLS found a {better} rating):")
    print(f"  within qEI : {adv(pls_qei['avg_bests'][-1], pca_qei['avg_bests'][-1]):+.4f}")
    print(f"  within Optuna TPE       : {adv(pls_tpe['avg_bests'][-1], pca_tpe['avg_bests'][-1]):+.4f}")

    print(f"\nqEI vs TPE advantage (same reduction method; positive = qEI found a {better} rating):")
    print(f"  within PCA : {adv(pca_qei['avg_bests'][-1], pca_tpe['avg_bests'][-1]):+.4f}")
    print(f"  within PLS : {adv(pls_qei['avg_bests'][-1], pls_tpe['avg_bests'][-1]):+.4f}")

    print("\nAvg wall-clock time per run:")
    print(f"  PCA + qEI : {pca_qei['avg_time_s']:.1f}s ± {pca_qei['std_time_s']:.1f}s")
    print(f"  PCA + Optuna TPE       : {pca_tpe['avg_time_s']:.1f}s ± {pca_tpe['std_time_s']:.1f}s")
    print(f"  PLS + qEI : {pls_qei['avg_time_s']:.1f}s ± {pls_qei['std_time_s']:.1f}s")
    print(f"  PLS + Optuna TPE       : {pls_tpe['avg_time_s']:.1f}s ± {pls_tpe['std_time_s']:.1f}s")


def main() -> None:
    ad_ids, ratings, embs = load_data()

    required = WARM_UP + N_ITERATIONS * BATCH_SIZE
    idx = np.arange(len(ratings))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_FRACTION, random_state=42)
    if len(test_idx) < required:
        raise ValueError(
            f"Search pool too small: need >= {required} test ads (WARM_UP + "
            f"N_ITERATIONS*BATCH_SIZE), got {len(test_idx)} from TEST_FRACTION={TEST_FRACTION} "
            f"on a pool of {len(ratings)}. Raise TEST_FRACTION or lower N_ITERATIONS/BATCH_SIZE."
        )
    test_ratings = ratings[test_idx]
    print(f"Split: {len(train_idx)} train ads (fit only) | {len(test_idx)} test ads "
          f"(shared search pool for both reduction methods)")

    # Both fit on the identical train split; both project the identical test split.
    pca = fit_pca(embs[train_idx])
    X_test_pca = pca.transform(embs[test_idx]).astype(np.float32)

    pls = fit_pls(embs[train_idx], ratings[train_idx])
    X_test_pls = pls.transform(embs[test_idx]).astype(np.float32)
    held_out_r2 = pls.score(embs[test_idx], test_ratings)
    print(f"PLS held-out R² (test ads, never seen during fit): {held_out_r2:.3f}")

    pca_qei = run_all_seeds(X_test_pca, test_ratings, use_qei=True,  reduction_label="PCA")
    pca_tpe = run_all_seeds(X_test_pca, test_ratings, use_qei=False, reduction_label="PCA")
    pls_qei = run_all_seeds(X_test_pls, test_ratings, use_qei=True,  reduction_label="PLS")
    pls_tpe = run_all_seeds(X_test_pls, test_ratings, use_qei=False, reduction_label="PLS")

    print_report(pca_qei, pca_tpe, pls_qei, pls_tpe)


if __name__ == "__main__":
    main()
