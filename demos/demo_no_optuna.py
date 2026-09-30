# demo_no_optuna.py — q-EI batch optimization with your own ask-tell loop.
#
# No Optuna, no BatchSampler, no DimSpec. You hold a fixed pool of candidate
# vectors and a score function; each round asks QEIClient for the q most
# promising untried items, evaluates them, and appends the results.
#
# Usage:
#   pip install quantecarlo scikit-learn
#   python demos/demo_no_optuna.py
#
# Pool: 400 digit images from sklearn (64-dim pixel vectors — real feature
# structure, no download). Score: negative distance to a hidden target image
# plus noise, so higher = better and the model never sees the digit labels.
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits

from quantecarlo import QEIClient

POOL_SIZE    = 400
WARM_UP      = 8      # random evaluations before the first q-EI round
N_ROUNDS     = 10
Q            = 4
SEED         = 0

rng = np.random.default_rng(SEED)
digits = load_digits()
pool = digits.data[rng.choice(len(digits.data), POOL_SIZE, replace=False)].astype(np.float32)
target = pool[rng.integers(POOL_SIZE)]


def score(x: np.ndarray) -> float:
    """Expensive black box stand-in. Higher = better."""
    return float(-np.linalg.norm(x - target) + rng.normal(0, 2.0))


client = QEIClient(train_steps=100)

observed:  list[int] = rng.choice(POOL_SIZE, WARM_UP, replace=False).tolist()
scores:    list[float] = [score(pool[i]) for i in observed]
remaining: list[int] = [i for i in range(POOL_SIZE) if i not in set(observed)]

print(f"warm-up best: {max(scores):.2f}")
for rnd in range(1, N_ROUNDS + 1):
    # ask — X/y are what you've seen, candidates are the real untried items
    picks = client.suggest(
        X=pool[observed], y=scores, candidates=pool[remaining],
        q=Q, direction="maximize",
    )
    chosen = sorted({remaining[p["index"]] for p in picks})   # pool indices

    # tell — evaluate and record
    for i in chosen:
        observed.append(i)
        scores.append(score(pool[i]))
    remaining = [i for i in remaining if i not in set(chosen)]

    print(f"round {rnd:2d}: picked {chosen}  best so far {max(scores):.2f}")

best = observed[int(np.argmax(scores))]
print(f"\nbest item index {best}, distance to target {np.linalg.norm(pool[best] - target):.2f} "
      f"(target is index {int(np.where((pool == target).all(1))[0][0])})")
