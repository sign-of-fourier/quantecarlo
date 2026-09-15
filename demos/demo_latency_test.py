# demo_latency_test.py — one-off latency probe for modal_suggest.
#
# Not a unit test — deliberately kept out of tests/ so it never runs on every
# `pytest`. This fires a single real call at the live Modal endpoint with a much
# larger candidate pool than demo7.py uses (10000 vs demo7's 512) and times how
# long the GP fit + q-EI scoring takes at that scale.
#
# Uses modal_suggest (the same continuous-relaxation call demo7.py's qEI arm
# makes) rather than call_modal_api directly, since the point is to time the
# call shape demo7.py actually issues, just with a bigger n_probe_points.
#
# Usage:
#   python demos/demo_latency_test.py

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from quantecarlo import DimSpec, modal_suggest

API_URL = "https://markshipman4273--bo-gp-service-gp-suggest.modal.run"

N_DIMS         = 256    # matches demo7.py's PCA_DIMS
N_OBSERVED     = 16     # matches demo7.py's WARM_UP
N_PROBE_POINTS = 10000
Q              = 4
TRAIN_STEPS    = 100
TIMEOUT_S      = 600.0  # generous — the whole point is to measure real latency,
                         # not to have the client cut the call off first


def main() -> None:
    rng = np.random.default_rng(0)
    search_space = [
        DimSpec(name=f"dim_{i}", type="float", low=-1.0, high=1.0)
        for i in range(N_DIMS)
    ]
    X = rng.uniform(-1.0, 1.0, size=(N_OBSERVED, N_DIMS)).tolist()
    y = rng.uniform(0.0, 1.0, size=N_OBSERVED).tolist()

    # n_candidate_batches isn't passed, so modal_suggest defaults it to
    # n_probe_points (10000) too — see the docstring in bo_sampler.py.
    print(f"Calling modal_suggest: n_probe_points={N_PROBE_POINTS}, dims={N_DIMS}, "
          f"n_observed={N_OBSERVED}, q={Q}, train_steps={TRAIN_STEPS}")
    t0 = time.perf_counter()
    result = modal_suggest(
        X, y, search_space, q=Q,
        direction="minimize", api_url=API_URL,
        n_probe_points=N_PROBE_POINTS, train_steps=TRAIN_STEPS,
        seed=0, timeout=TIMEOUT_S,
    )
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.2f}s — got {len(result)} candidates back")


if __name__ == "__main__":
    main()
