"""
Live smoke tests for modal_suggest against the real Modal GP endpoint.

Skipped automatically when MODAL_BO_API_URL is unset (defaults to the
markshipman4273 workspace URL which is always valid if the service is deployed).

Run:
    python -m pytest tests/test_modal_live.py -v -s
"""
import os

import pytest

from quantecarlo import DimSpec, modal_suggest

MODAL_API_URL = os.environ.get(
    "MODAL_BO_API_URL",
    "https://markshipman4273--bo-gp-service-gp-suggest.modal.run",
)

SEARCH_SPACE = [
    DimSpec("lr",       "float", 1e-4, 1e-1, log=True),
    DimSpec("n_hidden", "int",   16,   256),
    DimSpec("alpha",    "float", 1e-5, 1e-2, log=True),
]

X_OBS = [
    [1e-3,  32,  1e-4],
    [5e-3,  64,  5e-4],
    [1e-2,  128, 1e-3],
    [5e-2,  64,  2e-3],
    [1e-4,  256, 1e-5],
    [2e-3,  48,  3e-4],
]
Y_OBS = [0.25, 0.18, 0.15, 0.22, 0.30, 0.20]  # lower is better (minimize)


@pytest.mark.skipif(not MODAL_API_URL, reason="MODAL_BO_API_URL not set")
class TestModalSuggestLive:
    """Calls the live Modal endpoint. Requires a running deployment."""

    def test_returns_q_dicts(self):
        result = modal_suggest(
            X_OBS, Y_OBS, SEARCH_SPACE, q=2,
            direction="minimize", api_url=MODAL_API_URL,
            n_candidates=64, train_steps=20,
        )
        assert len(result) == 2
        assert all(isinstance(p, dict) for p in result)

    def test_param_names_match_search_space(self):
        result = modal_suggest(
            X_OBS, Y_OBS, SEARCH_SPACE, q=2,
            direction="minimize", api_url=MODAL_API_URL,
            n_candidates=64, train_steps=20,
        )
        expected_keys = {"lr", "n_hidden", "alpha"}
        for params in result:
            assert set(params.keys()) == expected_keys

    def test_float_params_within_bounds(self):
        result = modal_suggest(
            X_OBS, Y_OBS, SEARCH_SPACE, q=2,
            direction="minimize", api_url=MODAL_API_URL,
            n_candidates=64, train_steps=20,
        )
        for params in result:
            assert 1e-4 <= params["lr"] <= 1e-1
            assert 1e-5 <= params["alpha"] <= 1e-2

    def test_int_dims_are_integers_in_range(self):
        result = modal_suggest(
            X_OBS, Y_OBS, SEARCH_SPACE, q=2,
            direction="minimize", api_url=MODAL_API_URL,
            n_candidates=64, train_steps=20,
        )
        for params in result:
            assert isinstance(params["n_hidden"], int)
            assert 16 <= params["n_hidden"] <= 256

    def test_maximize_direction_accepted(self):
        """Maximize study: pass y as-is (no negation). Should not raise."""
        y_max = [-v for v in Y_OBS]  # flip sign to simulate a maximize study's raw values
        result = modal_suggest(
            X_OBS, y_max, SEARCH_SPACE, q=2,
            direction="maximize", api_url=MODAL_API_URL,
            n_candidates=64, train_steps=20,
        )
        assert len(result) == 2
