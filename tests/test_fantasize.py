"""Unit tests for quantecarlo.fantasize_suggest. Self-contained, no network."""
import numpy as np
import pytest

from quantecarlo import DimSpec, fantasize_suggest


def _float_dims():
    return [
        DimSpec("x", "float", -5.0, 5.0),
        DimSpec("y", "float", -5.0, 5.0),
    ]


def _obs(n=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5.0, 5.0, (n, 2)).tolist()
    y = rng.standard_normal(n).tolist()
    return X, y


class TestFantasizeSuggest:

    def test_returns_q_dicts(self):
        X, y = _obs()
        result = fantasize_suggest(X, y, _float_dims(), q=3, seed=0)
        assert len(result) == 3
        assert all(isinstance(p, dict) for p in result)

    def test_param_names_match_search_space(self):
        X, y = _obs()
        result = fantasize_suggest(X, y, _float_dims(), q=2, seed=0)
        for params in result:
            assert set(params.keys()) == {"x", "y"}

    def test_float_values_within_bounds(self):
        X, y = _obs()
        result = fantasize_suggest(X, y, _float_dims(), q=4, seed=0)
        for params in result:
            assert -5.0 <= params["x"] <= 5.0
            assert -5.0 <= params["y"] <= 5.0

    def test_log_float_dim_within_bounds(self):
        dims = [DimSpec("lr", "float", 1e-4, 1e-1, log=True)]
        rng = np.random.default_rng(1)
        X = [[float(rng.uniform(1e-4, 1e-1))] for _ in range(6)]
        y = rng.standard_normal(6).tolist()
        result = fantasize_suggest(X, y, dims, q=3, seed=1)
        for params in result:
            assert 1e-4 <= params["lr"] <= 1e-1

    def test_int_dim_values_are_integers_in_range(self):
        dims = [DimSpec("n", "int", 1, 50)]
        rng = np.random.default_rng(2)
        X = [[float(rng.integers(1, 51))] for _ in range(6)]
        y = rng.standard_normal(6).tolist()
        result = fantasize_suggest(X, y, dims, q=3, seed=2)
        for params in result:
            assert isinstance(params["n"], int)
            assert 1 <= params["n"] <= 50

    def test_single_observation_does_not_crash(self):
        """Degenerate case: only 1 obs. GP falls back gracefully."""
        result = fantasize_suggest([[0.0, 0.0]], [1.0], _float_dims(), q=2, seed=0)
        assert len(result) == 2

    def test_q_equals_one(self):
        X, y = _obs()
        result = fantasize_suggest(X, y, _float_dims(), q=1, seed=0)
        assert len(result) == 1

    def test_minimize_and_maximize_differ(self):
        """Minimize and maximize over the same data should steer to different regions."""
        X = [[-4.0, -4.0], [0.0, 0.0], [4.0, 4.0]]
        y = [1.0, 2.0, 3.0]
        dims = _float_dims()
        res_min = fantasize_suggest(X, y, dims, q=1, direction="minimize", seed=42)
        res_max = fantasize_suggest(X, y, dims, q=1, direction="maximize", seed=42)
        assert res_min != res_max

    def test_minimize_steers_away_from_best_known(self):
        """With minimize, suggestion should not collapse onto the already-best point."""
        # Best (lowest y) is at X[0]; GP should explore elsewhere.
        X = [[0.0, 0.0], [4.0, 4.0], [-4.0, -4.0]]
        y = [0.1, 2.0, 3.0]
        result = fantasize_suggest(X, y, _float_dims(), q=1, direction="minimize", seed=0)
        assert len(result) == 1
        # Suggestion should not land exactly on [0,0]
        assert not (result[0]["x"] == pytest.approx(0.0) and result[0]["y"] == pytest.approx(0.0))

    def test_constant_y_does_not_crash(self):
        """All identical y values — GP variance collapses; must not raise."""
        X = [[float(i), float(i)] for i in range(5)]
        y = [1.0] * 5
        result = fantasize_suggest(X, y, _float_dims(), q=2, seed=0)
        assert len(result) == 2
