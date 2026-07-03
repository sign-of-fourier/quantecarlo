"""Unit tests for quantecarlo.bo_sampler. No network — urlopen is mocked."""
import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantecarlo import DimSpec, modal_suggest
from quantecarlo.bo_sampler import _sample_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dims():
    return [
        DimSpec("lr", "float", 1e-4, 1e-1, log=True),
        DimSpec("n_hidden", "int", 16, 256),
        DimSpec("alpha", "float", 1e-5, 1e-2),
    ]


def _mock_urlopen(fake_candidates: list[dict]):
    """Return (fake_urlopen, captured). captured['body'] is set on each call."""
    captured = {}
    resp_bytes = json.dumps({"candidates": fake_candidates}).encode()

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = resp_bytes

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return mock_resp

    return fake_urlopen, captured


def _fake_candidates(dims, q, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "index": i,
            "x": [float(rng.uniform(dim.low, dim.high)) for dim in dims],
            "mu": 0.5,
            "sigma": 0.1,
        }
        for i in range(q)
    ]


# ---------------------------------------------------------------------------
# _sample_candidates
# ---------------------------------------------------------------------------

class TestSampleCandidates:

    def test_returns_n_rows(self):
        rng = np.random.default_rng(0)
        result = _sample_candidates(_dims(), 50, rng)
        assert len(result) == 50

    def test_each_row_has_d_columns(self):
        dims = _dims()
        rng = np.random.default_rng(0)
        result = _sample_candidates(dims, 10, rng)
        assert all(len(row) == len(dims) for row in result)

    def test_n_zero_returns_empty(self):
        rng = np.random.default_rng(0)
        assert _sample_candidates(_dims(), 0, rng) == []

    def test_log_float_dim_stays_in_bounds(self):
        dims = [DimSpec("lr", "float", 1e-4, 1e-1, log=True)]
        rng = np.random.default_rng(0)
        result = _sample_candidates(dims, 500, rng)
        assert all(1e-4 <= row[0] <= 1e-1 for row in result)

    def test_int_dim_values_are_whole_numbers_in_range(self):
        dims = [DimSpec("n", "int", 1, 100)]
        rng = np.random.default_rng(0)
        result = _sample_candidates(dims, 200, rng)
        for row in result:
            assert float(row[0]) == int(row[0])
            assert 1 <= row[0] <= 100

    def test_mixed_dims_each_in_bounds(self):
        dims = [
            DimSpec("x", "float", 0.0, 1.0),
            DimSpec("n", "int", 5, 10),
            DimSpec("lr", "float", 1e-3, 1.0, log=True),
        ]
        rng = np.random.default_rng(7)
        result = _sample_candidates(dims, 100, rng)
        for row in result:
            assert 0.0 <= row[0] <= 1.0
            assert 5 <= row[1] <= 10
            assert 1e-3 <= row[2] <= 1.0


# ---------------------------------------------------------------------------
# modal_suggest — mocked urlopen
# ---------------------------------------------------------------------------

class TestModalSuggest:

    def test_payload_has_required_keys(self):
        dims = _dims()
        X = [[0.01, 32, 1e-4], [0.05, 64, 5e-4]]
        y = [0.3, 0.2]
        fake_urlopen, captured = _mock_urlopen(_fake_candidates(dims, 2))

        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake_urlopen):
            modal_suggest(X, y, dims, q=2, direction="minimize", api_url="https://fake.run")

        body = captured["body"]
        for key in ("X", "y", "candidates", "q", "n_batches", "train_steps", "lr", "xi", "mode"):
            assert key in body, f"payload missing '{key}'"

    def test_y_negated_for_minimize(self):
        dims = _dims()
        X = [[0.01, 32, 1e-4]]
        y = [0.5]
        fake_urlopen, captured = _mock_urlopen(_fake_candidates(dims, 1))

        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake_urlopen):
            modal_suggest(X, y, dims, q=1, direction="minimize", api_url="https://fake.run")

        assert captured["body"]["y"] == pytest.approx([-0.5])

    def test_y_not_negated_for_maximize(self):
        dims = _dims()
        X = [[0.01, 32, 1e-4]]
        y = [0.5]
        fake_urlopen, captured = _mock_urlopen(_fake_candidates(dims, 1))

        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake_urlopen):
            modal_suggest(X, y, dims, q=1, direction="maximize", api_url="https://fake.run")

        assert captured["body"]["y"] == pytest.approx([0.5])

    def test_candidates_length_equals_n_candidates(self):
        dims = _dims()
        X = [[0.01, 32, 1e-4]]
        y = [0.3]
        fake_urlopen, captured = _mock_urlopen(_fake_candidates(dims, 1))

        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake_urlopen):
            modal_suggest(X, y, dims, q=1, direction="minimize",
                          api_url="https://fake.run", n_candidates=128)

        assert len(captured["body"]["candidates"]) == 128

    def test_returns_q_dicts_with_correct_param_names(self):
        dims = _dims()
        X = [[0.01, 32, 1e-4], [0.05, 64, 5e-4]]
        y = [0.3, 0.2]
        q = 3
        fake_urlopen, _ = _mock_urlopen(_fake_candidates(dims, q))

        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake_urlopen):
            result = modal_suggest(X, y, dims, q=q, direction="minimize", api_url="https://fake.run")

        assert len(result) == q
        for params in result:
            assert set(params.keys()) == {"lr", "n_hidden", "alpha"}

    def test_int_dim_rounded_in_output(self):
        dims = [DimSpec("n", "int", 1, 100)]
        X = [[10]]
        y = [0.5]
        fake_urlopen, _ = _mock_urlopen(
            [{"index": 0, "x": [42.7], "mu": 0.1, "sigma": 0.05}]
        )

        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake_urlopen):
            result = modal_suggest(X, y, dims, q=1, direction="minimize", api_url="https://fake.run")

        assert result[0]["n"] == 43
        assert isinstance(result[0]["n"], int)

    def test_http_error_body_surfaced_in_exception(self):
        dims = _dims()
        X = [[0.01, 32, 1e-4]]
        y = [0.3]

        def error_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(
                url="https://fake.run",
                code=500,
                msg="Internal Server Error",
                hdrs=MagicMock(),
                fp=io.BytesIO(b"GP fitting exploded"),
            )

        with patch("quantecarlo._modal_api.urllib.request.urlopen", error_urlopen):
            with pytest.raises(urllib.error.HTTPError, match="GP fitting exploded"):
                modal_suggest(X, y, dims, q=1, direction="minimize", api_url="https://fake.run")
