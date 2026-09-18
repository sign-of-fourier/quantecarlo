"""Unit tests for quantecarlo.qei.QEIClient. No network — urlopen is mocked."""
import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantecarlo import DimSpec, QEIClient, call_modal_api, sample_candidates


def _mock_urlopen(fake_candidates):
    captured = {"bodies": []}
    resp_bytes = json.dumps({"candidates": fake_candidates}).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = resp_bytes

    def fake_urlopen(req, timeout=None):
        captured["bodies"].append(json.loads(req.data.decode()))
        captured["timeout"] = timeout
        return mock_resp

    return fake_urlopen, captured


X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
Y = np.array([1.0, 2.0, 3.0])
CANDS = np.array([[0.7, 0.8], [0.9, 1.0], [0.2, 0.3]])
FAKE = [{"index": 1, "x": [0.9, 1.0], "mu": 0.5, "sigma": 0.1},
        {"index": 2, "x": [0.2, 0.3], "mu": 0.4, "sigma": 0.2}]


class TestSuggest:
    def test_payload_identical_to_call_modal_api(self):
        fake, cap = _mock_urlopen(FAKE)
        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake):
            QEIClient("http://x", train_steps=30, xi=0.05, n_prefilter=100).suggest(
                X, Y, CANDS, q=2, n_batches=64)
            call_modal_api("http://x", X.astype(np.float32), Y.astype(np.float32),
                           CANDS.astype(np.float32), q=2, n_batches=64,
                           train_steps=30, xi=0.05, n_prefilter=100)
        assert cap["bodies"][0] == cap["bodies"][1]

    def test_direction_minimize_negates_y(self):
        fake, cap = _mock_urlopen(FAKE)
        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake):
            QEIClient("http://x").suggest(X, Y, CANDS, q=2, direction="minimize")
            QEIClient("http://x").suggest(X, Y, CANDS, q=2, direction="maximize")
        assert cap["bodies"][0]["y"] == [-1.0, -2.0, -3.0]
        assert cap["bodies"][1]["y"] == [1.0, 2.0, 3.0]

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError):
            QEIClient("http://x").suggest(X, Y, CANDS, q=2, direction="up")

    def test_unknown_tuning_field_raises(self):
        with pytest.raises(TypeError):
            QEIClient("http://x", bogus=1).suggest(X, Y, CANDS, q=2)

    def test_accepts_plain_lists_and_returns_indices(self):
        fake, cap = _mock_urlopen(FAKE)
        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake):
            picks = QEIClient("http://x", timeout=7.0).suggest(X.tolist(), Y.tolist(), CANDS.tolist(), q=2)
        assert [p["index"] for p in picks] == [1, 2]
        assert isinstance(picks[0]["x"], np.ndarray)
        assert cap["timeout"] == 7.0

    def test_multioutput_sends_d_and_rho(self):
        fake, cap = _mock_urlopen(FAKE)
        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake):
            QEIClient("http://x").suggest_multioutput(
                X, Y, CANDS, [0, 1, 0], [1, 1, 0], q=2, rho=0.3, direction="minimize")
        b = cap["bodies"][0]
        assert b["d"] == [0, 1, 0] and b["d_candidates"] == [1, 1, 0]
        assert b["rho"] == pytest.approx(0.3)
        assert b["y"] == [-1.0, -2.0, -3.0]


class TestSampleCandidates:
    def test_shape_bounds_and_seed(self):
        dims = [DimSpec("lr", "float", 1e-4, 1e-1, log=True), DimSpec("n", "int", 2, 9)]
        a = sample_candidates(dims, 50, seed=0)
        b = sample_candidates(dims, 50, seed=0)
        assert a.shape == (50, 2) and np.array_equal(a, b)
        assert (a[:, 0] >= 1e-4).all() and (a[:, 0] <= 1e-1).all()
        assert (a[:, 1] == np.round(a[:, 1])).all() and a[:, 1].min() >= 2 and a[:, 1].max() <= 9


def test_import_does_not_require_optuna():
    with patch.dict(sys.modules, {"optuna": None, "optunahub": None}):
        for m in [k for k in sys.modules if k.startswith("quantecarlo")]:
            del sys.modules[m]
        import quantecarlo  # noqa: F401
        assert quantecarlo.QEIClient
