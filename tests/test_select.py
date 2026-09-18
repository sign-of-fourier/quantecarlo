"""Unit tests for QEIClient.select / call_select_api. No network — urlopen is mocked."""
import io
import json
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantecarlo import QEIClient, call_select_api
from quantecarlo._select_api import build_select_body, pack_tril

RESULT = {"indices": [2, 0], "qei": 0.31, "regime": "exact",
          "n_cands": 4, "n_sampled": 6, "n_batches": 6}


def _mock_urlopen(result=RESULT):
    captured = {"bodies": [], "headers": []}
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = json.dumps(result).encode()

    def fake_urlopen(req, timeout=None):
        npz = np.load(io.BytesIO(req.data), allow_pickle=False)
        captured["bodies"].append({k: npz[k] for k in npz.files})
        captured["headers"].append(dict(req.headers))
        captured["timeout"] = timeout
        return mock_resp

    return fake_urlopen, captured


def _posterior(n=4, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return rng.normal(size=n), a @ a.T + np.eye(n)


def test_pack_tril_round_trip():
    _, cov = _posterior(5)
    packed = pack_tril(cov)
    assert packed.shape == (15,)
    full = np.zeros((5, 5)); full[np.tril_indices(5)] = packed
    np.testing.assert_array_equal(full + np.tril(full, -1).T, cov)
    with pytest.raises(ValueError):
        pack_tril(np.zeros((3, 4)))


def test_body_layout_and_dtype():
    mu, cov = _posterior()
    body = build_select_body(mu, cov, {"q": 2, "best_y": 0.1}, dtype="float32")
    npz = np.load(io.BytesIO(body))
    assert set(npz.files) == {"mu", "cov_tril", "params"}
    assert npz["mu"].dtype == np.float32 and npz["cov_tril"].dtype == np.float32
    assert json.loads(str(npz["params"])) == {"q": 2, "best_y": 0.1}
    with pytest.raises(ValueError):
        build_select_body(mu, cov, {}, dtype="float16")
    with pytest.raises(ValueError):
        build_select_body(mu[:3], cov, {})


def test_call_select_api_posts_octet_stream_and_parses():
    mu, cov = _posterior()
    fake, cap = _mock_urlopen()
    diag = {}
    with patch("quantecarlo._select_api.urllib.request.urlopen", fake):
        out = call_select_api("http://x", mu, cov, best_y=0.5, q=2, seed=3, timeout=7,
                              mode="debug", diagnostics=diag)
    assert out == RESULT
    assert cap["headers"][0]["Content-type"] == "application/octet-stream"
    assert cap["timeout"] == 7
    body = cap["bodies"][0]
    assert body["mu"].dtype == np.float64
    np.testing.assert_array_equal(body["mu"], mu)
    np.testing.assert_array_equal(body["cov_tril"], pack_tril(cov))
    assert json.loads(str(body["params"])) == {"q": 2, "best_y": 0.5, "mode": "debug", "seed": 3}
    assert diag == {}                      # nothing beyond the result keys came back


def test_debug_keys_land_in_diagnostics():
    mu, cov = _posterior()
    fake, _ = _mock_urlopen({**RESULT, "ei_all": [0.1, 0.3], "timing_s": {"total": 0.2}, "prefilter": None})
    diag = {}
    with patch("quantecarlo._select_api.urllib.request.urlopen", fake):
        out = call_select_api("http://x", mu, cov, 0.0, q=2, diagnostics=diag)
    assert out == RESULT
    assert diag == {"ei_all": [0.1, 0.3], "timing_s": {"total": 0.2}}


def test_unknown_field_raises():
    mu, cov = _posterior()
    with pytest.raises(TypeError):
        call_select_api("http://x", mu, cov, 0.0, q=2, n_prefilter=10)


def test_large_n_float64_warns(caplog):
    n = 1200
    mu = np.zeros(n); cov = np.eye(n)
    fake, cap = _mock_urlopen()
    with patch("quantecarlo._select_api.urllib.request.urlopen", fake):
        with caplog.at_level(logging.WARNING, logger="quantecarlo._select_api"):
            call_select_api("http://x", mu, cov, 0.0, q=2)
            assert any("float32" in r.message for r in caplog.records)
            caplog.clear()
            call_select_api("http://x", mu, cov, 0.0, q=2, dtype="float32")
            assert not caplog.records
    assert cap["bodies"][1]["cov_tril"].dtype == np.float32


class TestClient:
    def test_select_uses_select_url_and_client_defaults(self):
        mu, cov = _posterior()
        fake, cap = _mock_urlopen()
        seen = {}

        def spy(req, timeout=None):
            seen["url"] = req.full_url
            return fake(req, timeout)

        with patch("quantecarlo._select_api.urllib.request.urlopen", spy):
            c = QEIClient("http://gp", select_url="http://sel", dtype="float32", mode="debug",
                          n_prefilter=100, ei_budget=500, gh_nodes=9)
            out = c.select(mu, cov, best_y=0.2, q=2, seed=1)
        assert out["indices"] == [2, 0]
        assert seen["url"] == "http://sel"
        body = cap["bodies"][0]
        assert body["cov_tril"].dtype == np.float32
        params = json.loads(str(body["params"]))
        # select fields and shared orthant fields forwarded; suggest-only ones not
        assert params == {"q": 2, "best_y": 0.2, "mode": "debug", "ei_budget": 500, "gh_nodes": 9, "seed": 1}

    def test_suggest_does_not_forward_select_fields(self):
        from tests.test_qei import _mock_urlopen as mock_gp, X, Y, CANDS, FAKE
        fake, cap = mock_gp(FAKE)
        with patch("quantecarlo._modal_api.urllib.request.urlopen", fake):
            QEIClient("http://x", ei_budget=500, n_prefilter=100).suggest(X, Y, CANDS, q=2)
        assert "ei_budget" not in cap["bodies"][0] and cap["bodies"][0]["n_prefilter"] == 100

    def test_unknown_tuning_rejected_at_construction(self):
        with pytest.raises(TypeError):
            QEIClient("http://x", bogus=1)

    def test_per_call_dtype_override(self):
        mu, cov = _posterior()
        fake, cap = _mock_urlopen()
        with patch("quantecarlo._select_api.urllib.request.urlopen", fake):
            QEIClient().select(mu, cov, 0.0, q=2, dtype="float32")
        assert cap["bodies"][0]["mu"].dtype == np.float32
