"""Unit tests for orthant_cdf. No network -- urlopen is mocked."""
import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantecarlo import orthant_cdf
from quantecarlo._cdf_api import build_cdf_body


def _mock_urlopen(p):
    captured = {}
    buf = io.BytesIO()
    np.savez(buf, p=np.asarray(p))
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = buf.getvalue()

    def fake_urlopen(req, timeout=None):
        npz = np.load(io.BytesIO(req.data), allow_pickle=False)
        captured["body"] = {k: npz[k] for k in npz.files}
        captured["headers"] = dict(req.headers)
        captured["timeout"] = timeout
        return mock_resp

    return fake_urlopen, captured


R = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])


def test_posts_npz_and_returns_p():
    upper = np.arange(12.0).reshape(4, 3)
    fake, cap = _mock_urlopen([0.1, 0.2, 0.3, 0.4])
    with patch("quantecarlo._cdf_api.urllib.request.urlopen", fake):
        p = orthant_cdf(upper, R, signs=[1, -1, 1], resolution="low", timeout=9)
    np.testing.assert_allclose(p, [0.1, 0.2, 0.3, 0.4])
    assert p.dtype == np.float64
    body = cap["body"]
    np.testing.assert_array_equal(body["upper"], upper)
    np.testing.assert_array_equal(body["cov_tril"], R[np.tril_indices(3)])
    np.testing.assert_array_equal(body["signs"], [1.0, -1.0, 1.0])
    assert json.loads(str(body["params"])) == {"resolution": "low"}
    assert cap["headers"]["Content-type"] == "application/octet-stream"
    assert cap["timeout"] == 9


def test_single_row_and_no_signs():
    fake, cap = _mock_urlopen([0.5])
    with patch("quantecarlo._cdf_api.urllib.request.urlopen", fake):
        orthant_cdf([0.0, 0.0, 0.0], R)
    assert cap["body"]["upper"].shape == (1, 3)
    assert cap["body"]["upper"].dtype == np.float32      # the default
    assert cap["body"]["cov_tril"].dtype == np.float64   # always
    assert "signs" not in cap["body"]


@pytest.mark.parametrize("kwargs", [
    {"cov": np.eye(2)},
    {"signs": [1, 0, 1]},
    {"signs": [1, -1]},
    {"resolution": "medium"},
    {"dtype": "float16"},
])
def test_rejects_bad_input(kwargs):
    args = {"upper": np.zeros((2, 3)), "cov": R, **kwargs}
    with pytest.raises(ValueError):
        build_cdf_body(**args)


def test_float64_on_request_and_body_is_uncompressed():
    import zipfile
    body = build_cdf_body(np.zeros((2, 3)), R, dtype="float64")
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())
    with np.load(io.BytesIO(body)) as npz:
        assert npz["upper"].dtype == np.float64
