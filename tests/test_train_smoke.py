"""End-to-end smoke test of train_gender.py on a tiny synthetic dataset (CPU, no downloads)."""
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402


def _make_dataset(root, n_originals=12, copies=3):
    rng = np.random.default_rng(0)
    for i in range(n_originals):
        cls = "female" if i % 2 == 0 else "male"
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        base = (rng.random((40, 60, 3)) * 255).astype("uint8")
        Image.fromarray(base).save(d / f"{i:02d}.png")
        for k in range(copies - 1):
            Image.fromarray(np.roll(base, k + 1, axis=1)).save(d / f"{i:02d}_aug_{k:04d}.png")


def test_training_runs_and_writes_leak_free_metrics(tmp_path, monkeypatch):
    data, out = tmp_path / "data", tmp_path / "out"
    _make_dataset(data)
    import train_gender

    monkeypatch.setattr(sys, "argv", ["train_gender.py", "--data-dir", str(data), "--epochs", "1",
                                      "--out-dir", str(out), "--weights", "none", "--batch-size", "4"])
    train_gender.main()
    m = json.loads((out / "metrics.json").read_text())
    assert m["test_originals_only"]["n"] == m["n_test_source_xrays"]
    assert 0.0 <= m["test_originals_only"]["accuracy"] <= 1.0
    assert (out / "best_gender_model.pth").exists()
