import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.eval_metrics import (
    resolve_eval_unit,
    window_level_metrics_from_scores,
)


def test_model_uses_d_model_and_dropout():
    torch = pytest.importorskip("torch")
    from model.AnomalyTransformer import AnomalyTransformer

    model = AnomalyTransformer(
        win_size=8,
        enc_in=3,
        c_out=3,
        d_model=128,
        n_heads=8,
        e_layers=1,
        dropout=0.2,
    )
    assert model.embedding.value_embedding.tokenConv.out_channels == 128
    assert model.encoder.attn_layers[0].dropout.p == 0.2


def test_eval_unit_auto_policy():
    assert resolve_eval_unit("ST330IR001_CP001", "auto") == "window"
    assert resolve_eval_unit("MSL", "auto") == "point"
    assert resolve_eval_unit("SMAP", "point") == "point"


def test_window_threshold_uses_only_validation_scores():
    val_scores = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=float)
    test_scores = np.array([100, 100, 200, 200, 300, 300, 400, 400], dtype=float)
    labels_a = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    labels_b = 1 - labels_a

    a = window_level_metrics_from_scores(test_scores, labels_a, val_scores, win_size=2, percentile=95.0)
    b = window_level_metrics_from_scores(test_scores * 10, labels_b, val_scores, win_size=2, percentile=95.0)

    assert a["threshold"] == b["threshold"]
    assert a["threshold_details"]["source"] == "normal_validation_windows"
    assert not a["threshold_details"]["uses_test_scores"]
    assert not a["threshold_details"]["uses_test_labels"]
    assert not a["threshold_details"]["uses_test_anomaly_ratio"]


def test_prepare_st_split_has_no_leakage(tmp_path):
    raw = tmp_path / "raw"
    normal = raw / "normal"
    fault = raw / "fault_1"
    normal.mkdir(parents=True)
    fault.mkdir()
    columns = [f"f{i}" for i in range(29)]
    for i in range(10):
        np.savetxt(normal / f"normal_{i}.csv", np.full((56, 29), i), delimiter=",", header=",".join(columns), comments="")
    for i in range(3):
        np.savetxt(fault / f"fault_{i}.csv", np.full((56, 29), 100 + i), delimiter=",", header=",".join(columns), comments="")

    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_st330ir001_cp001.py",
            "--raw_root",
            str(raw),
            "--output_dir",
            str(out),
        ],
        check=True,
    )
    manifest = json.loads((out / "ST330IR001_CP001_manifest.json").read_text(encoding="utf-8"))
    train = set(manifest["files"]["train"])
    val = set(manifest["files"]["val"])
    test = set(manifest["files"]["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert manifest["split"]["fault_test"] == 3
    assert np.load(out / "ST330IR001_CP001_test_window_label.npy").sum() == 3
