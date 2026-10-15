import hashlib
import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from credit_risk import finalize
from credit_risk.features import RAW
from credit_risk.data import TARGET


class RejectTestRows(DummyClassifier):
    def fit(self, X, y, sample_weight=None):
        assert (X.AGE < 90).all(), "Test row reached fit"
        return super().fit(X, y, sample_weight)

    def predict_proba(self, X):
        assert (X.AGE < 90).all(), "Test row reached development scoring"
        return super().predict_proba(X)


def test_full_development_excludes_test_and_freezes_before_audit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = Path("artifacts/run-001")
    original.mkdir(parents=True)
    np.savez(
        original / "splits.npz",
        train=np.arange(60),
        validation=np.arange(60, 80),
        test=np.arange(80, 100),
        groups=np.arange(100),
    )
    data = pd.DataFrame(0.0, index=np.arange(100), columns=RAW)
    data["AGE"] = 30
    data.loc[80:, "AGE"] = 99
    data[TARGET] = np.tile([0, 1], 50)
    incumbent = dict(
        metadata=dict(data_sha256="test", decision_threshold=0.28),
        background=data.iloc[:3][RAW],
    )
    monkeypatch.setattr(finalize, "load_bundle", lambda _: incumbent)
    monkeypatch.setattr(finalize, "load_data", lambda _: data)
    monkeypatch.setattr(finalize, "fingerprint", lambda _: "test")
    model = Pipeline([("model", RejectTestRows())])
    monkeypatch.setattr(finalize, "candidate_models", lambda: {"incumbent_xgb": model})
    monkeypatch.setattr(
        finalize, "behavior_candidates", lambda _: {"incumbent_xgb": model}
    )
    args = Namespace(output="artifacts/new", report="reports/new", data="unused")
    finalize.train(args)
    frozen = json.loads(Path("reports/new/frozen_selection.json").read_text())
    assert frozen["original_test_scored"] is False
    assert (
        frozen["model_sha256"]
        == hashlib.sha256(Path("artifacts/new/model.joblib").read_bytes()).hexdigest()
    )
    np.testing.assert_array_equal(
        np.load("artifacts/new/development.npz")["row_indices"], np.arange(80)
    )
    assert not Path("reports/new/audit.json").exists()
    with pytest.raises(ValueError, match="empty"):
        finalize.train(args)
    Path("reports/new/audit_started.json").write_text("{}")
    with pytest.raises(ValueError, match="one-time audit"):
        finalize.audit(args)
    Path("artifacts/new/model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after freezing"):
        finalize.audit(args)
