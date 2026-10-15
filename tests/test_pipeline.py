import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from credit_risk.features import (
    RAW,
    NUM,
    FeatureBuilder,
    make_pipeline,
    validate_payload,
)
from credit_risk.data import groups_for, split_data, TARGET
from credit_risk.artifacts import load_bundle, save_bundle
from credit_risk.metrics import metrics, threshold_study
from credit_risk.explain import local_explanation


@pytest.fixture
def sample():
    d = {k: 0 for k in RAW}
    d.update(LIMIT_BAL=200000, SEX=1, EDUCATION=2, MARRIAGE=1, AGE=35)
    return d


@pytest.fixture
def trained(sample):
    rng = np.random.default_rng(42)
    X = pd.DataFrame([sample] * 80)
    X["LIMIT_BAL"] = rng.integers(10000, 500000, 80)
    X["PAY_0"] = rng.integers(-2, 5, 80)
    X["BILL_AMT1"] = rng.integers(-5000, 100000, 80)
    y = (X.PAY_0 > 1).astype(int)
    model = make_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    return (
        dict(
            schema_version=1,
            model=model,
            background=X.iloc[:5],
            metadata=dict(
                model_version="test-v1",
                decision_threshold=0.3,
                risk_boundaries=[0.2, 0.5],
            ),
        ),
        X,
        y,
    )


def test_feature_values_and_safe_zero(sample):
    sample.update(LIMIT_BAL=0, BILL_AMT1=-1, PAY_0=-2, PAY_2=3)
    d = FeatureBuilder().fit_transform(pd.DataFrame([sample]))
    assert np.isnan(d.UTILIZATION_RATIO.iloc[0])
    assert d.MAX_PAY_DELAY.iloc[0] == 3 and d.DELAY_COUNT.iloc[0] == 1
    assert d.BILL_CHANGE.iloc[0] == -1
    assert not np.isinf(d.to_numpy()).any()


def test_validation_order(sample):
    assert list(validate_payload(dict(reversed(list(sample.items())))).columns) == RAW


@pytest.mark.parametrize(
    "field,value",
    [
        ("AGE", None),
        ("AGE", True),
        ("AGE", 20.5),
        ("SEX", 3),
        ("PAY_0", -3),
        ("LIMIT_BAL", 0),
        ("PAY_AMT1", -1),
        ("AGE", ""),
        ("AGE", float("nan")),
        ("AGE", float("inf")),
        ("AGE", {}),
        ("AGE", []),
    ],
)
def test_reject_bad_inputs(sample, field, value):
    sample[field] = value
    with pytest.raises(ValueError):
        validate_payload(sample)


def test_missing_extra(sample):
    with pytest.raises(ValueError):
        validate_payload({**sample, "ID": 1})
    sample.pop("AGE")
    with pytest.raises(ValueError):
        validate_payload(sample)


def test_train_only_imputation(trained):
    b, X, y = trained
    X.loc[0, "LIMIT_BAL"] = np.nan
    model = make_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    imputer = (
        model.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
    )
    expected = np.nanmedian(X.LIMIT_BAL)
    assert imputer.statistics_[NUM.index("LIMIT_BAL")] == expected
    heldout = X.iloc[:2].copy()
    heldout["LIMIT_BAL"] = 9000000
    model.predict_proba(heldout)
    assert imputer.statistics_[NUM.index("LIMIT_BAL")] == expected


def test_single_batch_and_missing_values(trained):
    b, X, y = trained
    p = b["model"].predict_proba(X)[:, 1]
    assert np.all((p >= 0) & (p <= 1))
    assert b["model"].predict_proba(X.iloc[[0]])[0, 1] == pytest.approx(p[0])
    missing = X.iloc[:2].copy()
    missing["BILL_AMT1"] = np.nan
    assert np.isfinite(b["model"].predict_proba(missing)).all()


def test_groups_and_splits(trained):
    _, X, y = trained
    d = X.copy()
    d["ID"] = np.arange(len(d))
    d[TARGET] = y
    d = pd.concat([d, d.iloc[:5]], ignore_index=True)
    # Repeated ID with changed profile must join the same group too.
    d.loc[81, "BILL_AMT1"] += 123
    splits, groups = split_data(d)
    assert groups[1] == groups[81]
    assert groups[0] == groups[80]
    a, b, c = [set(groups[splits[k]]) for k in ["train", "validation", "test"]]
    assert not (a & b or a & c or b & c)
    assert sum(len(v) for v in splits.values()) == len(d)


def test_artifact_integrity(trained, tmp_path):
    b, X, _ = trained
    p = tmp_path / "model.joblib"
    save_bundle(b, p)
    loaded = load_bundle(p)
    np.testing.assert_allclose(
        loaded["model"].predict_proba(X), b["model"].predict_proba(X)
    )
    manifest = p.with_suffix(".manifest.json")
    m = json.loads(manifest.read_text())
    m["versions"]["numpy"] = "invalid"
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="versions"):
        load_bundle(p)
    save_bundle(b, p)
    p.write_bytes(p.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(p)


def test_threshold_and_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    threshold, rows = threshold_study(y, p)
    assert 0.2 < threshold <= 0.3
    assert metrics(y, p, threshold)["confusion_matrix"] == [[2, 0], [0, 2]]


def test_shap_exact_served_probability(trained):
    b, X, _ = trained
    e = local_explanation(b, X.iloc[[0]])
    assert e["output"] == "class_1_probability"
    assert len(e["contributions"]) == 23
    assert e["reconstructed_probability"] == pytest.approx(
        b["model"].predict_proba(X.iloc[[0]])[0, 1], abs=1e-6
    )
