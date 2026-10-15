import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from credit_risk.behavior import RepaymentFeatures, behavior_candidates
from credit_risk.features import RAW, STATUS
from credit_risk.improve import candidate_models


def fixture():
    row = json.loads(Path("examples/applicant.json").read_text())
    data = pd.DataFrame([row] * 60, columns=RAW)
    data["AGE"] = np.arange(60) + 20
    data["PAY_0"] = np.tile([0, 2], 30)
    return data


def test_features_are_row_local_and_ignore_target_and_id():
    X = fixture()
    model = RepaymentFeatures().fit(X.iloc[:20], np.zeros(20))
    a = model.transform(X)
    b = model.transform(X.assign(ID=123, **{"default payment next month": 1}))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a.iloc[[15]], model.transform(X.iloc[[15]]))
    pd.testing.assert_frame_equal(
        a.sort_index(), model.transform(X.iloc[::-1]).sort_index()
    )
    pd.testing.assert_frame_equal(
        a, RepaymentFeatures().fit(X, np.ones(60)).transform(X)
    )


def test_temporal_direction_ratios_and_zero_denominators():
    X = fixture().iloc[[0]].copy()
    X.loc[:, STATUS] = [3, 2, 0, 1, 0, 0]
    X["LIMIT_BAL"] = 100
    X["BILL_AMT1"], X["BILL_AMT2"], X["PAY_AMT1"] = 60, 50, 10
    row = RepaymentFeatures().transform(X).iloc[0]
    assert row["CURRENT_DELAY_STREAK"] == 2
    assert row["PAYMENT_PRIOR_BILL_1"] == pytest.approx(0.2)
    assert row["BALANCE_FLOW_LIMIT_1"] == pytest.approx(0.2)
    assert row["DELAY_TREND"] == pytest.approx(4 / 3)
    X["LIMIT_BAL"], X["BILL_AMT2"] = 0, -10
    result = RepaymentFeatures().transform(X)
    assert np.isnan(result["PAYMENT_PRIOR_BILL_1"].iloc[0])
    assert not np.isinf(result.to_numpy()).any()


@pytest.mark.parametrize(
    "name", list(behavior_candidates(candidate_models()["incumbent_xgb"]))
)
def test_candidate_clone_fit_and_batch_single_parity(name):
    model = clone(behavior_candidates(candidate_models()["incumbent_xgb"])[name])
    updates = {
        key: 2
        for key in model.get_params()
        if key.endswith(("__n_estimators", "__max_iter"))
    }
    model.set_params(**updates)
    X = fixture()
    model.fit(X, np.tile([0, 1], 30))
    p = model.predict_proba(X)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p[[3]], model.predict_proba(X.iloc[[3]]))
