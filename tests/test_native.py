import json
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from credit_risk.native import NativeCategories, native_candidates
from credit_risk.features import FeatureBuilder, RAW, CAT
from credit_risk.artifacts import save_bundle, load_bundle


def test_categories_match_api_float_and_workbook_integer():
    raw = pd.read_json("examples/applicant.json", typ="series").to_frame().T[RAW]
    transform = NativeCategories()
    integer = transform.fit_transform(FeatureBuilder().fit_transform(raw.astype(int)))
    floating = transform.transform(FeatureBuilder().fit_transform(raw.astype(float)))
    pd.testing.assert_frame_equal(integer, floating)
    missing = integer.copy()
    missing[CAT[0]] = np.nan
    assert transform.transform(missing)[CAT[0]].iloc[0] == "missing"


def test_native_pipeline_clone_predict_and_dependency_manifest(tmp_path):
    raw = pd.read_json("examples/applicant.json", typ="series").to_frame().T[RAW]
    X = pd.concat([raw] * 20, ignore_index=True)
    X["AGE"] = np.arange(20) + 20
    y = np.array([0, 1] * 10)
    model = clone(native_candidates()["catboost_depth4"])
    model.set_params(model__iterations=3).fit(X, y)
    before = model.predict_proba(raw)
    path = tmp_path / "model.joblib"
    save_bundle(dict(schema_version=1, model=model, extra_packages=["catboost"]), path)
    np.testing.assert_allclose(load_bundle(path)["model"].predict_proba(raw), before)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["versions"]["catboost"] = "wrong-version"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dependency"):
        load_bundle(path)
