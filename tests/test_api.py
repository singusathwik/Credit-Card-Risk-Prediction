import pytest
from test_pipeline import sample, trained
from server import create_app


@pytest.fixture
def client(trained):
    return create_app(bundle=trained[0]).test_client()


def test_api_consistency(client, trained, sample):
    from credit_risk.features import validate_payload

    r = client.post("/api/predict", json=sample)
    assert r.status_code == 200
    d = r.get_json()
    assert d["default_probability"] == pytest.approx(
        trained[0]["model"].predict_proba(validate_payload(sample))[0, 1]
    )
    assert d["predicted_label"] == int(d["default_probability"] >= 0.3)
    assert client.get("/health").status_code == 200
    assert client.get("/model-info").get_json()["model_version"] == "test-v1"


@pytest.mark.parametrize("payload", [None, [], {}, {"AGE": "bad"}])
def test_bad_json(client, payload):
    assert (
        client.post(
            "/predict", json=payload, content_type="application/json"
        ).status_code
        == 400
    )


def test_explain_and_alias(client, sample):
    p = client.post("/predict", json=sample).get_json()
    r = client.post("/explain", json=sample)
    assert r.status_code == 200
    assert r.get_json()["explanation"]["reconstructed_probability"] == pytest.approx(
        p["default_probability"], abs=1e-6
    )


def test_no_artifact_no_leak(tmp_path):
    c = create_app(artifact_path=tmp_path / "absent.joblib").test_client()
    assert c.get("/health").status_code == 503
    assert c.post("/predict", json={}).status_code == 503
    assert c.get("/server.py").status_code == 404
    assert c.get("/default%20of%20credit%20card%20clients.xls").status_code == 404


def test_protocol_errors(client):
    assert (
        client.post("/predict", data="oops", content_type="text/plain").status_code
        == 415
    )
    assert (
        client.post("/predict", data="{", content_type="application/json").status_code
        == 400
    )
    assert (
        client.post(
            "/predict", data="x" * 20000, content_type="application/json"
        ).status_code
        == 413
    )
    assert client.get("/api/plots/server.py").status_code == 404
    assert "Access-Control-Allow-Origin" not in client.get("/health").headers


def test_configurable_cors(monkeypatch, trained):
    monkeypatch.setenv("CORS_ORIGINS", "https://demo.example")
    c = create_app(bundle=trained[0]).test_client()
    assert (
        c.get("/health", headers={"Origin": "https://demo.example"}).headers[
            "Access-Control-Allow-Origin"
        ]
        == "https://demo.example"
    )
    assert (
        "Access-Control-Allow-Origin"
        not in c.get("/health", headers={"Origin": "https://untrusted.example"}).headers
    )


def test_safe_internal_error(client, monkeypatch, trained):
    def fail(*args):
        raise RuntimeError("private applicant details")

    monkeypatch.setattr(trained[0]["model"], "predict_proba", fail)
    from test_pipeline import RAW
    from credit_risk.features import validate_payload

    sample = trained[1].iloc[0].to_dict()
    r = client.post("/predict", json=sample)
    assert r.status_code == 500
    assert "private" not in r.get_data(as_text=True)


def test_evaluation_is_strict_browser_json(trained, tmp_path):
    import json

    (tmp_path / "evaluation.json").write_text(
        json.dumps({"metadata": trained[0]["metadata"], "test": {"accuracy": 0.8}})
    )
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "models": [
                    {"algorithm": "xgboost", "hyperparameters": {"missing": "NaN"}}
                ]
            }
        )
    )
    c = create_app(bundle=trained[0], report_path=tmp_path).test_client()
    r = c.get("/api/evaluation")
    assert r.status_code == 200

    def reject(value):
        raise ValueError(value)

    result = json.loads(r.data, parse_constant=reject)
    assert result["models"][0]["hyperparameters"]["missing"] == "NaN"
