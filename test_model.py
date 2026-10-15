"""Assert that the actual saved model and API return identical probabilities."""

import json
from pathlib import Path
from credit_risk.artifacts import load_bundle
from credit_risk.features import validate_payload
from server import create_app


def main():
    bundle = load_bundle("artifacts/run-001/model.joblib")
    sample = json.loads(Path("examples/applicant.json").read_text())
    expected = float(bundle["model"].predict_proba(validate_payload(sample))[0, 1])
    result = create_app(bundle=bundle).test_client().post("/predict", json=sample)
    assert result.status_code == 200
    assert abs(result.json["default_probability"] - expected) < 1e-10
    explanation = create_app(bundle=bundle).test_client().post("/explain", json=sample)
    assert explanation.status_code == 200
    assert (
        abs(explanation.json["explanation"]["reconstructed_probability"] - expected)
        < 1e-6
    )
    print(
        json.dumps(
            {
                "model_version": bundle["metadata"]["model_version"],
                "probability": expected,
                "prediction_and_shap": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
