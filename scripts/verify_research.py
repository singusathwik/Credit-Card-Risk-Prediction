"""Check a saved research run without scoring the original test partition."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from credit_risk.artifacts import load_bundle
from credit_risk.features import validate_payload
from server import create_app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    artifact, report = Path(args.artifact), Path(args.report)
    bundle = load_bundle(artifact / "model.joblib")
    payload = json.loads(Path("examples/applicant.json").read_text())
    frame = validate_payload(payload)
    probability = float(bundle["model"].predict_proba(frame)[0, 1])
    client = create_app(bundle=bundle, report_path=report).test_client()
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert abs(response.json["default_probability"] - probability) < 1e-10
    explanation = client.post("/api/explain", json=payload)
    assert explanation.status_code == 200, explanation.json
    assert (
        abs(explanation.json["explanation"]["reconstructed_probability"] - probability)
        < 1e-6
    )
    original = np.load("artifacts/run-001/splits.npz")
    predictions = np.load(artifact / "nested_predictions.npz")
    assert np.array_equal(predictions["row_indices"], original["train"])
    for partition in ("validation", "test"):
        assert not set(predictions["row_indices"]) & set(original[partition])
        assert not set(predictions["groups"]) & set(
            original["groups"][original[partition]]
        )
    for fold in np.unique(predictions["outer_fold"]):
        mask = predictions["outer_fold"] == fold
        assert not set(predictions["groups"][mask]) & set(predictions["groups"][~mask])
    preserved = {
        "artifacts/run-001/model.joblib": "66ad159f6d0d2b4c1b4ba955e153b66cedf9ffeb14f866cbb55fa209520b670e",
        "reports/run-001/evaluation.json": "eb9da0d41efe63bf05f7c11b36a75369fb4c1c2652db97a26cad501eaab1179e",
    }
    for path, digest in preserved.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    result = dict(
        model_version=bundle["metadata"]["model_version"],
        example_probability=probability,
        api_probability_parity=True,
        shap_reconstruction=True,
        training_partition_only=True,
        outer_groups_disjoint=True,
        original_files_unchanged=preserved,
    )
    (report / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
