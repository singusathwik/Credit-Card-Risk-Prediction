"""Flask API and built React application. Never serve the repository root."""

import json
import logging
import os
from pathlib import Path
from threading import Lock
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from credit_risk.artifacts import load_bundle
from credit_risk.features import validate_payload, RAW
from credit_risk.explain import local_explanation

ROOT = Path(__file__).resolve().parent


def create_app(bundle=None, artifact_path=None, report_path=None):
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 16384
    report_dir = Path(
        report_path or os.environ.get("REPORT_DIR", ROOT / "reports/run-001")
    )
    if bundle is None:
        try:
            bundle = load_bundle(
                artifact_path
                or os.environ.get("MODEL_PATH", ROOT / "artifacts/run-001/model.joblib")
            )
        except Exception as exc:
            app.logger.warning(
                "Model unavailable (%s); train a compatible local artifact.",
                type(exc).__name__,
            )
    app.config["MODEL_BUNDLE"] = bundle
    origins = [
        s.strip() for s in os.environ.get("CORS_ORIGINS", "").split(",") if s.strip()
    ]
    if origins:
        CORS(
            app,
            origins=origins,
            methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    explanation_lock = Lock()

    def ready():
        return app.config["MODEL_BUNDLE"]

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def internal_error(error):
        app.logger.error("Request failed (%s)", type(error).__name__)
        return jsonify(error="Internal service error."), 500

    @app.get("/health")
    @app.get("/api/health")
    def health():
        return jsonify(
            status="ready" if ready() else "unavailable",
            model_loaded=ready() is not None,
        ), (200 if ready() else 503)

    @app.get("/model-info")
    @app.get("/api/model-info")
    def model_info():
        if not ready():
            return jsonify(error="Model unavailable. Train a compatible artifact."), 503
        return jsonify(
            **ready()["metadata"],
            input_fields=RAW,
            notice="Educational prototype; probability estimates are not guarantees or lending recommendations."
        )

    def report():
        result = json.loads((report_dir / "evaluation.json").read_text())
        if (
            not ready()
            or result["metadata"]["model_version"]
            != ready()["metadata"]["model_version"]
        ):
            raise ValueError("Report version mismatch")
        return result

    @app.get("/api/evaluation")
    def evaluation():
        try:
            result = report()
            result["models"] = json.loads((report_dir / "selection.json").read_text())[
                "models"
            ]
            return jsonify(result)
        except (OSError, ValueError):
            return jsonify(error="Matching evaluation report unavailable."), 503

    @app.get("/accuracy")
    @app.get("/api/accuracy")
    def accuracy():
        try:
            a = report()["test"]["accuracy"]
            return jsonify(accuracy=a, accuracy_percent=a * 100)
        except (OSError, ValueError):
            return jsonify(error="Matching evaluation report unavailable."), 503

    def prediction(explain=False):
        if not ready():
            return jsonify(error="Model unavailable. Run training first."), 503
        if not request.is_json:
            return jsonify(error="Content-Type must be application/json."), 415
        payload = request.get_json(silent=True)
        try:
            X = validate_payload(payload)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        b = ready()
        meta = b["metadata"]
        probability = float(b["model"].predict_proba(X)[0, 1])
        lo, hi = meta["risk_boundaries"]
        response = dict(
            default_probability=probability,
            will_default=probability >= meta["decision_threshold"],
            predicted_label=int(probability >= meta["decision_threshold"]),
            decision_threshold=meta["decision_threshold"],
            risk_category=(
                "Low"
                if probability < lo
                else "Moderate" if probability < hi else "High"
            ),
            risk_boundaries=[lo, hi],
            risk_boundaries_illustrative=True,
            model_version=meta["model_version"],
            notice="Educational estimate; not a guarantee, causal conclusion or lending decision.",
        )
        if explain:
            if not explanation_lock.acquire(blocking=False):
                return jsonify(error="Explanation service busy; retry shortly."), 429
            try:
                response["explanation"] = local_explanation(b, X)
            finally:
                explanation_lock.release()
        return jsonify(response)

    app.add_url_rule("/predict", "predict", prediction, methods=["POST"])
    app.add_url_rule("/api/predict", "api_predict", prediction, methods=["POST"])
    app.add_url_rule("/explain", "explain", lambda: prediction(True), methods=["POST"])
    app.add_url_rule(
        "/api/explain", "api_explain", lambda: prediction(True), methods=["POST"]
    )

    @app.get("/api/plots/<name>")
    def plot(name):
        allowed = {
            "confusion_matrix.png",
            "roc.png",
            "precision_recall.png",
            "calibration.png",
            "comparison.png",
            "thresholds.png",
            "distributions.png",
            "shap_beeswarm.png",
            "shap_importance.png",
            "shap_relationship.png",
        }
        if name not in allowed:
            return jsonify(error="Unknown plot."), 404
        try:
            report()
        except (OSError, ValueError):
            return jsonify(error="Matching evaluation unavailable."), 503
        return send_from_directory(report_dir, name)

    @app.get("/")
    def index():
        if not (ROOT / "dist/index.html").exists():
            return jsonify(error="Build frontend with npm run build."), 503
        return send_from_directory(ROOT / "dist", "index.html")

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(ROOT / "dist/assets", name)

    return app


app = create_app()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from waitress import serve

    serve(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        threads=4,
    )
