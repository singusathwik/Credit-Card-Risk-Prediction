import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
from .features import RAW

PACKAGES = ["scikit-learn", "numpy", "pandas", "scipy", "joblib", "shap", "xgboost"]


def versions(extra_packages=()):
    result = {}
    for p in [*PACKAGES, *extra_packages]:
        try:
            result[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            result[p] = None
    return result


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "sha256": digest,
                "versions": versions(bundle.get("extra_packages", [])),
                "extra_packages": bundle.get("extra_packages", []),
                "schema": RAW,
            },
            indent=2,
        )
    )


def load_bundle(path):
    # Only operator-configured local artifacts. Hash detects corruption, not trust.
    path = Path(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if manifest["schema"] != RAW or manifest["versions"] != versions(
        manifest.get("extra_packages", [])
    ):
        raise ValueError("Artifact schema or dependency versions differ; retrain.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Artifact checksum mismatch.")
    b = joblib.load(path)
    if b.get("schema_version") != 1 or list(b["model"].classes_) != [0, 1]:
        raise ValueError("Unsupported artifact format or output classes.")
    return b
