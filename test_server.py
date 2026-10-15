"""Live smoke check; failures raise rather than silently printing success."""

import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    with urlopen("http://127.0.0.1:5000/health", timeout=10) as r:
        assert json.load(r)["model_loaded"] is True
    payload = Path("examples/applicant.json").read_bytes()
    req = Request(
        "http://127.0.0.1:5000/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=10) as r:
        result = json.load(r)
        assert 0 <= result["default_probability"] <= 1
    print("Live health and prediction assertions passed.")


if __name__ == "__main__":
    main()
