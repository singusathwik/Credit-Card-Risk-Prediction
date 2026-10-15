"""Inspect a hypothetical delayed-payment profile without asserting risk monotonicity."""

import json
from pathlib import Path
from server import create_app


def main():
    d = json.loads(Path("examples/applicant.json").read_text())
    d.update(PAY_0=2, PAY_2=3, PAY_3=2, PAY_4=1)
    response = create_app().test_client().post("/predict", json=d)
    assert response.status_code == 200
    print(json.dumps(response.json, indent=2))


if __name__ == "__main__":
    main()
