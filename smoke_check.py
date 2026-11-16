"""Small HTTP contract check for an already running prediction service."""

import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def check(base_url):
    evidence = []

    def request(path, data=None, expected=200, content_type="application/json"):
        headers = {"Content-Type": content_type} if data is not None else {}
        req = Request(base_url.rstrip("/") + path, data=data, headers=headers)
        try:
            with urlopen(req, timeout=20) as response:
                status, body = response.status, response.read()
        except HTTPError as error:
            status, body = error.code, error.read()
        assert status == expected, (path, status, expected, body[:200])
        result = json.loads(body)
        assert isinstance(result, dict)
        evidence.append({"path": path, "expected_status": expected, "status": status})
        return result

    health = request("/health")
    assert health["status"] == "ok"
    sample = json.loads(Path(__file__).with_name("sample_request.json").read_text())
    prediction = request("/predict", json.dumps(sample).encode())
    value = prediction["strength_mpa"]
    assert type(value) in (int, float) and math.isfinite(value) and value >= 0
    assert prediction["unit"] == "MPa"
    cases = [
        {key: val for key, val in sample.items() if key != "age"},
        {**sample, "unexpected": 1},
        {**sample, "age": True},
        {**sample, "age": "28"},
        {**sample, "water": -1},
        {**sample, "cement": float("nan")},
        [],
    ]
    for payload in cases:
        result = request("/predict", json.dumps(payload).encode(), expected=400)
        assert "error" in result
    request("/predict", b"{broken", expected=400)
    request("/predict", b"{}", expected=415, content_type="text/plain")
    request("/predict", b" " * 9000, expected=413)
    request("/missing", expected=404)
    return {"health": health, "prediction": prediction, "checks": evidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9696")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.url)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
