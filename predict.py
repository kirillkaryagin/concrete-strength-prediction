"""JSON prediction API. The model file must be a trusted local project artifact."""

import math
import pickle
import warnings
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from sklearn.exceptions import InconsistentVersionWarning
from werkzeug.exceptions import HTTPException

FEATURES = [
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age",
]


def create_app():
    # Never load model files from requests or arbitrary remote URLs.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        with Path(__file__).with_name("model.pkl").open("rb") as stream:
            artifact = pickle.load(stream)
    model, metadata = artifact["model"], artifact["metadata"]
    if metadata["features"] != FEATURES:
        raise ValueError("Model feature order differs from the API contract.")
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=8192, JSON_SORT_KEYS=False)

    @app.errorhandler(HTTPException)
    def http_error(error):
        response = error.get_response()
        response.data = app.json.dumps({"error": error.name})
        response.content_type = "application/json"
        return response

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify(error="Prediction service error"), 500

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=metadata["selected_model"], unit="MPa")

    @app.post("/predict")
    def predict():
        payload = request.get_json()
        if not isinstance(payload, dict) or set(payload) != set(FEATURES):
            return jsonify(error="Provide exactly the eight documented feature names.",
                           required=FEATURES), 400
        for key, value in payload.items():
            if type(value) not in (int, float):
                return jsonify(error=f"{key} must be a finite number, not a string or boolean."), 400
            # Check bounds before float conversion; gigantic JSON integers are invalid too.
            upper = 365 if key == "age" else 2000
            lower = 1 if key == "age" else 0
            if not lower <= value <= upper or not math.isfinite(value):
                return jsonify(error=f"{key} must be between {lower} and {upper}."), 400
        if any(payload[key] <= 0 for key in ["cement", "water", "coarse_aggregate", "fine_aggregate"]):
            return jsonify(error="Cement, water and both aggregates must be positive."), 400
        if not float(payload["age"]).is_integer():
            return jsonify(error="age must be a whole number of days."), 400
        messages = []
        for key in FEATURES:
            limits = metadata["training_ranges"][key]
            if not limits["min"] <= payload[key] <= limits["max"]:
                messages.append(f"{key} is outside the observed training range.")
        frame = pd.DataFrame([[payload[key] for key in FEATURES]], columns=FEATURES)
        value = float(model.predict(frame)[0])
        if not math.isfinite(value) or value < 0:
            return jsonify(error="Model produced an invalid strength estimate."), 500
        return jsonify(strength_mpa=value, unit="MPa", model=metadata["selected_model"],
                       warnings=messages, educational_use_only=True)

    return app
