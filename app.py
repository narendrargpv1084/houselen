"""
HouseLens — Flask Application Entry Point
==========================================
Gunicorn start command (Render):
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120

Local dev:
    python app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ── Absolute root of the repository ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "web", "templates"),
    static_folder=os.path.join(ROOT, "web", "static"),
    static_url_path="/static",
)
CORS(app)

# ── Load ML artifacts ─────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(ROOT, "models")

try:
    model         = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    oe            = joblib.load(os.path.join(MODEL_DIR, "ordinal_encoder.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    print("[OK] Model artifacts loaded.")
except FileNotFoundError as exc:
    print(f"[ERROR] {exc}")
    print("Run  python src/train.py  first to generate model artifacts.")
    sys.exit(1)

BINARY_COLS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea",
]


def preprocess(data: dict) -> np.ndarray:
    """Apply identical transforms to those used in src/train.py."""
    df = pd.DataFrame([data])

    for col in BINARY_COLS:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})

    if df["furnishingstatus"].dtype == object:
        df["furnishingstatus"] = oe.transform(df[["furnishingstatus"]])

    df["price_per_sqft_proxy"] = df["area"] / (df["bedrooms"] + 1)
    df["total_rooms"]          = df["bedrooms"] + df["bathrooms"]
    df["amenity_score"]        = df[BINARY_COLS].sum(axis=1)

    df = df[feature_names]
    return scaler.transform(df)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        required = [
            "area", "bedrooms", "bathrooms", "stories",
            "mainroad", "guestroom", "basement", "hotwaterheating",
            "airconditioning", "parking", "prefarea", "furnishingstatus",
        ]
        missing = [k for k in required if k not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        data = {
            "area":             int(payload["area"]),
            "bedrooms":         int(payload["bedrooms"]),
            "bathrooms":        int(payload["bathrooms"]),
            "stories":          int(payload["stories"]),
            "mainroad":         str(payload["mainroad"]),
            "guestroom":        str(payload["guestroom"]),
            "basement":         str(payload["basement"]),
            "hotwaterheating":  str(payload["hotwaterheating"]),
            "airconditioning":  str(payload["airconditioning"]),
            "parking":          int(payload["parking"]),
            "prefarea":         str(payload["prefarea"]),
            "furnishingstatus": str(payload["furnishingstatus"]),
        }

        X     = preprocess(data)
        price = float(model.predict(X)[0])

        # Confidence band ±1 RMSE (1,327,681 from training evaluation)
        RMSE = 1_327_681
        low  = max(0, price - RMSE)
        high = price + RMSE

        return jsonify({
            "predicted_price": round(price),
            "range_low":       round(low),
            "range_high":      round(high),
            "model":           type(model).__name__,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": type(model).__name__})


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  HouseLens  →  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
