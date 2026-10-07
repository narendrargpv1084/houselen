"""
House Price Prediction — Inference Script
==========================================
Usage:
    python src/predict.py                     ← interactive prompt
    python src/predict.py --batch sample.csv  ← batch CSV prediction
"""

import argparse
import sys
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Load saved artifacts
# ─────────────────────────────────────────────────────────────────────────────
try:
    model         = joblib.load("models/best_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    oe            = joblib.load("models/ordinal_encoder.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
except FileNotFoundError:
    print("[ERROR] Model artifacts not found. Run  python src/train.py  first.")
    sys.exit(1)

BINARY_COLS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea",
]


def preprocess(df_raw: pd.DataFrame) -> np.ndarray:
    """Apply the same transforms used during training."""
    df = df_raw.copy()

    # Normalise yes/no columns
    for col in BINARY_COLS:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})

    # Ordinal encode furnishingstatus
    if df["furnishingstatus"].dtype == object:
        df["furnishingstatus"] = oe.transform(df[["furnishingstatus"]])

    # Derived features (must match train.py)
    df["price_per_sqft_proxy"] = df["area"] / (df["bedrooms"] + 1)
    df["total_rooms"]          = df["bedrooms"] + df["bathrooms"]
    df["amenity_score"]        = (
        df["mainroad"] + df["guestroom"] + df["basement"] +
        df["hotwaterheating"] + df["airconditioning"] + df["prefarea"]
    )

    df = df[feature_names]          # enforce column order
    return scaler.transform(df)


def predict_single() -> None:
    """Interactive single-house prediction via CLI prompts."""
    print("\n" + "=" * 50)
    print("  HOUSE PRICE PREDICTOR  (type 'q' to quit)")
    print("=" * 50)

    def ask(prompt, cast=str, choices=None):
        while True:
            val = input(f"  {prompt}: ").strip()
            if val.lower() == "q":
                print("Exiting.")
                sys.exit(0)
            try:
                val = cast(val)
                if choices and val not in choices:
                    raise ValueError
                return val
            except (ValueError, TypeError):
                hint = f" ({'/'.join(str(c) for c in choices)})" if choices else ""
                print(f"    Invalid input{hint}. Try again.")

    yes_no = ["yes", "no"]

    area            = ask("Area (sq ft)", int)
    bedrooms        = ask("Bedrooms", int)
    bathrooms       = ask("Bathrooms", int)
    stories         = ask("Stories", int)
    mainroad        = ask("Main road access? (yes/no)", str, yes_no)
    guestroom       = ask("Guest room? (yes/no)", str, yes_no)
    basement        = ask("Basement? (yes/no)", str, yes_no)
    hotwaterheating = ask("Hot water heating? (yes/no)", str, yes_no)
    airconditioning = ask("Air conditioning? (yes/no)", str, yes_no)
    parking         = ask("Parking spaces", int)
    prefarea        = ask("Preferred area? (yes/no)", str, yes_no)
    furnishing      = ask(
        "Furnishing (furnished/semi-furnished/unfurnished)",
        str, ["furnished", "semi-furnished", "unfurnished"],
    )

    raw = pd.DataFrame([{
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "stories": stories, "mainroad": mainroad, "guestroom": guestroom,
        "basement": basement, "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning, "parking": parking,
        "prefarea": prefarea, "furnishingstatus": furnishing,
    }])

    X = preprocess(raw)
    predicted_price = model.predict(X)[0]

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Predicted House Price: {predicted_price:>16,.0f}  │")
    print(f"  └─────────────────────────────────────────────┘\n")


def predict_batch(csv_path: str) -> None:
    """Predict prices for all rows in a CSV file."""
    df_raw = pd.read_csv(csv_path)
    print(f"[Batch] Loaded {len(df_raw)} records from '{csv_path}'")

    has_price = "price" in df_raw.columns
    X = preprocess(df_raw.drop(columns=["price"], errors="ignore"))
    preds = model.predict(X)

    df_raw["predicted_price"] = preds.round(0).astype(int)

    if has_price:
        from sklearn.metrics import r2_score, mean_absolute_error
        r2  = r2_score(df_raw["price"], preds)
        mae = mean_absolute_error(df_raw["price"], preds)
        print(f"[Batch] R² = {r2:.4f}  |  MAE = {mae:,.0f}")

    out_path = csv_path.replace(".csv", "_predictions.csv")
    df_raw.to_csv(out_path, index=False)
    print(f"[Batch] Results saved to '{out_path}'")
    print(df_raw[["predicted_price"]].describe().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="House Price Predictor")
    parser.add_argument(
        "--batch", metavar="CSV_PATH",
        help="Path to a CSV file for batch prediction",
    )
    args = parser.parse_args()

    if args.batch:
        predict_batch(args.batch)
    else:
        predict_single()
