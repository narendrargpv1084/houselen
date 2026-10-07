"""
House Price Prediction — Training Pipeline
==========================================
Dataset  : Dataset/Housing.csv
Target   : price (continuous, regression)
Models   : Linear Regression, Ridge, Random Forest, Gradient Boosting
Output   : models/best_model.pkl  +  models/preprocessor.pkl
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  HOUSE PRICE PREDICTION — TRAINING PIPELINE")
print("=" * 60)

df = pd.read_csv("Dataset/Housing.csv")
print(f"\n[1] Dataset loaded  ->  {df.shape[0]} rows x {df.shape[1]} cols")
print(f"    Columns : {list(df.columns)}")
print(f"    Missing  : {df.isnull().sum().sum()} total NaN values")
print(f"\n    Target (price) stats:")
print(df["price"].describe().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
binary_cols = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea",
]

# Map yes/no → 1/0
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# Ordinal encoding for furnishingstatus
furnishing_order = [["unfurnished", "semi-furnished", "furnished"]]
oe = OrdinalEncoder(categories=furnishing_order)
df["furnishingstatus"] = oe.fit_transform(df[["furnishingstatus"]])

# Derived features
df["price_per_sqft_proxy"] = df["area"] / (df["bedrooms"] + 1)
df["total_rooms"]          = df["bedrooms"] + df["bathrooms"]
df["amenity_score"]        = (
    df["mainroad"] + df["guestroom"] + df["basement"] +
    df["hotwaterheating"] + df["airconditioning"] + df["prefarea"]
)

print(f"\n[2] Feature engineering done  ->  {df.shape[1]} columns after engineering")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXPLORATORY CHARTS (saved to outputs/)
# ─────────────────────────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != "price"]

# 3a. Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14, pad=12)
plt.tight_layout()
fig.savefig("outputs/correlation_heatmap.png", dpi=120)
plt.close(fig)

# 3b. Price distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["price"], bins=30, kde=True, ax=axes[0], color="#3b82d4")
axes[0].set_title("Price Distribution")
axes[0].set_xlabel("Price")
sns.histplot(np.log1p(df["price"]), bins=30, kde=True,
             ax=axes[1], color="#7c5cd8")
axes[1].set_title("Log-Price Distribution")
axes[1].set_xlabel("log(Price + 1)")
plt.tight_layout()
fig.savefig("outputs/price_distribution.png", dpi=120)
plt.close(fig)

# 3c. Price vs area scatter
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["area"], df["price"], alpha=0.5, color="#3b82d4", edgecolors="none")
ax.set_xlabel("Area (sq ft)")
ax.set_ylabel("Price")
ax.set_title("Price vs Area")
plt.tight_layout()
fig.savefig("outputs/price_vs_area.png", dpi=120)
plt.close(fig)

print("[3] EDA charts saved to outputs/")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPARE TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[4] Train / Test split  →  {len(X_train)} train | {len(X_test)} test")

# Scale numerical features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN & EVALUATE MULTIPLE MODELS
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "Linear Regression":    LinearRegression(),
    "Ridge (α=10)":         Ridge(alpha=10),
    "Random Forest":        RandomForestRegressor(
                                n_estimators=200, max_depth=None,
                                min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingRegressor(
                                n_estimators=300, learning_rate=0.05,
                                max_depth=4, subsample=0.8, random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
print(f"\n[5] Model evaluation (5-fold CV on training set)\n")
print(f"    {'Model':<25}  {'CV R²':>8}  {'Test R²':>8}  {'Test MAE':>12}  {'Test RMSE':>12}")
print(f"    {'-'*25}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}")

for name, model in models.items():
    cv_r2   = cross_val_score(model, X_train_sc, y_train,
                               cv=kf, scoring="r2", n_jobs=-1).mean()
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    test_r2 = r2_score(y_test, y_pred)
    mae     = mean_absolute_error(y_test, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {
        "model": model,
        "cv_r2": cv_r2,
        "test_r2": test_r2,
        "mae": mae,
        "rmse": rmse,
        "y_pred": y_pred,
    }
    print(f"    {name:<25}  {cv_r2:>8.4f}  {test_r2:>8.4f}  {mae:>12,.0f}  {rmse:>12,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SELECT BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["test_r2"])
best      = results[best_name]
print(f"\n[6] Best model → {best_name}  (Test R² = {best['test_r2']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 7. DIAGNOSTIC PLOTS FOR BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
y_pred_best = best["y_pred"]
residuals   = y_test.values - y_pred_best

# 7a. Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 6))
lo, hi = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
ax.scatter(y_test, y_pred_best, alpha=0.6, color="#3b82d4", edgecolors="none")
ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title(f"Actual vs Predicted — {best_name}")
ax.legend()
plt.tight_layout()
fig.savefig("outputs/actual_vs_predicted.png", dpi=120)
plt.close(fig)

# 7b. Residual distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred_best, residuals, alpha=0.5, color="#7c5cd8", edgecolors="none")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_xlabel("Predicted Price")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residuals vs Predicted")
sns.histplot(residuals, bins=25, kde=True, ax=axes[1], color="#7c5cd8")
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual")
plt.tight_layout()
fig.savefig("outputs/residuals.png", dpi=120)
plt.close(fig)

# 7c. Feature importance (tree models only)
if hasattr(best["model"], "feature_importances_"):
    importances = best["model"].feature_importances_
    fi = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    fi.plot(kind="barh", ax=ax, color="#3b82d4")
    ax.set_title(f"Feature Importances — {best_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig("outputs/feature_importances.png", dpi=120)
    plt.close(fig)

# 7d. Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
names   = list(results.keys())
r2_vals = [results[n]["test_r2"] for n in names]
colors  = ["#3b82d4" if n == best_name else "#aac4e8" for n in names]
bars    = ax.barh(names, r2_vals, color=colors)
ax.set_xlabel("Test R²")
ax.set_title("Model Comparison — Test R²")
ax.set_xlim(0, 1)
for bar, val in zip(bars, r2_vals):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
fig.savefig("outputs/model_comparison.png", dpi=120)
plt.close(fig)

print("[7] Diagnostic plots saved to outputs/")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(best["model"], "models/best_model.pkl")
joblib.dump(scaler,        "models/scaler.pkl")
joblib.dump(oe,            "models/ordinal_encoder.pkl")

# Save feature column order so predict.py can replicate it exactly
feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.pkl")

print(f"\n[8] Artifacts saved:")
print(f"    models/best_model.pkl       ← {best_name}")
print(f"    models/scaler.pkl")
print(f"    models/ordinal_encoder.pkl")
print(f"    models/feature_names.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"  Best model    : {best_name}")
print(f"  CV R²         : {best['cv_r2']:.4f}")
print(f"  Test R²       : {best['test_r2']:.4f}")
print(f"  Test MAE      : {best['mae']:,.0f}")
print(f"  Test RMSE     : {best['rmse']:,.0f}")
print("=" * 60)
print("\nTraining complete.\n")
