# HouseLens 🏠

> AI-powered house price prediction web application — trained on 545 real
> property transactions, deployed as a Flask + Gunicorn service on Render.

---

## Live Demo

Once deployed, your app will be available at:
```
https://houselen.onrender.com
```


---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn · Linear Regression |
| Backend | Python 3.11 · Flask 3 · Gunicorn |
| Frontend | Vanilla HTML/CSS/JS (no frameworks) |
| Deployment | Render (Web Service) |

---

## Project Structure

```
houselens/
├── app.py                  ← Flask application (root entry point)
├── wsgi.py                 ← Gunicorn WSGI entry point
├── requirements.txt        ← Pinned Python dependencies
├── render.yaml             ← Render Blueprint (one-click deploy)
│
├── models/                 ← Trained model artifacts (committed)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── ordinal_encoder.pkl
│   └── feature_names.pkl
│
├── web/
│   ├── templates/
│   │   └── index.html      ← Single-page frontend
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
├── src/
│   ├── train.py            ← Full training pipeline
│   └── predict.py          ← CLI inference script
│
└── Dataset/
    └── Housing.csv         ← Source dataset
```

---

## Deploy to Render (3 Steps)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "feat: initial HouseLens deployment"
git remote add origin https://github.com/<your-username>/houselens.git
git push -u origin main
```

### Step 2 — Create a Render Web Service

1. Go to **[render.com](https://render.com)** → **New** → **Web Service**
2. Connect your GitHub repository
3. Render auto-detects `render.yaml` — click **Apply**

Or configure manually:

| Setting | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |
| **Health Check Path** | `/api/health` |

### Step 3 — Set Environment Variable (optional)

In Render dashboard → **Environment**:

```
PYTHON_VERSION = 3.11.0
```

Render will build, install dependencies, and start the service automatically.

---

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (only needed once)
python src/train.py

# 3a. Dev server (Flask auto-reload)
python app.py

# 3b. Production-equivalent (Gunicorn)
gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 2

# Open browser
open http://localhost:5000
```

---

## API Reference

### `POST /api/predict`

**Request body (JSON):**

```json
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "parking": 2,
  "prefarea": "yes",
  "furnishingstatus": "furnished"
}
```

**Response (JSON):**

```json
{
  "predicted_price": 8054989,
  "range_low": 6727308,
  "range_high": 9382670,
  "model": "LinearRegression"
}
```

### `GET /api/health`

```json
{ "status": "ok", "model": "LinearRegression" }
```

---

## Model Performance

| Model | CV R² | Test R² | MAE | RMSE |
|---|---|---|---|---|
| **Linear Regression** ✅ | 0.6495 | **0.6513** | 9,76,736 | 13,27,681 |
| Ridge (α=10) | 0.6490 | 0.6489 | 9,77,425 | 13,32,257 |
| Gradient Boosting | 0.6062 | 0.6362 | 9,77,803 | 13,56,055 |
| Random Forest | 0.6222 | 0.6198 | 10,08,891 | 13,86,347 |

---

## Retrain the Model

```bash
python src/train.py
```

Artifacts are saved to `models/` and automatically picked up by the Flask app
on next restart. Commit the updated `.pkl` files before redeploying.

---

## License

MIT
