import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)

CSV_PATH = os.path.join(os.path.dirname(__file__), "USA_Housing.csv")

# Short keys used by the API <-> the dataset's actual column names.
# If you rename columns in the CSV, update this map — nothing else needs to change.
FEATURE_MAP = {
    "income": "Avg. Area Income",
    "age": "Avg. Area House Age",
    "rooms": "Avg. Area Number of Rooms",
    "bedrooms": "Avg. Area Number of Bedrooms",
    "population": "Area Population",
}
FEATURE_ORDER = list(FEATURE_MAP.keys())
TARGET_COL = "Price"


def train_model():
    """Loads USA_Housing.csv fresh and retrains the model.
    Runs once at startup — restart the server after changing the CSV
    and everything downstream (coefficients, metrics, the website) updates automatically."""
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=[c for c in ["Address"] if c in df.columns])

    cols = [FEATURE_MAP[k] for k in FEATURE_ORDER]
    X = df[cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)

    metrics = {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    feature_info = {}
    for i, key in enumerate(FEATURE_ORDER):
        feature_info[key] = {
            "label": cols[i],
            "coefficient": round(float(lr.coef_[i]), 4),
            "mean": round(float(scaler.mean_[i]), 6),
            "std": round(float(scaler.scale_[i]), 6),
            "min": round(float(X[cols[i]].min()), 4),
            "max": round(float(X[cols[i]].max()), 4),
        }

    return {
        "model": lr,
        "scaler": scaler,
        "metrics": metrics,
        "features": feature_info,
        "intercept": round(float(lr.intercept_), 2),
        "trained_rows": len(df),
    }


# Train once when the server starts.
STATE = train_model()


@app.get("/")
def health():
    return jsonify({"status": "ok", "trained_rows": STATE["trained_rows"]})


@app.get("/model-info")
def model_info():
    """Everything the frontend needs to render itself: coefficients, feature
    ranges, and performance metrics — all derived live from the CSV."""
    return jsonify({
        "intercept": STATE["intercept"],
        "features": STATE["features"],
        "metrics": STATE["metrics"],
        "feature_order": FEATURE_ORDER,
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Send a JSON body with income, age, rooms, bedrooms, population."}), 400

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        row = np.array([[float(data[f]) for f in FEATURE_ORDER]])
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric."}), 400

    scaled = STATE["scaler"].transform(row)
    prediction = float(STATE["model"].predict(scaled)[0])

    return jsonify({
        "predicted_price": round(prediction, 2),
        "inputs": data,
    })


@app.post("/retrain")
def retrain():
    """Call this after replacing USA_Housing.csv on the server to retrain
    without a full redeploy/restart."""
    global STATE
    STATE = train_model()
    return jsonify({"status": "retrained", "trained_rows": STATE["trained_rows"], "metrics": STATE["metrics"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
