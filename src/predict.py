import joblib
import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/churn_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "../models/features.pkl")

model = None
feature_names = None

# -------------------------
# LOAD OR FALLBACK
# -------------------------
def load_model():
    global model, feature_names

    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        return True
    return False


# -------------------------
# PREDICTION
# -------------------------
def predict_churn(X: pd.DataFrame):

    global model, feature_names

    # Try loading model
    if model is None:
        if not load_model():
            raise ValueError("Model not found. Please retrain.")

    X = X.copy()

    # Align features
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]

    return model.predict_proba(X)[:, 1]


# -------------------------
# RISK
# -------------------------
def assign_risk(prob):
    if prob > 0.75:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"


# -------------------------
# ACTION
# -------------------------
def recommend_action(risk):
    if risk == "High":
        return "Offer 1GB bonus / discount"
    elif risk == "Medium":
        return "Send promotional SMS"
    else:
        return "Maintain engagement"
