import joblib
import os
import pandas as pd

# Absolute paths for Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/churn_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "../models/features.pkl")

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

def predict_churn(X: pd.DataFrame):
    X = X.copy()
    # Align features
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    probs = model.predict_proba(X)[:, 1]
    return probs

def assign_risk(prob):
    if prob > 0.75:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"

def recommend_action(risk):
    if risk == "High":
        return "Offer 1GB bonus / discount"
    elif risk == "Medium":
        return "Send promotional SMS"
    else:
        return "Maintain engagement"
