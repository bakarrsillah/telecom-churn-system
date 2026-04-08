import joblib
import pandas as pd

# -------------------------
# LOAD MODEL + FEATURES
# -------------------------
model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/features.pkl")


# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_churn(X: pd.DataFrame):

    # Ensure same features as training
    X = X.copy()

    # Add missing columns
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    # Remove extra columns
    X = X[feature_names]

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]

    return probs


# -------------------------
# RISK SCORING
# -------------------------
def assign_risk(prob):
    if prob > 0.75:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"


# -------------------------
# RECOMMENDATION ENGINE
# -------------------------
def recommend_action(risk):
    if risk == "High":
        return "Offer 1GB bonus / discount"
    elif risk == "Medium":
        return "Send promotional SMS"
    else:
        return "Maintain engagement"
