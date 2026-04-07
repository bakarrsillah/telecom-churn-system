import joblib
import pandas as pd

# -------------------------
# LOAD MODEL
# -------------------------
def load_model():
    return joblib.load("models/churn_model.pkl")

# -------------------------
# PREDICT PROBABILITY
# -------------------------
def predict_churn(df):
    model = load_model()
    probs = model.predict_proba(df)[:, 1]
    return probs

# -------------------------
# RISK CLASSIFICATION
# -------------------------
def assign_risk(prob):
    if prob < 0.4:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    else:
        return "High"

# -------------------------
# BUSINESS RECOMMENDATION
# -------------------------
def recommend_action(risk):
    if risk == "High":
        return "Offer discount / retention package"
    elif risk == "Medium":
        return "Send targeted promotion"
    else:
        return "Maintain engagement"