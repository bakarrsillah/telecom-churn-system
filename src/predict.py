import joblib
import os

# -------------------------
# LOAD PIPELINE
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "../models/churn_pipeline.pkl")

pipeline = None


def load_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = joblib.load(PIPELINE_PATH)


# -------------------------
# PREDICT CHURN (ONLY PROBABILITY)
# -------------------------
def predict_churn(df):

    load_pipeline()

    probs = pipeline.predict_proba(df)[:, 1]

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
