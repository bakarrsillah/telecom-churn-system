import joblib
import os

PIPELINE_PATH = "models/churn_pipeline.pkl"

pipeline = None


def load_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = joblib.load(PIPELINE_PATH)


# -------------------------
# PREDICTION
# -------------------------
def predict_churn(df):
    load_pipeline()
    return pipeline.predict_proba(df)[:, 1]


# -------------------------
# RISK LEVEL
# -------------------------
def assign_risk(prob):
    if prob > 0.75:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"


# -------------------------
# GET MODEL FOR SHAP (IMPORTANT FIX)
# -------------------------
def get_model():
    load_pipeline()

    # IMPORTANT: extract ONLY model (not pipeline)
    return pipeline.named_steps["model"]
