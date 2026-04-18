import joblib
import os

MODEL_PATH = "models/churn_pipeline.pkl"

pipeline = None


# -------------------------
# LOAD MODEL
# -------------------------
def load_model():
    global pipeline
    if pipeline is None:
        pipeline = joblib.load(MODEL_PATH)
    return pipeline


# -------------------------
# PREDICTION
# -------------------------
def predict_churn(X):
    model = load_model()
    return model.predict_proba(X)[:, 1]


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
# SHAP SUPPORT (IMPORTANT)
# -------------------------
def get_pipeline():
    """
    Used ONLY for SHAP + feature extraction
    """
    return load_model()
