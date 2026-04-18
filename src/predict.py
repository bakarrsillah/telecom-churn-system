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
# PREDICT
# -------------------------
def predict_churn(df):
    load_pipeline()
    return pipeline.predict_proba(df)[:, 1]


# -------------------------
# GET MODEL FOR SHAP
# -------------------------
def get_model():
    load_pipeline()
    return pipeline
