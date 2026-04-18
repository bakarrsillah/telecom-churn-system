import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# CUSTOM FEATURE ENGINEERING
# -------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Fill missing values
        df = df.ffill()

        # Feature: tenure group
        if "tenure" in df.columns:
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 60],
                labels=[0, 1, 2]
            )

        # Feature: avg monthly usage
        if "total_charges" in df.columns and "tenure" in df.columns:
            df["avg_monthly_usage"] = df["total_charges"] / (df["tenure"] + 1)

        # Feature: engagement score
        if "contract" in df.columns:
            df["engagement_score"] = df["contract"].map({
                "Month-to-month": 1,
                "One year": 2,
                "Two year": 3
            }).fillna(1)

        # Ensure required columns exist
        for col in ["complaint_ratio", "revenue_risk", "usage_intensity"]:
            if col not in df.columns:
                df[col] = 0

        # Convert all object columns → numeric
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category").cat.codes

        return df


# -------------------------
# PIPELINE BUILDER
# -------------------------
def build_pipeline():

    pipeline = Pipeline([
        ("features", FeatureEngineer()),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier())
    ])

    return pipeline
