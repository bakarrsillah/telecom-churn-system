import pandas as pd

def create_features(df):
    df = df.copy()

    # Tenure grouping
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 60],
            labels=["0-12", "12-24", "24+"]
        )

    # Avg monthly usage
    if "total_charges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_usage"] = df["total_charges"] / (df["tenure"] + 1)

    # Engagement score
    if "contract" in df.columns:
        df["engagement_score"] = df["contract"].map({
            "Month-to-month": 1,
            "One year": 2,
            "Two year": 3
        }).fillna(1)

    return df
