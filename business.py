import pandas as pd

def calculate_revenue(df):
    df = df.copy()

    if "monthly_charges" in df.columns:
        df["revenue_at_risk"] = df["monthly_charges"] * 3
    else:
        df["revenue_at_risk"] = 0

    return df


def calculate_priority(df):
    df = df.copy()

    if "monthly_charges" in df.columns:
        df["priority_score"] = df["churn_probability"] * df["monthly_charges"]
    else:
        df["priority_score"] = df["churn_probability"]

    return df


def segment_customers(df):
    df = df.copy()

    if "monthly_charges" in df.columns:
        df["customer_segment"] = pd.cut(
            df["monthly_charges"],
            bins=[0, 50, 100, 1000],
            labels=["Low", "Medium", "High"]
        )
    else:
        df["customer_segment"] = "Unknown"

    return df


def smart_action(row):

    if row["risk"] == "High" and row.get("customer_segment") == "High":
        return "Call + Premium Offer"

    elif row["risk"] == "High":
        return "Discount Offer"

    elif row["risk"] == "Medium":
        return "SMS Campaign"

    else:
        return "No Action"
