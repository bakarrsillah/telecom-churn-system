import pandas as pd

def create_features(df):
    # -------------------------
    # USAGE INTENSITY
    # -------------------------
    df['usage_intensity'] = df['calls_per_day'] * df['avg_call_duration']

    # -------------------------
    # CUSTOMER ENGAGEMENT SCORE
    # -------------------------
    df['engagement_score'] = (
        df['calls_per_day'] + df['data_usage_mb_per_day']
    ) / (df['days_since_last_activity'] + 1)

    # -------------------------
    # REVENUE RISK
    # -------------------------
    df['revenue_risk'] = df['monthly_charge'] / (df['tenure_months'] + 1)

    # -------------------------
    # COMPLAINT RATIO
    # -------------------------
    df['complaint_ratio'] = df['num_complaints'] / (df['tenure_months'] + 1)

    return df