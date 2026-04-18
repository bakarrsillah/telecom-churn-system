# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import shap

# project root fix
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import predict
import business
import train_model

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Telecom Churn Intelligence System",
    layout="wide"
)

st.title("📡 Telecom Churn Intelligence System")
st.markdown("Churn prediction + revenue risk + explainability (SHAP)")

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload CSV")
use_sample = st.button("Use Sample Data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

elif use_sample:
    df = pd.read_csv("data/raw.csv")

else:
    st.stop()

# =========================
# TRAIN MODEL IF NEEDED
# =========================
if not os.path.exists("models/churn_pipeline.pkl"):
    st.warning("Training model...")

    if "churn" not in df.columns:
        st.error("Dataset must contain 'churn'")
        st.stop()

    train_model.train_pipeline(df)

    st.success("Model trained")

# =========================
# PREDICTIONS
# =========================
X = df.drop(columns=["churn"], errors="ignore")

probs = predict.predict_churn(X)

df["churn_probability"] = probs
df["risk"] = df["churn_probability"].apply(predict.assign_risk)

# =========================
# BUSINESS LOGIC
# =========================
df = business.calculate_revenue(df)
df = business.calculate_priority(df)
df = business.segment_customers(df)

df["action"] = df.apply(business.smart_action, axis=1)

# =========================
# DASHBOARD METRICS
# =========================
st.subheader("📊 Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Customers", len(df))
c2.metric("High Risk", (df["risk"] == "High").sum())
c3.metric("Avg Churn", round(df["churn_probability"].mean(), 3))

# =========================
# BUSINESS IMPACT
# =========================
st.subheader("💰 Business Impact")

c1, c2, c3 = st.columns(3)

c1.metric("Revenue at Risk", f"${df['revenue_at_risk'].sum():,.2f}")
c2.metric("High Risk Customers", (df["risk"] == "High").sum())
c3.metric("Avg Priority Score", round(df["priority_score"].mean(), 2))

# =========================
# TOP CUSTOMERS
# =========================
st.subheader("🎯 Top Customers to Retain")

st.dataframe(df.sort_values("priority_score", ascending=False).head(10))

# =========================
# FULL DATA
# =========================
st.subheader("📈 Full Predictions")
st.dataframe(df)

# =========================
# VISUALS
# =========================
st.subheader("📊 Risk Distribution")
st.bar_chart(df["risk"].value_counts())

st.subheader("💰 Revenue by Segment")
st.bar_chart(df.groupby("customer_segment")["revenue_at_risk"].sum())

# =====================================================
# 🧠 SHAP (FULLY FIXED, NO SHAPE ERRORS)
# =====================================================

st.subheader("🧠 SHAP Explainability Dashboard")

try:
    pipeline = predict.get_pipeline()

    # sample raw input
    X_sample_raw = X.sample(min(100, len(X)), random_state=42)

    # transform safely
    X_transformed = pipeline[:-1].transform(X_sample_raw)

    model = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_transformed)

    # -------------------------
    # FIX SHAP OUTPUT TYPE
    # -------------------------
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary classification fix

    # -------------------------
    # SELECT CUSTOMER
    # -------------------------
    idx = st.selectbox("Select Customer Index", X_sample_raw.index)

    st.write("### 👤 Customer Data")
    st.write(X_sample_raw.loc[idx])

    pos = X_sample_raw.index.get_loc(idx)

    row = np.array(shap_values[pos]).reshape(-1)

    # -------------------------
    # FEATURE NAMES SAFE
    # -------------------------
    try:
        feature_names = pipeline[:-1].get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(len(row))]

    feature_names = feature_names[:len(row)]

    # -------------------------
    # BUILD EXPLANATION TABLE
    # -------------------------
    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": row
    })

    explanation_df = explanation_df.sort_values(
        by="Impact",
        key=abs,
        ascending=False
    )

    st.write("### 📊 Feature Impact Ranking")
    st.dataframe(explanation_df)

    # -------------------------
    # BUSINESS INSIGHT
    # -------------------------
    top = explanation_df.iloc[0]

    direction = "increases churn risk" if top["Impact"] > 0 else "reduces churn risk"

    st.info(f"Primary driver: **{top['Feature']}** → {direction}")

    # -------------------------
    # VISUALIZATION
    # -------------------------
    st.write("### 🔍 Top 5 Drivers")

    st.bar_chart(explanation_df.head(5).set_index("Feature"))

except Exception as e:
    st.warning(f"SHAP not available: {e}")

# =========================
# DOWNLOAD OUTPUT
# =========================
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Results",
    csv,
    "churn_results.csv"
)
