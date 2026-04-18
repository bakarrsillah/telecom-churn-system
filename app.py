# =========================
# IMPORTS (MUST BE FIRST)
# =========================
import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import shap

# Ensure project root is accessible
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

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
st.markdown("End-to-end churn prediction + revenue risk + explainable AI")

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
    st.info("Upload a dataset or click sample data")
    st.stop()

# =========================
# TRAIN MODEL IF NEEDED
# =========================
if not os.path.exists("models/churn_pipeline.pkl"):
    st.warning("Training model...")

    if "churn" not in df.columns:
        st.error("Dataset must contain 'churn' column")
        st.stop()

    train_model.train_pipeline(df)
    st.success("Model trained successfully")

# =========================
# PREDICTION
# =========================
X = df.drop(columns=["churn"], errors="ignore")

probs = predict.predict_churn(X)

df["churn_probability"] = probs
df["risk"] = df["churn_probability"].apply(predict.assign_risk)

# =========================
# BUSINESS LAYER
# =========================
df = business.calculate_revenue(df)
df = business.calculate_priority(df)
df = business.segment_customers(df)

df["action"] = df.apply(business.smart_action, axis=1)

# =========================
# METRICS
# =========================
st.subheader("📊 Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(df))
col2.metric("High Risk", (df["risk"] == "High").sum())
col3.metric("Avg Churn", round(df["churn_probability"].mean(), 2))

# =========================
# BUSINESS IMPACT
# =========================
st.subheader("💰 Business Impact")

col1, col2, col3 = st.columns(3)

col1.metric("Revenue at Risk", f"${df['revenue_at_risk'].sum():,.2f}")
col2.metric("High Risk Customers", (df["risk"] == "High").sum())
col3.metric("Avg Priority Score", round(df["priority_score"].mean(), 2))

# =========================
# TOP CUSTOMERS
# =========================
st.subheader("🎯 Top Customers to Retain")

top = df.sort_values("priority_score", ascending=False).head(10)
st.dataframe(top)

# =========================
# FULL DATA
# =========================
st.subheader("📈 Predictions")
st.dataframe(df)

# =========================
# VISUALS
# =========================
st.subheader("📊 Risk Distribution")
st.bar_chart(df["risk"].value_counts())

st.subheader("💰 Revenue by Segment")
st.bar_chart(df.groupby("customer_segment")["revenue_at_risk"].sum())

# =====================================================
# 🧠 SHAP DASHBOARD (FULLY FIXED)
# =====================================================

st.subheader("🧠 SHAP Explainability Dashboard")

try:
    model = predict.get_model()

    # sample for speed
    X_sample = X.sample(min(100, len(X)), random_state=42)

    # IMPORTANT: transform safely if pipeline exists
    if hasattr(model, "predict"):
        explainer = shap.TreeExplainer(model)
    else:
        st.warning("Model not SHAP compatible")
        st.stop()

    shap_values = explainer.shap_values(X_sample)

    # -------------------------
    # SELECT CUSTOMER
    # -------------------------
    customer_idx = st.selectbox("Select Customer Index", X_sample.index)

    st.write("### 👤 Customer Data")
    st.write(X_sample.loc[customer_idx])

    pos = X_sample.index.get_loc(customer_idx)

    values = shap_values[pos]

    explanation_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Impact": values
    })

    explanation_df = explanation_df.sort_values(
        by="Impact",
        key=abs,
        ascending=False
    )

    st.write("### 📊 Feature Impact")
    st.dataframe(explanation_df)

    # -------------------------
    # BUSINESS INSIGHT
    # -------------------------
    top_feature = explanation_df.iloc[0]

    direction = "increases churn risk" if top_feature["Impact"] > 0 else "reduces churn risk"

    st.info(f"Primary driver: **{top_feature['Feature']}** → {direction}")

    # -------------------------
    # VISUALIZATION
    # -------------------------
    st.write("### 🔍 Top 5 Drivers")

    st.bar_chart(explanation_df.head(5).set_index("Feature"))

except Exception as e:
    st.warning(f"SHAP not available: {e}")

# =========================
# DOWNLOAD
# =========================
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Results",
    csv,
    "churn_results.csv"
)
