import streamlit as st
import pandas as pd
import os

import predict
import business
import train_model

st.set_page_config(page_title="Telecom Churn Intelligence System", layout="wide")

st.title("📡 Telecom Churn Intelligence System")
st.markdown("Predict churn + revenue risk + retention strategy")

# -------------------------
# LOAD DATA
# -------------------------
uploaded_file = st.file_uploader("Upload CSV")
use_sample = st.button("Use Sample Data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

elif use_sample:
    df = pd.read_csv("data/raw.csv")

else:
    st.stop()

# -------------------------
# TRAIN MODEL IF MISSING
# -------------------------
if not os.path.exists("models/churn_pipeline.pkl"):
    st.warning("Training model...")

    if "churn" not in df.columns:
        st.error("Dataset must contain 'churn' column")
        st.stop()

    train_model.train_pipeline(df)
    st.success("Model trained")

# -------------------------
# PREDICTION
# -------------------------
X = df.drop(columns=["churn"], errors="ignore")

probs = predict.predict_churn(X)

df["churn_probability"] = probs
df["risk"] = df["churn_probability"].apply(predict.assign_risk)

# -------------------------
# BUSINESS LOGIC
# -------------------------
df = business.calculate_revenue(df)
df = business.calculate_priority(df)
df = business.segment_customers(df)

df["action"] = df.apply(business.smart_action, axis=1)

# -------------------------
# METRICS
# -------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(df))
col2.metric("High Risk", (df["risk"] == "High").sum())
col3.metric("Avg Churn", round(df["churn_probability"].mean(), 2))

# -------------------------
# BUSINESS IMPACT
# -------------------------
st.subheader("💰 Business Impact")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Revenue at Risk",
    f"${df['revenue_at_risk'].sum():,.2f}"
)

col2.metric(
    "High Risk Customers",
    (df["risk"] == "High").sum()
)

col3.metric(
    "Avg Priority Score",
    round(df["priority_score"].mean(), 2)
)

# -------------------------
# TOP CUSTOMERS
# -------------------------
st.subheader("🎯 Top Customers to Retain")

top = df.sort_values("priority_score", ascending=False).head(10)
st.dataframe(top)

# -------------------------
# FULL TABLE
# -------------------------
st.subheader("📈 Full Predictions")
st.dataframe(df)

# -------------------------
# VISUALS
# -------------------------
st.subheader("📊 Risk Distribution")
st.bar_chart(df["risk"].value_counts())

st.subheader("💰 Revenue by Segment")
st.bar_chart(df.groupby("customer_segment")["revenue_at_risk"].sum())

# -------------------------
# DOWNLOAD
# -------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Results",
    csv,
    "churn_results.csv"
)
