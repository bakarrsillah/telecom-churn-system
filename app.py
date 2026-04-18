import streamlit as st
import pandas as pd
import os

from src.predict import predict_churn, assign_risk
from src.train_model import train_pipeline
from src.business import (
    calculate_revenue,
    calculate_priority,
    segment_customers,
    smart_action
)

st.set_page_config(page_title="Telecom Churn System", layout="wide")

st.title("📡 Telecom Churn Intelligence System")

st.markdown("""
### 💡 Identify churn, quantify revenue risk, and optimize retention strategy
""")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV")
use_sample = st.sidebar.button("Use Sample Data")

risk_filter = st.sidebar.selectbox(
    "Filter by Risk",
    ["All", "High", "Medium", "Low"]
)

# -------------------------
# LOAD DATA
# -------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

elif use_sample:
    df = pd.read_csv("data/raw.csv")
    st.success("Using sample dataset")

else:
    st.info("Upload a dataset to begin")
    st.stop()

# -------------------------
# AUTO TRAIN PIPELINE
# -------------------------
if not os.path.exists("models/churn_pipeline.pkl"):

    st.warning("⚠️ Model not found. Training pipeline...")

    if "churn" not in df.columns:
        st.error("Dataset must contain 'churn' column to train model")
        st.stop()

    train_pipeline(df)

    st.success("✅ Pipeline trained successfully")

# -------------------------
# PREDICTION
# -------------------------
X = df.drop(columns=["churn"], errors="ignore")
probs = predict_churn(X)

df["churn_probability"] = probs
df["risk"] = df["churn_probability"].apply(assign_risk)

# -------------------------
# BUSINESS INTELLIGENCE
# -------------------------
df = calculate_revenue(df)
df = calculate_priority(df)
df = segment_customers(df)

df["action"] = df.apply(smart_action, axis=1)

# -------------------------
# FILTER
# -------------------------
if risk_filter != "All":
    df = df[df["risk"] == risk_filter]

# -------------------------
# METRICS
# -------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(df))

col2.metric(
    "High Risk",
    (df["risk"] == "High").sum()
)

col3.metric(
    "Avg Churn Probability",
    round(df["churn_probability"].mean(), 2)
)

# -------------------------
# BUSINESS METRICS
# -------------------------
st.subheader("💰 Business Impact")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Revenue at Risk",
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
st.subheader("🎯 Top Customers to Save")

top_customers = df.sort_values("priority_score", ascending=False).head(10)
st.dataframe(top_customers)

# -------------------------
# FULL TABLE
# -------------------------
st.subheader("📈 Predictions")
st.dataframe(df)

# -------------------------
# VISUALS
# -------------------------
st.subheader("📊 Risk Distribution")
st.bar_chart(df["risk"].value_counts())

st.subheader("💰 Revenue at Risk by Segment")
st.bar_chart(df.groupby("customer_segment")["revenue_at_risk"].sum())

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("💡 Insights")

high_pct = (df["risk"] == "High").mean() * 100
st.write(f"{round(high_pct,2)}% of customers are high risk")

# -------------------------
# DOWNLOAD
# -------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Results",
    data=csv,
    file_name="churn_results.csv"
)
