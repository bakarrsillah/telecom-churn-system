import streamlit as st
import pandas as pd
import os

from src.preprocess import clean_data, encode_data, scale_data
from src.features import create_features
from src.predict import predict_churn, assign_risk, recommend_action

st.set_page_config(page_title="Telecom Churn System", layout="wide")
st.title("📡 Telecom Churn Prediction System")
st.markdown("### 💡 Identify customers likely to churn and recommend retention actions")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV")
use_sample = st.sidebar.button("Use Sample Data")
risk_filter = st.sidebar.selectbox("Filter by Risk", ["All", "High", "Medium", "Low"])

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("data/raw.csv")
    st.success("Using sample dataset")
else:
    st.info("Upload a dataset to begin")
    st.stop()

# Validate data
if "tenure" in df.columns and (df["tenure"] < 0).any():
    st.error("Tenure cannot be negative")
    st.stop()
if "monthly_charges" in df.columns and (df["monthly_charges"] < 0).any():
    st.error("Monthly charges cannot be negative")
    st.stop()

# Processing
df = clean_data(df)
df = encode_data(df)
df = create_features(df)
df = scale_data(df)
X = df.drop(columns=["churn"], errors="ignore")
probs = predict_churn(X)

# Predictions
df["churn_probability"] = probs
df["risk_level"] = df["churn_probability"].apply(assign_risk)
df["action"] = df["risk_level"].apply(recommend_action)

# Filter
if risk_filter != "All":
    df = df[df["risk_level"] == risk_filter]

# Dashboard
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Customers", len(df))
col2.metric("High Risk", (df["risk_level"] == "High").sum())
col3.metric("Avg Risk", round(df["churn_probability"].mean(), 2))

st.subheader("📈 Predictions")
st.dataframe(df)

st.subheader("🚨 High Risk Customers")
st.dataframe(df[df["risk_level"] == "High"])

st.subheader("📊 Risk Distribution")
st.bar_chart(df["risk_level"].value_counts())

st.subheader("💡 Insights")
high_pct = (df["risk_level"] == "High").mean() * 100
st.write(f"{round(high_pct,2)}% customers are high risk")

# Download CSV
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download Predictions", data=csv, file_name="predictions.csv")
