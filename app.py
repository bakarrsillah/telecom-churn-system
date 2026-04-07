import streamlit as st
import pandas as pd

from src.preprocess import clean_data, encode_data, scale_data
from src.features import create_features
from src.predict import predict_churn, assign_risk, recommend_action

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Telecom Churn System", layout="wide")

st.title("📡 Telecom Churn Prediction System")
st.markdown("Predict customer churn and generate retention strategies.")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload Telecom Dataset (CSV)")

risk_filter = st.sidebar.selectbox(
    "Filter by Risk Level",
    ["All", "High", "Medium", "Low"]
)

# -------------------------
# MAIN APP
# -------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # -------------------------
    # PROCESSING PIPELINE
    # -------------------------
    df = clean_data(df)
    df = encode_data(df)
    df = create_features(df)
    df = scale_data(df)

    # Separate features
    X = df.drop(columns=["churn"])

    # -------------------------
    # PREDICTIONS
    # -------------------------
    probs = predict_churn(X)

    df["churn_probability"] = probs
    df["risk_level"] = df["churn_probability"].apply(assign_risk)
    df["action"] = df["risk_level"].apply(recommend_action)

    # -------------------------
    # METRICS (🔥 IMPORTANT)
    # -------------------------
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("High Risk Customers", (df["risk_level"] == "High").sum())
    col3.metric(
        "Avg Churn Probability",
        round(df["churn_probability"].mean(), 2)
    )

    # -------------------------
    # FILTER DATA
    # -------------------------
    if risk_filter != "All":
        df = df[df["risk_level"] == risk_filter]

    # -------------------------
    # PREDICTION TABLE
    # -------------------------
    st.subheader("📈 Predictions")
    st.dataframe(df)

    # -------------------------
    # HIGH RISK SECTION
    # -------------------------
    st.subheader("🚨 High Risk Customers")
    high_risk = df[df["risk_level"] == "High"]
    st.dataframe(high_risk)

    # -------------------------
    # VISUALIZATION
    # -------------------------
    st.subheader("📊 Risk Distribution")
    st.bar_chart(df["risk_level"].value_counts())

    # -------------------------
    # DOWNLOAD RESULTS (🔥 ELITE)
    # -------------------------
    st.subheader("⬇️ Download Results")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("⬅️ Upload a CSV file from the sidebar to begin.")