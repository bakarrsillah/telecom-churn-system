import streamlit as st
import pandas as pd

from src.preprocess import clean_data, encode_data, scale_data
from src.features import create_features
from src.predict import predict_churn, assign_risk, recommend_action

st.set_page_config(page_title="Telecom Churn System", layout="wide")

st.title("📡 Telecom Churn Prediction System")

st.markdown("""
### 💡 Business Objective
Identify customers likely to churn and recommend retention actions.
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
# VALIDATION
# -------------------------
def validate_data(df):
    errors = []

    if "tenure" in df.columns:
        if (df["tenure"] < 0).any():
            errors.append("Tenure cannot be negative")

    return errors


errors = validate_data(df)

if errors:
    for e in errors:
        st.error(e)
    st.stop()

# -------------------------
# PREPROCESSING
# -------------------------
try:
    df = clean_data(df)
    df = encode_data(df)
    df = create_features(df)
    df = scale_data(df)

    X = df.drop(columns=["churn"])

    probs = predict_churn(X)

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# -------------------------
# OUTPUT
# -------------------------
df["churn_probability"] = probs
df["risk_level"] = df["churn_probability"].apply(assign_risk)
df["action"] = df["risk_level"].apply(recommend_action)

# -------------------------
# METRICS
# -------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(df))
col2.metric("High Risk", (df["risk_level"] == "High").sum())
col3.metric("Avg Risk", round(df["churn_probability"].mean(), 2))

# -------------------------
# FILTER
# -------------------------
if risk_filter != "All":
    df = df[df["risk_level"] == risk_filter]

# -------------------------
# TABLES
# -------------------------
st.subheader("📈 Predictions")
st.dataframe(df)

st.subheader("🚨 High Risk Customers")
st.dataframe(df[df["risk_level"] == "High"])

# -------------------------
# VISUALS
# -------------------------
st.subheader("📊 Risk Distribution")
st.bar_chart(df["risk_level"].value_counts())

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("💡 Insights")

high_pct = (df["risk_level"] == "High").mean() * 100
st.write(f"{round(high_pct,2)}% customers are high risk")

# -------------------------
# DOWNLOAD
# -------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Predictions",
    data=csv,
    file_name="predictions.csv"
)
