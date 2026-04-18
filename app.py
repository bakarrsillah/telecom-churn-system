import streamlit as st
import pandas as pd
import os

from src.predict import predict_churn, assign_risk, recommend_action
from src.train_model import train_pipeline

st.set_page_config(page_title="Telecom Churn System", layout="wide")

st.title("📡 Telecom Churn Prediction System")

uploaded_file = st.file_uploader("Upload CSV")
use_sample = st.button("Use Sample Data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = pd.read_csv("data/raw.csv")
else:
    st.stop()

# Auto-train if pipeline missing
if not os.path.exists("models/churn_pipeline.pkl"):
    st.warning("Training pipeline...")
    train_pipeline(df)
    st.success("Pipeline trained!")

# Predict
probs = predict_churn(df.drop(columns=["churn"], errors="ignore"))

df["churn_probability"] = probs
df["risk"] = df["churn_probability"].apply(assign_risk)
df["action"] = df["risk"].apply(recommend_action)

st.dataframe(df)
st.bar_chart(df["risk"].value_counts())
