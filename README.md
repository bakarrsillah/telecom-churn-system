# 📡 Telecom Churn Prediction System

An end-to-end machine learning application that predicts customer churn in telecom datasets and provides actionable business insights through an interactive dashboard.

---

## 🚀 Live Demo
👉 https://telecom-churn-system-agzjb93krgke6zbevrfvuy.streamlit.app/

---

## 💡 Project Overview

Customer churn is a major challenge in the telecom industry. This system helps businesses:

- Identify customers likely to churn
- Understand risk levels
- Take proactive retention actions

The application combines **data preprocessing, feature engineering, machine learning, and a web dashboard** into a single deployable solution.

---

## 🧠 Key Features

### 🔍 Data Processing
- Cleans raw telecom data
- Encodes categorical variables
- Scales numerical features

### ⚙️ Feature Engineering
- Generates meaningful predictors for churn behavior

### 🤖 Machine Learning Model
- Trained classification model
- Predicts churn probability
- Achieves high accuracy (~98%)

### 📊 Interactive Dashboard
- Upload your own dataset
- View predictions in real-time
- Filter customers by risk level
- Visualize churn distribution

### 🚨 Risk Classification
- **High Risk** → Immediate retention action
- **Medium Risk** → Monitor closely
- **Low Risk** → Maintain engagement

### 📥 Export Results
- Download predictions as CSV

---

## 🖥️ App Preview

### 📊 Dashboard Sections
- Raw Data Preview
- Predictions Table
- High-Risk Customers
- Risk Distribution Chart

---

## 🏗️ Project Structure
churn_system/
│
├── data/
│ ├── raw.csv
│ └── processed.csv
│
├── models/
│ └── churn_model.pkl
│
├── src/
│ ├── preprocess.py
│ ├── features.py
│ ├── predict.py
│ └── train_model.py
│
├── app.py
├── run_pipeline.py
├── run_predict.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/telecom-churn-system.git
cd telecom-churn-system

python -m venv venv
venv\Scripts\activate   # Windows
