import joblib

model = joblib.load("models/churn_model.pkl")


def predict_churn(X):
    return model.predict_proba(X)[:, 1]


def assign_risk(prob):
    if prob > 0.75:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"


def recommend_action(risk):
    if risk == "High":
        return "Offer 1GB bonus / discount"
    elif risk == "Medium":
        return "Send promotional SMS"
    else:
        return "Maintain engagement"
