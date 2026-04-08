import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_models(X, y):

    # -------------------------
    # SPLIT DATA
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # DEFINE MODELS
    # -------------------------
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier()
    }

    best_model = None
    best_score = 0

    # -------------------------
    # TRAIN & COMPARE
    # -------------------------
    for name, model in models.items():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)

        print(f"{name} accuracy: {score}")

        if score > best_score:
            best_score = score
            best_model = model

    print(f"\n✅ Best Model Selected (Accuracy: {best_score})")

    # -------------------------
    # SAVE MODEL + FEATURES
    # -------------------------
    joblib.dump(best_model, "models/churn_model.pkl")

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "models/features.pkl")

    print("✅ Model saved to models/churn_model.pkl")
    print("✅ Features saved to models/features.pkl")

    return best_model
