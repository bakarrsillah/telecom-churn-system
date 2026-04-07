import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------------
# TRAIN MODEL
# -------------------------
def train_model(data_path):
    df = pd.read_csv(data_path)

    # Features & Target
    X = df.drop(columns=['churn'])
    y = df['churn']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("✅ Model Accuracy:", acc)
    print("\n📊 Classification Report:\n", report)

    # Save model
    joblib.dump(model, "models/churn_model.pkl")
    print("✅ Model saved to models/churn_model.pkl")

    return model