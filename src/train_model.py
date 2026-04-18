import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.pipeline import build_pipeline


def train_pipeline(df):

    # Split features and target
    y = df["churn"]
    X = df.drop(columns=["churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"✅ Pipeline Accuracy: {acc}")

    # Save FULL pipeline
    joblib.dump(pipeline, "models/churn_pipeline.pkl")

    print("✅ Pipeline saved to models/churn_pipeline.pkl")

    return pipeline
