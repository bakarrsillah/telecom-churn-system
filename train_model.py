import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_pipeline(df):

    y = df["churn"]
    X = df.drop(columns=["churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    joblib.dump(model, "models/churn_pipeline.pkl")

    return model
