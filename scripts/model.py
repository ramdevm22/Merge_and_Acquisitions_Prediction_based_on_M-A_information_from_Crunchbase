
import os, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def do_model_building():
    train = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))

    X_train = train.drop("y", axis=1)
    y_train = train["y"]
    X_test = test.drop("y", axis=1)
    y_test = test["y"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))

    joblib.dump(model, os.path.join(MODEL_DIR, "acquisition_model.joblib"))
    print("Model saved.")
