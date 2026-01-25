import pickle
import pandas as pd
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_logreg_model.pkl"
REPORT_PATH = BASE_DIR / "reports" / "robustness_report.csv"

@pytest.fixture(scope="session")
def model_artifacts():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict(texts, artifacts):
    X = artifacts["vectorizer"].transform(texts)
    return artifacts["model"].predict(X)

def log_result(test_name, text, prediction):
    REPORT_PATH.parent.mkdir(exist_ok=True)
    df = pd.DataFrame([{
        "test": test_name,
        "input": text,
        "prediction": int(prediction)
    }])
    df.to_csv(REPORT_PATH, mode="a", header=not REPORT_PATH.exists(), index=False)

def test_spelling_noise(model_artifacts):
    text = "Thiss prduct is amazng"
    pred = predict([text], model_artifacts)[0]
    log_result("spelling_noise", text, pred)
    assert pred in [0, 1]

def test_empty_text(model_artifacts):
    text = ""
    pred = predict([text], model_artifacts)[0]
    log_result("empty_text", text, pred)
    assert pred in [0, 1]

