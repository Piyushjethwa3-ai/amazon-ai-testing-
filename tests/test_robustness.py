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
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "test_case": test_name,
        "input_text": text,
        "prediction": int(prediction)
    }])
    df.to_csv(REPORT_PATH, mode="a", header=not REPORT_PATH.exists(), index=False)


# ------------------ ROBUSTNESS TESTS ------------------ #

def test_spelling_noise(model_artifacts):
    text = "Thiss prduct is amazng"
    pred = predict([text], model_artifacts)[0]
    log_result("spelling_noise", text, pred)
    assert pred in [0, 1]


def test_casing_variation(model_artifacts):
    text = "THIS PRODUCT IS AMAZING"
    pred = predict([text], model_artifacts)[0]
    log_result("casing_variation", text, pred)
    assert pred in [0, 1]


def test_long_text(model_artifacts):
    text = "good " * 1000
    pred = predict([text], model_artifacts)[0]
    log_result("long_text", text[:50] + "...", pred)
    assert pred in [0, 1]


def test_empty_text(model_artifacts):
    text = ""
    pred = predict([text], model_artifacts)[0]
    log_result("empty_text", text, pred)
    assert pred in [0, 1]


def test_injection_like_text(model_artifacts):
    text = "' OR 1=1; DROP TABLE users; --"
    pred = predict([text], model_artifacts)[0]
    log_result("injection_pattern", text, pred)
    assert pred in [0, 1]
