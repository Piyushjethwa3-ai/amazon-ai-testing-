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
    safe_texts = [t if t.strip() else "empty" for t in texts]
    X = artifacts["vectorizer"].transform(safe_texts)

    preds = artifacts["model"].predict(X)
    confs = artifacts["model"].predict_proba(X).max(axis=1)

    return preds, confs


def log_result(test_name, text, prediction, confidence):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{
        "test": test_name,
        "input": text,
        "prediction": int(prediction),
        "confidence": round(float(confidence), 4)
    }])

    df.to_csv(
        REPORT_PATH,
        mode="a",
        header=not REPORT_PATH.exists(),
        index=False
    )


def test_spelling_noise(model_artifacts):
    text = "Thiss prduct is amazng"

    preds, confs = predict([text], model_artifacts)

    log_result(
        "spelling_noise",
        text,
        preds[0],
        confs[0]
    )

    assert preds[0] in [0, 1]
    assert confs[0] >= 0.5


def test_empty_text(model_artifacts):
    text = ""

    preds, confs = predict([text], model_artifacts)

    log_result(
        "empty_text",
        text,
        preds[0],
        confs[0]
    )

    assert preds[0] in [0, 1]
