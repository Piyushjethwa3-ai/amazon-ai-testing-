import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sentiment_logreg_model.pkl"
INPUT_CSV = BASE_DIR / "Data" / "processed" / "reviews_processed.csv"
OUTPUT_CSV = BASE_DIR / "Data" / "predictions" / "predictions.csv"

def main():
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)

    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]

    df = pd.read_csv(INPUT_CSV)
    X = vectorizer.transform(df["clean_review"])
    df["prediction"] = model.predict(X)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

