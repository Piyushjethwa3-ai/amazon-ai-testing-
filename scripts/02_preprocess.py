import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = BASE_DIR / "Data" / "raw" / "reviews_raw.csv"
OUTPUT_CSV = BASE_DIR / "Data" / "processed" / "reviews_processed.csv"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rating_to_label(rating: int):
    if rating >= 4:
        return 1
    if rating <= 2:
        return 0
    return None


def main():
    df = pd.read_csv(INPUT_CSV)

    df = df.dropna(subset=["review", "rating"])
    df = df.drop_duplicates()

    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = df["rating"].apply(rating_to_label)

    df = df.dropna(subset=["label"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Processed data saved to {OUTPUT_CSV}")
    print("Class distribution:")
    print(df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
