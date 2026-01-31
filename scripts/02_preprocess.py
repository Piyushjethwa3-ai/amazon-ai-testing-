import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = BASE_DIR / "Data" / "raw" / "reviews_raw.csv"
OUTPUT_CSV = BASE_DIR / "Data" / "processed" / "reviews_processed.csv"

def main():
    df = pd.read_csv(INPUT_CSV)

    print("Columns found:", df.columns.tolist())

    # ---- FIX: map correct column names ----
    if "review" not in df.columns:
        if "review_text" in df.columns:
            df = df.rename(columns={"review_text": "review"})
        elif "clean_review" in df.columns:
            df = df.rename(columns={"clean_review": "review"})

    if "rating" not in df.columns:
        if "label" in df.columns:
            df["rating"] = df["label"].map({1: 5, 0: 1})

    # Safety check
    required = {"review", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["review", "rating"])

    df["label"] = df["rating"].apply(
        lambda x: 1 if x >= 4 else 0 if x <= 2 else None
    )

    df = df.dropna(subset=["label"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Preprocessing complete")
    print("Final columns:", df.columns.tolist())

if __name__ == "__main__":
    main()
