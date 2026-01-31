import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "Data" / "raw" / "reviews_raw.csv"
PROCESSED_PATH = BASE_DIR / "Data" / "processed" / "reviews_processed.csv"

def rating_to_label(rating):
    if rating >= 4:
        return 1
    if rating <= 2:
        return 0
    return None

def main():
    df = pd.read_csv(RAW_PATH)

    # Detect correct text column
    if "review" in df.columns:
        text_col = "review"
    elif "clean_review" in df.columns:
        text_col = "clean_review"
    else:
        raise ValueError("No valid review text column found")

    # Drop missing values
    df = df.dropna(subset=[text_col, "rating"])

    # Remove duplicates
    df = df.drop_duplicates(subset=[text_col])

    # Create label
    df["label"] = df["rating"].apply(rating_to_label)

    # Drop neutral reviews
    df = df[df["label"].notnull()]

    # Rename for downstream consistency
    df = df.rename(columns={text_col: "clean_review"})

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Preprocessing completed")
    print("Rows:", len(df))

if __name__ == "__main__":
    main()
