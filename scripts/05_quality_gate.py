from pathlib import Path
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = BASE_DIR / "reports" / "robustness_report.csv"

print(f"🔍 Looking for robustness report at: {REPORT_PATH}")

if not REPORT_PATH.exists():
    print("❌ Robustness report missing")
    sys.exit(1)

df = pd.read_csv(REPORT_PATH)

if df.empty:
    print("❌ Robustness report is empty")
    sys.exit(1)

print("✅ Robustness quality gate passed")
