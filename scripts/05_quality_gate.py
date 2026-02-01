import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = BASE_DIR / "reports" / "robustness_report.csv"

if not REPORT_PATH.exists():
    print("❌ Robustness report missing")
    sys.exit(1)

df = pd.read_csv(REPORT_PATH)

if len(df) < 5:
    print("❌ Not enough robustness tests executed")
    sys.exit(1)

print("✅ Quality gate passed")
