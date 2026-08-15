import pandas as pd
from pathlib import Path


# ============================================================
# CRISIS MMD - MASTER DATASET VERIFICATION
# ============================================================

DATA_FILE = Path("data/processed/crisis_mmd_master.csv")


print("=" * 70)
print("CRISIS MMD - MASTER DATASET VERIFICATION")
print("=" * 70)


# Check whether file exists
if not DATA_FILE.exists():

    print("❌ Master dataset not found!")
    print(f"Expected location: {DATA_FILE}")

else:

    print("✅ Master dataset found!")
    print(f"File: {DATA_FILE}")

    # Load dataset
    df = pd.read_csv(DATA_FILE)

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn names:")
    print("-" * 70)

    for column in df.columns:
        print(column)

    print("\nFirst 5 records:")
    print("-" * 70)

    print(df.head())

    print("\nMissing values:")
    print("-" * 70)

    print(df.isnull().sum())

    print("\nEvent distribution:")
    print("-" * 70)

    print(df["event"].value_counts())

    print("\nText informativeness:")
    print("-" * 70)

    print(df["text_info"].value_counts())

    print("\nHumanitarian categories:")
    print("-" * 70)

    print(df["text_human"].value_counts())

    print("\n" + "=" * 70)
    print("✅ MASTER DATASET VERIFICATION COMPLETED")
    print("=" * 70)