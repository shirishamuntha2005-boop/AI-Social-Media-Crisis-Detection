import pandas as pd
from pathlib import Path

DATA_FILE = Path(
    "data/raw/CrisisMMD/annotations/california_wildfires_final_data.tsv"
)

print("=" * 70)
print("CRISIS MMD - ANNOTATION FILE INSPECTION")
print("=" * 70)

if not DATA_FILE.exists():
    print("❌ Annotation file not found!")
else:
    print("✅ Annotation file found!")
    print(f"File: {DATA_FILE}")

    # Load TSV file
    df = pd.read_csv(DATA_FILE, sep="\t")

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn names:")
    print("-" * 70)

    for column in df.columns:
        print(column)

    print("\nFirst 5 rows:")
    print("-" * 70)

    print(df.head())

    print("\nData types:")
    print("-" * 70)

    print(df.dtypes)

    print("\nMissing values:")
    print("-" * 70)

    print(df.isnull().sum())

    print("\n" + "=" * 70)
    print("✅ Annotation inspection completed")
    print("=" * 70)