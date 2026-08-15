import pandas as pd
from pathlib import Path


# ============================================================
# CRISIS MMD - COMBINE ALL NLP ANNOTATIONS
# ============================================================

DATA_PATH = Path("data/raw/CrisisMMD/annotations")
OUTPUT_PATH = Path("data/processed")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# All 7 CrisisMMD annotation files
files = {
    "California Wildfires": "california_wildfires_final_data.tsv",
    "Hurricane Harvey": "hurricane_harvey_final_data.tsv",
    "Hurricane Irma": "hurricane_irma_final_data.tsv",
    "Hurricane Maria": "hurricane_maria_final_data.tsv",
    "Iraq-Iran Earthquake": "iraq_iran_earthquake_final_data.tsv",
    "Mexico Earthquake": "mexico_earthquake_final_data.tsv",
    "Sri Lanka Floods": "srilanka_floods_final_data.tsv"
}


print("=" * 70)
print("CRISIS MMD - COMBINING NLP DATASETS")
print("=" * 70)


all_data = []


for event, filename in files.items():

    file_path = DATA_PATH / filename

    print(f"\nReading: {filename}")

    if not file_path.exists():
        print("❌ File not found!")
        continue

    df = pd.read_csv(file_path, sep="\t")

    # Keep only NLP-related columns
    df = df[
        [
            "tweet_id",
            "tweet_text",
            "text_info",
            "text_info_conf",
            "text_human",
            "text_human_conf"
        ]
    ].copy()

    # Add disaster event
    df["event"] = event

    all_data.append(df)

    print(f"✅ Records: {len(df)}")


# Combine all events
master_df = pd.concat(all_data, ignore_index=True)


# Remove duplicate tweet IDs
before = len(master_df)

master_df = master_df.drop_duplicates(
    subset=["tweet_id"]
).reset_index(drop=True)

after = len(master_df)


print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(f"Records before duplicate removal: {before}")
print(f"Duplicates removed: {before - after}")
print(f"Final records: {after}")

print("\nEvent distribution:")
print(master_df["event"].value_counts())


# Save master dataset
output_file = OUTPUT_PATH / "crisis_mmd_master.csv"

master_df.to_csv(output_file, index=False)


print("\n" + "=" * 70)
print("✅ DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Saved to:")
print(output_file)