import pandas as pd
import re
from pathlib import Path


# ============================================================
# CRISIS MMD - TEXT CLEANING
# ============================================================

INPUT_FILE = Path("data/processed/crisis_mmd_master.csv")
OUTPUT_FILE = Path("data/processed/crisis_mmd_cleaned.csv")


print("=" * 70)
print("CRISIS MMD - TEXT CLEANING")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("✅ Dataset loaded")
print(f"Records: {len(df)}")


# ------------------------------------------------------------
# 2. Check duplicate tweets
# ------------------------------------------------------------

duplicates = df["tweet_text"].duplicated().sum()

print(f"\nDuplicate tweet texts: {duplicates}")


# ------------------------------------------------------------
# 3. Text cleaning function
# ------------------------------------------------------------

def clean_text(text):

    text = str(text)

    # Remove retweet marker
    text = re.sub(r"\bRT\b", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ------------------------------------------------------------
# 4. Apply cleaning
# ------------------------------------------------------------

print("\nCleaning tweet text...")

df["clean_text"] = df["tweet_text"].apply(clean_text)

print("✅ Text cleaning completed")


# ------------------------------------------------------------
# 5. Remove empty text
# ------------------------------------------------------------

before = len(df)

df = df[df["clean_text"].str.len() > 0].copy()

after = len(df)

print(f"\nEmpty texts removed: {before - after}")


# ------------------------------------------------------------
# 6. Save cleaned dataset
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print(f"Final records: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")


print("\nExample:")
print("-" * 70)

print("Original:")
print(df["tweet_text"].iloc[0])

print("\nCleaned:")
print(df["clean_text"].iloc[0])


print("\n" + "=" * 70)
print("✅ CLEANED DATASET CREATED SUCCESSFULLY")
print("=" * 70)