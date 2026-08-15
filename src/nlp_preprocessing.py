import pandas as pd
import re
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# CRISIS MMD - NLP TEXT PREPROCESSING
# ============================================================

INPUT_FILE = Path("data/processed/crisis_mmd_cleaned.csv")
OUTPUT_FILE = Path("data/processed/crisis_mmd_nlp.csv")


print("=" * 70)
print("CRISIS MMD - NLP TEXT PREPROCESSING")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("✅ Dataset loaded")
print(f"Records: {len(df)}")


# ------------------------------------------------------------
# 2. Load NLP resources
# ------------------------------------------------------------

print("\nLoading NLP resources...")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

print("✅ Stopwords loaded")
print("✅ Lemmatizer loaded")


# ------------------------------------------------------------
# 3. NLP preprocessing function
# ------------------------------------------------------------

def preprocess_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove RT
    text = re.sub(r"\brt\b", " ", text)

    # Keep only alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Join tokens
    return " ".join(tokens)


# ------------------------------------------------------------
# 4. Apply NLP preprocessing
# ------------------------------------------------------------

print("\nApplying NLP preprocessing...")

df["processed_text"] = df["clean_text"].apply(preprocess_text)

print("✅ NLP preprocessing completed")


# ------------------------------------------------------------
# 5. Remove empty processed text
# ------------------------------------------------------------

before = len(df)

df = df[df["processed_text"].str.len() > 0].copy()

after = len(df)

print(f"\nEmpty processed texts removed: {before - after}")


# ------------------------------------------------------------
# 6. Display examples
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NLP PREPROCESSING EXAMPLES")
print("=" * 70)

for i in range(3):

    print("\nOriginal:")
    print(df["tweet_text"].iloc[i])

    print("\nClean text:")
    print(df["clean_text"].iloc[i])

    print("\nProcessed text:")
    print(df["processed_text"].iloc[i])

    print("-" * 70)


# ------------------------------------------------------------
# 7. Save NLP dataset
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("NLP PREPROCESSING COMPLETED")
print("=" * 70)

print(f"Final records: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\n" + "=" * 70)
print("✅ NLP-READY DATASET CREATED SUCCESSFULLY")
print("=" * 70)