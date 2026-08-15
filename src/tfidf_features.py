import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# ============================================================
# CRISIS MMD - TF-IDF FEATURE EXTRACTION
# ============================================================

print("=" * 70)
print("CRISIS MMD - TF-IDF FEATURE EXTRACTION")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "crisis_mmd_nlp.csv"

TFIDF_FILE = DATA_DIR / "tfidf_features.npz"

LABEL_FILE = DATA_DIR / "tfidf_labels.csv"

VECTORIZER_FILE = DATA_DIR / "tfidf_vectorizer.pkl"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading NLP dataset...")

df = pd.read_csv(INPUT_FILE)

print("✅ Dataset loaded")
print("Records:", len(df))


# ============================================================
# 3. PREPARE TEXT
# ============================================================

print("\nPreparing text data...")

texts = df["processed_text"].fillna("").astype(str)

print("Texts available:", len(texts))


# ============================================================
# 4. CREATE TF-IDF VECTORIZER
# ============================================================

print("\nCreating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

print("✅ TF-IDF vectorizer created")


# ============================================================
# 5. TRANSFORM TEXT
# ============================================================

print("\nConverting text into numerical features...")

X = vectorizer.fit_transform(texts)

print("✅ TF-IDF transformation completed")


# ============================================================
# 6. FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF FEATURE INFORMATION")
print("=" * 70)

print("Number of documents :", X.shape[0])
print("Number of features  :", X.shape[1])
print("Matrix shape        :", X.shape)

print("\nFirst 20 TF-IDF features:")
print("-" * 70)

features = vectorizer.get_feature_names_out()

for feature in features[:20]:
    print(feature)


# ============================================================
# 7. SAVE TF-IDF MATRIX
# ============================================================

print("\nSaving TF-IDF features...")

save_npz(TFIDF_FILE, X)

print("✅ TF-IDF matrix saved to:")
print(TFIDF_FILE)


# ============================================================
# 8. SAVE LABELS
# ============================================================

labels = df[["text_info"]]

labels.to_csv(LABEL_FILE, index=False)

print("✅ Labels saved to:")
print(LABEL_FILE)


# ============================================================
# 9. SAVE VECTORIZER
# ============================================================

print("\nSaving TF-IDF vectorizer...")

joblib.dump(vectorizer, VECTORIZER_FILE)

print("✅ TF-IDF vectorizer saved to:")
print(VECTORIZER_FILE)


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print("Documents :", X.shape[0])
print("Features  :", X.shape[1])

print("\nFiles created:")
print("✅ tfidf_features.npz")
print("✅ tfidf_labels.csv")
print("✅ tfidf_vectorizer.pkl")

print("\n" + "=" * 70)
print("✅ ML-READY FEATURES CREATED SUCCESSFULLY")
print("=" * 70)