import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "crisis_mmd_nlp.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("HYBRID NLP HYPERPARAMETER TUNING")
print("=" * 70)


# ================================================================
# CHECK DATASET
# ================================================================

print("\nChecking dataset...")

if not os.path.exists(DATA_PATH):
    print("❌ Dataset not found:")
    print(DATA_PATH)
    raise FileNotFoundError(DATA_PATH)

print("✅ Dataset found")


# ================================================================
# LOAD DATASET
# ================================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nAvailable columns:")
print(list(df.columns))


# ================================================================
# SELECT TEXT COLUMN
# ================================================================

if "tweet_text" in df.columns:
    TEXT_COLUMN = "tweet_text"
elif "clean_text" in df.columns:
    TEXT_COLUMN = "clean_text"
elif "processed_text" in df.columns:
    TEXT_COLUMN = "processed_text"
else:
    raise ValueError(
        "❌ No valid text column found."
    )


# ================================================================
# SELECT LABEL COLUMN
# ================================================================

if "text_info" in df.columns:
    LABEL_COLUMN = "text_info"
else:
    raise ValueError(
        "❌ text_info column not found."
    )


print("\nText column :", TEXT_COLUMN)
print("Label column:", LABEL_COLUMN)


# ================================================================
# CLEAN DATA
# ================================================================

print("\nCleaning data...")

df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("").astype(str)

# Remove empty text
df = df[df[TEXT_COLUMN].str.strip() != ""]

# Remove invalid labels
df = df[df[LABEL_COLUMN].isin([
    "informative",
    "not_informative"
])]

# Remove duplicates
before = len(df)

df = df.drop_duplicates(
    subset=[TEXT_COLUMN]
).reset_index(drop=True)

after = len(df)

print("Rows before cleaning :", before)
print("Rows after cleaning  :", after)
print("Duplicates removed   :", before - after)


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print("\nClass distribution:")
print(df[LABEL_COLUMN].value_counts())


# ================================================================
# TRAIN TEST SPLIT
# ================================================================

print("\nSplitting dataset...")

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train_text))
print("Testing samples :", len(X_test_text))


# ================================================================
# WORD TF-IDF
# ================================================================

print("\n" + "=" * 70)
print("CREATING WORD TF-IDF")
print("=" * 70)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=40000,
    strip_accents="unicode"
)

print("Training word vectorizer...")

X_train_word = word_vectorizer.fit_transform(X_train_text)

X_test_word = word_vectorizer.transform(X_test_text)

print("Word features:", X_train_word.shape)


# ================================================================
# CHARACTER TF-IDF
# ================================================================

print("\n" + "=" * 70)
print("CREATING CHARACTER TF-IDF")
print("=" * 70)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=40000,
    sublinear_tf=True
)

print("Training character vectorizer...")

X_train_char = char_vectorizer.fit_transform(X_train_text)

X_test_char = char_vectorizer.transform(X_test_text)

print("Character features:", X_train_char.shape)


# ================================================================
# COMBINE FEATURES
# ================================================================

print("\n" + "=" * 70)
print("COMBINING WORD + CHARACTER FEATURES")
print("=" * 70)

X_train_hybrid = hstack(
    [X_train_word, X_train_char],
    format="csr"
)

X_test_hybrid = hstack(
    [X_test_word, X_test_char],
    format="csr"
)

print("Hybrid training shape:", X_train_hybrid.shape)
print("Hybrid testing shape :", X_test_hybrid.shape)


# ================================================================
# HYPERPARAMETER TUNING
# ================================================================

print("\n" + "=" * 70)
print("STARTING LOGISTIC REGRESSION TUNING")
print("=" * 70)

C_VALUES = [
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    10.0
]


results = []

best_model = None
best_accuracy = 0
best_f1 = 0
best_C = None
best_predictions = None


# ================================================================
# TRAIN MODELS
# ================================================================

for index, C_VALUE in enumerate(C_VALUES, start=1):

    print("\n" + "-" * 70)
    print(f"MODEL {index}/{len(C_VALUES)}")
    print("Logistic Regression")
    print(f"C = {C_VALUE}")
    print("-" * 70)

    model = LogisticRegression(
        C=C_VALUE,
        max_iter=3000,
        solver="liblinear",
        class_weight=None,
        random_state=RANDOM_STATE
    )

    print("Training...")

    model.fit(
        X_train_hybrid,
        y_train
    )

    print("✅ Training completed")

    print("Making predictions...")

    predictions = model.predict(
        X_test_hybrid
    )

    print("✅ Predictions completed")


    # ============================================================
    # METRICS
    # ============================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    print("\nResults:")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")


    results.append({
        "C": C_VALUE,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })


    # ============================================================
    # SELECT BEST MODEL
    # ============================================================

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_f1 = f1
        best_C = C_VALUE
        best_model = model
        best_predictions = predictions

        print("\n🏆 NEW BEST MODEL!")


# ================================================================
# RESULTS TABLE
# ================================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("HYBRID NLP TUNING RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1_Score": "{:.4f}".format
        }
    )
)


# ================================================================
# BEST MODEL
# ================================================================

print("\n" + "=" * 70)
print("🏆 BEST HYBRID NLP MODEL")
print("=" * 70)

print("Model       : Logistic Regression")
print(f"Best C      : {best_C}")
print(f"Accuracy    : {best_accuracy * 100:.2f}%")
print(f"F1 Score    : {best_f1 * 100:.2f}%")


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    best_predictions
)

print(cm)


# ================================================================
# SAVE BEST MODEL
# ================================================================

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "tuned_hybrid_crisis_model.joblib"
)

print("\n" + "=" * 70)
print("SAVING BEST MODEL")
print("=" * 70)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("✅ Model saved successfully!")
print("Model path:")
print(MODEL_PATH)


# ================================================================
# SAVE WORD VECTORIZER
# ================================================================

WORD_VECTORIZER_PATH = os.path.join(
    OUTPUT_DIR,
    "tuned_hybrid_word_tfidf.pkl"
)

joblib.dump(
    word_vectorizer,
    WORD_VECTORIZER_PATH
)

print("\n✅ Word TF-IDF vectorizer saved")
print(WORD_VECTORIZER_PATH)


# ================================================================
# SAVE CHARACTER VECTORIZER
# ================================================================

CHAR_VECTORIZER_PATH = os.path.join(
    OUTPUT_DIR,
    "tuned_hybrid_char_tfidf.pkl"
)

joblib.dump(
    char_vectorizer,
    CHAR_VECTORIZER_PATH
)

print("\n✅ Character TF-IDF vectorizer saved")
print(CHAR_VECTORIZER_PATH)


# ================================================================
# SAVE RESULTS
# ================================================================

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "tuned_hybrid_nlp_results.csv"
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n✅ Results saved")
print(RESULTS_PATH)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("TUNING COMPLETED")
print("=" * 70)

print(f"""
🏆 Best Model       : Logistic Regression
🎯 Best C           : {best_C}
📊 Accuracy         : {best_accuracy * 100:.2f}%
📊 Weighted F1      : {best_f1 * 100:.2f}%

💾 Model saved      : YES
💾 Vectorizers saved: YES
📁 Results saved    : YES
""")

print("=" * 70)
print("🎉 HYBRID NLP OPTIMIZATION COMPLETED!")
print("=" * 70)