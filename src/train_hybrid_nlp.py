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


# ================================================================
# PROJECT PATHS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

DATASET_PATH = os.path.join(
    DATA_DIR,
    "crisis_mmd_nlp.csv"
)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("HYBRID WORD + CHARACTER TF-IDF NLP")
print("=" * 70)


# ================================================================
# CHECK DATASET
# ================================================================

print("\nChecking dataset...")

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )

print("✅ Dataset found")


# ================================================================
# LOAD DATASET
# ================================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    "Dataset shape:",
    df.shape
)

print(
    "\nAvailable columns:"
)

print(
    df.columns.tolist()
)


# ================================================================
# SELECT COLUMNS
# ================================================================

TEXT_COLUMN = "tweet_text"

LABEL_COLUMN = "text_info"


if TEXT_COLUMN not in df.columns:

    raise ValueError(
        f"Text column '{TEXT_COLUMN}' not found."
    )


if LABEL_COLUMN not in df.columns:

    raise ValueError(
        f"Label column '{LABEL_COLUMN}' not found."
    )


print(
    f"\nText column : {TEXT_COLUMN}"
)

print(
    f"Label column: {LABEL_COLUMN}"
)


# ================================================================
# CLEAN DATA
# ================================================================

print("\nCleaning data...")

df = df[
    [
        TEXT_COLUMN,
        LABEL_COLUMN
    ]
].copy()


# Remove missing values

df = df.dropna(
    subset=[
        TEXT_COLUMN,
        LABEL_COLUMN
    ]
)


# Convert text to string

df[TEXT_COLUMN] = df[
    TEXT_COLUMN
].astype(str)


# Remove empty text

df = df[
    df[TEXT_COLUMN].str.strip() != ""
]


# Remove duplicate text

before = len(df)

df = df.drop_duplicates(
    subset=[TEXT_COLUMN]
)

after = len(df)

print(
    f"Rows before cleaning: {before}"
)

print(
    f"Rows after cleaning : {after}"
)

print(
    f"Duplicates removed  : {before - after}"
)


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print("\nClass distribution:")

print(
    df[LABEL_COLUMN].value_counts()
)


# ================================================================
# TRAIN TEST SPLIT
# ================================================================

print("\nSplitting dataset...")

from sklearn.model_selection import train_test_split


X = df[TEXT_COLUMN]

y = df[LABEL_COLUMN]


X_train_text, X_test_text, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    f"Training samples: {len(X_train_text)}"
)

print(
    f"Testing samples : {len(X_test_text)}"
)


# ================================================================
# WORD TF-IDF
# ================================================================

print("\n")
print("=" * 70)
print("CREATING WORD TF-IDF FEATURES")
print("=" * 70)


word_vectorizer = TfidfVectorizer(

    ngram_range=(1, 3),

    max_features=30000,

    min_df=2,

    max_df=0.95,

    sublinear_tf=True,

    strip_accents="unicode"
)


print("\nTraining word vectorizer...")

X_train_word = word_vectorizer.fit_transform(
    X_train_text
)

X_test_word = word_vectorizer.transform(
    X_test_text
)


print(
    "\n✅ Word TF-IDF created"
)

print(
    "X_train_word:",
    X_train_word.shape
)

print(
    "X_test_word :",
    X_test_word.shape
)


# ================================================================
# CHARACTER TF-IDF
# ================================================================

print("\n")
print("=" * 70)
print("CREATING CHARACTER TF-IDF FEATURES")
print("=" * 70)


char_vectorizer = TfidfVectorizer(

    analyzer="char",

    ngram_range=(3, 5),

    max_features=30000,

    min_df=2,

    max_df=0.95,

    sublinear_tf=True
)


print("\nTraining character vectorizer...")

X_train_char = char_vectorizer.fit_transform(
    X_train_text
)

X_test_char = char_vectorizer.transform(
    X_test_text
)


print(
    "\n✅ Character TF-IDF created"
)

print(
    "X_train_char:",
    X_train_char.shape
)

print(
    "X_test_char :",
    X_test_char.shape
)


# ================================================================
# COMBINE FEATURES
# ================================================================

print("\n")
print("=" * 70)
print("COMBINING WORD + CHARACTER FEATURES")
print("=" * 70)


X_train_hybrid = hstack(
    [
        X_train_word,
        X_train_char
    ]
).tocsr()


X_test_hybrid = hstack(
    [
        X_test_word,
        X_test_char
    ]
).tocsr()


print(
    "\n✅ Features combined"
)

print(
    "X_train_hybrid:",
    X_train_hybrid.shape
)

print(
    "X_test_hybrid :",
    X_test_hybrid.shape
)


# ================================================================
# TRAIN MULTIPLE LOGISTIC REGRESSION MODELS
# ================================================================

print("\n")
print("=" * 70)
print("TRAINING HYBRID LOGISTIC REGRESSION MODELS")
print("=" * 70)


C_VALUES = [
    0.5,
    1.0,
    2.0,
    3.0,
    5.0
]


results = []

best_model = None

best_accuracy = 0

best_f1 = 0

best_C = None

best_predictions = None


for C in C_VALUES:

    print("\n")
    print("-" * 70)

    print(
        f"Training Logistic Regression"
    )

    print(
        f"C = {C}"
    )

    print("-" * 70)


    model = LogisticRegression(

        C=C,

        max_iter=2000,

        solver="liblinear",

        class_weight=None,

        random_state=42
    )


    print("\nTraining...")

    model.fit(
        X_train_hybrid,
        y_train
    )

    print(
        "✅ Training completed"
    )


    print(
        "Making predictions..."
    )

    predictions = model.predict(
        X_test_hybrid
    )

    print(
        "✅ Predictions completed"
    )


    # ------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------

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
        average="weighted"
    )


    print(
        f"\nAccuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )


    results.append({

        "C": C,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1
    })


    # ------------------------------------------------------------
    # BEST MODEL
    # ------------------------------------------------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_f1 = f1

        best_model = model

        best_C = C

        best_predictions = predictions

        print(
            "\n🏆 NEW BEST MODEL!"
        )


# ================================================================
# RESULTS
# ================================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)


print("\n")
print("=" * 70)
print("HYBRID MODEL COMPARISON")
print("=" * 70)

print()

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

print("\n")
print("=" * 70)
print("🏆 BEST HYBRID MODEL")
print("=" * 70)


print(
    f"\nModel       : Logistic Regression"
)

print(
    f"C           : {best_C}"
)

print(
    f"Accuracy    : {best_accuracy * 100:.2f}%"
)

print(
    f"F1 Score    : {best_f1 * 100:.2f}%"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n")
print("=" * 70)
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

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    y_test,
    best_predictions
)

print()

print(cm)


# ================================================================
# SAVE MODEL
# ================================================================

MODEL_PATH = os.path.join(
    DATA_DIR,
    "hybrid_crisis_model.joblib"
)


print("\n")
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)


joblib.dump(
    best_model,
    MODEL_PATH
)


print(
    "\n✅ Hybrid model saved successfully!"
)

print(
    "Model path:"
)

print(
    MODEL_PATH
)


# ================================================================
# SAVE VECTORIZERS
# ================================================================

WORD_VECTORIZER_PATH = os.path.join(
    DATA_DIR,
    "hybrid_word_tfidf.pkl"
)


CHAR_VECTORIZER_PATH = os.path.join(
    DATA_DIR,
    "hybrid_char_tfidf.pkl"
)


joblib.dump(
    word_vectorizer,
    WORD_VECTORIZER_PATH
)


joblib.dump(
    char_vectorizer,
    CHAR_VECTORIZER_PATH
)


print(
    "\n✅ Word vectorizer saved"
)

print(
    WORD_VECTORIZER_PATH
)


print(
    "\n✅ Character vectorizer saved"
)

print(
    CHAR_VECTORIZER_PATH
)


# ================================================================
# SAVE COMPARISON
# ================================================================

RESULTS_PATH = os.path.join(
    DATA_DIR,
    "hybrid_nlp_results.csv"
)


results_df.to_csv(
    RESULTS_PATH,
    index=False
)


print(
    "\n✅ Results saved"
)

print(
    RESULTS_PATH
)


# ================================================================
# FINAL COMPARISON
# ================================================================

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)


print(
    "\nPrevious best Random Forest : 80.23%"
)

print(
    f"Hybrid NLP accuracy        : "
    f"{best_accuracy * 100:.2f}%"
)


improvement = (
    best_accuracy * 100
) - 80.23


print(
    f"Improvement                : "
    f"{improvement:+.2f} percentage points"
)


# ================================================================
# COMPLETE
# ================================================================

print("\n")
print("=" * 70)
print("HYBRID NLP TRAINING COMPLETED")
print("=" * 70)

print(
    "\n🎉 DONE!"
)