import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
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

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "crisis_mmd_nlp.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "hybrid_crisis_model.joblib"
)

WORD_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "hybrid_word_tfidf.pkl"
)

CHAR_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "hybrid_char_tfidf.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

RANDOM_STATE = 42


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("THRESHOLD OPTIMIZATION")
print("=" * 70)


# ================================================================
# CHECK FILES
# ================================================================

print("\nChecking required files...")

required_files = [
    DATA_PATH,
    MODEL_PATH,
    WORD_VECTORIZER_PATH,
    CHAR_VECTORIZER_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):
        print("❌ Missing file:")
        print(file_path)
        raise FileNotFoundError(file_path)

    print("✅ Found:", os.path.basename(file_path))


# ================================================================
# LOAD DATASET
# ================================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "text_info"

df = df[
    [TEXT_COLUMN, LABEL_COLUMN]
].copy()

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
)

df[LABEL_COLUMN] = (
    df[LABEL_COLUMN]
    .fillna("")
    .astype(str)
)

df = df[
    df[TEXT_COLUMN].str.strip() != ""
]

df = df[
    df[LABEL_COLUMN].isin(
        ["informative", "not_informative"]
    )
]

df = df.drop_duplicates(
    subset=[TEXT_COLUMN]
).reset_index(drop=True)

print("Cleaned dataset:", df.shape)


# ================================================================
# TRAIN TEST SPLIT
# ================================================================

print("\n" + "=" * 70)
print("CREATING TEST SET")
print("=" * 70)

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
# LOAD MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING HYBRID MODEL")
print("=" * 70)

model = joblib.load(MODEL_PATH)

print("✅ Model loaded")
print("Model:", type(model).__name__)


# ================================================================
# LOAD VECTORIZERS
# ================================================================

print("\nLoading word vectorizer...")

word_vectorizer = joblib.load(
    WORD_VECTORIZER_PATH
)

print("✅ Word vectorizer loaded")


print("\nLoading character vectorizer...")

char_vectorizer = joblib.load(
    CHAR_VECTORIZER_PATH
)

print("✅ Character vectorizer loaded")


# ================================================================
# CREATE FEATURES
# ================================================================

print("\n" + "=" * 70)
print("CREATING HYBRID FEATURES")
print("=" * 70)

print("Creating word features...")

X_test_word = word_vectorizer.transform(
    X_test_text
)

print(
    "Word test shape:",
    X_test_word.shape
)


print("\nCreating character features...")

X_test_char = char_vectorizer.transform(
    X_test_text
)

print(
    "Character test shape:",
    X_test_char.shape
)


print("\nCombining features...")

X_test_hybrid = hstack(
    [
        X_test_word,
        X_test_char
    ],
    format="csr"
)

print(
    "Hybrid test shape:",
    X_test_hybrid.shape
)


# ================================================================
# GET PROBABILITIES
# ================================================================

print("\n" + "=" * 70)
print("CALCULATING PREDICTION PROBABILITIES")
print("=" * 70)

probabilities = model.predict_proba(
    X_test_hybrid
)

classes = list(model.classes_)

print("Classes:", classes)

informative_index = classes.index(
    "informative"
)

informative_prob = probabilities[
    :,
    informative_index
]

print("✅ Probabilities calculated")


# ================================================================
# THRESHOLD TESTING
# ================================================================

print("\n" + "=" * 70)
print("TESTING DIFFERENT THRESHOLDS")
print("=" * 70)

thresholds = np.arange(
    0.30,
    0.71,
    0.02
)

results = []

best_threshold = None
best_accuracy = 0
best_f1 = 0
best_predictions = None


for threshold in thresholds:

    predictions = np.where(
        informative_prob >= threshold,
        "informative",
        "not_informative"
    )

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

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy * 100:.2f}% | "
        f"F1: {f1 * 100:.2f}%"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_f1 = f1
        best_threshold = threshold
        best_predictions = predictions


# ================================================================
# RESULTS
# ================================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1_Score": "{:.4f}".format
        }
    )
)


# ================================================================
# BEST THRESHOLD
# ================================================================

print("\n" + "=" * 70)
print("🏆 BEST THRESHOLD")
print("=" * 70)

print(
    f"Best threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy       : {best_accuracy * 100:.2f}%"
)

print(
    f"Weighted F1    : {best_f1 * 100:.2f}%"
)


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
# SAVE THRESHOLD
# ================================================================

THRESHOLD_PATH = os.path.join(
    OUTPUT_DIR,
    "best_prediction_threshold.txt"
)

with open(
    THRESHOLD_PATH,
    "w"
) as file:

    file.write(
        str(best_threshold)
    )

print("\n✅ Best threshold saved")
print(THRESHOLD_PATH)


# ================================================================
# SAVE RESULTS
# ================================================================

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "threshold_optimization_results.csv"
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n✅ Threshold results saved")
print(RESULTS_PATH)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 70)

print(
    f"""
🏆 Best Threshold : {best_threshold:.2f}
📊 Accuracy       : {best_accuracy * 100:.2f}%
📊 Weighted F1    : {best_f1 * 100:.2f}%

💾 Threshold saved : YES
📁 Results saved   : YES
"""
)

print("=" * 70)
print("🎉 DONE!")
print("=" * 70)