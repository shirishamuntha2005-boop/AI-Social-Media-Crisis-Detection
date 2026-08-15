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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

DATASET_PATH = os.path.join(
    DATA_DIR,
    "crisis_mmd_nlp.csv"
)

MODEL_PATH = os.path.join(
    DATA_DIR,
    "ensemble_hybrid_crisis_model.joblib"
)

WORD_VECTORIZER_PATH = os.path.join(
    DATA_DIR,
    "ensemble_word_tfidf.pkl"
)

CHAR_VECTORIZER_PATH = os.path.join(
    DATA_DIR,
    "ensemble_char_tfidf.pkl"
)

THRESHOLD_PATH = os.path.join(
    DATA_DIR,
    "best_ensemble_prediction_threshold.txt"
)

RESULTS_PATH = os.path.join(
    DATA_DIR,
    "ensemble_threshold_optimization_results.csv"
)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("ENSEMBLE HYBRID NLP THRESHOLD OPTIMIZATION")
print("=" * 70)


# ================================================================
# CHECK REQUIRED FILES
# ================================================================

print("\nChecking required files...")

required_files = {
    "Dataset": DATASET_PATH,
    "Ensemble model": MODEL_PATH,
    "Word vectorizer": WORD_VECTORIZER_PATH,
    "Character vectorizer": CHAR_VECTORIZER_PATH
}

for name, path in required_files.items():

    if os.path.exists(path):
        print(f"✅ Found: {name}")
    else:
        print(f"❌ Missing: {name}")
        print(path)
        raise FileNotFoundError(path)


# ================================================================
# LOAD DATASET
# ================================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape: {df.shape}")

# ------------------------------------------------
# Select required columns
# ------------------------------------------------

df = df[["tweet_text", "text_info"]].copy()

df = df.dropna(
    subset=["tweet_text", "text_info"]
)

df["tweet_text"] = df["tweet_text"].astype(str)

df["text_info"] = df["text_info"].astype(str)

print(f"Cleaned dataset: {df.shape}")


# ================================================================
# CREATE SAME TRAIN/TEST SPLIT
# ================================================================

print("\n" + "=" * 70)
print("CREATING TEST SET")
print("=" * 70)

X = df["tweet_text"]

y = df["text_info"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ================================================================
# LOAD ENSEMBLE MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING ENSEMBLE MODEL")
print("=" * 70)

model = joblib.load(MODEL_PATH)

print("✅ Ensemble model loaded")

print(
    f"Model type: {type(model).__name__}"
)


# ================================================================
# LOAD WORD VECTORIZER
# ================================================================

print("\nLoading Word TF-IDF vectorizer...")

word_vectorizer = joblib.load(
    WORD_VECTORIZER_PATH
)

print("✅ Word vectorizer loaded")


# ================================================================
# LOAD CHARACTER VECTORIZER
# ================================================================

print("\nLoading Character TF-IDF vectorizer...")

char_vectorizer = joblib.load(
    CHAR_VECTORIZER_PATH
)

print("✅ Character vectorizer loaded")


# ================================================================
# CREATE WORD FEATURES
# ================================================================

print("\n" + "=" * 70)
print("CREATING WORD TF-IDF FEATURES")
print("=" * 70)

X_test_word = word_vectorizer.transform(
    X_test
)

print(
    f"Word test shape: {X_test_word.shape}"
)


# ================================================================
# CREATE CHARACTER FEATURES
# ================================================================

print("\n" + "=" * 70)
print("CREATING CHARACTER TF-IDF FEATURES")
print("=" * 70)

X_test_char = char_vectorizer.transform(
    X_test
)

print(
    f"Character test shape: {X_test_char.shape}"
)


# ================================================================
# COMBINE FEATURES
# ================================================================

print("\n" + "=" * 70)
print("COMBINING WORD + CHARACTER FEATURES")
print("=" * 70)

X_test_hybrid = hstack(
    [
        X_test_word,
        X_test_char
    ]
)

print(
    f"Hybrid test shape: {X_test_hybrid.shape}"
)


# ================================================================
# CALCULATE PROBABILITIES
# ================================================================

print("\n" + "=" * 70)
print("CALCULATING ENSEMBLE PROBABILITIES")
print("=" * 70)

probabilities = model.predict_proba(
    X_test_hybrid
)

classes = list(model.classes_)

print(f"Classes: {classes}")

print("✅ Probabilities calculated")


# ================================================================
# FIND INFORMATIVE CLASS
# ================================================================

if "informative" not in classes:

    raise ValueError(
        "The model does not contain the 'informative' class."
    )

informative_index = classes.index(
    "informative"
)

informative_probabilities = probabilities[
    :, informative_index
]


# ================================================================
# TEST DIFFERENT THRESHOLDS
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
best_accuracy = -1
best_f1 = -1


for threshold in thresholds:

    # ------------------------------------------------
    # Prediction using threshold
    # ------------------------------------------------

    predictions = np.where(
        informative_probabilities >= threshold,
        "informative",
        "not_informative"
    )

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="informative",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="informative",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy * 100:.2f}% | "
        f"F1: {f1 * 100:.2f}%"
    )

    results.append({

        "Threshold": round(
            float(threshold),
            2
        ),

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1
    })


    # ------------------------------------------------
    # Select best model
    # ------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_accuracy = accuracy

        best_threshold = threshold


# ================================================================
# RESULTS DATAFRAME
# ================================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)


# ================================================================
# DISPLAY RESULTS
# ================================================================

print("\n" + "=" * 70)
print("ENSEMBLE THRESHOLD RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ================================================================
# BEST THRESHOLD
# ================================================================

print("\n" + "=" * 70)
print("🏆 BEST ENSEMBLE THRESHOLD")
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
# FINAL PREDICTIONS
# ================================================================

best_predictions = np.where(
    informative_probabilities >= best_threshold,
    "informative",
    "not_informative"
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
    best_predictions,
    labels=[
        "informative",
        "not_informative"
    ]
)

print(cm)


# ================================================================
# SAVE BEST THRESHOLD
# ================================================================

with open(
    THRESHOLD_PATH,
    "w"
) as f:

    f.write(
        f"{best_threshold:.2f}"
    )

print("\n✅ Best threshold saved")

print(
    THRESHOLD_PATH
)


# ================================================================
# SAVE RESULTS
# ================================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n✅ Threshold results saved")

print(
    RESULTS_PATH
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("ENSEMBLE THRESHOLD OPTIMIZATION COMPLETED")
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