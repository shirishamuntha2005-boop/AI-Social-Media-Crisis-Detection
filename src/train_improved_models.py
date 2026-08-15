# ================================================================
# AI SOCIAL MEDIA CRISIS DETECTION
# IMPROVED NLP MODEL COMPARISON
# ================================================================

import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ================================================================
# 1. PROJECT PATHS
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

X_TRAIN_PATH = os.path.join(
    DATA_DIR,
    "X_train.npz"
)

X_TEST_PATH = os.path.join(
    DATA_DIR,
    "X_test.npz"
)

Y_TRAIN_PATH = os.path.join(
    DATA_DIR,
    "y_train.csv"
)

Y_TEST_PATH = os.path.join(
    DATA_DIR,
    "y_test.csv"
)


# ================================================================
# 2. HEADER
# ================================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("IMPROVED NLP MODEL COMPARISON")
print("=" * 70)


# ================================================================
# 3. CHECK FILES
# ================================================================

print("\nChecking required files...")

required_files = [
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):

        print(f"❌ Missing file: {file_path}")

        raise FileNotFoundError(file_path)

    print(
        f"✅ Found: {os.path.basename(file_path)}"
    )


# ================================================================
# 4. LOAD TF-IDF FEATURES
# ================================================================

print("\n" + "=" * 70)
print("LOADING TF-IDF FEATURES")
print("=" * 70)

print("\nLoading X_train...")

X_train = load_npz(
    X_TRAIN_PATH
)

print(
    f"✅ X_train loaded: {X_train.shape}"
)


print("\nLoading X_test...")

X_test = load_npz(
    X_TEST_PATH
)

print(
    f"✅ X_test loaded: {X_test.shape}"
)


# ================================================================
# 5. LOAD LABELS
# ================================================================

print("\n" + "=" * 70)
print("LOADING LABELS")
print("=" * 70)

y_train_df = pd.read_csv(
    Y_TRAIN_PATH
)

y_test_df = pd.read_csv(
    Y_TEST_PATH
)


print(
    "\ny_train columns:",
    list(y_train_df.columns)
)

print(
    "y_test columns:",
    list(y_test_df.columns)
)


# ================================================================
# 6. SELECT LABEL COLUMN
# ================================================================

LABEL_COLUMN = "text_info"


if LABEL_COLUMN not in y_train_df.columns:

    raise ValueError(
        f"Column '{LABEL_COLUMN}' not found in y_train."
    )


y_train = y_train_df[
    LABEL_COLUMN
].astype(str)


y_test = y_test_df[
    LABEL_COLUMN
].astype(str)


print(
    "\n✅ Label column:",
    LABEL_COLUMN
)

print(
    "y_train shape:",
    y_train.shape
)

print(
    "y_test shape:",
    y_test.shape
)


# ================================================================
# 7. DATA VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)


if X_train.shape[0] != len(y_train):

    raise ValueError(
        "X_train and y_train sizes do not match."
    )


if X_test.shape[0] != len(y_test):

    raise ValueError(
        "X_test and y_test sizes do not match."
    )


print(
    "✅ Training data size matches"
)

print(
    "✅ Testing data size matches"
)


# ================================================================
# 8. CLASS DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining classes:")

print(
    y_train.value_counts()
)


print("\nTesting classes:")

print(
    y_test.value_counts()
)


# ================================================================
# 9. DEFINE MODELS
# ================================================================

models = {

    "Logistic Regression": LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        C=1.5,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ),

    "Complement Naive Bayes": ComplementNB(
        alpha=0.5
    )
}


# ================================================================
# 10. STORE RESULTS
# ================================================================

results = []

trained_models = {}


# ================================================================
# 11. TRAIN MODELS
# ================================================================

print("\n" + "=" * 70)
print("TRAINING IMPROVED MODELS")
print("=" * 70)


for model_name, model in models.items():

    print("\n" + "-" * 70)

    print(
        f"Training: {model_name}"
    )

    print("-" * 70)


    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    print(
        "✅ Training completed"
    )


    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    print(
        "✅ Predictions completed"
    )


    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1

    })


    trained_models[
        model_name
    ] = model


    # ------------------------------------------------------------
    # Display
    # ------------------------------------------------------------

    print("\nResults:")

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
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


    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# ================================================================
# 12. RESULTS TABLE
# ================================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
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
# 13. FIND BEST MODEL
# ================================================================

best_model_name = results_df.iloc[
    0
]["Model"]


best_model = trained_models[
    best_model_name
]


best_row = results_df.iloc[
    0
]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)


print(
    f"\nBest Model: {best_model_name}"
)

print(
    f"Accuracy  : {best_row['Accuracy'] * 100:.2f}%"
)

print(
    f"Precision : {best_row['Precision'] * 100:.2f}%"
)

print(
    f"Recall    : {best_row['Recall'] * 100:.2f}%"
)

print(
    f"F1 Score  : {best_row['F1_Score'] * 100:.2f}%"
)


# ================================================================
# 14. BEST MODEL CLASSIFICATION REPORT
# ================================================================

best_predictions = best_model.predict(
    X_test
)


print("\n" + "=" * 70)
print("BEST MODEL CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)


# ================================================================
# 15. CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("BEST MODEL CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    y_test,
    best_predictions
)


print()

print(cm)


# ================================================================
# 16. SAVE BEST MODEL
# ================================================================

BEST_MODEL_PATH = os.path.join(
    DATA_DIR,
    "best_crisis_detection_model.joblib"
)


print("\n" + "=" * 70)
print("SAVING BEST MODEL")
print("=" * 70)


joblib.dump(
    best_model,
    BEST_MODEL_PATH
)


print(
    "\n✅ Best model saved successfully!"
)

print(
    f"Model path:\n{BEST_MODEL_PATH}"
)


# ================================================================
# 17. SAVE MODEL COMPARISON
# ================================================================

COMPARISON_PATH = os.path.join(
    DATA_DIR,
    "improved_model_comparison.csv"
)


results_df.to_csv(
    COMPARISON_PATH,
    index=False
)


print(
    "\n✅ Model comparison saved!"
)

print(
    f"Comparison path:\n{COMPARISON_PATH}"
)


# ================================================================
# 18. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("IMPROVED MODEL TRAINING COMPLETED")
print("=" * 70)


print(
    f"\n🏆 Best Model       : {best_model_name}"
)

print(
    f"📊 Accuracy         : "
    f"{best_row['Accuracy'] * 100:.2f}%"
)

print(
    f"📊 Weighted F1      : "
    f"{best_row['F1_Score'] * 100:.2f}%"
)

print(
    "\n💾 Model saved      : YES"
)

print(
    "📁 Comparison saved : YES"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)