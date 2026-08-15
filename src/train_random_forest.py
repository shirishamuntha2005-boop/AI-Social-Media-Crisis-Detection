# ================================================================
# AI SOCIAL MEDIA CRISIS DETECTION
# CRISIS MMD - RANDOM FOREST CLASSIFIER
# ================================================================

import os
import joblib
import pandas as pd

from scipy.sparse import load_npz

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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

MODEL_PATH = os.path.join(
    DATA_DIR,
    "final_random_forest_model.joblib"
)


# ================================================================
# 2. HEADER
# ================================================================

print("=" * 70)
print("CRISIS MMD - RANDOM FOREST CLASSIFIER")
print("=" * 70)


# ================================================================
# 3. CHECK REQUIRED FILES
# ================================================================

print("\nChecking required files...")

required_files = {
    "X_train": X_TRAIN_PATH,
    "X_test": X_TEST_PATH,
    "y_train": Y_TRAIN_PATH,
    "y_test": Y_TEST_PATH
}

for name, path in required_files.items():

    if os.path.exists(path):

        print(f"✅ {name} found")

    else:

        print(f"❌ {name} NOT found")
        print(f"Path: {path}")

        raise FileNotFoundError(path)


# ================================================================
# 4. LOAD SPARSE TF-IDF FEATURES
# ================================================================

print("\n" + "=" * 70)
print("LOADING TF-IDF FEATURES")
print("=" * 70)

print("\nLoading X_train.npz...")

X_train = load_npz(
    X_TRAIN_PATH
)

print("✅ X_train loaded")

print(
    "X_train shape:",
    X_train.shape
)

print(
    "X_train type:",
    type(X_train)
)


print("\nLoading X_test.npz...")

X_test = load_npz(
    X_TEST_PATH
)

print("✅ X_test loaded")

print(
    "X_test shape:",
    X_test.shape
)

print(
    "X_test type:",
    type(X_test)
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

print("✅ y_train loaded")
print("✅ y_test loaded")

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

    raise KeyError(
        f"'{LABEL_COLUMN}' not found in y_train.csv"
    )


if LABEL_COLUMN not in y_test_df.columns:

    raise KeyError(
        f"'{LABEL_COLUMN}' not found in y_test.csv"
    )


y_train = y_train_df[
    LABEL_COLUMN
].values

y_test = y_test_df[
    LABEL_COLUMN
].values


print(
    f"\n✅ Label column: {LABEL_COLUMN}"
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
# 7. DATA INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("DATA INFORMATION")
print("=" * 70)

print(
    "\nX_train shape:",
    X_train.shape
)

print(
    "X_test shape :",
    X_test.shape
)

print(
    "y_train shape:",
    y_train.shape
)

print(
    "y_test shape :",
    y_test.shape
)


# ================================================================
# 8. VERIFY DATA SIZE
# ================================================================

print("\nChecking data consistency...")

if X_train.shape[0] != len(y_train):

    print("\n❌ Training data mismatch")

    print(
        "X_train samples:",
        X_train.shape[0]
    )

    print(
        "y_train labels:",
        len(y_train)
    )

    raise ValueError(
        "Training data size mismatch."
    )


if X_test.shape[0] != len(y_test):

    print("\n❌ Testing data mismatch")

    print(
        "X_test samples:",
        X_test.shape[0]
    )

    print(
        "y_test labels:",
        len(y_test)
    )

    raise ValueError(
        "Testing data size mismatch."
    )


print(
    "✅ Training data size matches"
)

print(
    "✅ Testing data size matches"
)


# ================================================================
# 9. DISPLAY CLASSES
# ================================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining classes:")

print(
    pd.Series(y_train).value_counts()
)


print("\nTesting classes:")

print(
    pd.Series(y_test).value_counts()
)


# ================================================================
# 10. CREATE RANDOM FOREST
# ================================================================

print("\n" + "=" * 70)
print("CREATING RANDOM FOREST MODEL")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\n✅ Random Forest model created")

print(
    "Number of trees:",
    model.n_estimators
)

print(
    "Random state:",
    model.random_state
)

print(
    "CPU jobs:",
    model.n_jobs
)

print(
    "Class weight:",
    model.class_weight
)


# ================================================================
# 11. TRAIN MODEL
# ================================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

print("\nTraining started...")
print("Please wait...")
print("This may take several minutes.\n")


model.fit(
    X_train,
    y_train
)


print("\n✅ Training completed successfully!")


# ================================================================
# 12. PREDICTION
# ================================================================

print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

print("\nPredicting test data...")

y_pred = model.predict(
    X_test
)

print(
    "✅ Predictions completed"
)


# ================================================================
# 13. ACCURACY
# ================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ================================================================
# 14. CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


# ================================================================
# 15. CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n", cm)


# ================================================================
# 16. MODEL INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(
    "\nNumber of features:",
    model.n_features_in_
)

print(
    "Number of trees:",
    model.n_estimators
)

print(
    "Number of classes:",
    len(model.classes_)
)

print(
    "Classes:",
    model.classes_
)


# ================================================================
# 17. SAVE MODEL
# ================================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(
    model,
    MODEL_PATH
)

print(
    "\n✅ Model saved successfully!"
)

print(
    "\nModel path:"
)

print(
    MODEL_PATH
)


# ================================================================
# 18. VERIFY MODEL
# ================================================================

if os.path.exists(MODEL_PATH):

    file_size = os.path.getsize(
        MODEL_PATH
    )

    print(
        "\n✅ Model file verified"
    )

    print(
        f"Model size: "
        f"{file_size / (1024 * 1024):.2f} MB"
    )

else:

    print(
        "\n❌ Model file was not created!"
    )


# ================================================================
# 19. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFINAL RESULTS")
print("-" * 70)

print(
    f"Model           : Random Forest"
)

print(
    f"Training samples : {X_train.shape[0]}"
)

print(
    f"Testing samples  : {X_test.shape[0]}"
)

print(
    f"Features         : {X_train.shape[1]}"
)

print(
    f"Accuracy         : {accuracy * 100:.2f}%"
)

print(
    f"Trees            : {model.n_estimators}"
)

print(
    f"Model saved      : YES"
)

print("\n" + "=" * 70)
print(
    "🎉 CRISIS DETECTION MODEL READY!"
)
print("=" * 70)