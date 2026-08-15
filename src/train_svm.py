import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CRISIS MMD - LINEAR SVM CLASSIFIER
# ============================================================

DATA_DIR = Path("data/processed")

print("=" * 70)
print("CRISIS MMD - LINEAR SVM CLASSIFIER")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load training and testing data
# ------------------------------------------------------------

print("\nLoading training and testing data...")

X_train = load_npz(DATA_DIR / "X_train.npz")
X_test = load_npz(DATA_DIR / "X_test.npz")

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
)["text_info"]

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
)["text_info"]

print("✅ Training data loaded")
print("✅ Testing data loaded")

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")


# ------------------------------------------------------------
# 2. Create Linear SVM model
# ------------------------------------------------------------

print("\nCreating Linear SVM model...")

model = LinearSVC(
    random_state=42,
    max_iter=5000
)

print("✅ Model created")


# ------------------------------------------------------------
# 3. Train model
# ------------------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("✅ Model training completed")


# ------------------------------------------------------------
# 4. Make predictions
# ------------------------------------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("✅ Predictions completed")


# ------------------------------------------------------------
# 5. Calculate accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------------------------
# 6. Classification report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ------------------------------------------------------------
# 7. Confusion matrix
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ------------------------------------------------------------
# 8. Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("✅ LINEAR SVM TRAINING COMPLETED")
print("=" * 70)