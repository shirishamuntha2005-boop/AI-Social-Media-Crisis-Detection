import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CRISIS MMD - BALANCED LOGISTIC REGRESSION
# ============================================================

print("=" * 70)
print("CRISIS MMD - BALANCED LOGISTIC REGRESSION")
print("=" * 70)


# ============================================================
# 1. DATA PATH
# ============================================================

DATA_DIR = Path("data/processed")


# ============================================================
# 2. LOAD TRAINING AND TESTING DATA
# ============================================================

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


# ============================================================
# 3. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(y_train.value_counts())


# ============================================================
# 4. CREATE BALANCED LOGISTIC REGRESSION
# ============================================================

print("\nCreating balanced Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

print("✅ Balanced model created")


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("✅ Model training completed")


# ============================================================
# 6. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("✅ Predictions completed")


# ============================================================
# 7. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ============================================================
# 10. SAVE MODEL
# ============================================================

import joblib

MODEL_PATH = DATA_DIR / "balanced_logistic_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 70)
print("MODEL SAVING")
print("=" * 70)

print(f"✅ Model saved to: {MODEL_PATH}")


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("✅ BALANCED LOGISTIC REGRESSION COMPLETED")
print("=" * 70)