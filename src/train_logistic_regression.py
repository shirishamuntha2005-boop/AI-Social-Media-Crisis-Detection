import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATA_DIR = Path("data/processed")

print("=" * 70)
print("CRISIS MMD - LOGISTIC REGRESSION CLASSIFIER")
print("=" * 70)


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


print("\nCreating Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("✅ Model created")


print("\nTraining model...")

model.fit(X_train, y_train)

print("✅ Model training completed")


print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("✅ Predictions completed")


accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(classification_report(y_test, y_pred))


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred)

print(cm)


print("\n" + "=" * 70)
print("✅ LOGISTIC REGRESSION TRAINING COMPLETED")
print("=" * 70)