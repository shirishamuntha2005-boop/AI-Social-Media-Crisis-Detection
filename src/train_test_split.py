import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz, save_npz
from sklearn.model_selection import train_test_split


# ============================================================
# CRISIS MMD - TRAIN TEST SPLIT
# ============================================================

FEATURE_FILE = Path("data/processed/tfidf_features.npz")
LABEL_FILE = Path("data/processed/tfidf_labels.csv")

OUTPUT_DIR = Path("data/processed")

print("=" * 70)
print("CRISIS MMD - TRAIN / TEST SPLIT")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load TF-IDF features
# ------------------------------------------------------------

print("\nLoading TF-IDF features...")

X = load_npz(FEATURE_FILE)

print("✅ TF-IDF features loaded")
print(f"Feature matrix shape: {X.shape}")


# ------------------------------------------------------------
# 2. Load labels
# ------------------------------------------------------------

print("\nLoading labels...")

labels = pd.read_csv(LABEL_FILE)

print("✅ Labels loaded")
print(f"Number of labels: {len(labels)}")


# ------------------------------------------------------------
# 3. Select target
# ------------------------------------------------------------

y = labels["text_info"]

print("\nTarget variable:")
print("text_info")

print("\nClass distribution:")
print(y.value_counts())


# ------------------------------------------------------------
# 4. Train/Test split
# ------------------------------------------------------------

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("✅ Train/test split completed")


# ------------------------------------------------------------
# 5. Display dataset sizes
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA SPLIT INFORMATION")
print("=" * 70)

print(f"Total samples : {len(y)}")
print(f"Training data : {len(y_train)}")
print(f"Testing data  : {len(y_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ------------------------------------------------------------
# 6. Save training and testing features
# ------------------------------------------------------------

print("\nSaving split datasets...")

save_npz(
    OUTPUT_DIR / "X_train.npz",
    X_train
)

save_npz(
    OUTPUT_DIR / "X_test.npz",
    X_test
)

y_train.to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)

print("✅ X_train saved")
print("✅ X_test saved")
print("✅ y_train saved")
print("✅ y_test saved")


# ------------------------------------------------------------
# 7. Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT COMPLETED")
print("=" * 70)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")
print(f"Features        : {X_train.shape[1]}")

print("\n" + "=" * 70)
print("✅ DATA READY FOR MACHINE LEARNING")
print("=" * 70)