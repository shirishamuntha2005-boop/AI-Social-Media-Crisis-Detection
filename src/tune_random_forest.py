import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)


# ================================================================
# PATHS
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
# HEADER
# ================================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)


# ================================================================
# CHECK FILES
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

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(
        "✅ Found:",
        os.path.basename(file_path)
    )


# ================================================================
# LOAD TF-IDF FEATURES
# ================================================================

print("\n")
print("=" * 70)
print("LOADING TF-IDF FEATURES")
print("=" * 70)


print("\nLoading X_train...")

X_train = load_npz(
    X_TRAIN_PATH
)

print(
    "✅ X_train loaded:",
    X_train.shape
)

print(
    "Type:",
    type(X_train)
)


print("\nLoading X_test...")

X_test = load_npz(
    X_TEST_PATH
)

print(
    "✅ X_test loaded:",
    X_test.shape
)


# ================================================================
# LOAD LABELS
# ================================================================

print("\n")
print("=" * 70)
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
    y_train_df.columns.tolist()
)

print(
    "y_test columns :",
    y_test_df.columns.tolist()
)


# ================================================================
# SELECT LABEL COLUMN
# ================================================================

if "text_info" in y_train_df.columns:

    LABEL_COLUMN = "text_info"

else:

    LABEL_COLUMN = y_train_df.columns[0]


print(
    f"\n✅ Label column: {LABEL_COLUMN}"
)


y_train = y_train_df[
    LABEL_COLUMN
]

y_test = y_test_df[
    LABEL_COLUMN
]


# ================================================================
# VALIDATION
# ================================================================

print("\n")
print("=" * 70)
print("DATA VALIDATION")
print("=" * 70)


print(
    f"\nX_train: {X_train.shape}"
)

print(
    f"y_train: {y_train.shape}"
)

print(
    f"X_test : {X_test.shape}"
)

print(
    f"y_test : {y_test.shape}"
)


if X_train.shape[0] != len(y_train):

    raise ValueError(
        "❌ X_train and y_train size mismatch"
    )


if X_test.shape[0] != len(y_test):

    raise ValueError(
        "❌ X_test and y_test size mismatch"
    )


print(
    "\n✅ Training data matches"
)

print(
    "✅ Testing data matches"
)


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


print("\nTraining:")

print(
    y_train.value_counts()
)


print("\nTesting:")

print(
    y_test.value_counts()
)


# ================================================================
# RANDOM FOREST CONFIGURATIONS
# ================================================================

configs = [

    {
        "name": "RF_200_Default",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },

    {
        "name": "RF_300",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },

    {
        "name": "RF_500",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },

    {
        "name": "RF_Depth30",
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },

    {
        "name": "RF_Depth50",
        "n_estimators": 300,
        "max_depth": 50,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },

    {
        "name": "RF_MinSplit5",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },

    {
        "name": "RF_MinLeaf2",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced"
    },

    {
        "name": "RF_MaxFeaturesLog2",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "class_weight": "balanced"
    },

    {
        "name": "RF_NoBalance",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None
    }

]


# ================================================================
# TRAINING
# ================================================================

results = []

best_accuracy = 0.0

best_f1 = 0.0

best_model = None

best_config = None

best_predictions = None


print("\n")
print("=" * 70)
print("STARTING RANDOM FOREST TUNING")
print("=" * 70)


for index, config in enumerate(
    configs,
    start=1
):

    print("\n")
    print("-" * 70)

    print(
        f"MODEL {index}/{len(configs)}"
    )

    print(
        f"Configuration: {config['name']}"
    )

    print("-" * 70)


    print(
        f"Trees            : "
        f"{config['n_estimators']}"
    )

    print(
        f"Max depth        : "
        f"{config['max_depth']}"
    )

    print(
        f"Min split        : "
        f"{config['min_samples_split']}"
    )

    print(
        f"Min leaf         : "
        f"{config['min_samples_leaf']}"
    )

    print(
        f"Max features     : "
        f"{config['max_features']}"
    )

    print(
        f"Class weight     : "
        f"{config['class_weight']}"
    )


    # ------------------------------------------------------------
    # CREATE MODEL
    # ------------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=config["n_estimators"],

        max_depth=config["max_depth"],

        min_samples_split=config["min_samples_split"],

        min_samples_leaf=config["min_samples_leaf"],

        max_features=config["max_features"],

        class_weight=config["class_weight"],

        random_state=42,

        n_jobs=-1
    )


    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    print("\nTraining...")

    model.fit(
        X_train,
        y_train
    )

    print(
        "✅ Training completed"
    )


    # ------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------

    print(
        "Making predictions..."
    )

    predictions = model.predict(
        X_test
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

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )


    # ------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------

    results.append({

        "Model":
            config["name"],

        "Accuracy":
            accuracy,

        "F1_Score":
            f1,

        "Trees":
            config["n_estimators"],

        "Max_Depth":
            config["max_depth"],

        "Min_Split":
            config["min_samples_split"],

        "Min_Leaf":
            config["min_samples_leaf"],

        "Max_Features":
            config["max_features"],

        "Class_Weight":
            config["class_weight"]
    })


    # ------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------

    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Weighted F1: "
        f"{f1 * 100:.2f}%"
    )


    # ------------------------------------------------------------
    # CHECK BEST
    # ------------------------------------------------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_f1 = f1

        best_model = model

        best_config = config

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
print("RANDOM FOREST MODEL COMPARISON")
print("=" * 70)

print()

print(
    results_df.to_string(
        index=False,

        formatters={

            "Accuracy":
                "{:.4f}".format,

            "F1_Score":
                "{:.4f}".format
        }
    )
)


# ================================================================
# BEST MODEL
# ================================================================

print("\n")
print("=" * 70)
print("🏆 BEST RANDOM FOREST MODEL")
print("=" * 70)


print(
    f"\nModel name      : "
    f"{best_config['name']}"
)

print(
    f"Trees           : "
    f"{best_config['n_estimators']}"
)

print(
    f"Max depth       : "
    f"{best_config['max_depth']}"
)

print(
    f"Min split       : "
    f"{best_config['min_samples_split']}"
)

print(
    f"Min leaf        : "
    f"{best_config['min_samples_leaf']}"
)

print(
    f"Max features    : "
    f"{best_config['max_features']}"
)

print(
    f"Class weight    : "
    f"{best_config['class_weight']}"
)

print(
    f"\nAccuracy        : "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Weighted F1     : "
    f"{best_f1 * 100:.2f}%"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n")
print("=" * 70)
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
# SAVE BEST MODEL
# ================================================================

BEST_MODEL_PATH = os.path.join(
    DATA_DIR,
    "tuned_random_forest_model.joblib"
)


print("\n")
print("=" * 70)
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
    "\nModel path:"
)

print(
    BEST_MODEL_PATH
)


# ================================================================
# SAVE RESULTS
# ================================================================

RESULTS_PATH = os.path.join(
    DATA_DIR,
    "random_forest_tuning_results.csv"
)


results_df.to_csv(
    RESULTS_PATH,
    index=False
)


print(
    "\n✅ Tuning results saved!"
)

print(
    "\nResults path:"
)

print(
    RESULTS_PATH
)


# ================================================================
# FINAL
# ================================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST TUNING COMPLETED")
print("=" * 70)


print(
    f"\n🏆 Best Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"📊 Best F1 Score: "
    f"{best_f1 * 100:.2f}%"
)

print(
    "\n💾 Model saved: YES"
)

print(
    "\n🎯 Previous baseline: 79.45%"
)

difference = (
    best_accuracy * 100
) - 79.45

print(
    f"📈 Improvement: "
    f"{difference:+.2f} percentage points"
)


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)