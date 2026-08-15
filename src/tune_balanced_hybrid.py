import os
import joblib
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

from sklearn.model_selection import train_test_split


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "crisis_mmd_nlp.csv"
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
print("BALANCED HYBRID NLP MODEL TUNING")
print("=" * 70)


# ================================================================
# CHECK DATASET
# ================================================================

print("\nChecking dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

print("✅ Dataset found")


# ================================================================
# LOAD DATASET
# ================================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

print("\nAvailable columns:")
print(df.columns.tolist())


# ================================================================
# SELECT COLUMNS
# ================================================================

TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "text_info"

print(f"\nText column  : {TEXT_COLUMN}")
print(f"Label column : {LABEL_COLUMN}")


# ================================================================
# VALIDATE COLUMNS
# ================================================================

if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"Text column '{TEXT_COLUMN}' not found."
    )

if LABEL_COLUMN not in df.columns:
    raise ValueError(
        f"Label column '{LABEL_COLUMN}' not found."
    )


# ================================================================
# CLEAN DATA
# ================================================================

print("\nCleaning data...")

rows_before = len(df)

df = df[
    df[TEXT_COLUMN].notna() &
    df[LABEL_COLUMN].notna()
].copy()

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .astype(str)
    .str.strip()
)

df[LABEL_COLUMN] = (
    df[LABEL_COLUMN]
    .astype(str)
    .str.strip()
)

df = df[
    (df[TEXT_COLUMN] != "") &
    (df[LABEL_COLUMN] != "")
]

# Remove duplicate tweets
df = df.drop_duplicates(
    subset=[TEXT_COLUMN]
)

rows_after = len(df)

print(
    f"Rows before cleaning: {rows_before}"
)

print(
    f"Rows after cleaning : {rows_after}"
)

print(
    f"Duplicates removed  : "
    f"{rows_before - rows_after}"
)


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print("\nClass distribution:")

print(
    df[LABEL_COLUMN].value_counts()
)


# ================================================================
# PREPARE X AND Y
# ================================================================

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]


# ================================================================
# SPLIT DATA
# ================================================================

print("\nSplitting dataset...")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
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

print("\n" + "=" * 70)
print("CREATING WORD TF-IDF")
print("=" * 70)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=40000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode"
)

print("Training word vectorizer...")

X_train_word = word_vectorizer.fit_transform(
    X_train_text
)

X_test_word = word_vectorizer.transform(
    X_test_text
)

print(
    f"Word training shape: "
    f"{X_train_word.shape}"
)

print(
    f"Word testing shape : "
    f"{X_test_word.shape}"
)


# ================================================================
# CHARACTER TF-IDF
# ================================================================

print("\n" + "=" * 70)
print("CREATING CHARACTER TF-IDF")
print("=" * 70)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=40000,
    min_df=2,
    max_df=0.98,
    sublinear_tf=True
)

print("Training character vectorizer...")

X_train_char = char_vectorizer.fit_transform(
    X_train_text
)

X_test_char = char_vectorizer.transform(
    X_test_text
)

print(
    f"Character training shape: "
    f"{X_train_char.shape}"
)

print(
    f"Character testing shape : "
    f"{X_test_char.shape}"
)


# ================================================================
# COMBINE WORD + CHARACTER FEATURES
# ================================================================

print("\n" + "=" * 70)
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
    f"Hybrid training shape: "
    f"{X_train_hybrid.shape}"
)

print(
    f"Hybrid testing shape : "
    f"{X_test_hybrid.shape}"
)


# ================================================================
# MODEL CONFIGURATIONS
# ================================================================

models = [

    {
        "name": "Balanced_C_0.5",
        "C": 0.5,
        "class_weight": "balanced"
    },

    {
        "name": "Balanced_C_0.75",
        "C": 0.75,
        "class_weight": "balanced"
    },

    {
        "name": "Balanced_C_1.0",
        "C": 1.0,
        "class_weight": "balanced"
    },

    {
        "name": "Balanced_C_1.5",
        "C": 1.5,
        "class_weight": "balanced"
    },

    {
        "name": "Balanced_C_2.0",
        "C": 2.0,
        "class_weight": "balanced"
    },

    {
        "name": "Balanced_C_3.0",
        "C": 3.0,
        "class_weight": "balanced"
    },

    {
        "name": "Balanced_C_5.0",
        "C": 5.0,
        "class_weight": "balanced"
    },

    {
        "name": "Custom_Weight_1.10",
        "C": 1.0,
        "class_weight": {
            "informative": 1.0,
            "not_informative": 1.10
        }
    },

    {
        "name": "Custom_Weight_1.25",
        "C": 1.0,
        "class_weight": {
            "informative": 1.0,
            "not_informative": 1.25
        }
    },

    {
        "name": "Custom_Weight_1.50",
        "C": 1.0,
        "class_weight": {
            "informative": 1.0,
            "not_informative": 1.50
        }
    },

    {
        "name": "Custom_Weight_1.75",
        "C": 1.0,
        "class_weight": {
            "informative": 1.0,
            "not_informative": 1.75
        }
    }
]


# ================================================================
# START TUNING
# ================================================================

print("\n" + "=" * 70)
print("STARTING BALANCED MODEL TUNING")
print("=" * 70)


results = []

best_model = None
best_model_name = None

best_accuracy = 0.0
best_f1 = 0.0


# ================================================================
# TRAIN EACH MODEL
# ================================================================

for index, config in enumerate(
    models,
    start=1
):

    print("\n" + "-" * 70)

    print(
        f"MODEL {index}/{len(models)}"
    )

    print(
        f"Configuration: {config['name']}"
    )

    print("-" * 70)

    print(
        f"C            : {config['C']}"
    )

    print(
        f"Class weight : {config['class_weight']}"
    )

    # ------------------------------------------------------------
    # CREATE MODEL
    # ------------------------------------------------------------

    model = LogisticRegression(
        C=config["C"],
        class_weight=config["class_weight"],
        max_iter=3000,
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    print("\nTraining...")

    model.fit(
        X_train_hybrid,
        y_train
    )

    print(
        "✅ Training completed"
    )

    # ------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------

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
        average="weighted",
        zero_division=0
    )

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
    # SAVE RESULTS
    # ------------------------------------------------------------

    results.append(
        {
            "Model": config["name"],
            "C": config["C"],
            "Class_Weight": str(
                config["class_weight"]
            ),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        }
    )

    # ------------------------------------------------------------
    # BEST MODEL
    # ------------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1
        best_accuracy = accuracy

        best_model = model
        best_model_name = config["name"]

        print(
            "\n🏆 NEW BEST MODEL!"
        )


# ================================================================
# RESULTS TABLE
# ================================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

print("\n" + "=" * 70)
print("BALANCED MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ================================================================
# BEST MODEL INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("🏆 BEST BALANCED MODEL")
print("=" * 70)

print(
    f"Model    : {best_model_name}"
)

print(
    f"Accuracy : {best_accuracy * 100:.2f}%"
)

print(
    f"F1 Score : {best_f1 * 100:.2f}%"
)


# ================================================================
# BEST MODEL PREDICTIONS
# ================================================================

best_predictions = best_model.predict(
    X_test_hybrid
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

print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)


# ================================================================
# SAVE MODEL
# ================================================================

print("\n" + "=" * 70)
print("SAVING BEST BALANCED MODEL")
print("=" * 70)


model_path = os.path.join(
    OUTPUT_DIR,
    "balanced_hybrid_crisis_model.joblib"
)

word_vectorizer_path = os.path.join(
    OUTPUT_DIR,
    "balanced_hybrid_word_tfidf.pkl"
)

char_vectorizer_path = os.path.join(
    OUTPUT_DIR,
    "balanced_hybrid_char_tfidf.pkl"
)

results_path = os.path.join(
    OUTPUT_DIR,
    "balanced_hybrid_results.csv"
)


# Save model
joblib.dump(
    best_model,
    model_path
)

print(
    "✅ Model saved successfully!"
)

print(
    f"Model path:\n{model_path}"
)


# Save word vectorizer
joblib.dump(
    word_vectorizer,
    word_vectorizer_path
)

print(
    "\n✅ Word TF-IDF vectorizer saved!"
)

print(
    word_vectorizer_path
)


# Save character vectorizer
joblib.dump(
    char_vectorizer,
    char_vectorizer_path
)

print(
    "\n✅ Character TF-IDF vectorizer saved!"
)

print(
    char_vectorizer_path
)


# Save results
results_df.to_csv(
    results_path,
    index=False
)

print(
    "\n✅ Results saved!"
)

print(
    results_path
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("BALANCED HYBRID NLP TUNING COMPLETED")
print("=" * 70)

print(
    f"""
🏆 Best Model       : {best_model_name}

📊 Accuracy         : {best_accuracy * 100:.2f}%
📊 Weighted F1      : {best_f1 * 100:.2f}%

🔤 Word Features     : {X_train_word.shape[1]:,}
🔤 Character Features: {X_train_char.shape[1]:,}
🔤 Hybrid Features   : {X_train_hybrid.shape[1]:,}

💾 Model saved       : YES
💾 Vectorizers saved : YES
📁 Results saved     : YES
"""
)

print("=" * 70)
print("🎉 BALANCED HYBRID NLP OPTIMIZATION COMPLETED!")
print("=" * 70)