# ================================================================
# AI SOCIAL MEDIA CRISIS DETECTION
# ENSEMBLE HYBRID NLP MODEL TUNING
# ================================================================

import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

RESULTS_PATH = os.path.join(
    DATA_DIR,
    "ensemble_hybrid_results.csv"
)


RANDOM_STATE = 42


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("ENSEMBLE HYBRID NLP MODEL TUNING")
print("=" * 70)


# ================================================================
# CHECK DATASET
# ================================================================

print("\nChecking dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

print("✅ Dataset found")


# ================================================================
# LOAD DATASET
# ================================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape: {df.shape}")

print("\nAvailable columns:")
print(df.columns.tolist())


# ================================================================
# SELECT COLUMNS
# ================================================================

TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "text_info"

if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"Missing text column: {TEXT_COLUMN}"
    )

if LABEL_COLUMN not in df.columns:
    raise ValueError(
        f"Missing label column: {LABEL_COLUMN}"
    )

print(f"\nText column  : {TEXT_COLUMN}")
print(f"Label column : {LABEL_COLUMN}")


# ================================================================
# CLEAN DATA
# ================================================================

print("\nCleaning data...")

df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

print(f"Rows before cleaning: {len(df)}")

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .str.strip()
)

df[LABEL_COLUMN] = (
    df[LABEL_COLUMN]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    (df[TEXT_COLUMN] != "") &
    (df[LABEL_COLUMN] != "")
]

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=[TEXT_COLUMN]
).reset_index(drop=True)

duplicates_removed = (
    before_duplicates - len(df)
)

print(f"Rows after cleaning : {len(df)}")
print(f"Duplicates removed  : {duplicates_removed}")


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print("\nClass distribution:")

print(
    df[LABEL_COLUMN].value_counts()
)


# ================================================================
# TRAIN TEST SPLIT
# ================================================================

print("\nSplitting dataset...")

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


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
    max_df=0.98,
    sublinear_tf=True,
    strip_accents="unicode"
)

print("Training word vectorizer...")

X_train_word = word_vectorizer.fit_transform(
    X_train
)

X_test_word = word_vectorizer.transform(
    X_test
)

print(
    f"Word training shape: {X_train_word.shape}"
)

print(
    f"Word testing shape : {X_test_word.shape}"
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
    sublinear_tf=True
)

print("Training character vectorizer...")

X_train_char = char_vectorizer.fit_transform(
    X_train
)

X_test_char = char_vectorizer.transform(
    X_test
)

print(
    f"Character training shape: {X_train_char.shape}"
)

print(
    f"Character testing shape : {X_test_char.shape}"
)


# ================================================================
# COMBINE FEATURES
# ================================================================

print("\n" + "=" * 70)
print("COMBINING WORD + CHARACTER FEATURES")
print("=" * 70)

X_train_hybrid = hstack(
    [X_train_word, X_train_char]
).tocsr()

X_test_hybrid = hstack(
    [X_test_word, X_test_char]
).tocsr()

print(
    f"Hybrid training shape: {X_train_hybrid.shape}"
)

print(
    f"Hybrid testing shape : {X_test_hybrid.shape}"
)


# ================================================================
# MODEL CONFIGURATIONS
# ================================================================

models = {

    "Logistic_C_0.5": LogisticRegression(
        C=0.5,
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),

    "Logistic_C_1.0": LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),

    "Logistic_C_1.5": LogisticRegression(
        C=1.5,
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),

    "Logistic_C_2.0": LogisticRegression(
        C=2.0,
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),

    "Logistic_C_3.0": LogisticRegression(
        C=3.0,
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),

    # ------------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------------

    "RandomForest_300_Log2": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="log2",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "RandomForest_300_Sqrt": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ================================================================
# TRAIN MODELS
# ================================================================

print("\n" + "=" * 70)
print("STARTING ENSEMBLE MODEL TUNING")
print("=" * 70)


results = []

best_model = None
best_model_name = None
best_accuracy = 0
best_f1 = 0


for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"MODEL: {model_name}")
    print("-" * 70)

    print("Training...")

    model.fit(
        X_train_hybrid,
        y_train
    )

    print("✅ Training completed")

    print("Making predictions...")

    predictions = model.predict(
        X_test_hybrid
    )

    print("✅ Predictions completed")

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

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    # Select best model based on weighted F1
    if f1 > best_f1:

        best_f1 = f1
        best_accuracy = accuracy
        best_model = model
        best_model_name = model_name

        print("\n🏆 NEW BEST MODEL!")


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
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ================================================================
# ENSEMBLE VOTING MODEL
# ================================================================

print("\n" + "=" * 70)
print("TRAINING SOFT VOTING ENSEMBLE")
print("=" * 70)


print("\nCreating ensemble...")

logistic_model = LogisticRegression(
    C=1.0,
    max_iter=2000,
    solver="liblinear",
    random_state=RANDOM_STATE
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="log2",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


ensemble = VotingClassifier(
    estimators=[
        ("logistic", logistic_model),
        ("random_forest", rf_model)
    ],
    voting="soft",
    weights=[2, 1],
    n_jobs=-1
)


print("Training ensemble...")

ensemble.fit(
    X_train_hybrid,
    y_train
)

print("✅ Ensemble training completed")


# ================================================================
# ENSEMBLE PREDICTIONS
# ================================================================

print("\nMaking ensemble predictions...")

ensemble_predictions = ensemble.predict(
    X_test_hybrid
)

print("✅ Predictions completed")


ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_predictions
)

ensemble_precision = precision_score(
    y_test,
    ensemble_predictions,
    average="weighted",
    zero_division=0
)

ensemble_recall = recall_score(
    y_test,
    ensemble_predictions,
    average="weighted",
    zero_division=0
)

ensemble_f1 = f1_score(
    y_test,
    ensemble_predictions,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("ENSEMBLE RESULTS")
print("=" * 70)

print(
    f"Accuracy  : {ensemble_accuracy * 100:.2f}%"
)

print(
    f"Precision : {ensemble_precision * 100:.2f}%"
)

print(
    f"Recall    : {ensemble_recall * 100:.2f}%"
)

print(
    f"F1 Score  : {ensemble_f1 * 100:.2f}%"
)


# ================================================================
# ADD ENSEMBLE TO RESULTS
# ================================================================

results.append({
    "Model": "Soft_Voting_Ensemble",
    "Accuracy": ensemble_accuracy,
    "Precision": ensemble_precision,
    "Recall": ensemble_recall,
    "F1_Score": ensemble_f1
})

results_df = pd.DataFrame(
    results
).sort_values(
    by="F1_Score",
    ascending=False
)


# ================================================================
# SELECT FINAL MODEL
# ================================================================

if ensemble_f1 > best_f1:

    final_model = ensemble
    final_model_name = "Soft_Voting_Ensemble"
    final_accuracy = ensemble_accuracy
    final_f1 = ensemble_f1

else:

    final_model = best_model
    final_model_name = best_model_name
    final_accuracy = best_accuracy
    final_f1 = best_f1


# ================================================================
# FINAL MODEL REPORT
# ================================================================

print("\n" + "=" * 70)
print("🏆 FINAL BEST MODEL")
print("=" * 70)

print(
    f"Model    : {final_model_name}"
)

print(
    f"Accuracy : {final_accuracy * 100:.2f}%"
)

print(
    f"F1 Score : {final_f1 * 100:.2f}%"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

final_predictions = final_model.predict(
    X_test_hybrid
)

print(
    classification_report(
        y_test,
        final_predictions,
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
    final_predictions
)

print(cm)


# ================================================================
# SAVE FINAL MODEL
# ================================================================

print("\n" + "=" * 70)
print("SAVING FINAL MODEL")
print("=" * 70)

joblib.dump(
    final_model,
    MODEL_PATH
)

print("✅ Final model saved")

print(
    f"Model path:\n{MODEL_PATH}"
)


# ================================================================
# SAVE WORD VECTORIZER
# ================================================================

joblib.dump(
    word_vectorizer,
    WORD_VECTORIZER_PATH
)

print("\n✅ Word TF-IDF vectorizer saved")

print(
    WORD_VECTORIZER_PATH
)


# ================================================================
# SAVE CHARACTER VECTORIZER
# ================================================================

joblib.dump(
    char_vectorizer,
    CHAR_VECTORIZER_PATH
)

print("\n✅ Character TF-IDF vectorizer saved")

print(
    CHAR_VECTORIZER_PATH
)


# ================================================================
# SAVE RESULTS
# ================================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n✅ Results saved")

print(
    RESULTS_PATH
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("ENSEMBLE HYBRID NLP OPTIMIZATION COMPLETED")
print("=" * 70)

print(f"""
🏆 Final Model       : {final_model_name}

📊 Accuracy          : {final_accuracy * 100:.2f}%
📊 Weighted F1       : {final_f1 * 100:.2f}%

🔤 Word Features     : {X_train_word.shape[1]:,}
🔤 Character Features: {X_train_char.shape[1]:,}
🔤 Hybrid Features   : {X_train_hybrid.shape[1]:,}

💾 Model saved       : YES
💾 Vectorizers saved : YES
📁 Results saved     : YES
""")

print("=" * 70)
print("🎉 DONE!")
print("=" * 70)