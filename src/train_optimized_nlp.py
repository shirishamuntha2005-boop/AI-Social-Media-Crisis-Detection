import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score


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

DATA_PATH = os.path.join(
    DATA_DIR,
    "crisis_mmd_nlp.csv"
)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("ADVANCED NLP OPTIMIZATION")
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

print(
    f"Dataset shape: {df.shape}"
)

print("\nAvailable columns:")

print(
    df.columns.tolist()
)


# ================================================================
# FIND TEXT COLUMN
# ================================================================

possible_text_columns = [
    "tweet_text",
    "text",
    "clean_text"
]

text_column = None

for column in possible_text_columns:

    if column in df.columns:

        text_column = column
        break


if text_column is None:

    raise ValueError(
        "❌ Could not find the tweet text column."
    )


# ================================================================
# FIND LABEL COLUMN
# ================================================================

label_column = "text_info"

if label_column not in df.columns:

    raise ValueError(
        "❌ Column 'text_info' not found."
    )


print(
    f"\nText column : {text_column}"
)

print(
    f"Label column: {label_column}"
)


# ================================================================
# SELECT REQUIRED COLUMNS
# ================================================================

df = df[
    [text_column, label_column]
].copy()


# ================================================================
# REMOVE MISSING VALUES
# ================================================================

print("\nCleaning data...")

before = len(df)

df = df.dropna()

df[text_column] = (
    df[text_column]
    .astype(str)
    .str.strip()
)

df = df[
    df[text_column] != ""
]

after = len(df)

print(
    f"Rows before cleaning: {before}"
)

print(
    f"Rows after cleaning : {after}"
)


# ================================================================
# REMOVE DUPLICATE POSTS
# ================================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=[text_column]
)

after_duplicates = len(df)

print(
    f"Duplicate rows removed: "
    f"{before_duplicates - after_duplicates}"
)


# ================================================================
# X AND Y
# ================================================================

X = df[text_column]

y = df[label_column].astype(str)


# ================================================================
# CLASS DISTRIBUTION
# ================================================================

print("\nClass distribution:")

print(
    y.value_counts()
)


# ================================================================
# TRAIN TEST SPLIT
# ================================================================

print("\nSplitting dataset...")

X_train_text, X_test_text, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    f"Training samples: {len(X_train_text)}"
)

print(
    f"Testing samples : {len(X_test_text)}"
)


# ================================================================
# TF-IDF CONFIGURATIONS
# ================================================================

vectorizers = {

    "TFIDF_WORD_BIGRAM":

        TfidfVectorizer(

            lowercase=True,

            strip_accents="unicode",

            ngram_range=(1, 2),

            min_df=2,

            max_df=0.95,

            max_features=20000,

            sublinear_tf=True
        ),


    "TFIDF_WORD_TRIGRAM":

        TfidfVectorizer(

            lowercase=True,

            strip_accents="unicode",

            ngram_range=(1, 3),

            min_df=2,

            max_df=0.95,

            max_features=30000,

            sublinear_tf=True
        ),


    "TFIDF_CHAR":

        TfidfVectorizer(

            analyzer="char",

            ngram_range=(3, 5),

            min_df=2,

            max_features=30000,

            sublinear_tf=True
        )
}


# ================================================================
# MODELS
# ================================================================

models = {

    "Logistic Regression":

        LogisticRegression(

            C=2.0,

            max_iter=3000,

            class_weight="balanced",

            solver="liblinear",

            random_state=42
        ),


    "Linear SVM":

        LinearSVC(

            C=1.0,

            class_weight="balanced",

            max_iter=10000,

            random_state=42
        )
}


# ================================================================
# RESULTS
# ================================================================

results = []

best_accuracy = 0.0

best_f1 = 0.0

best_model = None

best_vectorizer = None

best_model_name = None

best_vectorizer_name = None

best_predictions = None


# ================================================================
# TRAINING LOOP
# ================================================================

for vectorizer_name, vectorizer in vectorizers.items():

    print("\n")

    print("=" * 70)

    print(
        f"TF-IDF CONFIGURATION: "
        f"{vectorizer_name}"
    )

    print("=" * 70)


    # ------------------------------------------------------------
    # CREATE TF-IDF FEATURES
    # ------------------------------------------------------------

    print("\nCreating TF-IDF features...")

    X_train = vectorizer.fit_transform(
        X_train_text
    )

    X_test = vectorizer.transform(
        X_test_text
    )


    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"X_test shape : {X_test.shape}"
    )


    # ------------------------------------------------------------
    # TRAIN MODELS
    # ------------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "-" * 70)

        print(
            f"Training model: {model_name}"
        )

        print("-" * 70)


        model.fit(
            X_train,
            y_train
        )


        print(
            "✅ Training completed"
        )


        # --------------------------------------------------------
        # PREDICTIONS
        # --------------------------------------------------------

        predictions = model.predict(
            X_test
        )


        print(
            "✅ Predictions completed"
        )


        # --------------------------------------------------------
        # METRICS
        # --------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )


        f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )


        # --------------------------------------------------------
        # STORE RESULTS
        # --------------------------------------------------------

        results.append({

            "Vectorizer": vectorizer_name,

            "Model": model_name,

            "Accuracy": accuracy,

            "F1_Score": f1,

            "Features": X_train.shape[1]
        })


        # --------------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------------

        print(
            f"\nAccuracy : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"F1 Score : "
            f"{f1 * 100:.2f}%"
        )


        # --------------------------------------------------------
        # CHECK BEST MODEL
        # --------------------------------------------------------

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_f1 = f1

            best_model = model

            best_vectorizer = vectorizer

            best_model_name = model_name

            best_vectorizer_name = vectorizer_name

            best_predictions = predictions


# ================================================================
# RESULTS TABLE
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

print("FINAL MODEL COMPARISON")

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

print("🏆 BEST MODEL")

print("=" * 70)


print(
    f"\nVectorizer : "
    f"{best_vectorizer_name}"
)

print(
    f"Model      : "
    f"{best_model_name}"
)

print(
    f"Accuracy   : "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"F1 Score   : "
    f"{best_f1 * 100:.2f}%"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n")

print("=" * 70)

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
# SAVE BEST MODEL
# ================================================================

BEST_MODEL_PATH = os.path.join(

    DATA_DIR,

    "optimized_crisis_model.joblib"
)


BEST_VECTORIZER_PATH = os.path.join(

    DATA_DIR,

    "optimized_tfidf_vectorizer.pkl"
)


print("\nSaving best model...")

joblib.dump(

    best_model,

    BEST_MODEL_PATH
)


print(
    "✅ Best model saved"
)


print("\nSaving TF-IDF vectorizer...")

joblib.dump(

    best_vectorizer,

    BEST_VECTORIZER_PATH
)


print(
    "✅ Vectorizer saved"
)


# ================================================================
# SAVE RESULTS
# ================================================================

RESULTS_PATH = os.path.join(

    DATA_DIR,

    "optimized_model_comparison.csv"
)


results_df.to_csv(

    RESULTS_PATH,

    index=False
)


print(
    "✅ Comparison results saved"
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")

print("=" * 70)

print("OPTIMIZATION COMPLETED")

print("=" * 70)


print(
    f"\n🏆 Best Model: "
    f"{best_model_name}"
)

print(
    f"🏆 Best TF-IDF: "
    f"{best_vectorizer_name}"
)

print(
    f"📊 Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"📊 F1 Score: "
    f"{best_f1 * 100:.2f}%"
)


print(
    "\n💾 Model:"
)

print(
    BEST_MODEL_PATH
)


print(
    "\n💾 Vectorizer:"
)

print(
    BEST_VECTORIZER_PATH
)


print("\n")

print("=" * 70)

print("DONE")

print("=" * 70)