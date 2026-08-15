



import os
import joblib
import numpy as np
from scipy.sparse import hstack


# ================================================================
# AI SOCIAL MEDIA CRISIS DETECTION
# OPTIMIZED HYBRID NLP PREDICTION SYSTEM
# ================================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("OPTIMIZED HYBRID NLP PREDICTION SYSTEM")
print("=" * 70)


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_PATH = os.path.join(
    PROCESSED_DIR,
    "tuned_hybrid_crisis_model.joblib"
)

WORD_VECTORIZER_PATH = os.path.join(
    PROCESSED_DIR,
    "tuned_hybrid_word_tfidf.pkl"
)

CHAR_VECTORIZER_PATH = os.path.join(
    PROCESSED_DIR,
    "tuned_hybrid_char_tfidf.pkl"
)

THRESHOLD_PATH = os.path.join(
    PROCESSED_DIR,
    "best_prediction_threshold.txt"
)


# ================================================================
# CHECK REQUIRED FILES
# ================================================================

print("\nChecking required files...")

required_files = {
    "Hybrid model": MODEL_PATH,
    "Word TF-IDF vectorizer": WORD_VECTORIZER_PATH,
    "Character TF-IDF vectorizer": CHAR_VECTORIZER_PATH,
    "Prediction threshold": THRESHOLD_PATH
}

for name, path in required_files.items():

    if os.path.exists(path):
        print(f"✅ {name} found")
    else:
        print(f"❌ {name} NOT FOUND")
        print(f"Expected path: {path}")
        raise FileNotFoundError(path)


# ================================================================
# LOAD MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING OPTIMIZED MODEL")
print("=" * 70)

model = joblib.load(MODEL_PATH)

print("✅ Hybrid Logistic Regression model loaded successfully")


# ================================================================
# LOAD WORD VECTORIZER
# ================================================================

print("\nLoading Word TF-IDF vectorizer...")

word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)

print("✅ Word TF-IDF vectorizer loaded successfully")


# ================================================================
# LOAD CHARACTER VECTORIZER
# ================================================================

print("\nLoading Character TF-IDF vectorizer...")

char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)

print("✅ Character TF-IDF vectorizer loaded successfully")


# ================================================================
# LOAD OPTIMIZED THRESHOLD
# ================================================================

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read().strip())

print(f"✅ Optimized threshold loaded: {threshold}")


# ================================================================
# MODEL INFORMATION
# ================================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(f"Model type              : {type(model).__name__}")
print(f"Word features           : {len(word_vectorizer.vocabulary_)}")
print(f"Character features      : {len(char_vectorizer.vocabulary_)}")
print(f"Total hybrid features   : "
      f"{len(word_vectorizer.vocabulary_) + len(char_vectorizer.vocabulary_)}")
print(f"Prediction threshold    : {threshold}")
print(f"Classes                 : {list(model.classes_)}")


# ================================================================
# TEXT PREPROCESSING
# ================================================================

def clean_text(text):

    text = str(text).lower()

    return text


# ================================================================
# PREDICTION FUNCTION
# ================================================================

def predict_crisis(text):

    # Clean text
    cleaned_text = clean_text(text)

    # ------------------------------------------------------------
    # WORD TF-IDF
    # ------------------------------------------------------------

    word_features = word_vectorizer.transform([cleaned_text])

    # ------------------------------------------------------------
    # CHARACTER TF-IDF
    # ------------------------------------------------------------

    char_features = char_vectorizer.transform([cleaned_text])

    # ------------------------------------------------------------
    # COMBINE WORD + CHARACTER FEATURES
    # ------------------------------------------------------------

    hybrid_features = hstack(
        [word_features, char_features]
    )

    # ------------------------------------------------------------
    # GET PROBABILITIES
    # ------------------------------------------------------------

    probabilities = model.predict_proba(hybrid_features)[0]

    classes = list(model.classes_)

    informative_index = classes.index("informative")
    not_informative_index = classes.index("not_informative")

    informative_probability = probabilities[informative_index]
    not_informative_probability = probabilities[not_informative_index]

    # ------------------------------------------------------------
    # APPLY OPTIMIZED THRESHOLD
    # ------------------------------------------------------------

    if informative_probability >= threshold:

        prediction = "informative"
        confidence = informative_probability

    else:

        prediction = "not_informative"
        confidence = not_informative_probability

    return (
        prediction,
        confidence,
        informative_probability,
        not_informative_probability
    )


# ================================================================
# CRISIS DETECTION
# ================================================================

print("\n" + "=" * 70)
print("CRISIS DETECTION")
print("=" * 70)

print("""
Enter a social media post below.
Type 'exit' to close the program.
""")

print("-" * 70)


while True:

    try:

        post = input("Enter post: ").strip()

        # --------------------------------------------------------
        # EXIT
        # --------------------------------------------------------

        if post.lower() == "exit":

            print("\nExiting program...")
            print("Thank you for using AI Social Media Crisis Detection.")
            break

        # --------------------------------------------------------
        # EMPTY INPUT
        # --------------------------------------------------------

        if not post:

            print("⚠️ Please enter a social media post.")
            print("-" * 70)
            continue

        # --------------------------------------------------------
        # PREDICTION
        # --------------------------------------------------------

        (
            prediction,
            confidence,
            informative_probability,
            not_informative_probability
        ) = predict_crisis(post)

        # --------------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------------

        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)

        print(f"\nPost:")
        print(post)

        print(f"\nPrediction: {prediction.upper()}")

        print(f"Confidence: {confidence * 100:.2f}%")

        print("\nClass Probabilities:")

        print(
            f"  informative: "
            f"{informative_probability * 100:.2f}%"
        )

        print(
            f"  not_informative: "
            f"{not_informative_probability * 100:.2f}%"
        )

        # --------------------------------------------------------
        # CRISIS MESSAGE
        # --------------------------------------------------------

        if prediction == "informative":

            print("\n🚨 RESULT: INFORMATIVE")
            print(
                "This post may contain useful crisis-related "
                "information."
            )

        else:

            print("\nℹ️ RESULT: NOT INFORMATIVE")
            print(
                "This post does not appear to contain "
                "useful crisis-related information."
            )

        print("=" * 70)
        print("-" * 70)

    except KeyboardInterrupt:

        print("\n\nProgram stopped by user.")
        break

    except Exception as e:

        print("\n❌ Prediction error:")
        print(e)
        print("-" * 70)