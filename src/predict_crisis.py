import pickle
import joblib
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# AI SOCIAL MEDIA CRISIS DETECTION
# CRISIS PREDICTION SYSTEM
# ============================================================

print("=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("CRISIS PREDICTION SYSTEM")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_PATH = DATA_DIR / "final_random_forest_model.pkl"


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading trained model...")

try:
    # Model was saved using joblib
    model = joblib.load(MODEL_PATH)

    print("✅ Random Forest model loaded successfully")

except Exception as e:
    print("❌ Model loading failed")
    print("Error:", e)
    exit()


# ============================================================
# 3. LOAD TF-IDF VECTORIZER
# ============================================================

print("\nLoading TF-IDF vectorizer...")

VECTORIZER_PATH = DATA_DIR / "tfidf_vectorizer.pkl"

try:
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("✅ TF-IDF vectorizer loaded successfully")

except Exception:
    print("⚠️ TF-IDF vectorizer file not found.")
    print("Creating a new vectorizer is not recommended for prediction.")
    print("Please make sure tfidf_vectorizer.pkl exists.")
    exit()


# ============================================================
# 4. LOAD NLTK RESOURCES
# ============================================================

print("\nLoading NLP resources...")

try:
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    print("✅ Stopwords loaded")
    print("✅ Lemmatizer loaded")

except Exception as e:
    print("❌ NLTK resources not available")
    print("Error:", e)
    exit()


# ============================================================
# 5. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but keep word
    text = re.sub(r"#", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = text.split()

    # Stopword removal + lemmatization
    processed_words = []

    for word in words:

        if word not in stop_words:

            lemma = lemmatizer.lemmatize(word)

            processed_words.append(lemma)

    return " ".join(processed_words)


# ============================================================
# 6. USER INPUT
# ============================================================

print("\n" + "=" * 70)
print("ENTER SOCIAL MEDIA TEXT")
print("=" * 70)

text = input("\nEnter tweet/post: ")


# ============================================================
# 7. PREPROCESS TEXT
# ============================================================

print("\nProcessing text...")

cleaned_text = clean_text(text)

print("\nCleaned text:")
print(cleaned_text)


# ============================================================
# 8. TF-IDF TRANSFORMATION
# ============================================================

print("\nConverting text into TF-IDF features...")

X = vectorizer.transform([cleaned_text])

print("✅ TF-IDF transformation completed")


# ============================================================
# 9. PREDICTION
# ============================================================

print("\nMaking prediction...")

prediction = model.predict(X)[0]

print("✅ Prediction completed")


# ============================================================
# 10. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("CRISIS DETECTION RESULT")
print("=" * 70)

print("\nPrediction:", prediction)

if prediction == "informative":

    print("\n🚨 CRISIS-RELATED / INFORMATIVE CONTENT")
    print("⚠️ This post contains potentially useful crisis information.")

else:

    print("\nℹ️ NOT INFORMATIVE")
    print("This post is classified as not informative for crisis response.")


# ============================================================
# 11. CONFIDENCE
# ============================================================

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(X)[0]

    confidence = max(probabilities) * 100

    print(f"\nPrediction Confidence: {confidence:.2f}%")


print("\n" + "=" * 70)
print("✅ CRISIS PREDICTION COMPLETED")
print("=" * 70)