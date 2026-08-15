import os
import joblib
import numpy as np
import streamlit as st

from scipy.sparse import hstack


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="AI Social Media Crisis Detection",
    page_icon="🚨",
    layout="wide"
)


# ================================================================
# PATH CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ensemble_hybrid_crisis_model.joblib"
)

WORD_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ensemble_word_tfidf.pkl"
)

CHAR_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ensemble_char_tfidf.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "best_ensemble_prediction_threshold.txt"
)


# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    word_vectorizer = joblib.load(
        WORD_VECTORIZER_PATH
    )

    char_vectorizer = joblib.load(
        CHAR_VECTORIZER_PATH
    )

    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read().strip())

    return (
        model,
        word_vectorizer,
        char_vectorizer,
        threshold
    )


# ================================================================
# LOAD RESOURCES
# ================================================================

try:

    model, word_vectorizer, char_vectorizer, threshold = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        "❌ Failed to load the model files."
    )

    st.error(str(e))


# ================================================================
# HEADER
# ================================================================

st.title(
    "🚨 AI-Powered Social Media Crisis Detection"
)

st.subheader(
    "Early Warning System Using Natural Language Processing"
)

st.write(
    """
    This application analyzes social media posts and predicts
    whether the post contains useful crisis-related information.
    """
)


# ================================================================
# MODEL INFORMATION
# ================================================================

st.markdown("## 📊 Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Model",
        "Soft Voting Ensemble"
    )

with col2:

    st.metric(
        "Word Features",
        f"{len(word_vectorizer.vocabulary_):,}"
        if model_loaded
        else "N/A"
    )

with col3:

    st.metric(
        "Character Features",
        f"{len(char_vectorizer.vocabulary_):,}"
        if model_loaded
        else "N/A"
    )

with col4:

    st.metric(
        "Threshold",
        f"{threshold:.2f}"
        if model_loaded
        else "N/A"
    )


# ================================================================
# MODEL PERFORMANCE
# ================================================================

st.markdown("## 🏆 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Best Accuracy",
        "82.16%"
    )

with col2:

    st.metric(
        "Weighted F1",
        "81.58%"
    )

with col3:

    st.metric(
        "Hybrid Features",
        "70,945"
    )


# ================================================================
# INPUT SECTION
# ================================================================

st.markdown("## 📝 Enter Social Media Post")

post = st.text_area(
    "Social media post:",
    height=150,
    placeholder="Example: Flood water has entered several houses and people need emergency rescue"
)


# ================================================================
# EXAMPLES
# ================================================================

st.markdown("### 💡 Try an Example")

example1, example2, example3 = st.columns(3)

with example1:

    if st.button("🚨 Crisis Example"):

        st.session_state["post"] = (
            "Flood water has entered several houses "
            "and people need emergency rescue"
        )

with example2:

    if st.button("🌧️ Disaster Example"):

        st.session_state["post"] = (
            "Heavy rainfall has caused severe flooding "
            "in several areas"
        )

with example3:

    if st.button("ℹ️ Normal Example"):

        st.session_state["post"] = (
            "I watched a great movie today"
        )


# Use example from session state

if "post" in st.session_state:

    post = st.session_state["post"]


# ================================================================
# PREDICTION
# ================================================================

if st.button(
    "🔍 Analyze Post",
    type="primary"
):

    if not model_loaded:

        st.error(
            "❌ Model files could not be loaded."
        )

    elif not post.strip():

        st.warning(
            "⚠️ Please enter a social media post."
        )

    else:

        with st.spinner(
            "Analyzing social media post..."
        ):

            # ----------------------------------------------------
            # WORD TF-IDF
            # ----------------------------------------------------

            word_features = word_vectorizer.transform(
                [post]
            )

            # ----------------------------------------------------
            # CHARACTER TF-IDF
            # ----------------------------------------------------

            char_features = char_vectorizer.transform(
                [post]
            )

            # ----------------------------------------------------
            # HYBRID FEATURES
            # ----------------------------------------------------

            hybrid_features = hstack(
                [
                    word_features,
                    char_features
                ]
            )

            # ----------------------------------------------------
            # PROBABILITIES
            # ----------------------------------------------------

            probabilities = model.predict_proba(
                hybrid_features
            )[0]

            classes = model.classes_

            # Find probability of informative class

            informative_index = list(
                classes
            ).index("informative")

            informative_probability = (
                probabilities[informative_index]
            )

            not_informative_probability = (
                1 - informative_probability
            )

            # ----------------------------------------------------
            # THRESHOLD DECISION
            # ----------------------------------------------------

            if informative_probability >= threshold:

                prediction = "informative"

            else:

                prediction = "not_informative"


        # ========================================================
        # RESULT
        # ========================================================

        st.markdown("---")

        st.markdown(
            "## 🎯 Prediction Result"
        )


        if prediction == "informative":

            st.success(
                """
                🚨 **INFORMATIVE**

                This post may contain useful
                crisis-related information.
                """
            )

        else:

            st.info(
                """
                ℹ️ **NOT INFORMATIVE**

                This post does not appear to contain
                useful crisis-related information.
                """
            )


        # ========================================================
        # CONFIDENCE
        # ========================================================

        confidence = max(
            informative_probability,
            not_informative_probability
        )

        st.markdown(
            "### 🎯 Confidence"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ========================================================
        # CLASS PROBABILITIES
        # ========================================================

        st.markdown(
            "### 📊 Class Probabilities"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🚨 Informative",
                f"{informative_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "ℹ️ Not Informative",
                f"{not_informative_probability * 100:.2f}%"
            )


        # ========================================================
        # PROBABILITY BAR
        # ========================================================

        st.markdown(
            "### 📈 Probability"
        )

        st.progress(
            float(informative_probability)
        )


        # ========================================================
        # ANALYZED POST
        # ========================================================

        st.markdown(
            "### 📝 Analyzed Post"
        )

        st.code(
            post
        )


        # ========================================================
        # PREDICTION HISTORY
        # ========================================================

        if "prediction_history" not in st.session_state:

            st.session_state[
                "prediction_history"
            ] = []


        st.session_state[
            "prediction_history"
        ].append(
            {
                "post": post,
                "prediction": prediction,
                "confidence": confidence
            }
        )


# ================================================================
# PREDICTION HISTORY
# ================================================================

if (
    "prediction_history"
    in st.session_state
    and len(
        st.session_state["prediction_history"]
    ) > 0
):

    st.markdown("---")

    st.markdown(
        "## 📋 Prediction History"
    )

    history = st.session_state[
        "prediction_history"
    ]

    total_posts = len(history)

    informative_count = sum(
        1
        for item in history
        if item["prediction"] == "informative"
    )

    not_informative_count = (
        total_posts - informative_count
    )


    # ------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------

    st.markdown(
        "### 📊 Prediction Statistics"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Posts",
            total_posts
        )

    with col2:

        st.metric(
            "🚨 Informative",
            informative_count
        )

    with col3:

        st.metric(
            "ℹ️ Not Informative",
            not_informative_count
        )


    # ------------------------------------------------------------
    # HISTORY TABLE
    # ------------------------------------------------------------

    for item in reversed(history):

        if item["prediction"] == "informative":

            icon = "🚨"

            label = "INFORMATIVE"

        else:

            icon = "ℹ️"

            label = "NOT INFORMATIVE"


        st.markdown(
            f"""
            **{icon} {label}**
            — {item["confidence"] * 100:.2f}%

            {item["post"]}
            """
        )

        st.divider()


# ================================================================
# ABOUT PROJECT
# ================================================================

st.markdown("---")

st.markdown(
    "## ℹ️ About This Project"
)

st.write(
    """
    This project uses Natural Language Processing (NLP)
    to identify crisis-related social media posts.
    """
)

st.write(
    """
    The system combines:
    """
)

st.markdown(
    """
    - Word-level TF-IDF features
    - Character-level TF-IDF features
    - Logistic Regression
    - Random Forest
    - Soft Voting Ensemble
    - Optimized prediction threshold
    """
)

st.write(
    """
    The final ensemble model achieved approximately
    **82.16% accuracy** on the test dataset.
    """
)


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

st.caption(
    "AI-Powered Social Media Crisis Detection | "
    "NLP + Hybrid TF-IDF + Ensemble Learning"
)