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

# app.py is inside:
# AI-Social-Media-Crisis-Detection/src/
#
# Therefore, go one level up to:
# AI-Social-Media-Crisis-Detection/

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


# ================================================================
# MODEL FILE PATHS
# ================================================================

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

THRESHOLD_PATH = os.path.join(
    DATA_DIR,
    "best_ensemble_prediction_threshold.txt"
)


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    model = joblib.load(MODEL_PATH)

    word_vectorizer = joblib.load(
        WORD_VECTORIZER_PATH
    )

    char_vectorizer = joblib.load(
        CHAR_VECTORIZER_PATH
    )

    with open(
        THRESHOLD_PATH,
        "r"
    ) as f:

        threshold = float(
            f.read().strip()
        )

    return (
        model,
        word_vectorizer,
        char_vectorizer,
        threshold
    )


# ================================================================
# MODEL LOADING STATUS
# ================================================================

model_loaded = False

model = None
word_vectorizer = None
char_vectorizer = None
threshold = None

try:

    (
        model,
        word_vectorizer,
        char_vectorizer,
        threshold
    ) = load_model()

    model_loaded = True

except Exception as e:

    st.error(
        "❌ Failed to load the model files."
    )

    st.code(
        str(e)
    )


# ================================================================
# TITLE
# ================================================================

st.markdown(
    "# 🚨 AI-Powered Social Media Crisis Detection"
)

st.markdown(
    "### Early Warning System Using Natural Language Processing"
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

st.markdown(
    "## 📊 Model Information"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Model",
        "Soft Voting Ensemble"
    )


with col2:

    if model_loaded:

        try:

            word_features = word_vectorizer.get_feature_names_out()

            st.metric(
                "Word Features",
                len(word_features)
            )

        except Exception:

            st.metric(
                "Word Features",
                "Available"
            )

    else:

        st.metric(
            "Word Features",
            "N/A"
        )


with col3:

    if model_loaded:

        try:

            char_features = char_vectorizer.get_feature_names_out()

            st.metric(
                "Character Features",
                len(char_features)
            )

        except Exception:

            st.metric(
                "Character Features",
                "Available"
            )

    else:

        st.metric(
            "Character Features",
            "N/A"
        )


with col4:

    if model_loaded:

        st.metric(
            "Threshold",
            f"{threshold:.3f}"
        )

    else:

        st.metric(
            "Threshold",
            "N/A"
        )


# ================================================================
# MODEL PERFORMANCE
# ================================================================

st.markdown(
    "## 🏆 Model Performance"
)

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
# USER INPUT
# ================================================================

st.markdown(
    "## 📝 Enter Social Media Post"
)

user_text = st.text_area(
    "Social media post:",
    height=150,
    placeholder="Enter a social media post here..."
)


# ================================================================
# EXAMPLES
# ================================================================

st.markdown(
    "### 💡 Try an Example"
)

example_col1, example_col2 = st.columns(2)


with example_col1:

    if st.button(
        "🌧️ Flood Example"
    ):

        user_text = (
            "Heavy rain has caused severe flooding "
            "and people are trapped."
        )

        st.info(user_text)


with example_col2:

    if st.button(
        "☀️ Normal Example"
    ):

        user_text = (
            "Beautiful weather today, "
            "going out with friends."
        )

        st.info(user_text)


# ================================================================
# PREDICTION
# ================================================================

if st.button(
    "🔍 Analyze Post",
    use_container_width=True
):

    if not model_loaded:

        st.error(
            "❌ Model files could not be loaded."
        )

    elif not user_text.strip():

        st.warning(
            "⚠️ Please enter a social media post."
        )

    else:

        try:

            # ----------------------------------------------------
            # WORD TF-IDF
            # ----------------------------------------------------

            word_features = word_vectorizer.transform(
                [user_text]
            )

            # ----------------------------------------------------
            # CHARACTER TF-IDF
            # ----------------------------------------------------

            char_features = char_vectorizer.transform(
                [user_text]
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
            # PREDICT PROBABILITY
            # ----------------------------------------------------

            probabilities = model.predict_proba(
                hybrid_features
            )[0]

            classes = model.classes_

            # Probability of class 1
            if 1 in classes:

                informative_index = list(
                    classes
                ).index(1)

                informative_probability = probabilities[
                    informative_index
                ]

            else:

                informative_probability = probabilities[-1]

            # ----------------------------------------------------
            # THRESHOLD PREDICTION
            # ----------------------------------------------------

            if informative_probability >= threshold:

                prediction = "Informative"

            else:

                prediction = "Not Informative"

            confidence = (
                informative_probability
                if prediction == "Informative"
                else 1 - informative_probability
            )

            # ----------------------------------------------------
            # RESULT
            # ----------------------------------------------------

            st.markdown(
                "## 🎯 Prediction Result"
            )

            if prediction == "Informative":

                st.success(
                    f"### ✅ {prediction}"
                )

            else:

                st.info(
                    f"### ℹ️ {prediction}"
                )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Prediction",
                    prediction
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            # ----------------------------------------------------
            # PROBABILITY
            # ----------------------------------------------------

            st.markdown(
                "### 📈 Prediction Probability"
            )

            probability_data = {

                "Not Informative":
                    (1 - informative_probability) * 100,

                "Informative":
                    informative_probability * 100
            }

            st.bar_chart(
                probability_data
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.code(
                str(e)
            )


# ================================================================
# REAL-TIME CRISIS MONITORING
# ================================================================

st.markdown("---")

st.markdown(
    "# 🚨 Real-Time Crisis Monitoring"
)

st.write(
    """
    This section displays results generated by the
    real-time RoBERTa crisis detection pipeline.
    """
)


# ================================================================
# REAL-TIME FILE PATHS
# ================================================================

REALTIME_DIR = os.path.join(
    BASE_DIR,
    "realtime"
)

REALTIME_POSTS = os.path.join(
    REALTIME_DIR,
    "realtime_posts.csv"
)

REALTIME_PREDICTIONS = os.path.join(
    REALTIME_DIR,
    "realtime_predictions.csv"
)

REALTIME_SEVERITY = os.path.join(
    REALTIME_DIR,
    "realtime_severity.csv"
)

REALTIME_ALERTS = os.path.join(
    REALTIME_DIR,
    "realtime_alerts.csv"
)

REALTIME_EVENTS = os.path.join(
    REALTIME_DIR,
    "realtime_events.csv"
)


# ================================================================
# CHECK REAL-TIME FILES
# ================================================================

realtime_files = [

    REALTIME_POSTS,

    REALTIME_PREDICTIONS,

    REALTIME_SEVERITY,

    REALTIME_ALERTS,

    REALTIME_EVENTS

]

missing_realtime_files = [

    file

    for file in realtime_files

    if not os.path.exists(file)

]


if missing_realtime_files:

    st.warning(
        "⚠️ Some real-time pipeline output files are missing."
    )

    for file in missing_realtime_files:

        st.write(
            f"❌ {file}"
        )

else:

    # ============================================================
    # LOAD REAL-TIME DATA
    # ============================================================

    import pandas as pd

    realtime_posts = pd.read_csv(
        REALTIME_POSTS
    )

    realtime_predictions = pd.read_csv(
        REALTIME_PREDICTIONS
    )

    realtime_severity = pd.read_csv(
        REALTIME_SEVERITY
    )

    realtime_alerts = pd.read_csv(
        REALTIME_ALERTS
    )

    realtime_events = pd.read_csv(
        REALTIME_EVENTS
    )


    # ============================================================
    # REAL-TIME OVERVIEW
    # ============================================================

    st.markdown(
        "## 📊 Real-Time Crisis Overview"
    )

    total_posts = len(
        realtime_predictions
    )

    informative_posts = len(
        realtime_predictions[
            realtime_predictions["prediction"]
            == "Informative"
        ]
    )

    urgent_alerts = len(
        realtime_alerts[
            realtime_alerts["priority"]
            == "URGENT"
        ]
    )

    high_severity = len(
        realtime_severity[
            realtime_severity["severity"]
            == "High"
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📱 Total Posts",
            total_posts
        )

    with col2:

        st.metric(
            "📰 Informative",
            informative_posts
        )

    with col3:

        st.metric(
            "🚨 Urgent Alerts",
            urgent_alerts
        )

    with col4:

        st.metric(
            "🔴 High Severity",
            high_severity
        )


    # ============================================================
    # SEVERITY SUMMARY
    # ============================================================

    st.markdown(
        "## ⚠️ Severity Summary"
    )

    high_count = len(
        realtime_severity[
            realtime_severity["severity"]
            == "High"
        ]
    )

    medium_count = len(
        realtime_severity[
            realtime_severity["severity"]
            == "Medium"
        ]
    )

    low_count = len(
        realtime_severity[
            realtime_severity["severity"]
            == "Low"
        ]
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🔴 High",
            high_count
        )

    with col2:

        st.metric(
            "🟠 Medium",
            medium_count
        )

    with col3:

        st.metric(
            "🟢 Low",
            low_count
        )


    # ============================================================
    # SEVERITY CHART
    # ============================================================

    st.markdown(
        "### Severity Distribution"
    )

    severity_chart = pd.DataFrame(

        {

            "Severity": [
                "High",
                "Medium",
                "Low"
            ],

            "Count": [
                high_count,
                medium_count,
                low_count
            ]

        }

    )

    st.bar_chart(
        severity_chart.set_index(
            "Severity"
        )
    )


    # ============================================================
    # URGENT ALERTS
    # ============================================================

    st.markdown(
        "## 🚨 Urgent Crisis Alerts"
    )

    urgent = realtime_alerts[
        realtime_alerts["priority"]
        == "URGENT"
    ]

    if len(urgent) == 0:

        st.success(
            "✅ No urgent crisis alerts detected."
        )

    else:

        st.error(
            f"🚨 {len(urgent)} urgent crisis alert(s) detected!"
        )

        for _, row in urgent.iterrows():

            st.warning(
                f"""
**Post ID:** {row['id']}

**Post:** {row['text']}

**Confidence:** {row['confidence']}%

**Severity:** {row['severity']}

**Priority:** {row['priority']}

**Alert:** {row['alert']}
"""
            )


    # ============================================================
    # ALL ALERTS
    # ============================================================

    st.markdown(
        "## 📢 All Crisis Alerts"
    )

    st.dataframe(
        realtime_alerts,
        use_container_width=True
    )


    # ============================================================
    # CRISIS EVENTS
    # ============================================================

    st.markdown(
        "## 🌐 Crisis Event Groups"
    )

    if len(realtime_events) > 0:

        event_summary = (

            realtime_events

            .groupby("event_id")

            .agg(

                post_count=(
                    "id",
                    "count"
                ),

                highest_severity=(
                    "severity_score",
                    "max"
                )

            )

            .reset_index()

        )

        st.dataframe(
            event_summary,
            use_container_width=True
        )

    else:

        st.info(
            "No crisis events detected."
        )


    # ============================================================
    # REAL-TIME POSTS
    # ============================================================

    st.markdown(
        "## 📱 Real-Time Social Media Posts"
    )

    st.dataframe(
        realtime_predictions,
        use_container_width=True
    )


    # ============================================================
    # SYSTEM STATUS
    # ============================================================

    st.markdown(
        "## 🟢 System Status"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(
            "RoBERTa Model: ONLINE"
        )

    with col2:

        st.success(
            "Real-Time Pipeline: ACTIVE"
        )

    with col3:

        st.success(
            "Alert System: ACTIVE"
        )


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

st.markdown(
    """
    The system combines:

    - Word-level TF-IDF features
    - Character-level TF-IDF features
    - Logistic Regression
    - Random Forest
    - Soft Voting Ensemble
    - Optimized prediction threshold
    - RoBERTa real-time crisis classification
    - Crisis severity detection
    - Automated crisis alerts
    - Crisis event grouping
    """
)

st.markdown(
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
    "NLP + Hybrid TF-IDF + Ensemble Learning + "
    "Real-Time RoBERTa"
)