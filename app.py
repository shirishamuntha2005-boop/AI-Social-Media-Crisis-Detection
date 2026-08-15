# ============================================================
# AI-POWERED SOCIAL MEDIA CRISIS DETECTION
# DistilBERT + PyTorch + Streamlit
# ============================================================

import os
import json
import torch
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Social Media Crisis Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "distilbert_crisis_model"
)

METRICS_FILE = os.path.join(
    MODEL_DIR,
    "distilbert_metrics.json"
)

HISTORY_FILE = os.path.join(
    MODEL_DIR,
    "training_history.json"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

DISTILBERT_ACCURACY = 82.72
DISTILBERT_PRECISION = 86.90
DISTILBERT_RECALL = 89.35
DISTILBERT_F1 = 88.11

HYBRID_ACCURACY = 82.16
HYBRID_F1 = 81.58


# ============================================================
# CONFUSION MATRIX
# ============================================================

# Actual results from DistilBERT evaluation:
#
#                  Predicted
#                  Not Info   Informative
#
# Actual Not Info      601        310
# Actual Informative  245       2056

CONFUSION_MATRIX = [
    [601, 310],
    [245, 2056]
]


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

CLASSIFICATION_REPORT = [
    {
        "Class": "Not Informative",
        "Precision": 0.71,
        "Recall": 0.66,
        "F1 Score": 0.68,
        "Support": 911
    },
    {
        "Class": "Informative",
        "Precision": 0.87,
        "Recall": 0.89,
        "F1 Score": 0.88,
        "Support": 2301
    },
    {
        "Class": "Macro Average",
        "Precision": 0.79,
        "Recall": 0.78,
        "F1 Score": 0.78,
        "Support": 3212
    },
    {
        "Class": "Weighted Average",
        "Precision": 0.82,
        "Recall": 0.83,
        "F1 Score": 0.83,
        "Support": 3212
    }
]


# ============================================================
# SESSION STATE
# ============================================================

if "post_text" not in st.session_state:
    st.session_state.post_text = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "analyzed_text" not in st.session_state:
    st.session_state.analyzed_text = None


# ============================================================
# LOAD DISTILBERT MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_DIR):
        return None, None, None

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        return tokenizer, model, device

    except Exception as e:

        st.error(f"Error loading model: {e}")

        return None, None, None


tokenizer, model, device = load_model()


# ============================================================
# LOAD METRICS FILE IF AVAILABLE
# ============================================================

metrics_data = {}

if os.path.exists(METRICS_FILE):

    try:

        with open(
            METRICS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            metrics_data = json.load(f)

    except Exception:
        metrics_data = {}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_text(text):

    if model is None or tokenizer is None:

        return None, None

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    # Label mapping used during training:
    # 0 = not_informative
    # 1 = informative

    if predicted_class == 1:

        label = "INFORMATIVE"

    else:

        label = "NOT_INFORMATIVE"

    return label, confidence


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Model Information")

    st.markdown("### Model")
    st.write("**DistilBERT**")

    st.markdown("### Task")
    st.write("**Crisis Detection**")

    st.markdown("### Device")

    if device is not None:

        st.write(f"**{str(device).upper()}**")

    else:

        st.write("**Model not loaded**")

    st.divider()

    st.title("🏆 Model Performance")

    st.metric(
        "Accuracy",
        f"{DISTILBERT_ACCURACY:.2f}%"
    )

    st.metric(
        "F1 Score",
        f"{DISTILBERT_F1:.2f}%"
    )

    st.metric(
        "Precision",
        f"{DISTILBERT_PRECISION:.2f}%"
    )

    st.metric(
        "Recall",
        f"{DISTILBERT_RECALL:.2f}%"
    )

    st.divider()

    st.info(
        "The model analyzes social-media text "
        "and classifies it as informative or "
        "not informative."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🚨 AI-Powered Social Media Crisis Detection"
)

st.markdown(
    """
    ### Early Warning System Using Natural Language Processing
    """
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📝 Enter Social Media Post")

st.write("Social media post:")


# ============================================================
# EXAMPLE BUTTONS
# ============================================================

st.subheader("💡 Try an Example")

col1, col2 = st.columns(2)


with col1:

    if st.button(
        "🚨 Crisis Example",
        use_container_width=True
    ):

        st.session_state.post_text = (
            "Flood water has entered several houses "
            "and people need immediate help."
        )


with col2:

    if st.button(
        "😊 Normal Example",
        use_container_width=True
    ):

        st.session_state.post_text = (
            "I watched a movie with my friends yesterday."
        )


# ============================================================
# TEXT AREA
# ============================================================

post_text = st.text_area(
    "Social media post:",
    key="post_text",
    height=160,
    placeholder=(
        "Example: A major earthquake has damaged "
        "buildings and many people need help."
    )
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Post",
    type="primary",
    use_container_width=True
):

    if post_text.strip() == "":

        st.warning(
            "⚠️ Please enter a social media post first."
        )

    else:

        with st.spinner(
            "Analyzing the social media post..."
        ):

            label, confidence = predict_text(
                post_text.strip()
            )

        if label is not None:

            st.session_state.prediction = label
            st.session_state.confidence = confidence
            st.session_state.analyzed_text = post_text.strip()

        else:

            st.error(
                "❌ Unable to load the DistilBERT model."
            )


# ============================================================
# PREDICTION RESULT
# ============================================================

if st.session_state.prediction is not None:

    st.divider()

    st.header("🎯 Prediction Result")

    prediction = st.session_state.prediction
    confidence = st.session_state.confidence

    if prediction == "INFORMATIVE":

        st.error(
            "🚨 INFORMATIVE / POTENTIAL CRISIS CONTENT"
        )

        st.write(
            "The model identified this post as "
            "potentially useful crisis-related information."
        )

    else:

        st.success(
            "✅ NOT INFORMATIVE"
        )

        st.write(
            "The model identified this post as "
            "not containing useful crisis-related information."
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    st.subheader("📊 Prediction Confidence")

    confidence_percentage = confidence * 100

    st.metric(
        "Confidence",
        f"{confidence_percentage:.2f}%"
    )

    st.progress(
        float(confidence)
    )

    # ========================================================
    # TEXT ANALYZED
    # ========================================================

    st.subheader("📝 Text Analyzed")

    st.info(
        st.session_state.analyzed_text
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

st.header("ℹ️ About This Project")

st.write(
    """
    This project uses Natural Language Processing (NLP)
    and a fine-tuned DistilBERT transformer model to
    identify informative social-media posts related to
    crisis situations.
    """
)


# ============================================================
# TECHNOLOGIES USED
# ============================================================

st.subheader("🔧 Technologies Used")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:

    st.markdown("🐍 **Python**")
    st.markdown("🤗 **Transformers**")

with tech_col2:

    st.markdown("🧠 **DistilBERT**")
    st.markdown("🔥 **PyTorch**")

with tech_col3:

    st.markdown("📊 **Streamlit**")
    st.markdown("📝 **NLP**")


# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

st.divider()

st.header("📈 Model Performance Comparison")


# ============================================================
# DISTILBERT EVALUATION
# ============================================================

st.header("📊 DistilBERT Model Evaluation")

st.write(
    "Performance of the trained DistilBERT model "
    "on the test dataset."
)


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Accuracy",
        f"{DISTILBERT_ACCURACY:.2f}%"
    )


with metric2:

    st.metric(
        "Precision",
        f"{DISTILBERT_PRECISION:.2f}%"
    )


with metric3:

    st.metric(
        "Recall",
        f"{DISTILBERT_RECALL:.2f}%"
    )


with metric4:

    st.metric(
        "F1 Score",
        f"{DISTILBERT_F1:.2f}%"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🔲 Confusion Matrix")

st.write(
    "The confusion matrix shows how the model "
    "classified informative and non-informative posts."
)


st.subheader("🔥 Confusion Matrix Heatmap")


confusion_fig = go.Figure(
    data=go.Heatmap(
        z=CONFUSION_MATRIX,
        x=[
            "Not Informative",
            "Informative"
        ],
        y=[
            "Not Informative",
            "Informative"
        ],
        text=CONFUSION_MATRIX,
        texttemplate="%{text}",
        textfont={
            "size": 18
        },
        colorscale="Blues",
        colorbar=dict(
            title="Number of Posts"
        ),
        hovertemplate=(
            "Predicted Label: %{x}<br>"
            "Actual Label: %{y}<br>"
            "Number of Posts: %{z}"
            "<extra></extra>"
        )
    )
)

confusion_fig.update_layout(
    title="DistilBERT Confusion Matrix",
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label",
    height=550
)

st.plotly_chart(
    confusion_fig,
    use_container_width=True
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader("📋 Classification Report")

classification_df = pd.DataFrame(
    CLASSIFICATION_REPORT
)

st.dataframe(
    classification_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

st.header("🏆 Final Model Comparison")

st.write(
    "Comparison between the previous Hybrid Ensemble "
    "and the new DistilBERT model."
)


comparison_df = pd.DataFrame(
    {
        "Model": [
            "Hybrid Ensemble",
            "DistilBERT"
        ],
        "Accuracy": [
            "82.16%",
            "82.72%"
        ],
        "F1 Score": [
            "81.58%",
            "88.11%"
        ]
    }
)


st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ACCURACY COMPARISON
# ============================================================

st.subheader("📊 Accuracy Comparison")


accuracy_fig = go.Figure()

accuracy_fig.add_trace(
    go.Bar(
        x=[
            "Hybrid Ensemble",
            "DistilBERT"
        ],
        y=[
            HYBRID_ACCURACY,
            DISTILBERT_ACCURACY
        ],
        text=[
            f"{HYBRID_ACCURACY:.2f}%",
            f"{DISTILBERT_ACCURACY:.2f}%"
        ],
        textposition="outside"
    )
)

accuracy_fig.update_layout(
    title="Model Accuracy Comparison",
    xaxis_title="Model",
    yaxis_title="Accuracy (%)",
    yaxis=dict(
        range=[0, 100]
    ),
    height=500
)

st.plotly_chart(
    accuracy_fig,
    use_container_width=True
)


# ============================================================
# F1 SCORE COMPARISON
# ============================================================

st.subheader("🏆 F1 Score Comparison")


f1_fig = go.Figure()

f1_fig.add_trace(
    go.Bar(
        x=[
            "Hybrid Ensemble",
            "DistilBERT"
        ],
        y=[
            HYBRID_F1,
            DISTILBERT_F1
        ],
        text=[
            f"{HYBRID_F1:.2f}%",
            f"{DISTILBERT_F1:.2f}%"
        ],
        textposition="outside"
    )
)

f1_fig.update_layout(
    title="Model F1 Score Comparison",
    xaxis_title="Model",
    yaxis_title="F1 Score (%)",
    yaxis=dict(
        range=[0, 100]
    ),
    height=500
)

st.plotly_chart(
    f1_fig,
    use_container_width=True
)


# ============================================================
# IMPROVEMENT CALCULATION
# ============================================================

accuracy_improvement = (
    DISTILBERT_ACCURACY - HYBRID_ACCURACY
)

f1_improvement = (
    DISTILBERT_F1 - HYBRID_F1
)


st.success(
    f"🎉 DistilBERT achieved the best F1 Score of "
    f"{DISTILBERT_F1:.2f}%, improving significantly "
    f"over the previous Hybrid Ensemble F1 Score of "
    f"{HYBRID_F1:.2f}%."
)


# ============================================================
# PROJECT CONCLUSION
# ============================================================

st.header("🎯 Project Conclusion")

st.write(
    """
    The fine-tuned DistilBERT model provides strong
    performance for identifying informative crisis-related
    social-media posts.
    """
)

st.write(
    f"""
    Compared with the previous Hybrid Ensemble,
    DistilBERT achieved a higher accuracy and a
    substantially higher F1 score.

    Accuracy improved from {HYBRID_ACCURACY:.2f}% to
    {DISTILBERT_ACCURACY:.2f}%.

    F1 Score improved from {HYBRID_F1:.2f}% to
    {DISTILBERT_F1:.2f}%.
    """
)

st.write(
    """
    This model can therefore be used as the NLP
    prediction component of an early-warning crisis
    detection system.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">

    <h4>AI-Powered Social Media Crisis Detection
    & Early Warning System</h4>

    <p>
    NLP + DistilBERT + PyTorch + Streamlit
    </p>

    </div>
    """,
    unsafe_allow_html=True
)