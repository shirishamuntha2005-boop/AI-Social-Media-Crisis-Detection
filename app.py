# ============================================================
# 🚨 AI-POWERED SOCIAL MEDIA CRISIS DETECTION
# ============================================================
# DistilBERT + Hugging Face + PyTorch + Streamlit
# CrisisMMD Real Social Media Dataset
# ============================================================

import os
import json

import torch
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "crisis_mmd_master.csv"
)

METRICS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "distilbert_crisis_model",
    "distilbert_metrics.json"
)


# ============================================================
# HUGGING FACE MODEL
# ============================================================

MODEL_SOURCE = (
    "shirishamuntha2005/"
    "crisis-detection-distilbert"
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

if "dataset_label" not in st.session_state:
    st.session_state.dataset_label = None

if "selected_event" not in st.session_state:
    st.session_state.selected_event = None


# ============================================================
# LOAD DISTILBERT FROM HUGGING FACE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SOURCE
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_SOURCE
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model.to(device)
        model.eval()

        return tokenizer, model, device, None

    except Exception as e:

        return None, None, None, str(e)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_FILE):
        return None, (
            "CrisisMMD dataset file not found: "
            f"{DATASET_FILE}"
        )

    try:

        df = pd.read_csv(
            DATASET_FILE
        )

        return df, None

    except Exception as e:

        return None, str(e)


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner(
    "Loading AI model from Hugging Face..."
):

    tokenizer, model, device, model_error = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

df, dataset_error = load_dataset()


# ============================================================
# LOAD METRICS
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
# DATASET COLUMN DETECTION
# ============================================================

def find_column(dataframe, possible_names):

    if dataframe is None:
        return None

    columns_lower = {
        str(column).lower(): column
        for column in dataframe.columns
    }

    for name in possible_names:

        if name.lower() in columns_lower:
            return columns_lower[name.lower()]

    return None


TEXT_COLUMN = find_column(
    df,
    [
        "text",
        "tweet_text",
        "tweet",
        "post",
        "content",
        "text_info"
    ]
)

EVENT_COLUMN = find_column(
    df,
    [
        "event_name",
        "event",
        "crisis_event",
        "crisis"
    ]
)

LABEL_COLUMN = find_column(
    df,
    [
        "text_info",
        "label",
        "class",
        "informative",
        "target"
    ]
)


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

        outputs = model(
            **encoded
        )

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

    # Training label mapping
    # 0 = not_informative
    # 1 = informative

    if predicted_class == 1:

        label = "INFORMATIVE"

    else:

        label = "NOT INFORMATIVE"

    return label, confidence


# ============================================================
# PAGE HEADER
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
# MODEL STATUS
# ============================================================

if model_error:

    st.error(
        "❌ Error loading Hugging Face model"
    )

    with st.expander(
        "Show technical error"
    ):

        st.code(model_error)

else:

    st.success(
        "✅ DistilBERT model loaded successfully from Hugging Face"
    )


# ============================================================
# REAL CRISISMMD DATA
# ============================================================

st.header(
    "📰 Real Social Media Crisis Data"
)

if df is None:

    st.error(
        f"❌ {dataset_error}"
    )

else:

    total_posts = len(df)

    if EVENT_COLUMN:

        crisis_events = (
            df[EVENT_COLUMN]
            .dropna()
            .astype(str)
            .unique()
        )

        crisis_event_count = len(
            crisis_events
        )

    else:

        crisis_events = []
        crisis_event_count = 0

    informative_posts = 0

    if LABEL_COLUMN:

        labels = (
            df[LABEL_COLUMN]
            .astype(str)
            .str.lower()
        )

        informative_posts = labels.isin(
            [
                "informative",
                "1",
                "true"
            ]
        ).sum()

    st.success(
        f"Loaded {total_posts:,} real historical "
        "social-media posts from the CrisisMMD dataset."
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Total Posts",
            f"{total_posts:,}"
        )

    with metric2:

        st.metric(
            "Crisis Events",
            f"{crisis_event_count:,}"
        )

    with metric3:

        st.metric(
            "Informative Posts",
            f"{informative_posts:,}"
        )


# ============================================================
# EXPLORE CRISIS DATA
# ============================================================

if df is not None and EVENT_COLUMN and TEXT_COLUMN:

    st.subheader(
        "🔎 Explore Real CrisisMMD Posts"
    )

    event_options = sorted(
        df[EVENT_COLUMN]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if event_options:

        selected_event = st.selectbox(
            "Select a crisis event",
            event_options
        )

        event_df = df[
            df[EVENT_COLUMN].astype(str)
            == selected_event
        ].copy()

        if not event_df.empty:

            post_indices = event_df.index.tolist()

            selected_index = st.selectbox(
                "Select a social media post",
                post_indices,
                format_func=lambda x:
                    str(
                        event_df.loc[
                            x,
                            TEXT_COLUMN
                        ]
                    )[:120]
            )

            selected_row = event_df.loc[
                selected_index
            ]

            selected_post = str(
                selected_row[TEXT_COLUMN]
            )

            selected_label = None

            if LABEL_COLUMN:

                selected_label = str(
                    selected_row[LABEL_COLUMN]
                ).strip().upper()

                if selected_label in [
                    "1",
                    "TRUE"
                ]:

                    selected_label = "INFORMATIVE"

                elif selected_label in [
                    "0",
                    "FALSE"
                ]:

                    selected_label = "NOT INFORMATIVE"

            st.session_state.post_text = (
                selected_post
            )

            st.session_state.dataset_label = (
                selected_label
            )

            st.session_state.selected_event = (
                selected_event
            )

            st.subheader(
                "📝 Real Social Media Post"
            )

            st.info(
                selected_post
            )

            st.markdown(
                f"**Event:** {selected_event}"
            )

            st.subheader(
                "🏷️ Dataset Information"
            )

            info_col1, info_col2 = st.columns(2)

            with info_col1:

                st.markdown(
                    "**Crisis Event**"
                )

                st.write(
                    selected_event
                )

            with info_col2:

                st.markdown(
                    "**Dataset / Human Label**"
                )

                if selected_label:

                    if selected_label == "INFORMATIVE":

                        st.success(
                            selected_label
                        )

                    else:

                        st.warning(
                            selected_label
                        )

                else:

                    st.write(
                        "Not available"
                    )


# ============================================================
# ANALYZE POST
# ============================================================

st.divider()

st.header(
    "📝 Analyze Social Media Post"
)

post_text = st.text_area(
    "Enter Social Media Post",
    value=st.session_state.post_text,
    height=150,
    placeholder=(
        "Example: Flood water has entered "
        "several houses and people need rescue."
    )
)

analyze_button = st.button(
    "🔍 Analyze Post",
    type="primary",
    use_container_width=True
)


if analyze_button:

    if not post_text.strip():

        st.warning(
            "⚠️ Please enter a social media post first."
        )

    elif model is None:

        st.error(
            "❌ AI model is not available."
        )

    else:

        with st.spinner(
            "Analyzing the social media post..."
        ):

            label, confidence = predict_text(
                post_text.strip()
            )

        st.session_state.prediction = label

        st.session_state.confidence = (
            confidence
        )

        st.session_state.analyzed_text = (
            post_text.strip()
        )

        st.session_state.post_text = (
            post_text.strip()
        )


# ============================================================
# PREDICTION RESULT
# ============================================================

if (
    st.session_state.prediction
    is not None
):

    st.divider()

    st.header(
        "🎯 Prediction Result"
    )

    prediction = (
        st.session_state.prediction
    )

    confidence = (
        st.session_state.confidence
    )

    if prediction == "INFORMATIVE":

        st.success(
            "📢 INFORMATIVE"
        )

        st.write(
            "The model identified this post "
            "as containing useful crisis-related "
            "information."
        )

    else:

        st.info(
            "ℹ️ NOT INFORMATIVE"
        )

        st.write(
            "The model identified this post "
            "as not containing useful "
            "crisis-related information."
        )

    st.subheader(
        "📊 Prediction Confidence"
    )

    confidence_col1, confidence_col2 = st.columns(
        [1, 3]
    )

    with confidence_col1:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with confidence_col2:

        st.progress(
            float(confidence)
        )

    st.subheader(
        "📝 Text Analyzed"
    )

    st.info(
        st.session_state.analyzed_text
    )


    # ========================================================
    # DATASET LABEL VS PREDICTION
    # ========================================================

    dataset_label = (
        st.session_state.dataset_label
    )

    if dataset_label:

        st.subheader(
            "🔍 Dataset Label vs AI Prediction"
        )

        compare_col1, compare_col2 = st.columns(
            2
        )

        with compare_col1:

            st.markdown(
                "**Dataset / Human Label**"
            )

            if dataset_label == "INFORMATIVE":

                st.success(
                    dataset_label
                )

            else:

                st.warning(
                    dataset_label
                )

        with compare_col2:

            st.markdown(
                "**DistilBERT Prediction**"
            )

            if prediction == "INFORMATIVE":

                st.success(
                    prediction
                )

            else:

                st.info(
                    prediction
                )

        if dataset_label == prediction:

            st.success(
                "✅ The AI prediction matches "
                "the dataset label."
            )

        else:

            st.warning(
                "⚠️ The AI prediction does not "
                "match the dataset label."
            )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

st.header(
    "ℹ️ About This Project"
)

st.write(
    """
    This project uses Natural Language Processing (NLP)
    and a fine-tuned DistilBERT transformer model to
    identify informative social-media posts related to
    crisis situations.

    The system is trained and evaluated using the
    CrisisMMD dataset containing real historical
    social-media posts collected during major
    disaster events.
    """
)


# ============================================================
# TECHNOLOGIES
# ============================================================

st.subheader(
    "🔧 Technologies Used"
)

tech1, tech2, tech3 = st.columns(3)

with tech1:

    st.markdown(
        "🐍 **Python**"
    )

    st.markdown(
        "🤗 **Transformers**"
    )

with tech2:

    st.markdown(
        "🧠 **DistilBERT**"
    )

    st.markdown(
        "🔥 **PyTorch**"
    )

with tech3:

    st.markdown(
        "📊 **Streamlit**"
    )

    st.markdown(
        "📝 **NLP**"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header(
    "📈 Model Performance Comparison"
)

st.subheader(
    "📊 DistilBERT Model Evaluation"
)

st.write(
    "Performance of the trained DistilBERT "
    "model on the test dataset."
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

st.subheader(
    "🔥 Confusion Matrix"
)

confusion_df = pd.DataFrame(
    CONFUSION_MATRIX,
    index=[
        "Actual Not Informative",
        "Actual Informative"
    ],
    columns=[
        "Predicted Not Informative",
        "Predicted Informative"
    ]
)

confusion_fig = px.imshow(
    confusion_df,
    text_auto=True,
    aspect="auto",
    title="DistilBERT Confusion Matrix"
)

confusion_fig.update_layout(
    height=500
)

st.plotly_chart(
    confusion_fig,
    use_container_width=True
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader(
    "📋 Classification Report"
)

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

st.subheader(
    "📊 Final Model Comparison"
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

st.subheader(
    "📊 Accuracy Comparison"
)

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

st.subheader(
    "🏆 F1 Score Comparison"
)

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
# IMPROVEMENT
# ============================================================

accuracy_improvement = (
    DISTILBERT_ACCURACY
    - HYBRID_ACCURACY
)

f1_improvement = (
    DISTILBERT_F1
    - HYBRID_F1
)

st.success(
    f"""
    🎉 DistilBERT achieved the best F1 Score of
    {DISTILBERT_F1:.2f}%, improving by
    {f1_improvement:.2f} percentage points over
    the previous Hybrid Ensemble F1 Score of
    {HYBRID_F1:.2f}%.
    """
)


# ============================================================
# PROJECT CONCLUSION
# ============================================================

st.header(
    "🎯 Project Conclusion"
)

st.write(
    """
    The fine-tuned DistilBERT model provides strong
    performance for identifying informative
    crisis-related social-media posts.
    """
)

st.write(
    f"""
    Compared with the previous Hybrid Ensemble,
    DistilBERT achieved a higher accuracy and a
    substantially higher F1 score.

    Accuracy improved from
    {HYBRID_ACCURACY:.2f}% to
    {DISTILBERT_ACCURACY:.2f}%.

    F1 Score improved from
    {HYBRID_F1:.2f}% to
    {DISTILBERT_F1:.2f}%.
    """
)

st.write(
    """
    The model can therefore be used as the NLP
    prediction component of an early-warning
    crisis detection system.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">

    <h4>
    AI-Powered Social Media Crisis Detection
    & Early Warning System
    </h4>

    <p>
    NLP + DistilBERT + PyTorch + Streamlit
    </p>

    </div>
    """,
    unsafe_allow_html=True
)