import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ============================================================
# AI SOCIAL MEDIA CRISIS DETECTION
# Streamlit Application
# ============================================================

MODEL_PATH = "data/processed/distilbert_crisis_model"

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="AI Social Media Crisis Detection",
    page_icon="🚨",
    layout="centered"
)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------

def predict(text, tokenizer, model, device):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities[0][predicted_class].item()
            * 100
        )

    if predicted_class == 1:
        label = "INFORMATIVE"
    else:
        label = "NOT INFORMATIVE"

    return label, confidence


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("🚨 AI Social Media Crisis Detection")

st.write(
    "Analyze social media posts using a fine-tuned DistilBERT NLP model."
)

st.divider()

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

try:

    tokenizer, model, device = load_model()

    st.success("✅ DistilBERT model loaded successfully")

except Exception as e:

    st.error(f"❌ Error loading model: {e}")

    st.stop()


# ------------------------------------------------------------
# Text input
# ------------------------------------------------------------

st.subheader("Enter a Social Media Post")

text = st.text_area(
    "Post text",
    placeholder=(
        "Example: Flood water has entered several houses "
        "and people need immediate help."
    ),
    height=150
)


# ------------------------------------------------------------
# Analyze button
# ------------------------------------------------------------

if st.button(
    "🔍 Analyze Post",
    type="primary",
    use_container_width=True
):

    if not text.strip():

        st.warning("⚠️ Please enter some text first.")

    else:

        with st.spinner("Analyzing post..."):

            label, confidence = predict(
                text,
                tokenizer,
                model,
                device
            )

        st.divider()

        st.subheader("Prediction Result")

        if label == "INFORMATIVE":

            st.error("🚨 INFORMATIVE / POTENTIAL CRISIS CONTENT")

        else:

            st.success("✅ NOT INFORMATIVE")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            int(confidence)
        )

        # ----------------------------------------------------
        # Additional information
        # ----------------------------------------------------

        st.write("### Analysis")

        if label == "INFORMATIVE":

            st.write(
                "The model identified this post as informative "
                "crisis-related content."
            )

        else:

            st.write(
                "The model identified this post as "
                "non-informative content."
            )


# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

st.divider()

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:

    st.write("**Model:** DistilBERT")
    st.write("**Task:** Crisis Detection")

with col2:

    st.write("**Accuracy:** 82.72%")
    st.write("**F1 Score:** 88.11%")


st.caption(
    "AI-Powered Social Media Crisis Detection & Early Warning System"
)