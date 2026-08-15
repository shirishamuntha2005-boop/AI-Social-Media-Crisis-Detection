import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Social Media Crisis Detection",
    page_icon="🚨",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚨 AI Social Media Crisis Detection")
st.markdown(
    "### Analyze Social Media Text"
)

st.write(
    "Enter a social media post below to determine whether "
    "it contains informative crisis-related information."
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    "data",
    "processed",
    "distilbert_crisis_model"
)


# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error("❌ DistilBERT model folder could not be found.")

    st.write("Expected model location:")
    st.code(MODEL_PATH)

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model.eval()

    return tokenizer, model


# ============================================================
# LOAD MODEL WITH ERROR HANDLING
# ============================================================

try:

    tokenizer, model = load_model()

    st.success("✅ DistilBERT model loaded successfully.")

except Exception as e:

    st.error("❌ DistilBERT model could not be loaded.")

    st.write(
        "Please check the model files inside:"
    )

    st.code(MODEL_PATH)

    st.exception(e)

    st.stop()


# ============================================================
# TEXT INPUT
# ============================================================

st.subheader("Enter Social Media Text")

text = st.text_area(
    "Social media text:",
    height=150,
    placeholder="Example: Flood water has entered several houses..."
)


# ============================================================
# EXAMPLE TEXTS
# ============================================================

st.subheader("Try an Example")

example = st.selectbox(
    "Choose an example:",
    [
        "Select an example",
        "Flood water has entered several houses and people need immediate rescue.",
        "The weather is beautiful today.",
        "People are trapped after the earthquake and need emergency assistance.",
        "I am watching a movie tonight."
    ]
)


# ============================================================
# USE SELECTED EXAMPLE
# ============================================================

if example != "Select an example":

    text = example

    st.text_area(
        "Selected text:",
        value=text,
        height=100,
        disabled=True
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    return predicted_class, confidence


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Text",
    type="primary",
    use_container_width=True
):

    if not text or not text.strip():

        st.warning(
            "⚠️ Please enter some social media text."
        )

    else:

        try:

            predicted_class, confidence = predict_text(
                text
            )

            confidence_percent = confidence * 100


            # ------------------------------------------------
            # LABEL MAPPING
            # ------------------------------------------------
            #
            # IMPORTANT:
            # Change these labels only if your training
            # dataset used the opposite class mapping.
            #
            # Usually:
            # 0 = not informative
            # 1 = informative
            # ------------------------------------------------

            if predicted_class == 1:

                label = "INFORMATIVE"

                st.success(
                    f"✅ Prediction: {label}"
                )

            else:

                label = "NOT INFORMATIVE"

                st.warning(
                    f"⚠️ Prediction: {label}"
                )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            st.metric(
                "Confidence",
                f"{confidence_percent:.2f}%"
            )


            # ------------------------------------------------
            # DETAILS
            # ------------------------------------------------

            st.subheader("Prediction Details")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.write("**Prediction**")

                st.write(label)

            with col2:

                st.write("**Class ID**")

                st.write(predicted_class)

            with col3:

                st.write("**Confidence**")

                st.write(
                    f"{confidence_percent:.2f}%"
                )


        except Exception as e:

            st.error(
                "❌ An error occurred while making the prediction."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Powered Social Media Crisis Detection & Early Warning System"
)



