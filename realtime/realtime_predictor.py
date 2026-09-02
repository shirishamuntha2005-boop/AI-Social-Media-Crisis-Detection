
# ============================================================
# AI-POWERED SOCIAL MEDIA CRISIS DETECTION
# REAL-TIME RoBERTa PREDICTOR
# ============================================================

import os
import pandas as pd
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "outputs/roberta_crisis_model/checkpoint-2810"

INPUT_FILE = "realtime/realtime_posts.csv"

OUTPUT_FILE = "realtime/realtime_predictions.csv"


# ============================================================
# LABEL MAPPING
# ============================================================

# Your trained model:
# 0 = Not Informative
# 1 = Informative

LABEL_NAMES = {
    0: "Not Informative",
    1: "Informative"
}


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# LOAD RoBERTa MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("LOADING RoBERTa MODEL")
    print("=" * 60)

    print("Model path:", MODEL_PATH)
    print("Device:", DEVICE)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"RoBERTa model not found:\n{MODEL_PATH}"
        )

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    print("Loading RoBERTa model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(DEVICE)

    model.eval()

    print("RoBERTa model loaded successfully.")

    return tokenizer, model


# ============================================================
# PREDICT ONE POST
# ============================================================

def predict_post(
    text,
    tokenizer,
    model
):

    # --------------------------------------------------------
    # Tokenize text
    # --------------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # --------------------------------------------------------
    # Move inputs to device
    # --------------------------------------------------------

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    # --------------------------------------------------------
    # Get confidence
    # --------------------------------------------------------

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    # --------------------------------------------------------
    # Convert class number to label
    # --------------------------------------------------------

    label = LABEL_NAMES.get(
        predicted_class,
        "Unknown"
    )

    return label, confidence


# ============================================================
# LOAD REAL-TIME POSTS
# ============================================================

def load_posts():

    print()
    print("=" * 60)
    print("LOADING REAL-TIME POSTS")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Real-time posts file not found:\n{INPUT_FILE}"
        )

    # Read CSV
    df = pd.read_csv(
        INPUT_FILE
    )

    # Required columns
    required_columns = [
        "id",
        "text",
        "timestamp"
    ]

    # Check columns
    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    # Check empty file
    if df.empty:

        raise ValueError(
            "The real-time posts CSV is empty."
        )

    print(
        "Posts loaded:",
        len(df)
    )

    return df


# ============================================================
# PREDICT ALL POSTS
# ============================================================

def predict_all_posts(
    df,
    tokenizer,
    model
):

    print()
    print("=" * 60)
    print("RoBERTa REAL-TIME PREDICTIONS")
    print("=" * 60)

    results = []

    # --------------------------------------------------------
    # Process each post
    # --------------------------------------------------------

    for _, row in df.iterrows():

        post_id = row["id"]

        text = str(
            row["text"]
        )

        timestamp = row["timestamp"]

        # Predict
        label, confidence = predict_post(
            text,
            tokenizer,
            model
        )

        # Convert confidence to percentage
        confidence_percent = (
            confidence * 100
        )

        # Display
        print()
        print("Post ID:", post_id)
        print("Text:", text)
        print("Timestamp:", timestamp)
        print("Prediction:", label)
        print(
            "Confidence:",
            f"{confidence_percent:.2f}%"
        )

        print("-" * 60)

        # Store result
        results.append({
            "id": post_id,
            "text": text,
            "timestamp": timestamp,
            "prediction": label,
            "confidence": round(
                confidence_percent,
                2
            )
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results
    )

    return results_df


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    results_df
):

    # Save predictions
    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("PREDICTIONS SAVED")
    print("=" * 60)

    print(
        "Output file:",
        OUTPUT_FILE
    )

    print(
        "Total predictions:",
        len(results_df)
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(
    results_df
):

    print()
    print("=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    print(
        results_df[
            [
                "id",
                "prediction",
                "confidence"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 60)
    print("PREDICTION COUNTS")
    print("=" * 60)

    counts = (
        results_df[
            "prediction"
        ]
        .value_counts()
    )

    print(
        counts.to_string()
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 60)
    print("AI-POWERED SOCIAL MEDIA CRISIS DETECTION")
    print("=" * 60)

    print(
        "REAL-TIME RoBERTa PREDICTION SYSTEM"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: Load RoBERTa
    # --------------------------------------------------------

    tokenizer, model = load_model()

    # --------------------------------------------------------
    # STEP 2: Load real-time posts
    # --------------------------------------------------------

    posts_df = load_posts()

    # --------------------------------------------------------
    # STEP 3: Predict posts
    # --------------------------------------------------------

    results_df = predict_all_posts(
        posts_df,
        tokenizer,
        model
    )

    # --------------------------------------------------------
    # STEP 4: Save predictions
    # --------------------------------------------------------

    save_predictions(
        results_df
    )

    # --------------------------------------------------------
    # STEP 5: Display summary
    # --------------------------------------------------------

    display_summary(
        results_df
    )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Prediction stopped by user.")

    except Exception as error:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(
            type(error).__name__,
            ":",
            error
        )

