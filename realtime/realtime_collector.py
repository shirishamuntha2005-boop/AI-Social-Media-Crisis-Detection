```python
# ============================================================
# AI-POWERED SOCIAL MEDIA CRISIS DETECTION
# REAL-TIME RoBERTa PREDICTOR
# ============================================================
#
# Purpose:
#   1. Load the existing trained RoBERTa model
#   2. Read posts from realtime_posts.csv
#   3. Predict Informative / Not Informative
#   4. Calculate prediction confidence
#   5. Display results
#
# IMPORTANT:
#   This file does NOT train the model.
#   This file only uses the already-trained RoBERTa model.
#
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PROJECT PATHS
# ============================================================

# Best RoBERTa checkpoint from your previous training
MODEL_PATH = "outputs/roberta_crisis_model/checkpoint-2810"

# Real-time posts collected by realtime_collector.py
INPUT_FILE = "realtime/realtime_posts.csv"


# ============================================================
# LABEL MAPPING
# ============================================================

# Your dataset uses:
#
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

    # --------------------------------------------------------
    # Check whether model exists
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "\nRoBERTa model was not found.\n\n"
            f"Expected location:\n{MODEL_PATH}\n\n"
            "Please check that checkpoint-2810 exists."
        )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Load trained RoBERTa model
    # --------------------------------------------------------

    print("Loading RoBERTa model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Move model to CPU or GPU
    # --------------------------------------------------------

    model.to(DEVICE)

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    print("\nRoBERTa model loaded successfully.")

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
    # Move tensors to same device as model
    # --------------------------------------------------------

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Make prediction
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
    # Find predicted class
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
    # Convert numeric class to label
    # --------------------------------------------------------

    label = LABEL_NAMES.get(
        predicted_class,
        f"Unknown ({predicted_class})"
    )

    return label, confidence


# ============================================================
# LOAD REAL-TIME POSTS
# ============================================================

def load_posts():

    print("\n" + "=" * 60)
    print("LOADING REAL-TIME POSTS")
    print("=" * 60)

    # --------------------------------------------------------
    # Check CSV file
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            "\nReal-time posts file was not found.\n\n"
            f"Expected location:\n{INPUT_FILE}\n\n"
            "Please run realtime_collector.py first."
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "id",
        "text",
        "timestamp"
    ]

    missing_columns = []

    for column in required_columns:

        if column not in df.columns:

            missing_columns.append(column)

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Check empty dataset
    # --------------------------------------------------------

    if len(df) == 0:

        raise ValueError(
            "The real-time posts CSV is empty."
        )

    print("Posts loaded:", len(df))

    return df


# ============================================================
# PREDICT ALL POSTS
# ============================================================

def predict_all_posts(
    df,
    tokenizer,
    model
):

    print("\n" + "=" * 60)
    print("RoBERTa REAL-TIME PREDICTIONS")
    print("=" * 60)

    results = []

    # --------------------------------------------------------
    # Process every post
    # --------------------------------------------------------

    for _, row in df.iterrows():

        post_id = row["id"]

        text = str(
            row["text"]
        )

        timestamp = row["timestamp"]

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        label, confidence = predict_post(
            text,
            tokenizer,
            model
        )

        confidence_percentage = (
            confidence * 100
        )

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        print("\nPost ID:", post_id)

        print(
            "Text:",
            text
        )

        print(
            "Timestamp:",
            timestamp
        )

        print(
            "Prediction:",
            label
        )

        print(
            "Confidence:",
            f"{confidence_percentage:.2f}%"
        )

        print("-" * 60)

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "id": post_id,

            "text": text,

            "timestamp": timestamp,

            "prediction": label,

            "confidence": round(
                confidence_percentage,
                2
            )

        })

    # --------------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    return results_df


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(
    results_df
):

    print("\n" + "=" * 60)
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

    # --------------------------------------------------------
    # Prediction counts
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREDICTION COUNTS")
    print("=" * 60)

    prediction_counts = (
        results_df[
            "prediction"
        ]
        .value_counts()
    )

    print(
        prediction_counts.to_string()
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")

    print("=" * 60)
    print("AI-POWERED SOCIAL MEDIA CRISIS DETECTION")
    print("=" * 60)

    print("REAL-TIME RoBERTa PREDICTION SYSTEM")

    print("=" * 60)

    try:

        # ----------------------------------------------------
        # STEP 1
        # Load tokenizer and model
        # ----------------------------------------------------

        tokenizer, model = load_model()

        # ----------------------------------------------------
        # STEP 2
        # Load real-time posts
        # ----------------------------------------------------

        posts_df = load_posts()

        # ----------------------------------------------------
        # STEP 3
        # Predict all posts
        # ----------------------------------------------------

        results_df = predict_all_posts(
            posts_df,
            tokenizer,
            model
        )

        # ----------------------------------------------------
        # STEP 4
        # Display summary
        # ----------------------------------------------------

        display_summary(
            results_df
        )

        # ----------------------------------------------------
        # Finished
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("PREDICTION COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except FileNotFoundError as error:

        print("\n" + "=" * 60)
        print("FILE ERROR")
        print("=" * 60)

        print(error)

    except ValueError as error:

        print("\n" + "=" * 60)
        print("DATA ERROR")
        print("=" * 60)

        print(error)

    except Exception as error:

        print("\n" + "=" * 60)
        print("UNEXPECTED ERROR")
        print("=" * 60)

        print(
            type(error).__name__,
            ":",
            error
        )
```
