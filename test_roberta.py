import os
import json
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "./outputs/roberta_crisis_model/checkpoint-2810"
DATA_PATH = "./data/processed/crisis_mmd_nlp.csv"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("Columns:")
print(df.columns.tolist())


# ============================================================
# 2. CREATE LABEL
# ============================================================

if "label" not in df.columns:

    df["label"] = df["text_info"].map({
        "not_informative": 0,
        "informative": 1
    })

df["label"] = df["label"].astype(int)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 3. REMOVE MISSING TEXT
# ============================================================

df = df.dropna(
    subset=["processed_text"]
).copy()

df["processed_text"] = df["processed_text"].astype(str)

print("\nDataset after removing missing text:")
print(df.shape)


# ============================================================
# 4. CREATE SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Testing samples:", len(test_df))

print("\nExpected:")
print("Training      = 70%")
print("Validation    = 15%")
print("Testing       = 15%")

print("\nTesting distribution:")
print(test_df["label"].value_counts())


# ============================================================
# 5. CREATE HUGGING FACE TEST DATASET
# ============================================================

test_dataset = Dataset.from_pandas(
    test_df[["processed_text", "label"]],
    preserve_index=False
)


# ============================================================
# 6. LOAD ROBERTA TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING ROBERTA TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


# ============================================================
# 7. TOKENIZE TEST DATA
# ============================================================

print("\nTokenizing test data...")


def tokenize_function(examples):

    return tokenizer(
        examples["processed_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# ============================================================
# 8. REMOVE ORIGINAL TEXT COLUMN
# ============================================================

test_dataset = test_dataset.remove_columns(
    ["processed_text"]
)


# ============================================================
# 9. SET PYTORCH FORMAT
# ============================================================

test_dataset.set_format(
    type="torch"
)


# ============================================================
# 10. LOAD BEST ROBERTA MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING BEST ROBERTA MODEL")
print("=" * 60)

print("Model path:")
print(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


# ============================================================
# 11. CREATE TRAINER
# ============================================================

print("\nCreating Trainer...")

# IMPORTANT:
# Do NOT use tokenizer=tokenizer here.
# New Transformers versions do not accept that argument.

trainer = Trainer(
    model=model
)


# ============================================================
# 12. RUN TEST PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("RUNNING TEST PREDICTION")
print("=" * 60)

predictions = trainer.predict(
    test_dataset
)


# ============================================================
# 13. GET PREDICTIONS
# ============================================================

logits = predictions.predictions

predicted_labels = np.argmax(
    logits,
    axis=1
)

true_labels = np.array(
    test_df["label"]
)


# ============================================================
# 14. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predicted_labels,
    average="binary",
    zero_division=0
)


# ============================================================
# 15. DISPLAY MAIN RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ROBERTA TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=[
        "not_informative",
        "informative"
    ],
    digits=4
)

print(report)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print(cm)

print("\nConfusion Matrix:")
print(
    "                 Predicted"
)

print(
    "              Not Info  Info"
)

print(
    f"Actual Not Info   {cm[0][0]:5d}   {cm[0][1]:5d}"
)

print(
    f"Actual Info       {cm[1][0]:5d}   {cm[1][1]:5d}"
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results = {
    "model": "RoBERTa",
    "checkpoint": "checkpoint-2810",
    "test_samples": int(len(test_df)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}


output_folder = (
    "./outputs/roberta_crisis_model"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


results_file = os.path.join(
    output_folder,
    "test_metrics.json"
)


with open(
    results_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# 19. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TESTING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nResults saved to:")
print(results_file)