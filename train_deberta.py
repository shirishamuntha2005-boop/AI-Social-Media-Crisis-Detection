import os
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
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
# 1. PATHS
# ============================================================

DATA_PATH = "./data/processed/crisis_mmd_master.csv"

OUTPUT_DIR = "./outputs/deberta_crisis_model"

MODEL_NAME = "microsoft/deberta-v3-base"

MAX_LENGTH = 128


# ============================================================
# 2. DEVICE INFORMATION
# ============================================================

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Device: GPU")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("Device: CPU")
    print("GPU not available.")
    print("Training will use CPU.")

print()


# ============================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 5. CREATE LABEL
# ============================================================

print("\n" + "=" * 60)
print("CREATING LABELS")
print("=" * 60)

# CrisisMMD text_info contains informative / not_informative
df["label"] = df["text_info"].map({
    "informative": 1,
    "not_informative": 0
})

# Remove rows where label could not be created
df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 6. SELECT TEXT COLUMN
# ============================================================

# Use processed_text if available.
# If unavailable, use clean_text.

if "processed_text" in df.columns:
    TEXT_COLUMN = "processed_text"
elif "clean_text" in df.columns:
    TEXT_COLUMN = "clean_text"
else:
    TEXT_COLUMN = "tweet_text"

print("\nText column being used:", TEXT_COLUMN)


# Convert text to string
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

# Remove empty text
df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)

print("Dataset after removing missing/empty text:")
print(df.shape)


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

X = df[TEXT_COLUMN]
y = df["label"]

# First: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second: split temporary 50/50
# 15% validation + 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

print("\nExpected:")
print("Training      = 70%")
print("Validation    = 15%")
print("Testing       = 15%")


# ============================================================
# 8. CREATE HUGGING FACE DATASETS
# ============================================================

train_df = pd.DataFrame({
    "text": X_train.values,
    "label": y_train.values
})

val_df = pd.DataFrame({
    "text": X_val.values,
    "label": y_val.values
})

test_df = pd.DataFrame({
    "text": X_test.values,
    "label": y_test.values
})

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# ============================================================
# 9. LOAD DEBERTA TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING DEBERTA TOKENIZER")
print("=" * 60)

print("Model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 10. TOKENIZATION
# ============================================================

print("\n" + "=" * 60)
print("TOKENIZING DATA")
print("=" * 60)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("Tokenizing training data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("Tokenizing validation data...")

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("Tokenizing testing data...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


# ============================================================
# 11. LOAD DEBERTA MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING DEBERTA MODEL")
print("=" * 60)

print("Model:", MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "not_informative",
        1: "informative"
    },
    label2id={
        "not_informative": 0,
        "informative": 1
    }
)


# ============================================================
# 12. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 13. METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 14. TRAINING ARGUMENTS
# ============================================================

print("\n" + "=" * 60)
print("SETTING TRAINING PARAMETERS")
print("=" * 60)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Training
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    learning_rate=2e-5,
    weight_decay=0.01,

    # Evaluation
    eval_strategy="epoch",

    # Save checkpoints
    save_strategy="epoch",
    save_total_limit=2,

    # Select best model
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    # Logging
    logging_strategy="steps",
    logging_steps=100,

    # CPU
    fp16=False,

    # Reproducibility
    seed=42,

    # Reporting
    report_to="none"
)


# ============================================================
# 15. CREATE TRAINER
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAINER")
print("=" * 60)

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# 16. START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING DEBERTA TRAINING")
print("=" * 60)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Epochs:", 3)
print("Batch size:", 8)
print("Learning rate:", 2e-5)

print("\nIMPORTANT:")
print("Training is running on CPU.")
print("This may take several hours.")
print("Do NOT close PowerShell while training is running.")

train_result = trainer.train()


# ============================================================
# 17. SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING BEST DEBERTA MODEL")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# 18. VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION RESULTS")
print("=" * 60)

validation_results = trainer.evaluate(
    eval_dataset=val_dataset
)

for key, value in validation_results.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value}")


# ============================================================
# 19. TEST SET PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("RUNNING TEST PREDICTION")
print("=" * 60)

test_predictions = trainer.predict(
    test_dataset
)

predicted_labels = np.argmax(
    test_predictions.predictions,
    axis=1
)

true_labels = np.array(
    test_predictions.label_ids
)


# ============================================================
# 20. TEST METRICS
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

print("\n" + "=" * 60)
print("DEBERTA TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")


# ============================================================
# 21. CLASSIFICATION REPORT
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
    zero_division=0
)

print(report)


# ============================================================
# 22. CONFUSION MATRIX
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
print("                 Predicted")
print("              Not Info  Info")
print(
    f"Actual Not Info    {cm[0][0]:4d}    {cm[0][1]:4d}"
)
print(
    f"Actual Info        {cm[1][0]:4d}    {cm[1][1]:4d}"
)


# ============================================================
# 23. SAVE TEST METRICS
# ============================================================

metrics = {
    "model": MODEL_NAME,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "test_samples": int(len(test_dataset)),
    "confusion_matrix": cm.tolist()
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "test_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# 24. COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("DEBERTA TRAINING AND TESTING COMPLETED")
print("=" * 60)

print("\nModel saved to:")
print(OUTPUT_DIR)

print("\nMetrics saved to:")
print(metrics_path)

print("\nFinal Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nCurrent project best:")
print("BERT = 84.33%")