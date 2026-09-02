import os
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
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

MODEL_NAME = "roberta-base"

DATA_PATH = "./data/processed/crisis_mmd_nlp.csv"

OUTPUT_DIR = "./outputs/roberta_crisis_model"

RANDOM_STATE = 42

MAX_LENGTH = 128

NUM_LABELS = 2

NUM_EPOCHS = 3

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

TRAIN_BATCH_SIZE = 8

EVAL_BATCH_SIZE = 8


# ============================================================
# DEVICE INFORMATION
# ============================================================

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)

if torch.cuda.is_available():

    device = "cuda"

    print("Device: GPU")
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    device = "cpu"

    print("Device: CPU")
    print("GPU not available.")
    print("Training will use CPU.")


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset shape:",
    df.shape
)

print("\nColumns:")

print(
    df.columns.tolist()
)


# ============================================================
# 2. CREATE LABEL
# ============================================================

print("\n" + "=" * 60)
print("CREATING LABELS")
print("=" * 60)


if "label" not in df.columns:

    df["label"] = df["text_info"].map({

        "not_informative": 0,

        "informative": 1

    })


# Remove missing values

df = df.dropna(
    subset=[
        "processed_text",
        "label"
    ]
).copy()


# Convert text to string

df["processed_text"] = (
    df["processed_text"]
    .astype(str)
)


# Convert label to integer

df["label"] = (
    df["label"]
    .astype(int)
)


print("\nLabel mapping:")

print(
    "0 = not_informative"
)

print(
    "1 = informative"
)


print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)


train_df, temp_df = train_test_split(

    df,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=df["label"]
)


val_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=temp_df["label"]
)


print(
    "\nTraining samples:",
    len(train_df)
)

print(
    "Validation samples:",
    len(val_df)
)

print(
    "Testing samples:",
    len(test_df)
)


print("\nExpected:")

print(
    "Training      = 70%"
)

print(
    "Validation    = 15%"
)

print(
    "Testing       = 15%"
)


# ============================================================
# 4. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\nTraining distribution:")

print(
    train_df["label"].value_counts()
)


print("\nValidation distribution:")

print(
    val_df["label"].value_counts()
)


print("\nTesting distribution:")

print(
    test_df["label"].value_counts()
)


# ============================================================
# 5. CREATE HUGGING FACE DATASETS
# ============================================================

print("\n" + "=" * 60)
print("CREATING HUGGING FACE DATASETS")
print("=" * 60)


train_dataset = Dataset.from_pandas(

    train_df[
        [
            "processed_text",
            "label"
        ]
    ],

    preserve_index=False
)


val_dataset = Dataset.from_pandas(

    val_df[
        [
            "processed_text",
            "label"
        ]
    ],

    preserve_index=False
)


test_dataset = Dataset.from_pandas(

    test_df[
        [
            "processed_text",
            "label"
        ]
    ],

    preserve_index=False
)


print("Datasets created successfully.")


# ============================================================
# 6. LOAD ROBERTA TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING ROBERTA TOKENIZER")
print("=" * 60)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print(
    "Tokenizer loaded successfully."
)


# ============================================================
# 7. TOKENIZATION
# ============================================================

print("\n" + "=" * 60)
print("TOKENIZING DATASETS")
print("=" * 60)


def tokenize_function(examples):

    return tokenizer(

        examples["processed_text"],

        truncation=True,

        padding="max_length",

        max_length=MAX_LENGTH
    )


print("\nTokenizing training data...")

train_dataset = train_dataset.map(

    tokenize_function,

    batched=True
)


print("\nTokenizing validation data...")

val_dataset = val_dataset.map(

    tokenize_function,

    batched=True
)


print("\nTokenizing test data...")

test_dataset = test_dataset.map(

    tokenize_function,

    batched=True
)


# ============================================================
# 8. REMOVE ORIGINAL TEXT
# ============================================================

train_dataset = train_dataset.remove_columns(

    ["processed_text"]
)


val_dataset = val_dataset.remove_columns(

    ["processed_text"]
)


test_dataset = test_dataset.remove_columns(

    ["processed_text"]
)


# ============================================================
# 9. SET PYTORCH FORMAT
# ============================================================

train_dataset.set_format(
    type="torch"
)


val_dataset.set_format(
    type="torch"
)


test_dataset.set_format(
    type="torch"
)


print("\nPyTorch format configured.")


# ============================================================
# 10. CALCULATE CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING CLASS WEIGHTS")
print("=" * 60)


class_counts = np.bincount(

    train_df["label"].values
)


total_samples = len(
    train_df
)


class_weights = (

    total_samples /

    (
        NUM_LABELS *
        class_counts
    )
)


class_weights = torch.tensor(

    class_weights,

    dtype=torch.float
)


print(
    "Class counts:"
)

print(
    class_counts
)


print(
    "\nClass weights:"
)

print(
    class_weights
)


# ============================================================
# 11. LOAD ROBERTA MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING ROBERTA MODEL")
print("=" * 60)


model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=NUM_LABELS
)


print(
    "\nRoBERTa model loaded successfully."
)


# ============================================================
# 12. CUSTOM WEIGHTED TRAINER
# ============================================================

print("\n" + "=" * 60)
print("CREATING WEIGHTED TRAINER")
print("=" * 60)


class WeightedTrainer(Trainer):

    def compute_loss(

        self,

        model,

        inputs,

        return_outputs=False,

        num_items_in_batch=None

    ):

        labels = inputs.get(
            "labels"
        )


        outputs = model(
            **inputs
        )


        logits = outputs.get(
            "logits"
        )


        loss_function = (
            torch.nn.CrossEntropyLoss(
                weight=class_weights.to(
                    logits.device
                )
            )
        )


        loss = loss_function(

            logits.view(
                -1,
                NUM_LABELS
            ),

            labels.view(-1)
        )


        if return_outputs:

            return (
                loss,
                outputs
            )

        return loss


print(
    "Weighted Trainer created."
)


# ============================================================
# 13. METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred


    predictions = np.argmax(

        logits,

        axis=1
    )


    accuracy = accuracy_score(

        labels,

        predictions
    )


    precision, recall, f1, _ = (

        precision_recall_fscore_support(

            labels,

            predictions,

            average="binary",

            zero_division=0
        )
    )


    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1

    }


# ============================================================
# 14. TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)


print(
    "Model:",
    MODEL_NAME
)

print(
    "Epochs:",
    NUM_EPOCHS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Train batch size:",
    TRAIN_BATCH_SIZE
)

print(
    "Evaluation batch size:",
    EVAL_BATCH_SIZE
)

print(
    "Max sequence length:",
    MAX_LENGTH
)

print(
    "Weight decay:",
    WEIGHT_DECAY
)

print(
    "Best model metric: F1"
)


# ============================================================
# 15. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    weight_decay=WEIGHT_DECAY,

    logging_dir=os.path.join(

        OUTPUT_DIR,

        "logs"
    ),

    logging_steps=100,

    eval_strategy="steps",

    eval_steps=500,

    save_strategy="steps",

    save_steps=500,

    save_total_limit=3,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    report_to="none",

    fp16=False,

    dataloader_pin_memory=False
)


# ============================================================
# 16. CREATE TRAINER
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAINER")
print("=" * 60)


trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics,

    callbacks=[

        EarlyStoppingCallback(

            early_stopping_patience=2

        )

    ]
)


print(
    "Trainer created successfully."
)


# ============================================================
# 17. START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING ROBERTA TRAINING")
print("=" * 60)


print(
    "\nTraining started..."
)

print(
    "Because you are using CPU, this may take a long time."
)


train_result = trainer.train()


# ============================================================
# 18. SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING BEST ROBERTA MODEL")
print("=" * 60)


trainer.save_model(
    OUTPUT_DIR
)


tokenizer.save_pretrained(
    OUTPUT_DIR
)


print(
    "Best model saved successfully."
)


# ============================================================
# 19. VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)


validation_results = trainer.evaluate(

    eval_dataset=val_dataset
)


for key, value in validation_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# 20. FINAL TEST PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("RUNNING FINAL TEST PREDICTION")
print("=" * 60)


test_predictions = trainer.predict(

    test_dataset
)


test_logits = (
    test_predictions.predictions
)


predicted_labels = np.argmax(

    test_logits,

    axis=1
)


true_labels = np.array(

    test_df["label"]
)


# ============================================================
# 21. CALCULATE FINAL TEST METRICS
# ============================================================

accuracy = accuracy_score(

    true_labels,

    predicted_labels
)


precision, recall, f1, _ = (

    precision_recall_fscore_support(

        true_labels,

        predicted_labels,

        average="binary",

        zero_division=0
    )
)


# ============================================================
# 22. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL ROBERTA TEST RESULTS")
print("=" * 60)


print(

    f"Accuracy : "
    f"{accuracy * 100:.2f}%"
)


print(

    f"Precision: "
    f"{precision * 100:.2f}%"
)


print(

    f"Recall   : "
    f"{recall * 100:.2f}%"
)


print(

    f"F1 Score : "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# 23. CLASSIFICATION REPORT
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


print(
    report
)


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(

    true_labels,

    predicted_labels
)


print(
    cm
)


print("\nConfusion Matrix:")

print(
    "                 Predicted"
)

print(
    "              Not Info  Info"
)


print(

    f"Actual Not Info   "
    f"{cm[0][0]:5d}   "
    f"{cm[0][1]:5d}"
)


print(

    f"Actual Info       "
    f"{cm[1][0]:5d}   "
    f"{cm[1][1]:5d}"
)


# ============================================================
# 25. SAVE FINAL METRICS
# ============================================================

results = {

    "model": "RoBERTa",

    "model_name": MODEL_NAME,

    "accuracy": float(
        accuracy
    ),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1_score": float(
        f1
    ),

    "train_samples": int(
        len(train_df)
    ),

    "validation_samples": int(
        len(val_df)
    ),

    "test_samples": int(
        len(test_df)
    ),

    "epochs": NUM_EPOCHS,

    "learning_rate": LEARNING_RATE,

    "max_length": MAX_LENGTH,

    "train_batch_size": TRAIN_BATCH_SIZE,

    "evaluation_batch_size": EVAL_BATCH_SIZE,

    "class_weights": [

        float(x)

        for x in class_weights

    ],

    "confusion_matrix": (
        cm.tolist()
    )
}


# ============================================================
# 26. SAVE JSON
# ============================================================

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True
)


results_file = os.path.join(

    OUTPUT_DIR,

    "roberta_final_metrics.json"
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
# 27. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("ROBERTA TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)


print(
    "\nBest model saved to:"
)

print(
    OUTPUT_DIR
)


print(
    "\nMetrics saved to:"
)

print(
    results_file
)


print(
    "\nDone!"
)