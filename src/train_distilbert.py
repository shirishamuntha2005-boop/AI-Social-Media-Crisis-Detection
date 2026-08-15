# ================================================================
# AI-POWERED SOCIAL MEDIA CRISIS DETECTION
# DISTILBERT TRANSFORMER TRAINING
# ================================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ================================================================
# CONFIGURATION
# ================================================================

DATA_PATH = "data/processed/crisis_mmd_nlp.csv"

MODEL_NAME = "distilbert-base-uncased"

OUTPUT_DIR = "data/processed/distilbert_crisis_model"

MAX_LENGTH = 128

BATCH_SIZE = 8

EPOCHS = 3

LEARNING_RATE = 2e-5

RANDOM_STATE = 42


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ================================================================
# DEVICE
# ================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 GPU detected")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ GPU not detected")
    print("Using CPU")


# ================================================================
# HEADER
# ================================================================

print("\n" + "=" * 70)
print("AI SOCIAL MEDIA CRISIS DETECTION")
print("DISTILBERT TRANSFORMER MODEL TRAINING")
print("=" * 70)


# ================================================================
# CHECK DATASET
# ================================================================

print("\nChecking dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

print("✅ Dataset found")


# ================================================================
# LOAD DATASET
# ================================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

print("\nAvailable columns:")
print(df.columns.tolist())


# ================================================================
# SELECT COLUMNS
# ================================================================

TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "text_info"

if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"Text column '{TEXT_COLUMN}' not found."
    )

if LABEL_COLUMN not in df.columns:
    raise ValueError(
        f"Label column '{LABEL_COLUMN}' not found."
    )


print(f"\nText column  : {TEXT_COLUMN}")
print(f"Label column : {LABEL_COLUMN}")


# ================================================================
# CLEAN DATA
# ================================================================

print("\nCleaning data...")

df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

print(f"Rows before cleaning: {len(df)}")

df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna("").astype(str)

# Remove empty text
df = df[df[TEXT_COLUMN].str.strip() != ""]

# Keep only valid labels
df = df[
    df[LABEL_COLUMN].isin(
        ["informative", "not_informative"]
    )
]

# Remove duplicates
df = df.drop_duplicates(
    subset=[TEXT_COLUMN]
).reset_index(drop=True)

print(f"Rows after cleaning : {len(df)}")


# ================================================================
# LABEL ENCODING
# ================================================================

print("\nEncoding labels...")

label_mapping = {
    "not_informative": 0,
    "informative": 1
}

df["label"] = df[LABEL_COLUMN].map(label_mapping)

print("\nClass distribution:")
print(df[LABEL_COLUMN].value_counts())


# ================================================================
# TRAIN / TEST SPLIT
# ================================================================

print("\nSplitting dataset...")

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples : {len(test_df)}")


# ================================================================
# TOKENIZER
# ================================================================

print("\n" + "=" * 70)
print("LOADING DISTILBERT TOKENIZER")
print("=" * 70)

print("\nDownloading/loading tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_NAME
)

print("✅ Tokenizer loaded")


# ================================================================
# DATASET CLASS
# ================================================================

class CrisisDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        text = str(self.texts[index])

        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                label,
                dtype=torch.long
            )
        }


# ================================================================
# CREATE DATASETS
# ================================================================

print("\nCreating PyTorch datasets...")

train_dataset = CrisisDataset(
    train_df[TEXT_COLUMN].tolist(),
    train_df["label"].tolist(),
    tokenizer,
    MAX_LENGTH
)

test_dataset = CrisisDataset(
    test_df[TEXT_COLUMN].tolist(),
    test_df["label"].tolist(),
    tokenizer,
    MAX_LENGTH
)

print("✅ Training dataset created")
print("✅ Testing dataset created")


# ================================================================
# CREATE DATALOADERS
# ================================================================

print("\nCreating DataLoaders...")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Training batches: {len(train_loader)}")
print(f"Testing batches : {len(test_loader)}")


# ================================================================
# LOAD DISTILBERT MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING DISTILBERT MODEL")
print("=" * 70)

print("\nLoading pretrained DistilBERT...")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)

print("✅ DistilBERT model loaded")

print(f"Model device: {DEVICE}")


# ================================================================
# OPTIMIZER
# ================================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ================================================================
# TRAINING
# ================================================================

print("\n" + "=" * 70)
print("STARTING DISTILBERT TRAINING")
print("=" * 70)

print(f"\nEpochs       : {EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Max length   : {MAX_LENGTH}")
print(f"Device       : {DEVICE}")


training_history = []


for epoch in range(EPOCHS):

    print("\n" + "-" * 70)

    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )

    print("-" * 70)

    model.train()

    total_loss = 0

    correct = 0
    total = 0

    for batch_number, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        logits = outputs.logits

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if (
            batch_number + 1
        ) % 100 == 0:

            print(
                f"Batch {batch_number + 1}/"
                f"{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = (
        total_loss /
        len(train_loader)
    )

    train_accuracy = (
        correct / total
    )

    print("\nEpoch Training Results")

    print(
        f"Training Loss     : "
        f"{avg_loss:.4f}"
    )

    print(
        f"Training Accuracy : "
        f"{train_accuracy * 100:.2f}%"
    )


    # ============================================================
    # VALIDATION / TEST EVALUATION
    # ============================================================

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    epoch_precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    epoch_recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    print("\nTest Results")

    print(
        f"Accuracy  : "
        f"{epoch_accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{epoch_precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{epoch_recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{epoch_f1 * 100:.2f}%"
    )


    training_history.append({

        "epoch": epoch + 1,

        "train_loss": avg_loss,

        "train_accuracy": train_accuracy,

        "test_accuracy": epoch_accuracy,

        "precision": epoch_precision,

        "recall": epoch_recall,

        "f1_score": epoch_f1

    })


# ================================================================
# FINAL EVALUATION
# ================================================================

print("\n" + "=" * 70)

print("FINAL DISTILBERT RESULTS")

print("=" * 70)


final_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

final_precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

final_recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

final_f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


print(
    f"Accuracy  : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"Precision : "
    f"{final_precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{final_recall * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{final_f1 * 100:.2f}%"
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

print("\n" + "=" * 70)

print("CLASSIFICATION REPORT")

print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "not_informative",
            "informative"
        ],
        zero_division=0
    )
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

print("\n" + "=" * 70)

print("CONFUSION MATRIX")

print("=" * 70)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)


# ================================================================
# SAVE MODEL
# ================================================================

print("\n" + "=" * 70)

print("SAVING DISTILBERT MODEL")

print("=" * 70)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


print(
    f"\n✅ Model saved to:\n"
    f"{OUTPUT_DIR}"
)


# ================================================================
# SAVE TRAINING HISTORY
# ================================================================

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.json"
)

with open(
    history_path,
    "w"
) as file:

    json.dump(
        training_history,
        file,
        indent=4
    )


# ================================================================
# SAVE FINAL METRICS
# ================================================================

metrics = {

    "model": "DistilBERT",

    "base_model": MODEL_NAME,

    "accuracy": final_accuracy,

    "precision": final_precision,

    "recall": final_recall,

    "f1_score": final_f1,

    "epochs": EPOCHS,

    "batch_size": BATCH_SIZE,

    "learning_rate": LEARNING_RATE,

    "max_length": MAX_LENGTH,

    "training_samples": len(train_df),

    "testing_samples": len(test_df)

}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "distilbert_metrics.json"
)


with open(
    metrics_path,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    f"✅ Training history saved:\n"
    f"{history_path}"
)

print(
    f"✅ Metrics saved:\n"
    f"{metrics_path}"
)


# ================================================================
# FINAL COMPARISON
# ================================================================

print("\n" + "=" * 70)

print("MODEL COMPARISON")

print("=" * 70)

print(
    "\nCurrent Hybrid Ensemble:"
)

print(
    "Accuracy : 82.16%"
)

print(
    "F1 Score : 81.58%"
)

print(
    "\nNew DistilBERT:"
)

print(
    f"Accuracy : "
    f"{final_accuracy * 100:.2f}%"
)

print(
    f"F1 Score : "
    f"{final_f1 * 100:.2f}%"
)


if final_accuracy > 0.8216:

    print(
        "\n🎉 DistilBERT improved "
        "the previous accuracy!"
    )

else:

    print(
        "\nℹ️ DistilBERT did not exceed "
        "the current 82.16% accuracy "
        "yet."
    )


print("\n" + "=" * 70)

print("🎉 DISTILBERT TRAINING COMPLETED!")

print("=" * 70)