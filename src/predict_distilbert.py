import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# ============================================================
# DISTILBERT CRISIS DETECTION - PREDICTION
# ============================================================

MODEL_PATH = "data/processed/distilbert_crisis_model"

print("=" * 70)
print("DISTILBERT CRISIS DETECTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Select device
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")


# ------------------------------------------------------------
# 2. Load tokenizer
# ------------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

print("✅ Tokenizer loaded")


# ------------------------------------------------------------
# 3. Load trained model
# ------------------------------------------------------------

print("\nLoading trained DistilBERT model...")

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

print("✅ Model loaded successfully")


# ------------------------------------------------------------
# 4. Prediction function
# ------------------------------------------------------------

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()

        confidence = probabilities[0][predicted_class].item() * 100


    # IMPORTANT:
    # 0 = not_informative
    # 1 = informative

    if predicted_class == 1:
        label = "INFORMATIVE"
    else:
        label = "NOT_INFORMATIVE"


    return label, confidence


# ------------------------------------------------------------
# 5. Test messages
# ------------------------------------------------------------

test_messages = [

    "Flood water has entered several houses and people need immediate help.",

    "A major earthquake has damaged buildings and many people are injured.",

    "I watched a movie with my friends yesterday.",

    "The weather is beautiful today and I am enjoying my weekend."

]


# ------------------------------------------------------------
# 6. Run predictions
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("PREDICTIONS")
print("=" * 70)


for i, message in enumerate(test_messages, start=1):

    label, confidence = predict(message)

    print(f"\nTest {i}")
    print("-" * 70)

    print(f"Text       : {message}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}%")


print("\n")
print("=" * 70)
print("PREDICTION TEST COMPLETED")
print("=" * 70)