import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# LOAD ROBERTA MODEL
# ==========================================

MODEL_PATH = "outputs/roberta_crisis_model"

print("=" * 50)
print("LOADING ROBERTA MODEL")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_crisis(text):

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

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0][prediction].item()

    if prediction == 1:
        label = "INFORMATIVE"
    else:
        label = "NOT INFORMATIVE"

    return label, confidence


# ==========================================
# TEST EXAMPLES
# ==========================================

examples = [
    "A major earthquake has hit the city and emergency teams are rescuing people.",

    "I am having a great day with my friends!",

    "Flood waters have entered several homes and people need immediate rescue.",

    "Just watched a really good movie tonight.",

    "People are trapped after the building collapsed. Please send emergency help."
]


# ==========================================
# TEST THE MODEL
# ==========================================

print()
print("=" * 50)
print("TESTING ROBERTA")
print("=" * 50)

for text in examples:

    label, confidence = predict_crisis(text)

    print()
    print("Text:")
    print(text)

    print("Prediction:", label)
    print("Confidence:", round(confidence * 100, 2), "%")

    print("-" * 50)


# ==========================================
# MANUAL INPUT
# ==========================================

print()
print("=" * 50)
print("MANUAL TEST")
print("=" * 50)

while True:

    text = input("\nEnter a social media post (or type 'exit'): ")

    if text.lower() == "exit":
        print("Program stopped.")
        break

    if not text.strip():
        print("Please enter some text.")
        continue

    label, confidence = predict_crisis(text)

    print()
    print("RESULT")
    print("Prediction :", label)
    print("Confidence :", round(confidence * 100, 2), "%")