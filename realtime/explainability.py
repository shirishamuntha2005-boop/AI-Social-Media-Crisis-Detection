import re


# Crisis-related indicator words
CRISIS_KEYWORDS = {
    "flood",
    "flooding",
    "earthquake",
    "fire",
    "cyclone",
    "storm",
    "disaster",
    "emergency",
    "rescue",
    "evacuation",
    "evacuate",
    "injured",
    "injury",
    "dead",
    "death",
    "destroyed",
    "damage",
    "damaged",
    "stranded",
    "trapped",
    "victims",
    "warning",
    "danger",
    "urgent",
    "collapsed",
    "collapse",
    "explosion",
    "landslide",
    "tsunami"
}


def extract_crisis_indicators(text):
    """
    Find crisis-related words in a social-media post.
    """

    if not isinstance(text, str):
        return []

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    indicators = []

    for word in words:
        if word in CRISIS_KEYWORDS:
            if word not in indicators:
                indicators.append(word)

    return indicators


def generate_explanation(
    text,
    prediction,
    confidence
):
    """
    Generate a simple human-readable explanation.
    """

    indicators = extract_crisis_indicators(text)

    prediction = str(prediction)

    try:
        confidence = float(confidence)

        if confidence <= 1:
            confidence *= 100

    except (ValueError, TypeError):
        confidence = 0

    if indicators:

        explanation = (
            "Crisis-related indicators detected: "
            + ", ".join(indicators)
        )

    else:

        explanation = (
            "No strong predefined crisis indicators "
            "were detected."
        )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "indicators": indicators,
        "explanation": explanation
    }


def print_explanation(result):

    print("\n" + "=" * 70)
    print("CRISIS PREDICTION EXPLANATION")
    print("=" * 70)

    print(
        f"Prediction : {result['prediction']}"
    )

    print(
        f"Confidence : {result['confidence']}%"
    )

    if result["indicators"]:

        print("\nImportant Crisis Indicators:")

        for indicator in result["indicators"]:
            print(f"  ✓ {indicator}")

    print(
        f"\nExplanation: {result['explanation']}"
    )

    print("=" * 70)


if __name__ == "__main__":

    test_post = (
        "People are stranded after severe flooding "
        "and rescue teams are responding to the emergency."
    )

    result = generate_explanation(
        text=test_post,
        prediction="INFORMATIVE",
        confidence=94
    )

    print_explanation(result)