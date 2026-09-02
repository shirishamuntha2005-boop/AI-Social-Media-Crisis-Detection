
# ============================================================
# REAL-TIME CRISIS SEVERITY DETECTION
# ============================================================

import os
import re
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "realtime_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "realtime_severity.csv"
)


# ============================================================
# SEVERITY KEYWORDS
# ============================================================

HIGH_SEVERITY_KEYWORDS = [

    "people are trapped",
    "trapped",
    "rescue",
    "rescued",
    "rescuing",
    "emergency",
    "missing",
    "injured",
    "injuries",
    "death",
    "dead",
    "killed",
    "casualties",
    "evacuate",
    "evacuated",
    "evacuation",
    "life threatening",
    "life-threatening",
    "critical",
    "collapsed",
    "collapse",
    "destroyed",
    "destruction",
    "victims",
    "fatal",
    "fatalities"
]


MEDIUM_SEVERITY_KEYWORDS = [

    "flood",
    "flooding",
    "flood water",
    "water levels",
    "rising rapidly",
    "roads are blocked",
    "road blocked",
    "blocked",
    "heavy rain",
    "storm",
    "cyclone",
    "landslide",
    "damage",
    "damaged",
    "power outage",
    "power cut",
    "rising water",
    "overflow",
    "disaster",
    "crisis",
    "warning",
    "danger",
    "shelter",
    "displaced",
    "relief",
    "urgent need"
]


# ============================================================
# NON-CRISIS KEYWORDS
# ============================================================

NON_CRISIS_KEYWORDS = [

    "beautiful weather",
    "good weather",
    "nice weather",
    "normal traffic",
    "normal day",
    "going out",
    "with friends",
    "happy",
    "birthday",
    "celebration",
    "movie",
    "music",
    "shopping",
    "restaurant",
    "vacation",
    "holiday",
    "sports",
    "game",
    "concert"
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# CHECK KEYWORDS
# ============================================================

def find_matches(text, keywords):

    matches = []

    for keyword in keywords:

        if keyword in text:

            matches.append(keyword)

    return matches


# ============================================================
# DETERMINE SEVERITY
# ============================================================

def determine_severity(
    text,
    prediction,
    confidence
):

    text = normalize_text(text)

    prediction = (
        str(prediction)
        .strip()
        .lower()
    )

    try:

        confidence = float(confidence)

    except:

        confidence = 0.0


    # --------------------------------------------------------
    # NOT INFORMATIVE
    # --------------------------------------------------------

    if prediction != "informative":

        return (
            "Low",
            0,
            [],
            "Not Informative prediction"
        )


    # --------------------------------------------------------
    # NON-CRISIS CHECK
    # --------------------------------------------------------

    non_crisis_matches = find_matches(
        text,
        NON_CRISIS_KEYWORDS
    )

    if non_crisis_matches:

        return (
            "Low",
            0,
            non_crisis_matches,
            "Non-crisis content detected"
        )


    # --------------------------------------------------------
    # HIGH SEVERITY
    # --------------------------------------------------------

    high_matches = find_matches(
        text,
        HIGH_SEVERITY_KEYWORDS
    )

    if high_matches:

        return (
            "High",
            3,
            high_matches,
            "High-severity crisis indicators detected"
        )


    # --------------------------------------------------------
    # MEDIUM SEVERITY
    # --------------------------------------------------------

    medium_matches = find_matches(
        text,
        MEDIUM_SEVERITY_KEYWORDS
    )

    if medium_matches:

        return (
            "Medium",
            2,
            medium_matches,
            "Medium-severity crisis indicators detected"
        )


    # --------------------------------------------------------
    # INFORMATIVE BUT NO CRISIS EVIDENCE
    # --------------------------------------------------------

    return (
        "Low",
        0,
        [],
        "No crisis-specific evidence detected"
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    print()

    print("=" * 60)
    print("LOADING RoBERTa PREDICTIONS")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Prediction file not found:\n{INPUT_FILE}"
        )


    df = pd.read_csv(
        INPUT_FILE
    )


    required_columns = [

        "id",
        "text",
        "timestamp",
        "prediction",
        "confidence"

    ]


    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )


    if df.empty:

        raise ValueError(
            "Prediction file is empty."
        )


    print(
        "Predictions loaded:",
        len(df)
    )


    return df


# ============================================================
# PROCESS SEVERITY
# ============================================================

def process_severity(df):

    print()

    print("=" * 60)
    print("CRISIS SEVERITY DETECTION")
    print("=" * 60)


    severity_results = []


    for _, row in df.iterrows():

        post_id = row["id"]

        text = str(
            row["text"]
        )

        timestamp = row["timestamp"]

        prediction = row["prediction"]

        confidence = row["confidence"]


        # ----------------------------------------------------
        # Determine severity
        # ----------------------------------------------------

        (
            severity,
            severity_score,
            matches,
            reason

        ) = determine_severity(

            text,
            prediction,
            confidence

        )


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print()

        print(
            "Post ID:",
            post_id
        )

        print(
            "Text:",
            text
        )

        print(
            "Prediction:",
            prediction
        )

        print(
            "Confidence:",
            f"{float(confidence):.2f}%"
        )

        print(
            "Severity:",
            severity
        )

        print(
            "Severity Score:",
            severity_score
        )

        print(
            "Reason:",
            reason
        )


        if matches:

            print(
                "Matched Keywords:",
                ", ".join(matches)
            )

        else:

            print(
                "Matched Keywords: None"
            )


        print("-" * 60)


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        severity_results.append({

            "id":
                post_id,

            "text":
                text,

            "timestamp":
                timestamp,

            "prediction":
                prediction,

            "confidence":
                confidence,

            "severity":
                severity,

            "severity_score":
                severity_score,

            "severity_reason":
                reason,

            "severity_keywords":
                ", ".join(matches)

        })


    return pd.DataFrame(
        severity_results
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    df.to_csv(

        OUTPUT_FILE,

        index=False,

        encoding="utf-8"

    )


    print()

    print("=" * 60)
    print("SEVERITY RESULTS SAVED")
    print("=" * 60)


    print(
        "Output file:",
        OUTPUT_FILE
    )

    print(
        "Total records:",
        len(df)
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(df):

    print()

    print("=" * 60)
    print("SEVERITY SUMMARY")
    print("=" * 60)


    summary_columns = [

        "id",
        "prediction",
        "confidence",
        "severity",
        "severity_score"

    ]


    print(

        df[
            summary_columns
        ].to_string(
            index=False
        )

    )


    print()

    print("=" * 60)
    print("SEVERITY COUNTS")
    print("=" * 60)


    print(

        df[
            "severity"
        ]
        .value_counts()
        .to_string()

    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 60)
    print("AI-POWERED SOCIAL MEDIA CRISIS DETECTION")
    print("=" * 60)

    print(
        "REAL-TIME CRISIS SEVERITY MODULE"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    predictions_df = load_predictions()


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    severity_df = process_severity(
        predictions_df
    )


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    save_results(
        severity_df
    )


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    display_summary(
        severity_df
    )


    print()

    print("=" * 60)
    print(
        "SEVERITY DETECTION COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Program stopped by user."
        )

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
