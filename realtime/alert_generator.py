# ============================================================
# AI-POWERED SOCIAL MEDIA CRISIS DETECTION
# REAL-TIME CRISIS ALERT GENERATION
# ============================================================

import os
import pandas as pd
from datetime import datetime


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "realtime/realtime_severity.csv"
OUTPUT_FILE = "realtime/realtime_alerts.csv"


# ============================================================
# ALERT SETTINGS
# ============================================================

# Minimum confidence required for a low-severity
# informative post to generate an informational alert.
LOW_SEVERITY_CONFIDENCE_THRESHOLD = 85.0


# ============================================================
# LOAD SEVERITY DATA
# ============================================================

def load_severity_data():

    print()
    print("=" * 60)
    print("LOADING CRISIS SEVERITY DATA")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "id",
        "text",
        "timestamp",
        "prediction",
        "confidence",
        "severity",
        "severity_score"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    if df.empty:

        raise ValueError(
            "Severity file is empty."
        )

    print(
        "Records loaded:",
        len(df)
    )

    return df


# ============================================================
# GENERATE ONE ALERT
# ============================================================

def generate_alert(
    prediction,
    confidence,
    severity,
    severity_score
):

    prediction = str(
        prediction
    ).strip()

    severity = str(
        severity
    ).strip()

    confidence = float(
        confidence
    )

    severity_score = int(
        severity_score
    )

    # --------------------------------------------------------
    # NOT INFORMATIVE
    # --------------------------------------------------------

    if prediction.lower() != "informative":

        return (
            "No Alert",
            "NORMAL"
        )

    # --------------------------------------------------------
    # HIGH SEVERITY
    # --------------------------------------------------------

    if severity_score >= 3:

        return (
            "HIGH PRIORITY CRISIS ALERT",
            "URGENT"
        )

    # --------------------------------------------------------
    # MEDIUM SEVERITY
    # --------------------------------------------------------

    if severity_score == 2:

        return (
            "CRISIS MONITORING ALERT",
            "HIGH"
        )

    # --------------------------------------------------------
    # LOW SEVERITY
    # --------------------------------------------------------
    #
    # Low severity + weak confidence should not create
    # a crisis alert.
    #
    # This prevents cases such as:
    #
    # "normal traffic today"
    #
    # from becoming a crisis event.
    # --------------------------------------------------------

    if severity_score <= 1:

        if confidence >= LOW_SEVERITY_CONFIDENCE_THRESHOLD:

            return (
                "INFORMATIONAL ALERT",
                "MEDIUM"
            )

        else:

            return (
                "No Alert",
                "NORMAL"
            )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return (
        "No Alert",
        "NORMAL"
    )


# ============================================================
# GENERATE ALERTS
# ============================================================

def generate_alerts(df):

    print()
    print("=" * 60)
    print("GENERATING CRISIS ALERTS")
    print("=" * 60)

    results = []

    alert_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    for _, row in df.iterrows():

        post_id = row["id"]

        text = str(
            row["text"]
        )

        prediction = row["prediction"]

        confidence = float(
            row["confidence"]
        )

        severity = row["severity"]

        severity_score = int(
            row["severity_score"]
        )

        # ----------------------------------------------------
        # Generate alert
        # ----------------------------------------------------

        alert, priority = generate_alert(
            prediction,
            confidence,
            severity,
            severity_score
        )

        # ----------------------------------------------------
        # Display
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
            f"{confidence:.2f}%"
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
            "Alert:",
            alert
        )

        print(
            "Priority:",
            priority
        )

        print("-" * 60)

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "id": post_id,

            "text": text,

            "timestamp": row["timestamp"],

            "prediction": prediction,

            "confidence": confidence,

            "severity": severity,

            "severity_score": severity_score,

            "alert": alert,

            "priority": priority,

            "alert_generated_at": alert_time

        })

    return pd.DataFrame(
        results
    )


# ============================================================
# SAVE ALERTS
# ============================================================

def save_alerts(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("ALERTS SAVED")
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
    print("ALERT SUMMARY")
    print("=" * 60)

    columns = [
        "id",
        "prediction",
        "confidence",
        "severity",
        "alert",
        "priority"
    ]

    print(
        df[columns].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Priority counts
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PRIORITY COUNTS")
    print("=" * 60)

    priority_counts = (
        df["priority"]
        .value_counts()
    )

    print(
        priority_counts.to_string()
    )

    # --------------------------------------------------------
    # Alert counts
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ALERT COUNTS")
    print("=" * 60)

    alert_counts = (
        df["alert"]
        .value_counts()
    )

    print(
        alert_counts.to_string()
    )


# ============================================================
# DISPLAY URGENT ALERTS
# ============================================================

def display_urgent_alerts(df):

    print()
    print("=" * 60)
    print("🚨 URGENT CRISIS ALERTS")
    print("=" * 60)

    urgent_df = df[
        df["priority"] == "URGENT"
    ]

    if urgent_df.empty:

        print()
        print(
            "No urgent crisis alerts."
        )

        return

    for _, row in urgent_df.iterrows():

        print()
        print(
            "🚨 HIGH PRIORITY CRISIS ALERT"
        )

        print()
        print(
            "Post ID:",
            row["id"]
        )

        print(
            "Post:",
            row["text"]
        )

        print(
            "Confidence:",
            f'{row["confidence"]:.2f}%'
        )

        print(
            "Severity:",
            row["severity"]
        )

        print(
            "Priority:",
            row["priority"]
        )

        print("-" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("AI-POWERED SOCIAL MEDIA CRISIS DETECTION")
    print("=" * 60)

    print(
        "REAL-TIME CRISIS ALERT GENERATION"
    )

    print("=" * 60)

    # Step 1
    df = load_severity_data()

    # Step 2
    results_df = generate_alerts(
        df
    )

    # Step 3
    save_alerts(
        results_df
    )

    # Step 4
    display_summary(
        results_df
    )

    # Step 5
    display_urgent_alerts(
        results_df
    )

    print()
    print("=" * 60)
    print("ALERT GENERATION COMPLETED SUCCESSFULLY")
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