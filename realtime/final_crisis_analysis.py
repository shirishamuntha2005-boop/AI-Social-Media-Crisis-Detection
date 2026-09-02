import os
import pandas as pd


# ============================================================
# AI-POWERED SOCIAL MEDIA CRISIS DETECTION
# FINAL CRISIS ANALYSIS
# ============================================================

print("=" * 70)
print("AI-POWERED SOCIAL MEDIA CRISIS DETECTION")
print("FINAL CRISIS ANALYSIS")
print("=" * 70)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTIONS_FILE = os.path.join(
    BASE_DIR, "realtime_predictions.csv"
)

SEVERITY_FILE = os.path.join(
    BASE_DIR, "realtime_severity.csv"
)

ALERTS_FILE = os.path.join(
    BASE_DIR, "realtime_alerts.csv"
)

EVENTS_FILE = os.path.join(
    BASE_DIR, "realtime_events.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING PIPELINE OUTPUT FILES")
print("=" * 70)

files = {
    "realtime_predictions.csv": PREDICTIONS_FILE,
    "realtime_severity.csv": SEVERITY_FILE,
    "realtime_alerts.csv": ALERTS_FILE,
    "realtime_events.csv": EVENTS_FILE
}

for name, path in files.items():

    if os.path.exists(path):
        print(f"✓ Found: {name}")

    else:
        print(f"✗ Missing: {name}")
        raise FileNotFoundError(path)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING PIPELINE RESULTS")
print("=" * 70)

predictions = pd.read_csv(PREDICTIONS_FILE)
severity = pd.read_csv(SEVERITY_FILE)
alerts = pd.read_csv(ALERTS_FILE)
events = pd.read_csv(EVENTS_FILE)

print(f"Predictions loaded : {len(predictions)}")
print(f"Severity records   : {len(severity)}")
print(f"Alert records      : {len(alerts)}")
print(f"Event records      : {len(events)}")


# ============================================================
# 1. CRISIS DETECTION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("1. CRISIS DETECTION SUMMARY")
print("=" * 70)

total_posts = len(predictions)

informative_posts = (
    predictions["prediction"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("informative")
    .sum()
)

not_informative_posts = total_posts - informative_posts

print(f"Total Posts          : {total_posts}")
print(f"Informative Posts    : {informative_posts}")
print(f"Not Informative      : {not_informative_posts}")


# ============================================================
# 2. CRISIS SEVERITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("2. CRISIS SEVERITY SUMMARY")
print("=" * 70)

severity_clean = (
    severity["severity"]
    .astype(str)
    .str.strip()
    .str.lower()
)

high_severity = (severity_clean == "high").sum()
medium_severity = (severity_clean == "medium").sum()
low_severity = (severity_clean == "low").sum()

print(f"High Severity       : {high_severity}")
print(f"Medium Severity     : {medium_severity}")
print(f"Low Severity        : {low_severity}")


# ============================================================
# 3. CRISIS ALERT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("3. CRISIS ALERT SUMMARY")
print("=" * 70)

priority_clean = (
    alerts["priority"]
    .astype(str)
    .str.strip()
    .str.upper()
)

urgent_alerts = (priority_clean == "URGENT").sum()
high_alerts = (priority_clean == "HIGH").sum()
medium_alerts = (priority_clean == "MEDIUM").sum()
normal_alerts = (priority_clean == "NORMAL").sum()

print(f"Urgent Alerts       : {urgent_alerts}")
print(f"High Alerts         : {high_alerts}")
print(f"Medium Alerts       : {medium_alerts}")
print(f"Normal Alerts       : {normal_alerts}")


# ============================================================
# 4. CRISIS EVENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. CRISIS EVENT ANALYSIS")
print("=" * 70)

# Event ID 0 means no crisis event.
# Only count actual crisis events.

event_ids = pd.to_numeric(
    events["event_id"],
    errors="coerce"
).fillna(0)

crisis_events = events[event_ids > 0].copy()

unique_event_ids = sorted(
    crisis_events["event_id"].unique()
)

print(f"Crisis Events       : {len(unique_event_ids)}")

print("\nDetected Crisis Events:")

for event_id in unique_event_ids:

    event_posts = crisis_events[
        crisis_events["event_id"] == event_id
    ]

    post_count = len(event_posts)

    highest_score = event_posts["severity_score"].max()

    if highest_score >= 3:
        highest_severity = "High"

    elif highest_score >= 2:
        highest_severity = "Medium"

    else:
        highest_severity = "Low"

    print(
        f"Crisis Event {int(event_id)} -> "
        f"{post_count} related posts -> "
        f"Highest Severity: {highest_severity} -> "
        f"Score: {int(highest_score)}"
    )


# ============================================================
# 5. CRISIS TREND ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. CRISIS TREND ANALYSIS")
print("=" * 70)

informative_data = predictions[
    predictions["prediction"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("informative")
].copy()

if "timestamp" in informative_data.columns:

    informative_data["timestamp"] = pd.to_datetime(
        informative_data["timestamp"],
        errors="coerce"
    )

    informative_data = informative_data.sort_values(
        "timestamp"
    )


# Divide informative posts into two periods.
# This is a simple demonstration for the real-time dataset.

informative_count = len(informative_data)

if informative_count >= 2:

    split_point = informative_count // 2

    previous_period = split_point

    current_period = informative_count - split_point

else:

    previous_period = 0
    current_period = informative_count


if previous_period > 0:

    growth_rate = (
        (current_period - previous_period)
        / previous_period
    ) * 100

else:

    growth_rate = 0


if growth_rate > 10:

    trend = "INCREASING"

elif growth_rate < -10:

    trend = "DECREASING"

else:

    trend = "STABLE"


print(f"Previous Period    : {previous_period} posts")
print(f"Current Period     : {current_period} posts")
print(f"Growth Rate        : {growth_rate:.1f}%")
print(f"Crisis Trend       : {trend}")


# ============================================================
# 6. CRISIS RISK SCORE
# ============================================================

print("\n" + "=" * 70)
print("6. CRISIS RISK SCORE")
print("=" * 70)

average_confidence = predictions["confidence"].mean()

# Risk components
severity_component = (
    (high_severity / total_posts) * 40
)

alert_component = (
    (urgent_alerts / total_posts) * 30
)

informative_component = (
    (informative_posts / total_posts) * 20
)

event_component = (
    (len(unique_event_ids) / total_posts) * 10
)

risk_score = (
    severity_component
    + alert_component
    + informative_component
    + event_component
)

# Keep score between 0 and 100
risk_score = max(0, min(100, risk_score))


if risk_score >= 70:

    risk_level = "HIGH"

elif risk_score >= 40:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


print(f"Average Confidence : {average_confidence:.2f}%")
print(f"Risk Score         : {risk_score:.2f}/100")
print(f"Risk Level         : {risk_level}")


# ============================================================
# 7. EARLY WARNING SYSTEM
# ============================================================

print("\n" + "=" * 70)
print("7. EARLY WARNING SYSTEM")
print("=" * 70)

if urgent_alerts > 0 or high_severity >= 3:

    early_warning = "EARLY WARNING"

else:

    early_warning = "NO EARLY WARNING"


print(f"Early Warning      : {early_warning}")


# ============================================================
# 8. CRISIS EXPLANATION
# ============================================================

print("\n" + "=" * 70)
print("8. CRISIS EXPLANATION")
print("=" * 70)

explanation = (
    f"{high_severity} high-severity crisis posts detected; "
    f"{urgent_alerts} urgent alerts generated; "
    f"{len(unique_event_ids)} significant crisis events identified."
)

print(f"Explanation: {explanation}")


# ============================================================
# FINAL CRISIS ASSESSMENT
# ============================================================

print("\n" + "=" * 70)
print("FINAL CRISIS ASSESSMENT")
print("=" * 70)

print(f"\nTotal Posts       : {total_posts}")
print(f"Informative Posts : {informative_posts}")
print(f"Crisis Events     : {len(unique_event_ids)}")
print(f"High Severity     : {high_severity}")
print(f"Medium Severity   : {medium_severity}")
print(f"Low Severity      : {low_severity}")

print(f"\nTrend             : {trend}")
print(f"Growth Rate       : {growth_rate:.1f}%")

print(f"\nRisk Score        : {risk_score:.2f}/100")
print(f"Risk Level        : {risk_level}")

print(f"\nEarly Warning     : {early_warning}")

print(f"\nExplanation       : {explanation}")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("FINAL STATUS")
print("=" * 70)

if risk_level == "HIGH":

    print("🚨 HIGH CRISIS RISK: Immediate monitoring recommended.")

elif risk_level == "MEDIUM":

    print("⚠️ MEDIUM CRISIS RISK: Continued monitoring recommended.")

else:

    print("✅ LOW CRISIS RISK: Normal monitoring recommended.")


print("\n" + "=" * 70)
print("FINAL CRISIS ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)