# ============================================================
# AI-POWERED SOCIAL MEDIA CRISIS DETECTION
# REAL-TIME CRISIS MONITORING DASHBOARD
# ============================================================

import os
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Social Media Crisis Detection",
    page_icon="🚨",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

ALERT_FILE = "realtime/realtime_alerts.csv"

EVENT_FILE = "realtime/realtime_events.csv"


# ============================================================
# TITLE
# ============================================================

st.title(
    "🚨 AI-Powered Social Media Crisis Detection"
)

st.subheader(
    "Real-Time Crisis Monitoring Dashboard"
)

st.markdown(
    """
This dashboard displays real-time social-media posts,
AI predictions, crisis severity, alerts, and grouped
crisis events.
"""
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(ALERT_FILE):

    st.error(
        "realtime_alerts.csv not found."
    )

    st.info(
        "Run the prediction, severity, and alert modules first."
    )

    st.stop()


if not os.path.exists(EVENT_FILE):

    st.warning(
        "realtime_events.csv not found."
    )

    st.info(
        "Run crisis_event_grouping.py first."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    alerts_df = pd.read_csv(
        ALERT_FILE
    )

    events_df = pd.read_csv(
        EVENT_FILE
    )

except Exception as error:

    st.error(
        f"Error loading data: {error}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dashboard Controls"
)

st.sidebar.write(
    "Data source:"
)

st.sidebar.code(
    ALERT_FILE
)

if st.sidebar.button(
    "🔄 Refresh Dashboard"
):

    st.rerun()


# ============================================================
# BASIC COUNTS
# ============================================================

total_posts = len(alerts_df)

informative_count = len(
    alerts_df[
        alerts_df["prediction"]
        == "Informative"
    ]
)

not_informative_count = len(
    alerts_df[
        alerts_df["prediction"]
        == "Not Informative"
    ]
)

urgent_count = len(
    alerts_df[
        alerts_df["priority"]
        == "URGENT"
    ]
)

high_priority_count = len(
    alerts_df[
        alerts_df["priority"]
        == "HIGH"
    ]
)

normal_count = len(
    alerts_df[
        alerts_df["priority"]
        == "NORMAL"
    ]
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

st.markdown("## 📊 System Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Posts",
        total_posts
    )

with col2:

    st.metric(
        "Informative",
        informative_count
    )

with col3:

    st.metric(
        "Not Informative",
        not_informative_count
    )

with col4:

    st.metric(
        "Urgent Alerts",
        urgent_count
    )


# ============================================================
# SEVERITY METRICS
# ============================================================

st.markdown("## ⚠️ Crisis Severity")

severity_counts = (
    alerts_df["severity"]
    .value_counts()
)

high_count = severity_counts.get(
    "High",
    0
)

medium_count = severity_counts.get(
    "Medium",
    0
)

low_count = severity_counts.get(
    "Low",
    0
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "🔴 High Severity",
        high_count
    )

with col2:

    st.metric(
        "🟠 Medium Severity",
        medium_count
    )

with col3:

    st.metric(
        "🟢 Low Severity",
        low_count
    )


# ============================================================
# PRIORITY METRICS
# ============================================================

st.markdown("## 🚨 Alert Priority")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "URGENT",
        urgent_count
    )

with col2:

    st.metric(
        "HIGH",
        high_priority_count
    )

with col3:

    st.metric(
        "NORMAL",
        normal_count
    )


# ============================================================
# CRISIS EVENTS
# ============================================================

st.markdown("## 🌐 Crisis Events")

crisis_events = events_df[
    events_df["event_id"] != 0
]

event_count = (
    crisis_events["event_id"]
    .nunique()
)

st.metric(
    "Detected Crisis Events",
    event_count
)


if event_count > 0:

    event_summary = (
        crisis_events
        .groupby("event_id")
        .agg(
            Posts=("id", "count"),
            Highest_Severity=(
                "severity_score",
                "max"
            )
        )
        .reset_index()
    )

    event_summary[
        "Event"
    ] = event_summary[
        "event_id"
    ].apply(
        lambda x:
        f"Crisis Event {x}"
    )

    st.dataframe(
        event_summary[
            [
                "Event",
                "Posts",
                "Highest_Severity"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# URGENT ALERTS
# ============================================================

st.markdown("## 🚨 Urgent Crisis Alerts")

urgent_alerts = alerts_df[
    alerts_df["priority"]
    == "URGENT"
]

if len(urgent_alerts) == 0:

    st.success(
        "No urgent crisis alerts detected."
    )

else:

    for _, row in urgent_alerts.iterrows():

        st.error(
            f"""
🚨 HIGH PRIORITY CRISIS ALERT

Post ID: {row["id"]}

Post:
{row["text"]}

Confidence: {row["confidence"]}%

Severity: {row["severity"]}

Priority: {row["priority"]}
"""
        )


# ============================================================
# ALL POSTS
# ============================================================

st.markdown("## 📱 Real-Time Social Media Posts")

display_columns = [
    "id",
    "text",
    "timestamp",
    "prediction",
    "confidence",
    "severity",
    "severity_score",
    "alert",
    "priority"
]

available_columns = [
    column
    for column in display_columns
    if column in alerts_df.columns
]

st.dataframe(
    alerts_df[
        available_columns
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INFORMATIVE POSTS
# ============================================================

st.markdown(
    "## 📢 Informative Crisis Posts"
)

informative_df = alerts_df[
    alerts_df["prediction"]
    == "Informative"
]

if len(informative_df) > 0:

    st.dataframe(
        informative_df[
            available_columns
        ],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No informative crisis posts found."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Powered Social Media Crisis Detection "
    "and Early Warning System"
)

st.caption(
    "Model: RoBERTa | "
    "Processing: Real-Time Social Media Monitoring"
)