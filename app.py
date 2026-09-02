import os
import sys
import subprocess

import pandas as pd
import streamlit as st
import plotly.express as px
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Social Media Crisis Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# FILE PATHS
# ============================================================

PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "realtime",
    "realtime_predictions.csv"
)

SEVERITY_FILE = os.path.join(
    BASE_DIR,
    "realtime",
    "realtime_severity.csv"
)

ALERTS_FILE = os.path.join(
    BASE_DIR,
    "realtime",
    "realtime_alerts.csv"
)

EVENTS_FILE = os.path.join(
    BASE_DIR,
    "realtime",
    "realtime_events.csv"
)


# ============================================================
# ROBERTA MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "roberta_crisis_model",
    "checkpoint-2810"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SESSION STATE
# ============================================================

if "test_result" not in st.session_state:
    st.session_state.test_result = None

if "test_post" not in st.session_state:
    st.session_state.test_post = ""

if "pipeline_message" not in st.session_state:
    st.session_state.pipeline_message = None


# ============================================================
# LOAD ROBERTA MODEL
# ============================================================

@st.cache_resource
def load_roberta_model():

    if not os.path.exists(MODEL_PATH):

        return None, None, (
            f"RoBERTa model not found at:\n{MODEL_PATH}"
        )

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        model.to(DEVICE)
        model.eval()

        return tokenizer, model, None

    except Exception as error:

        return None, None, str(error)


tokenizer, roberta_model, model_error = load_roberta_model()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    return str(text).strip()


# ------------------------------------------------------------
# Severity Detection
# ------------------------------------------------------------

def calculate_severity(text, prediction):

    text_lower = text.lower()

    high_keywords = [
        "dead",
        "death",
        "killed",
        "missing",
        "trapped",
        "rescue",
        "rescued",
        "evacuated",
        "evacuation",
        "destroyed",
        "collapsed",
        "collapse",
        "severe",
        "severely",
        "major",
        "massive",
        "earthquake",
        "flood",
        "flooding",
        "tsunami",
        "cyclone",
        "hurricane",
        "wildfire",
        "explosion",
        "disaster",
        "emergency"
    ]

    medium_keywords = [
        "damage",
        "damaged",
        "injured",
        "injury",
        "warning",
        "danger",
        "storm",
        "heavy rain",
        "rising water",
        "power outage",
        "roads flooded",
        "road blocked",
        "disruption"
    ]

    if prediction.lower() != "informative":

        return "Low", 1

    high_matches = sum(
        keyword in text_lower
        for keyword in high_keywords
    )

    medium_matches = sum(
        keyword in text_lower
        for keyword in medium_keywords
    )

    if high_matches >= 2:

        return "High", 3

    if high_matches >= 1:

        return "High", 3

    if medium_matches >= 1:

        return "Medium", 2

    return "Low", 1


# ------------------------------------------------------------
# Alert Priority
# ------------------------------------------------------------

def calculate_priority(severity, confidence):

    if severity == "High" and confidence >= 80:

        return "URGENT"

    if severity == "High":

        return "HIGH"

    if severity == "Medium":

        return "HIGH"

    return "NORMAL"


# ------------------------------------------------------------
# RoBERTa Prediction
# ------------------------------------------------------------

def predict_post(text):

    if roberta_model is None or tokenizer is None:

        return None

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = roberta_model(
            **encoded
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    confidence = float(
        probabilities[predicted_class].item()
        * 100
    )

    # Your project mapping:
    # 0 = Not Informative
    # 1 = Informative

    if predicted_class == 1:

        prediction = "Informative"

    else:

        prediction = "Not Informative"

    not_informative_probability = float(
        probabilities[0].item() * 100
    )

    informative_probability = float(
        probabilities[1].item() * 100
    )

    severity, severity_score = calculate_severity(
        text,
        prediction
    )

    priority = calculate_priority(
        severity,
        confidence
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "not_informative_probability":
            not_informative_probability,
        "informative_probability":
            informative_probability,
        "severity": severity,
        "severity_score": severity_score,
        "priority": priority
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🚨 Crisis Detection")

    st.markdown(
        """
        ### AI-Powered System

        **Technology**

        - RoBERTa
        - NLP
        - PyTorch
        - Streamlit
        - Crisis Analytics

        ---

        **Pipeline**

        1. Social Media Posts
        2. Text Cleaning
        3. RoBERTa Classification
        4. Severity Detection
        5. Alert Generation
        6. Event Grouping
        7. Risk Assessment
        8. Early Warning
        """
    )

    st.divider()

    st.write("### System Information")

    if roberta_model is not None:

        st.success("Prediction Engine: ONLINE")

    else:

        st.error("Prediction Engine: OFFLINE")

    st.write(
        f"Device: `{DEVICE}`"
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🚨 AI-Powered Social Media Crisis Detection"
)

st.subheader(
    "Real-Time Crisis Monitoring & Early Warning System"
)

st.markdown(
    """
    This dashboard uses an AI-powered NLP pipeline to classify
    social media posts, estimate crisis severity, generate alerts,
    group related crisis events, calculate risk and provide
    early warnings.
    """
)

st.divider()


# ============================================================
# MODEL ERROR
# ============================================================

if model_error is not None:

    st.error(
        "❌ RoBERTa model could not be loaded."
    )

    st.code(
        model_error
    )


# ============================================================
# TEST SOCIAL MEDIA POST
# ============================================================

st.header(
    "🧪 Test a Social Media Post"
)

st.markdown(
    """
    Enter any social media post below and test the trained
    **RoBERTa crisis classification model**.
    """
)


with st.form("test_post_form"):

    test_input = st.text_area(
        "Enter social media post:",
        value=st.session_state.test_post,
        height=120,
        placeholder=(
            "Example: A powerful earthquake has damaged "
            "buildings and rescue teams are searching for survivors."
        )
    )

    test_button = st.form_submit_button(
        "🔍 Analyze Post",
        use_container_width=True
    )


if test_button:

    test_input = clean_text(
        test_input
    )

    if not test_input:

        st.warning(
            "⚠️ Please enter a social media post."
        )

    elif roberta_model is None:

        st.error(
            "❌ RoBERTa model is not available."
        )

    else:

        with st.spinner(
            "🤖 RoBERTa is analyzing the post..."
        ):

            result = predict_post(
                test_input
            )

        st.session_state.test_post = test_input
        st.session_state.test_result = result

        st.rerun()


# ============================================================
# DISPLAY TEST RESULT
# ============================================================

if st.session_state.test_result is not None:

    result = st.session_state.test_result

    test_post = st.session_state.test_post

    st.divider()

    st.header(
        "🤖 AI Prediction"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Prediction",
            result["prediction"]
        )

    with col2:

        st.metric(
            "Confidence",
            f"{result['confidence']:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # Crisis Assessment
    # --------------------------------------------------------

    st.header(
        "🚨 Crisis Assessment"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Severity",
            result["severity"]
        )

    with col2:

        st.metric(
            "Severity Score",
            result["severity_score"]
        )

    with col3:

        st.metric(
            "Alert Priority",
            result["priority"]
        )

    if result["priority"] == "URGENT":

        st.error(
            "🚨 URGENT CRISIS ALERT: "
            "Immediate attention recommended."
        )

    elif result["priority"] == "HIGH":

        st.warning(
            "⚠️ HIGH PRIORITY ALERT: "
            "Continued monitoring recommended."
        )

    else:

        st.success(
            "✅ NORMAL: "
            "No immediate crisis escalation detected."
        )

    st.divider()

    # --------------------------------------------------------
    # Probability Chart
    # --------------------------------------------------------

    st.header(
        "📊 Prediction Probability"
    )

    probability_data = pd.DataFrame(
        {
            "Class": [
                "Informative",
                "Not Informative"
            ],
            "Probability": [
                result["informative_probability"],
                result["not_informative_probability"]
            ]
        }
    )

    probability_fig = px.bar(
        probability_data,
        x="Class",
        y="Probability",
        text="Probability",
        title="RoBERTa Prediction Probability"
    )

    probability_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    probability_fig.update_yaxes(
        range=[
            0,
            100
        ],
        title="Probability (%)"
    )

    st.plotly_chart(
        probability_fig,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # Tested Post
    # --------------------------------------------------------

    st.header(
        "📝 Tested Post"
    )

    st.info(
        test_post
    )


st.divider()


# ============================================================
# REAL-TIME PROCESSING
# ============================================================

st.header(
    "🔄 Real-Time Processing"
)


if st.button(
    "🚀 Run Real-Time Crisis Analysis",
    use_container_width=True
):

    pipeline_file = os.path.join(
        BASE_DIR,
        "realtime",
        "realtime_pipeline.py"
    )

    if not os.path.exists(pipeline_file):

        st.error(
            f"❌ Pipeline file not found:\n{pipeline_file}"
        )

    else:

        with st.spinner(
            "Running AI crisis detection pipeline..."
        ):

            try:

                process = subprocess.run(
                    [
                        sys.executable,
                        pipeline_file
                    ],
                    capture_output=True,
                    text=True,
                    cwd=BASE_DIR
                )

                if process.returncode == 0:

                    st.success(
                        "✅ Real-time crisis analysis "
                        "completed successfully."
                    )

                    st.rerun()

                else:

                    st.error(
                        "❌ Pipeline execution failed."
                    )

                    if process.stderr:

                        st.code(
                            process.stderr
                        )

            except Exception as error:

                st.error(
                    f"❌ Pipeline error: {error}"
                )


st.divider()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    PREDICTIONS_FILE,
    SEVERITY_FILE,
    ALERTS_FILE,
    EVENTS_FILE
]


missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.warning(
        "⚠️ Real-time pipeline output files are not available yet."
    )

    st.info(
        "Click **Run Real-Time Crisis Analysis** above "
        "to generate the required files."
    )

    for file in missing_files:

        st.write(
            f"❌ {file}"
        )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    severity = pd.read_csv(
        SEVERITY_FILE
    )

    alerts = pd.read_csv(
        ALERTS_FILE
    )

    events = pd.read_csv(
        EVENTS_FILE
    )

except Exception as error:

    st.error(
        f"❌ Error loading pipeline files: {error}"
    )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

required_prediction_columns = [
    "id",
    "text",
    "timestamp",
    "prediction",
    "confidence"
]

required_severity_columns = [
    "id",
    "severity",
    "severity_score"
]

required_alert_columns = [
    "id",
    "alert",
    "priority"
]

required_event_columns = [
    "id",
    "event_id",
    "event_name"
]


for column in required_prediction_columns:

    if column not in predictions.columns:

        st.error(
            f"Missing column in predictions: {column}"
        )

        st.stop()


for column in required_severity_columns:

    if column not in severity.columns:

        st.error(
            f"Missing column in severity: {column}"
        )

        st.stop()


for column in required_alert_columns:

    if column not in alerts.columns:

        st.error(
            f"Missing column in alerts: {column}"
        )

        st.stop()


for column in required_event_columns:

    if column not in events.columns:

        st.error(
            f"Missing column in events: {column}"
        )

        st.stop()


# ============================================================
# MERGE DATA
# ============================================================

data = predictions.copy()


data = data.merge(
    severity[
        [
            "id",
            "severity",
            "severity_score"
        ]
    ],
    on="id",
    how="left"
)


data = data.merge(
    alerts[
        [
            "id",
            "alert",
            "priority"
        ]
    ],
    on="id",
    how="left"
)


data = data.merge(
    events[
        [
            "id",
            "event_id",
            "event_name"
        ]
    ],
    on="id",
    how="left"
)


# ============================================================
# BASIC STATISTICS
# ============================================================

total_posts = len(
    predictions
)


prediction_clean = (
    predictions["prediction"]
    .astype(str)
    .str.strip()
    .str.lower()
)


informative_posts = (
    prediction_clean == "informative"
).sum()


not_informative_posts = (
    total_posts - informative_posts
)


# ============================================================
# SEVERITY STATISTICS
# ============================================================

severity_clean = (
    severity["severity"]
    .astype(str)
    .str.strip()
    .str.lower()
)


high_severity = (
    severity_clean == "high"
).sum()


medium_severity = (
    severity_clean == "medium"
).sum()


low_severity = (
    severity_clean == "low"
).sum()


# ============================================================
# ALERT STATISTICS
# ============================================================

priority_clean = (
    alerts["priority"]
    .astype(str)
    .str.strip()
    .str.upper()
)


urgent_alerts = (
    priority_clean == "URGENT"
).sum()


high_alerts = (
    priority_clean == "HIGH"
).sum()


medium_alerts = (
    priority_clean == "MEDIUM"
).sum()


normal_alerts = (
    priority_clean == "NORMAL"
).sum()


# ============================================================
# EVENT STATISTICS
# ============================================================

event_ids = pd.to_numeric(
    events["event_id"],
    errors="coerce"
).fillna(0)


crisis_event_data = events[
    event_ids > 0
].copy()


unique_event_ids = (
    crisis_event_data["event_id"]
    .unique()
)


crisis_events = len(
    unique_event_ids
)


# ============================================================
# AVERAGE CONFIDENCE
# ============================================================

average_confidence = (
    pd.to_numeric(
        predictions["confidence"],
        errors="coerce"
    )
    .mean()
)


if pd.isna(average_confidence):

    average_confidence = 0


# ============================================================
# RISK SCORE
# ============================================================

if total_posts > 0:

    severity_component = (
        high_severity
        / total_posts
    ) * 40

    alert_component = (
        urgent_alerts
        / total_posts
    ) * 30

    informative_component = (
        informative_posts
        / total_posts
    ) * 20

    event_component = (
        crisis_events
        / total_posts
    ) * 10

else:

    severity_component = 0
    alert_component = 0
    informative_component = 0
    event_component = 0


risk_score = (
    severity_component
    + alert_component
    + informative_component
    + event_component
)


risk_score = max(
    0,
    min(
        100,
        risk_score
    )
)


# ============================================================
# RISK LEVEL
# ============================================================

if risk_score >= 70:

    risk_level = "HIGH"

elif risk_score >= 40:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# EARLY WARNING
# ============================================================

if (
    urgent_alerts > 0
    or high_severity >= 3
):

    early_warning = "EARLY WARNING"

else:

    early_warning = "NO EARLY WARNING"


# ============================================================
# CRISIS DETECTION OVERVIEW
# ============================================================

st.header(
    "📊 Crisis Detection Overview"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Posts",
        total_posts
    )


with col2:

    st.metric(
        "Informative Posts",
        informative_posts
    )


with col3:

    st.metric(
        "Not Informative",
        not_informative_posts
    )


with col4:

    st.metric(
        "Crisis Events",
        crisis_events
    )


st.divider()


# ============================================================
# CRISIS SEVERITY
# ============================================================

st.header(
    "🚨 Crisis Severity"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "High Severity",
        high_severity
    )


with col2:

    st.metric(
        "Medium Severity",
        medium_severity
    )


with col3:

    st.metric(
        "Low Severity",
        low_severity
    )


st.divider()


# ============================================================
# RISK ASSESSMENT
# ============================================================

st.header(
    "⚠️ Crisis Risk Assessment"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Risk Score",
        f"{risk_score:.2f}/100"
    )


with col2:

    st.metric(
        "Risk Level",
        risk_level
    )


with col3:

    st.metric(
        "Early Warning",
        early_warning
    )


with col4:

    st.metric(
        "Average Confidence",
        f"{average_confidence:.2f}%"
    )


if risk_level == "HIGH":

    st.error(
        "🚨 HIGH CRISIS RISK: "
        "Immediate monitoring recommended."
    )

elif risk_level == "MEDIUM":

    st.warning(
        "⚠️ MEDIUM CRISIS RISK: "
        "Continued monitoring recommended."
    )

else:

    st.success(
        "✅ LOW CRISIS RISK: "
        "Normal monitoring recommended."
    )


st.divider()


# ============================================================
# ALERT SUMMARY
# ============================================================

st.header(
    "🚨 Crisis Alert Summary"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "URGENT",
        urgent_alerts
    )


with col2:

    st.metric(
        "HIGH",
        high_alerts
    )


with col3:

    st.metric(
        "MEDIUM",
        medium_alerts
    )


with col4:

    st.metric(
        "NORMAL",
        normal_alerts
    )


st.divider()


# ============================================================
# CRISIS ANALYTICS
# ============================================================

st.header(
    "📈 Crisis Analytics"
)


chart_col1, chart_col2 = st.columns(2)


# ------------------------------------------------------------
# Severity Chart
# ------------------------------------------------------------

severity_data = pd.DataFrame(
    {
        "Severity": [
            "High",
            "Medium",
            "Low"
        ],
        "Count": [
            high_severity,
            medium_severity,
            low_severity
        ]
    }
)


with chart_col1:

    st.subheader(
        "Crisis Severity Distribution"
    )

    fig_severity = px.bar(
        severity_data,
        x="Severity",
        y="Count",
        text="Count",
        title="Severity Distribution"
    )

    fig_severity.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_severity,
        use_container_width=True
    )


# ------------------------------------------------------------
# Alert Chart
# ------------------------------------------------------------

alert_data = pd.DataFrame(
    {
        "Priority": [
            "URGENT",
            "HIGH",
            "MEDIUM",
            "NORMAL"
        ],
        "Count": [
            urgent_alerts,
            high_alerts,
            medium_alerts,
            normal_alerts
        ]
    }
)


with chart_col2:

    st.subheader(
        "Alert Priority Distribution"
    )

    fig_alert = px.bar(
        alert_data,
        x="Priority",
        y="Count",
        text="Count",
        title="Alert Distribution"
    )

    fig_alert.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_alert,
        use_container_width=True
    )


st.divider()


# ============================================================
# REAL-TIME CRISIS TREND
# ============================================================

st.header(
    "📈 Real-Time Crisis Trend"
)


informative_data = predictions[
    prediction_clean == "informative"
].copy()


if "timestamp" in informative_data.columns:

    informative_data["timestamp"] = pd.to_datetime(
        informative_data["timestamp"],
        errors="coerce"
    )

    informative_data = informative_data.sort_values(
        "timestamp"
    )


informative_count = len(
    informative_data
)


if informative_count >= 2:

    split_point = informative_count // 2

    previous_period = split_point

    current_period = (
        informative_count
        - split_point
    )

else:

    previous_period = 0

    current_period = informative_count


if previous_period > 0:

    growth_rate = (
        (
            current_period
            - previous_period
        )
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


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Current Trend",
        trend
    )


with col2:

    st.metric(
        "Growth Rate",
        f"{growth_rate:.1f}%"
    )


with col3:

    informative_ratio = (
        informative_posts
        / total_posts
    ) * 100 if total_posts > 0 else 0

    st.metric(
        "Informative Ratio",
        f"{informative_ratio:.2f}%"
    )


st.divider()


# ============================================================
# CRISIS EVENT ANALYSIS
# ============================================================

st.header(
    "🚨 Crisis Event Analysis"
)


if crisis_event_data.empty:

    st.info(
        "No crisis events detected."
    )

else:

    event_summary = (
        crisis_event_data
        .groupby(
            [
                "event_id",
                "event_name"
            ]
        )
        .agg(
            post_count=(
                "id",
                "count"
            ),
            highest_severity=(
                "severity_score",
                "max"
            )
        )
        .reset_index()
    )

    event_summary["severity"] = (
        event_summary[
            "highest_severity"
        ]
        .map(
            {
                1: "Low",
                2: "Medium",
                3: "High"
            }
        )
    )

    event_summary = event_summary[
        [
            "event_id",
            "event_name",
            "post_count",
            "severity",
            "highest_severity"
        ]
    ]

    st.dataframe(
        event_summary,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# URGENT CRISIS ALERTS
# ============================================================

st.header(
    "🚨 Urgent Crisis Alerts"
)


urgent_df = alerts[
    alerts["priority"]
    .astype(str)
    .str.strip()
    .str.upper()
    == "URGENT"
].copy()


if urgent_df.empty:

    st.success(
        "✅ No urgent crisis alerts."
    )

else:

    for _, row in urgent_df.iterrows():

        st.error(
            f"""
🚨 HIGH PRIORITY CRISIS ALERT

Post ID: {row['id']}

{row.get('text', 'Crisis-related social media post')}

Confidence: {row.get('confidence', 'N/A')}%

Severity: {row.get('severity', 'N/A')}

Priority: {row['priority']}
"""
        )


st.divider()


# ============================================================
# REAL-TIME SOCIAL MEDIA POSTS
# ============================================================

st.header(
    "📱 Real-Time Social Media Posts"
)


display_columns = [
    "id",
    "text",
    "timestamp",
    "prediction",
    "confidence",
    "severity",
    "severity_score",
    "alert",
    "priority",
    "event_id",
    "event_name"
]


available_columns = [
    column
    for column in display_columns
    if column in data.columns
]


st.dataframe(
    data[available_columns],
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# AI MODEL INFORMATION
# ============================================================

st.header(
    "🤖 AI Model Information"
)


info_col1, info_col2 = st.columns(2)


with info_col1:

    st.subheader(
        "RoBERTa Transformer"
    )

    st.markdown(
        """
        **Task:** Crisis Information Classification

        **Classes:**
        - Informative
        - Not Informative

        **Maximum Sequence Length:** 128

        **Model Type:** Transformer-based NLP

        **Application:** Social Media Crisis Detection
        """
    )


with info_col2:

    st.subheader(
        "Current Processing Statistics"
    )

    st.markdown(
        f"""
        **Average Prediction Confidence:**
        {average_confidence:.2f}%

        **Total Processed Posts:**
        {total_posts}

        **Informative Posts:**
        {informative_posts}

        **Crisis Events Detected:**
        {crisis_events}

        **Risk Score:**
        {risk_score:.2f}/100

        **Risk Level:**
        {risk_level}
        """
    )


st.divider()


# ============================================================
# REAL-TIME PIPELINE
# ============================================================

st.header(
    "🔄 Real-Time Processing Pipeline"
)


pipeline_steps = [
    "1️⃣ Social Media Posts",
    "2️⃣ Text Cleaning",
    "3️⃣ RoBERTa Classification",
    "4️⃣ Severity Detection",
    "5️⃣ Crisis Alert Generation",
    "6️⃣ Crisis Event Grouping",
    "7️⃣ Risk Assessment",
    "8️⃣ Early Warning Dashboard"
]


pipeline_col1, pipeline_col2 = st.columns(2)


for index, step in enumerate(
    pipeline_steps
):

    if index % 2 == 0:

        with pipeline_col1:

            st.info(step)

    else:

        with pipeline_col2:

            st.info(step)


st.divider()


# ============================================================
# SYSTEM STATUS
# ============================================================

st.header(
    "🟢 System Status"
)


col1, col2, col3 = st.columns(3)


with col1:

    if roberta_model is not None:

        st.success(
            "Prediction Engine\n\nONLINE"
        )

    else:

        st.error(
            "Prediction Engine\n\nOFFLINE"
        )


with col2:

    st.success(
        "Severity Engine\n\nONLINE"
    )


with col3:

    st.success(
        "Alert Engine\n\nONLINE"
    )


st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "AI-Powered Social Media Crisis Detection & Early Warning System"
)

st.caption(
    "RoBERTa • NLP • Severity Analysis • Event Detection • "
    "Risk Assessment • Early Warning"
)