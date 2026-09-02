# ============================================================
# REAL-TIME CRISIS EVENT GROUPING
# ============================================================

import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "realtime/realtime_alerts.csv"
OUTPUT_FILE = "realtime/realtime_events.csv"


# ============================================================
# SETTINGS
# ============================================================

# Higher threshold = stricter event grouping
SIMILARITY_THRESHOLD = 0.15


# ============================================================
# LOAD ALERT DATA
# ============================================================

def load_alerts():

    print()
    print("=" * 60)
    print("LOADING REAL-TIME ALERT DATA")
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
        "severity_score",
        "alert",
        "priority"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    print("Alerts loaded:", len(df))

    return df


# ============================================================
# CREATE CRISIS EVENT GROUPS
# ============================================================

def create_event_groups(df):

    print()
    print("=" * 60)
    print("GROUPING SIMILAR CRISIS POSTS")
    print("=" * 60)

    # --------------------------------------------------------
    # IMPORTANT:
    # Only Informative posts are allowed to create events.
    # --------------------------------------------------------

    event_ids = [0] * len(df)

    informative_indices = []

    for i in range(len(df)):

        prediction = str(
            df.iloc[i]["prediction"]
        ).strip().lower()

        if prediction == "informative":
            informative_indices.append(i)

    # --------------------------------------------------------
    # If there are no informative posts
    # --------------------------------------------------------

    if not informative_indices:

        print("No informative crisis posts found.")

        return event_ids

    # --------------------------------------------------------
    # Get informative texts
    # --------------------------------------------------------

    texts = [
        str(df.iloc[i]["text"])
        for i in informative_indices
    ]

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(
        texts
    )

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    # --------------------------------------------------------
    # Event assignment
    # --------------------------------------------------------

    current_event_id = 1

    assigned_events = {}

    for local_i, original_i in enumerate(
        informative_indices
    ):

        assigned_event = None

        # Compare with previous informative posts
        for local_j in range(local_i):

            original_j = informative_indices[local_j]

            similarity = similarity_matrix[
                local_i,
                local_j
            ]

            if similarity >= SIMILARITY_THRESHOLD:

                previous_event = assigned_events.get(
                    original_j
                )

                if previous_event is not None:

                    assigned_event = previous_event

                    break

        # ----------------------------------------------------
        # Existing event
        # ----------------------------------------------------

        if assigned_event is not None:

            event_ids[original_i] = assigned_event

        # ----------------------------------------------------
        # New event
        # ----------------------------------------------------

        else:

            assigned_event = current_event_id

            event_ids[original_i] = assigned_event

            current_event_id += 1

        assigned_events[original_i] = assigned_event

    return event_ids


# ============================================================
# ADD EVENT INFORMATION
# ============================================================

def add_event_information(
    df,
    event_ids
):

    df = df.copy()

    df["event_id"] = event_ids

    event_names = []

    for event_id in event_ids:

        if event_id == 0:

            event_names.append(
                "No Crisis Event"
            )

        else:

            event_names.append(
                f"Crisis Event {event_id}"
            )

    df["event_name"] = event_names

    return df


# ============================================================
# DISPLAY EVENTS
# ============================================================

def display_events(df):

    print()
    print("=" * 60)
    print("CRISIS EVENT GROUPS")
    print("=" * 60)

    event_ids = (
        df["event_id"]
        .drop_duplicates()
        .tolist()
    )

    crisis_event_found = False

    for event_id in event_ids:

        if event_id == 0:
            continue

        crisis_event_found = True

        event_df = df[
            df["event_id"] == event_id
        ]

        # Highest severity row
        highest_index = (
            event_df["severity_score"]
            .idxmax()
        )

        highest_row = event_df.loc[
            highest_index
        ]

        highest_severity = (
            highest_row["severity"]
        )

        highest_priority = (
            highest_row["priority"]
        )

        print()
        print(
            f"🚨 Crisis Event {event_id}"
        )

        print(
            "Number of related posts:",
            len(event_df)
        )

        print(
            "Highest severity:",
            highest_severity
        )

        print(
            "Highest priority:",
            highest_priority
        )

        print()

        for _, row in event_df.iterrows():

            print(
                f'Post {row["id"]}: {row["text"]}'
            )

    if not crisis_event_found:

        print()
        print(
            "No crisis events detected."
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
    print("EVENT GROUPING RESULTS SAVED")
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
# EVENT SUMMARY
# ============================================================

def display_summary(df):

    print()
    print("=" * 60)
    print("EVENT SUMMARY")
    print("=" * 60)

    summary = (
        df[
            df["event_id"] != 0
        ]
        .groupby("event_id")
        .agg(
            post_count=("id", "count"),
            highest_severity=(
                "severity_score",
                "max"
            )
        )
        .reset_index()
    )

    if summary.empty:

        print(
            "No crisis events detected."
        )

        return

    print(
        summary.to_string(
            index=False
        )
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
        "REAL-TIME CRISIS EVENT GROUPING"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    df = load_alerts()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    event_ids = create_event_groups(
        df
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    df = add_event_information(
        df,
        event_ids
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    display_events(
        df
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    display_summary(
        df
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    save_results(
        df
    )

    print()
    print("=" * 60)
    print("CRISIS EVENT GROUPING COMPLETED")
    print("=" * 60)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Program stopped by user.")

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