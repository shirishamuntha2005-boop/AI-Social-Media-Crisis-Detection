import os
import subprocess
import sys


# ============================================================
# AI-POWERED SOCIAL MEDIA CRISIS DETECTION
# COMPLETE REAL-TIME PIPELINE
# ============================================================

print("=" * 70)
print("AI-POWERED SOCIAL MEDIA CRISIS DETECTION")
print("COMPLETE REAL-TIME PROCESSING PIPELINE")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


# ============================================================
# PIPELINE MODULES
# ============================================================

PIPELINE_STEPS = [
    ("clean_realtime_posts.py", "REAL-TIME POST CLEANING"),
    ("realtime_predictor.py", "RoBERTa CRISIS CLASSIFICATION"),
    ("severity_detector.py", "CRISIS SEVERITY DETECTION"),
    ("alert_generator.py", "CRISIS ALERT GENERATION"),
    ("crisis_event_grouping.py", "CRISIS EVENT GROUPING"),
    ("final_crisis_analysis.py", "FINAL CRISIS ANALYSIS"),
]


# ============================================================
# RUN ONE MODULE
# ============================================================

def run_step(filename, description):

    print("\n")
    print("=" * 70)
    print(description)
    print("=" * 70)

    script_path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(script_path):

        print(f"ERROR: File not found: {script_path}")

        return False

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_DIR
    )

    if result.returncode != 0:

        print()
        print("=" * 70)
        print(f"PIPELINE FAILED: {filename}")
        print("=" * 70)

        return False

    print()
    print(f"✓ COMPLETED: {filename}")

    return True


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print()
    print("Pipeline stages:")
    
    for number, (filename, description) in enumerate(
        PIPELINE_STEPS,
        start=1
    ):

        print(f"{number}. {filename}")

    print()

    # --------------------------------------------------------
    # Run all stages
    # --------------------------------------------------------

    for filename, description in PIPELINE_STEPS:

        success = run_step(
            filename,
            description
        )

        if not success:

            print()
            print("=" * 70)
            print("REAL-TIME PIPELINE STOPPED")
            print("=" * 70)

            return

    # --------------------------------------------------------
    # Final success
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Generated files:")

    output_files = [
        "realtime_predictions.csv",
        "realtime_severity.csv",
        "realtime_alerts.csv",
        "realtime_events.csv"
    ]

    for filename in output_files:

        path = os.path.join(
            BASE_DIR,
            filename
        )

        if os.path.exists(path):

            print(f"✓ {filename}")

        else:

            print(f"✗ Missing: {filename}")

    print()
    print("=" * 70)
    print("AI SOCIAL MEDIA CRISIS DETECTION PIPELINE COMPLETED")
    print("=" * 70)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Pipeline stopped by user.")

    except Exception as error:

        print()
        print("=" * 70)
        print("PIPELINE ERROR")
        print("=" * 70)

        print(
            type(error).__name__,
            ":",
            error
        )