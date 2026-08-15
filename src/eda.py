import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CRISIS MMD - EXPLORATORY DATA ANALYSIS
# ============================================================

INPUT_FILE = Path("data/processed/crisis_mmd_cleaned.csv")
OUTPUT_DIR = Path("outputs/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("CRISIS MMD - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("✅ Dataset loaded")
print(f"Records: {len(df)}")


# ------------------------------------------------------------
# 2. Basic information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. Informative vs Not Informative
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEXT INFORMATIVENESS")
print("=" * 70)

info_counts = df["text_info"].value_counts()

print(info_counts)

print("\nPercentage:")
print(
    df["text_info"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# Plot
plt.figure(figsize=(8, 5))

info_counts.plot(kind="bar")

plt.title("Informative vs Not Informative Tweets")
plt.xlabel("Category")
plt.ylabel("Number of Tweets")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "text_informativeness.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 4. Humanitarian categories
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HUMANITARIAN CATEGORIES")
print("=" * 70)

human_counts = df["text_human"].value_counts()

print(human_counts)

print("\nPercentage:")
print(
    df["text_human"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# Plot
plt.figure(figsize=(10, 6))

human_counts.plot(kind="bar")

plt.title("Humanitarian Category Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Tweets")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "humanitarian_categories.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 5. Disaster event distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DISASTER EVENT DISTRIBUTION")
print("=" * 70)

event_counts = df["event"].value_counts()

print(event_counts)


# Plot
plt.figure(figsize=(10, 6))

event_counts.plot(kind="bar")

plt.title("Tweets by Disaster Event")
plt.xlabel("Disaster Event")
plt.ylabel("Number of Tweets")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "event_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 6. Tweet length analysis
# ------------------------------------------------------------

df["text_length"] = df["clean_text"].str.len()

print("\n" + "=" * 70)
print("TEXT LENGTH")
print("=" * 70)

print(df["text_length"].describe())


# Plot
plt.figure(figsize=(8, 5))

plt.hist(df["text_length"], bins=50)

plt.title("Tweet Text Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Number of Tweets")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "text_length_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Completed
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("✅ EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Charts saved in: {OUTPUT_DIR}")