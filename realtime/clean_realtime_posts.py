import pandas as pd

INPUT_FILE = "realtime/realtime_posts.csv"
OUTPUT_FILE = "realtime/realtime_posts.csv"


print("=" * 60)
print("CLEANING REAL-TIME SOCIAL MEDIA POSTS")
print("=" * 60)


# ------------------------------------------------------------
# LOAD POSTS
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print()
print("Records before cleaning:", len(df))


# ------------------------------------------------------------
# REMOVE DUPLICATE POSTS
# ------------------------------------------------------------

df = df.drop_duplicates(
    subset=["text"],
    keep="first"
)


# ------------------------------------------------------------
# RESET IDs
# ------------------------------------------------------------

df["id"] = range(1, len(df) + 1)


# ------------------------------------------------------------
# SAVE CLEAN DATA
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


print()
print("Records after cleaning:", len(df))

print()
print("=" * 60)
print("CLEANED POSTS")
print("=" * 60)

print(
    df.to_string(index=False)
)

print()
print("=" * 60)
print("CLEANING COMPLETED")
print("=" * 60)

print(
    "Clean file:",
    OUTPUT_FILE
)