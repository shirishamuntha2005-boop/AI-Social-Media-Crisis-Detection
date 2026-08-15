import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CRISIS MMD - MODEL COMPARISON
# ============================================================

print("=" * 70)
print("CRISIS MMD - MODEL COMPARISON")
print("=" * 70)


# ============================================================
# 1. MODEL RESULTS
# ============================================================

results = {
    "Model": [
        "Logistic Regression",
        "Linear SVM",
        "Random Forest",
        "Balanced Logistic Regression"
    ],

    "Accuracy": [
        0.8007,
        0.7886,
        0.8029,
        0.7783
    ],

    "Not_Informative_Recall": [
        0.45,
        0.53,
        0.47,
        0.72
    ],

    "Not_Informative_F1": [
        0.56,
        0.59,
        0.57,
        0.65
    ]
}


# ============================================================
# 2. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# 3. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print()

print(df.to_string(index=False))


# ============================================================
# 4. FIND BEST ACCURACY MODEL
# ============================================================

best_accuracy = df.loc[
    df["Accuracy"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL BY ACCURACY")
print("=" * 70)

print(
    f"\nModel   : {best_accuracy['Model']}"
)

print(
    f"Accuracy: {best_accuracy['Accuracy'] * 100:.2f}%"
)


# ============================================================
# 5. FIND BEST MODEL FOR MINORITY CLASS
# ============================================================

best_recall = df.loc[
    df["Not_Informative_Recall"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL FOR NOT-INFORMATIVE CLASS")
print("=" * 70)

print(
    f"\nModel : {best_recall['Model']}"
)

print(
    f"Recall: {best_recall['Not_Informative_Recall'] * 100:.2f}%"
)

print(
    f"F1    : {best_recall['Not_Informative_F1']:.2f}"
)


# ============================================================
# 6. CREATE ACCURACY GRAPH
# ============================================================

print("\nCreating accuracy comparison graph...")

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Accuracy"] * 100
)

plt.title("Model Accuracy Comparison")

plt.xlabel("Machine Learning Model")

plt.ylabel("Accuracy (%)")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.ylim(0, 100)

plt.tight_layout()


# ============================================================
# 7. SAVE GRAPH
# ============================================================

output_path = "data/processed/model_accuracy_comparison.png"

plt.savefig(output_path)

print(
    f"✅ Graph saved to: {output_path}"
)

plt.show()


# ============================================================
# 8. SAVE RESULTS
# ============================================================

csv_path = "data/processed/model_comparison.csv"

df.to_csv(
    csv_path,
    index=False
)

print(
    f"✅ Results saved to: {csv_path}"
)


# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)

print("\nRecommended models:")

print(
    f"🏆 Highest Accuracy : {best_accuracy['Model']}"
)

print(
    f"🛡️ Best Minority Recall : {best_recall['Model']}"
)

print("\n" + "=" * 70)