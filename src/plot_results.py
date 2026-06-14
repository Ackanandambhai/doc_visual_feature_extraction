import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = PROJECT_ROOT / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Find all comparison.csv files automatically
# --------------------------------------------------
csv_files = list(OUTPUTS_DIR.glob("*/comparison.csv"))

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No comparison.csv files found inside {OUTPUTS_DIR}"
    )

# --------------------------------------------------
# Read and combine all CSVs
# --------------------------------------------------
dfs = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# --------------------------------------------------
# Save combined CSV
# --------------------------------------------------
combined_csv = PLOTS_DIR / "combined_model_comparison.csv"
combined_df.to_csv(combined_csv, index=False)

print(f"Saved: {combined_csv}")

# --------------------------------------------------
# Sort by Test Accuracy (optional)
# --------------------------------------------------
combined_df = combined_df.sort_values(
    by="Test_Accuracy",
    ascending=False
)

# --------------------------------------------------
# Helper function
# --------------------------------------------------
def save_bar_plot(column, ylabel, filename):

    plt.figure(figsize=(10, 6))

    plt.bar(
        combined_df["Model"],
        combined_df[column]
    )

    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(ylabel + " Comparison")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=300
    )

    plt.close()

    print(f"Saved: {filename}")


# --------------------------------------------------
# Individual comparison plots
# --------------------------------------------------

save_bar_plot(
    "Test_Accuracy",
    "Test Accuracy",
    "test_accuracy_comparison.png"
)

save_bar_plot(
    "F1_Score",
    "F1 Score",
    "f1_score_comparison.png"
)

save_bar_plot(
    "Parameters",
    "Parameters",
    "parameter_comparison.png"
)

save_bar_plot(
    "Training_Time",
    "Training Time (seconds)",
    "training_time_comparison.png"
)

# --------------------------------------------------
# Accuracy vs Parameters
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    combined_df["Parameters"],
    combined_df["Test_Accuracy"]
)

for _, row in combined_df.iterrows():
    plt.text(
        row["Parameters"],
        row["Test_Accuracy"],
        row["Model"],
        fontsize=8
    )

plt.xlabel("Parameters")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Parameters")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "accuracy_vs_parameters.png",
    dpi=300
)

plt.close()

print("Saved: accuracy_vs_parameters.png")

# --------------------------------------------------
# Accuracy vs Training Time
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    combined_df["Training_Time"],
    combined_df["Test_Accuracy"]
)

for _, row in combined_df.iterrows():
    plt.text(
        row["Training_Time"],
        row["Test_Accuracy"],
        row["Model"],
        fontsize=8
    )

plt.xlabel("Training Time (seconds)")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Training Time")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "accuracy_vs_training_time.png",
    dpi=300
)

plt.close()

print("Saved: accuracy_vs_training_time.png")

print("\n====================================")
print("All plots generated successfully!")
print(f"Results saved in: {PLOTS_DIR}")
print("====================================")