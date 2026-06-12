"""
plot_results.py  -  Re-generate comparison plots from an existing comparison.csv

Usage
-----
python src/plot_results.py --csv outputs/comparison.csv --out_dir outputs
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def bar_chart(names, values, ylabel, title, out_path, fmt=".3f"):
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set(ylabel=ylabel, title=title)
    ax.set_ylim(0, max(values) * 1.15)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                format(v, fmt), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="outputs/comparison.csv")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    rows      = load_csv(args.csv)
    plots_dir = Path(args.out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    names = [r["Model"] for r in rows]

    bar_chart(names, [float(r["Test_Accuracy"])    for r in rows],
              "Test Accuracy",    "Test Accuracy Comparison",    plots_dir/"cmp_test_acc.png")
    bar_chart(names, [float(r["F1_Score"])         for r in rows],
              "F1-Score (macro)", "F1-Score Comparison",         plots_dir/"cmp_f1.png")
    bar_chart(names, [float(r["Training_Time"])    for r in rows],
              "Training Time (s)","Training Time Comparison",    plots_dir/"cmp_time.png", fmt=".0f")
    bar_chart(names, [int(r["Parameters"])/1e6     for r in rows],
              "Parameters (M)",   "Parameter Count Comparison",  plots_dir/"cmp_params.png", fmt=".1f")

    print("\n{:<22} {:>9} {:>8} {:>8} {:>8} {:>8} {:>9}".format(
        "Model","Params M","Train","Val","Test","F1","Time(s)"))
    print("-" * 77)
    for r in rows:
        print("{:<22} {:>9.1f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>9.1f}".format(
            r["Model"], int(r["Parameters"])/1e6,
            float(r["Train_Accuracy"]), float(r["Val_Accuracy"]),
            float(r["Test_Accuracy"]),  float(r["F1_Score"]),
            float(r["Training_Time"])))


if __name__ == "__main__":
    main()
