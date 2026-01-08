import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def _add_flavor_group(df, col="Flavor.intensity"):
    df = df.copy()
    median_val = df[col].median()
    df["FlavorGroup"] = np.where(df[col] >= median_val, 1, 0)
    return df, median_val


def compute_liking_stats(dataset):
    df = pd.read_csv(dataset)
    df, median_flavor = _add_flavor_group(df, col="Flavor.intensity")

    features = ["Bitter", "Fruit", "Astringent", "FlavorGroup"]
    rows = []

    for f1, f2 in combinations(features, 2):  # 順序を考えない全てのペア
        for v1 in (0, 1):
            for v2 in (0, 1):
                sub = df[(df[f1] == v1) & (df[f2] == v2)]
                if len(sub) == 0:
                    continue
                rows.append({
                    "Feature1": f1,
                    "Value1": v1,
                    "Feature2": f2,
                    "Value2": v2,
                    "N_samples": len(sub),
                    "Median_Liking": sub["Liking"].median(),
                    "Var_Liking": sub["Liking"].var()
                })

    result = pd.DataFrame(rows)
    return result


def plot_bitter_liking_box(dataset):
    df = pd.read_csv(dataset)

    low = df[df["Bitter"] == 0]["Liking"]
    high = df[df["Bitter"] == 1]["Liking"]

    plt.figure(figsize=(5, 5))
    plt.boxplot([low, high], tick_labels=["Low Bitter", "High Bitter"])
    plt.title("Liking vs Bitter Group")
    plt.ylabel("Liking")
    plt.grid(True)
    plt.show()

def plot_median_variance_scatter(dataset):
    result = compute_liking_stats(dataset)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        result["Median_Liking"],
        result["Var_Liking"],
        alpha=1
    )
    ax.set_xlabel("Median of Liking")
    ax.set_ylabel("Variance of Liking")
    ax.set_title("Liking Median vs Variance (Taste Conditions)")
    ax.grid(True)
    plt.show()