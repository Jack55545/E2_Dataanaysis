import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r
def boxplot0(dataset,value,itemlist=['Sweet', 'Fruit', 'Bitter', 'Astringent']):
    df0=pd.read_csv(dataset)
    df=df0[itemlist+[value]]
    y=df[value]
    swe = df['Sweet']
    fru = df['Fruit']
    bit = df['Bitter']
    ast = df['Astringent']
    fig,axs = plt.subplots( 1, 4, figsize=(10,3), tight_layout=True )
    for i,col in enumerate(itemlist):
        groups=[df[df[col]==v][value] for v in sorted(df[col].unique())]
        r=corrcoeff(df[col],y)
        axs[i].boxplot(groups)
        axs[i].set_xlabel(col)
        axs[i].set_title(f"r = {r:.3f}")
        if i==0:
            axs[i].set_ylabel(value)
        else:
            axs[i].set_yticklabels([])
    plt.show()   

def parabolaplot(dataset,value, item='Flavor.intensity'):
    df0=pd.read_csv(dataset)
    y=df0[value]
    fla=df0[item]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(fla, y, alpha=0.5, color='r')
    coeffs = np.polyfit(fla, y, 2)
    x_line = np.linspace(fla.min(), fla.max(), 100)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, color='k', ls='-', lw=2)
    ax.set_xlabel(item)
    ax.set_ylabel(value)
    ax.set_title("Fig.2 Relationship Between Flavor Intensity and Liking")
    plt.show()

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
    plt.title("Fig.3 Liking Distributions Stratified by Bitterness Level")
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
    ax.set_title("Fig.4 Scatter Plot of Median and Variance of Liking by Taste Condition")
    ax.grid(True)
    plt.show()