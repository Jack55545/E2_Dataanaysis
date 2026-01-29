import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import statistics as st


df = pd.read_csv("data.csv")

target = "Liking"
feature_cols = [
    "Dose", "Grind", "Brew Mass", "Percent Extraction", "pH", "Volume",
    "Brew Temperature", "Pour Temp", "90Sec Temp",
    "Flavor.intensity", "Acidity", "Mouthfeel",
    "Fruit", "Bitter", "Astringent", "Sour", "Sweet"
]
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
y = df[target].copy()
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")
if "Flavor.intensity" in X.columns:
    X["Flavor.intensity_sq"] = X["Flavor.intensity"] ** 2
feature_cols = X.columns.tolist()

cont_cols = [
    "Dose", "Grind", "Brew Mass", "Percent Extraction", "pH", "Volume",
    "Brew Temperature", "90Sec Temp",
    "Flavor.intensity", "Acidity", "Mouthfeel"]

bin_cols = ["Fruit", "Bitter", "Astringent", "Sour", "Sweet","Flavor.intensity_sq"]
preprocess = ColumnTransformer(
    transformers=[
        ("cont", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler()),
        ]), cont_cols),

        ("bin", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            # bin は何もしない（スケール不要）
        ]), bin_cols),
    ],
    remainder="drop"
)

pipe1 = Pipeline([
    ("pre", preprocess),
    ("model", LinearRegression())
])
pipe2 = Pipeline([
    ("model", LinearRegression())
])
cv = KFold(n_splits=5, shuffle=True, random_state=0)


def show_all():
    scores1 = list(cross_val_score(pipe1,X,y,cv=cv,scoring="r2"))
    scores2 = list(cross_val_score(pipe2,X,y,cv=cv,scoring="r2"))
    print(st.mean(scores1),st.variance(scores1))
    print(st.mean(scores2),st.variance(scores2))


median_PourTemp = df["Pour Temp"].median()
df["PourTempGroup"] = (df["Pour Temp"] >= median_PourTemp).astype(int)

low_temp  = df[df["PourTempGroup"] == 0]
high_temp = df[df["PourTempGroup"] == 1]

feature_cols2 = [
    "Dose", "Grind", "Brew Mass", "Percent Extraction", "pH", "Volume",
    "Brew Temperature", "90Sec Temp",
    "Flavor.intensity", "Acidity", "Mouthfeel",
    "Fruit", "Bitter", "Astringent", "Sour", "Sweet"
]

Xlow  = low_temp[feature_cols2].copy()
ylow  = low_temp[target]
if "Flavor.intensity" in Xlow.columns:
    Xlow["Flavor.intensity_sq"] = Xlow["Flavor.intensity"] ** 2
feature_cols = Xlow.columns.tolist()

Xhigh = high_temp[feature_cols2].copy()
yhigh = high_temp[target]
if "Flavor.intensity" in Xhigh.columns:
    Xhigh["Flavor.intensity_sq"] = Xhigh["Flavor.intensity"] ** 2
feature_cols = Xhigh.columns.tolist()

def show_temp_separate():
    scores_low1 = cross_val_score(pipe1,Xlow,ylow,cv=cv,scoring="r2")
    scores_low2 = cross_val_score(pipe2,Xlow,ylow,cv=cv,scoring="r2")
    scores_high1= cross_val_score(pipe1,Xhigh,yhigh,cv=cv,scoring="r2")
    scores_high2= cross_val_score(pipe2,Xhigh,yhigh,cv=cv,scoring="r2")
    print(st.mean(scores_low1),st.variance(scores_low1))
    print(st.mean(scores_low2),st.variance(scores_low2))
    print(st.mean(scores_high1),st.variance(scores_high1))
    print(st.mean(scores_high2),st.variance(scores_high2))

    pipe1.fit(Xlow, ylow)
    model = pipe1.named_steps["model"]
    coef_low = pd.Series(
        model.coef_,
        index=Xlow.columns,
        name="coef_low"
    ).sort_values(key=abs, ascending=False)
    pipe1.fit(Xhigh, yhigh)
    model = pipe1.named_steps["model"]
    coef_high = pd.Series(
        model.coef_,
        index=Xhigh.columns,
        name="coef_high"
    ).sort_values(key=abs, ascending=False)

    coef_df = pd.concat([coef_low, coef_high], axis=1)
    coef_df["diff_high_low"] = coef_df["coef_high"] - coef_df["coef_low"]

    print(coef_df.sort_values("diff_high_low", key=abs, ascending=False))
























