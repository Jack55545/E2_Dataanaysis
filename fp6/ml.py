import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error

from IPython.display import display


def Ml_beginner():
    DATA_PATH = "data.csv"
    df = pd.read_csv(DATA_PATH)
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

    # Ensure numeric; coerce errors to NaN (then impute)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    if "Flavor.intensity" in X.columns:
        X["Flavor.intensity_sq"] = X["Flavor.intensity"] ** 2
    feature_cols = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = feature_cols

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
        ],
        remainder="drop"
    )

    lin_model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression())
    ])

    ridge_model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # Baseline (mean predictor)
    baseline_pred = np.full_like(y_test, fill_value=np.nanmean(y_train), dtype=float)
    baseline_r2 = r2_score(y_test, baseline_pred)
    baseline_rmse = rmse(y_test, baseline_pred)

    # Linear regression
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred)
    lin_rmse = rmse(y_test, lin_pred)

    # Ridge regression
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_rmse = rmse(y_test, ridge_pred)

    results = pd.DataFrame({
        "Model": ["Baseline (mean)", "LinearRegression", "Ridge(alpha=1)"],
        "Test R2": [baseline_r2, lin_r2, ridge_r2],
        "Test RMSE": [baseline_rmse, lin_rmse, ridge_rmse],
    })
    display(results.round(3))
    return 0




def coef_show():

    DATA_PATH = "data.csv"
    df = pd.read_csv(DATA_PATH)
    target = "Liking"
    feature_cols = [
        "Dose", "Grind", "Brew Mass", "Percent Extraction", "pH", "Volume",
        "Brew Temperature", "Pour Temp", "90Sec Temp",
        "Flavor.intensity", "Acidity", "Mouthfeel",
        "Fruit", "Bitter", "Astringent", "Sour", "Sweet"
    ]

# Keep only columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target].copy()

# Ensure numeric; coerce errors to NaN (then impute)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    if "Flavor.intensity" in X.columns:
        X["Flavor.intensity_sq"] = X["Flavor.intensity"] ** 2
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )
    numeric_features = feature_cols

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
        ],
        remainder="drop"
    )

    lin_model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression())
    ])

    ridge_model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])

    
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

# Baseline (mean predictor)
    baseline_pred = np.full_like(y_test, fill_value=np.nanmean(y_train), dtype=float)
    baseline_r2 = r2_score(y_test, baseline_pred)
    baseline_rmse = rmse(y_test, baseline_pred)

# Linear regression
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred)
    lin_rmse = rmse(y_test, lin_pred)

# Ridge regression
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_rmse = rmse(y_test, ridge_pred)

    results = pd.DataFrame({
        "Model": ["Baseline (mean)", "LinearRegression", "Ridge(alpha=1)"],
        "Test R2": [baseline_r2, lin_r2, ridge_r2],
        "Test RMSE": [baseline_rmse, lin_rmse, ridge_rmse],
    })

    lin_fitted = lin_model.named_steps["model"]
    coef = lin_fitted.coef_
    coef_df = pd.DataFrame({"feature": numeric_features, "coef": coef}).sort_values("coef")
    display(coef_df.round(3))








































