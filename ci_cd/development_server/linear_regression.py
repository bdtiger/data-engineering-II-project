import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)

FEATURE_COLUMNS = [
    "forks_count",
    "subscribers_count",
    "open_issues_count",
    "size",
    "network_count",
    "has_wiki",
    "has_pages",
    "has_issues",
    "topics_count",
    "age_days",
    "language_encoded"
]

TARGET = "stargazers_count"

df = pd.read_csv("../../crawler/repos.csv")

df["language_encoded"] = pd.factorize(df["language"])[0]

df["log_stars"] = np.log1p(df["stargazers_count"])

X = df[FEATURE_COLUMNS].values
y = df["log_stars"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_prediction_log = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_prediction_log)

y_prediction_stars = np.expm1(y_prediction_log)
y_test_stars = np.expm1(y_test)


mae = mean_absolute_error(y_test_stars, y_prediction_stars)

mape = mean_absolute_percentage_error(
    y_test_stars + 1,
    y_prediction_stars + 1
)

accuracy = max(0.0, 1.0 - mape)

print("\n=============================================")
print("           Linear Regression")
print("=============================================")

print(f"R²               : {r2:.4f}")
print(f"MAE (stars)      : {mae:,.0f}")
print(f"MAPE             : {mape * 100:.2f}%")
print(f"Accuracy (1-MAPE): {accuracy * 100:.2f}%")

print("=============================================")