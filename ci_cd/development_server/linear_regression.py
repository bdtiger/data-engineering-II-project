import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from utils import evaluate_model

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

df = pd.read_csv("github-repository-data.csv")

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

evaluate_model(
    model,
    X_test_scaled,
    y_test,
    model_name="Linear Regression"
)