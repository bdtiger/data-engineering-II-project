import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features
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


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["language_encoded"] = pd.factorize(df["language"])[0]
    df["log_stars"] = np.log1p(df["stargazers_count"])
    X = df[FEATURE_COLUMNS].values
    y = df["log_stars"].values
    print(X.shape, y.shape)
    return X, y


def preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_model(n_features, output_bias_value):
    output_bias = keras.initializers.Constant(output_bias_value)
    model = Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, bias_initializer=output_bias)  # Output layer for regression
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()
    return model


def train_model(model, X_train, y_train):
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,     # 10% of train for val curve
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            )
        ]
    )
    return history


if __name__ == "__main__":
    X, y = load_data("github-repository-data.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
    model = build_model(len(FEATURE_COLUMNS), y_train.mean())
    history = train_model(model, X_train, y_train)
