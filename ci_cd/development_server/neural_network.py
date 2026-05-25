import time
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
    print(f"[1/4] Loading data from '{filepath}'...")
    df = pd.read_csv(filepath)
    df["language_encoded"] = pd.factorize(df["language"])[0]
    df["log_stars"] = np.log1p(df["stargazers_count"])
    X = df[FEATURE_COLUMNS].values
    y = df["log_stars"].values
    print(f"      Dataset loaded: {len(df)} rows | X shape: {X.shape} | y shape: {y.shape}")
    return X, y


def preprocess(X, y):
    print("[2/4] Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"      Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("      Feature scaling applied (StandardScaler)")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_model(n_features, output_bias_value):
    print("[3/4] Building model...")
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
    print("      Model compiled successfully")
    return model


def train_model(model, X_train, y_train):
    print("[4/4] Training model (max 100 epochs, early stopping patience=10)...")
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    epochs_run = len(history.history["loss"])
    print(f"      Training complete — ran {epochs_run} epoch(s) in {elapsed:.2f}s")
    print(f"      Final train loss: {history.history['loss'][-1]:.4f} | val loss: {history.history['val_loss'][-1]:.4f}")
    return history, elapsed


def save_timing_result(elapsed, epochs, num_vms=1, results_path="scalability_results.csv"):
    """Append a timing result row to a CSV for cross-VM scalability comparison."""
    import os
    row = pd.DataFrame([{"num_vms": num_vms, "training_time_s": round(elapsed, 4), "epochs": epochs}])
    write_header = not os.path.exists(results_path)
    row.to_csv(results_path, mode='a', header=write_header, index=False)
    print(f"[Timing] Result saved to '{results_path}' (num_vms={num_vms}, time={elapsed:.2f}s, epochs={epochs})")


def save_model(model, json_path="model.json", weights_path="model.weights.h5"):
    print(f"Saving model architecture to '{json_path}'...")
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    print(f"Saving model weights to '{weights_path}'...")
    model.save_weights(weights_path)
    print("Model saved to disk successfully")


if __name__ == "__main__":
    X, y = load_data("github-repository-data.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)
    model = build_model(len(FEATURE_COLUMNS), y_train.mean())
    history, training_time = train_model(model, X_train, y_train)
    print(f"\nTotal training time: {training_time:.2f}s")
    save_model(model)

    epochs_run = len(history.history["loss"])
    save_timing_result(training_time, epochs_run, num_vms=1)
