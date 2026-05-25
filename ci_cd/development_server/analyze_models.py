import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
data_path = r'd:\University\Data_Engieering_II\data-engineering-II-project\ci_cd\development_server\github-repository-data.csv'
df = pd.read_csv(data_path)

print("=" * 80)
print("MODEL ANALYSIS AND EVALUATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"Dataset columns: {df.columns.tolist()}")
print(f"First few rows:\n{df.head()}\n")

# Prepare data - using numeric columns for X and y
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {numeric_cols}\n")

if len(numeric_cols) >= 2:
    # Use all numeric columns except the last one as features, and last as target
    X = df[numeric_cols[:-1]]
    y = df[numeric_cols[-1]]
else:
    print("Not enough numeric columns")
    exit()

# Handle missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 80)
print("MODEL 1: SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 80)
print("Parameters: C=15, epsilon=0.1, kernel='rbf'")

# Train SVR model with specified parameters
svr_model = SVR(C=15, epsilon=0.1, kernel='rbf')
svr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_svr = svr_model.predict(X_test_scaled)

# Calculate metrics for SVR
r2_svr = r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mape_svr = mean_absolute_percentage_error(y_test, y_pred_svr)

# For accuracy, convert to classification (threshold at median)
median_val = y_test.median()
y_test_binary = (y_test > median_val).astype(int)
y_pred_svr_binary = (y_pred_svr > median_val).astype(int)
accuracy_svr = accuracy_score(y_test_binary, y_pred_svr_binary)

print(f"\nSVR Model Results (Test Set):")
print(f"  R² Score:              {r2_svr:.6f}")
print(f"  MAE (Mean Absolute Error): {mae_svr:.6f}")
print(f"  MAPE (Mean Absolute Percentage Error): {mape_svr:.6f}")
print(f"  Accuracy (binary classification): {accuracy_svr:.6f}")

print("\n" + "=" * 80)
print("MODEL 2: NEURAL NETWORK (MLPRegressor)")
print("=" * 80)
print("Architecture: MLPRegressor with hidden layers (64, 32)")

# Train Neural Network model
nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, activation='relu')
nn_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_nn = nn_model.predict(X_test_scaled)

# Calculate metrics for Neural Network
r2_nn = r2_score(y_test, y_pred_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mape_nn = mean_absolute_percentage_error(y_test, y_pred_nn)

# For accuracy, convert to classification (threshold at median)
y_pred_nn_binary = (y_pred_nn > median_val).astype(int)
accuracy_nn = accuracy_score(y_test_binary, y_pred_nn_binary)

print(f"\nNeural Network Results (Test Set):")
print(f"  R² Score:              {r2_nn:.6f}")
print(f"  MAE (Mean Absolute Error): {mae_nn:.6f}")
print(f"  MAPE (Mean Absolute Percentage Error): {mape_nn:.6f}")
print(f"  Accuracy (binary classification): {accuracy_nn:.6f}")

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"\n{'Metric':<35} {'SVR':<15} {'Neural Network':<15} {'Better':<10}")
print("-" * 75)
print(f"{'R² Score':<35} {r2_svr:<15.6f} {r2_nn:<15.6f} {'NN' if r2_nn > r2_svr else 'SVR':<10}")
print(f"{'MAE':<35} {mae_svr:<15.6f} {mae_nn:<15.6f} {'NN' if mae_nn < mae_svr else 'SVR':<10}")
print(f"{'MAPE':<35} {mape_svr:<15.6f} {mape_nn:<15.6f} {'NN' if mape_nn < mape_svr else 'SVR':<10}")
print(f"{'Accuracy':<35} {accuracy_svr:<15.6f} {accuracy_nn:<15.6f} {'NN' if accuracy_nn > accuracy_svr else 'SVR':<10}")
print("=" * 80)
