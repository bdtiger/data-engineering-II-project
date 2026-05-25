from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a regression model predicting log(stars).
    
    Args:
        model      : trained model with a .predict() method
        X_test     : scaled test features
        y_test     : true values in log scale (log1p transformed)
        model_name : label shown in the printed report
    
    Returns:
        dict with r2, rmsle, mae
    """
    # Prediction
    y_prediction_log = model.predict(X_test).flatten()

    # Metrics on log scale
    r2 = r2_score(y_test, y_prediction_log)
    rmsle = np.sqrt(np.mean((y_prediction_log - y_test) ** 2))

    # Convert back to real star counts
    y_prediction_stars = np.expm1(y_prediction_log)
    y_test_stars = np.expm1(y_test)

    # Metrics on real scale
    mae = mean_absolute_error(y_test_stars, y_prediction_stars)

    # Report
    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  R²               : {r2:.4f}")
    print(f"  RMSLE            : {rmsle:.4f}")
    print(f"  MAE (stars)      : {mae:>12,.0f}")
    print(f"{'='*45}")

    return {"model": model_name, "r2": r2, "rmsle": rmsle, "mae": mae}
