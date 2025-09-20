import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from Data_preprocess import Data
import joblib
import os
import pandas as pd

def adjusted_r2(r2, n, p):
    """calculate Adjusted R²"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def run_linear_regression():
    path = "cancer_reg-1.csv"
    data = Data(path)
    X, y = data.preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_data()

    # Combine train + val
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Use train+val construct linear model
    model = LinearRegression()
    model.fit(X_train_full, y_train_full)

    # Predict on test dataset
    y_test_pred = model.predict(X_test)

    # calculate R²
    r2_test = r2_score(y_test, y_test_pred)

    # calculate Adjusted R²
    n_train, p = X_train_full.shape
    n_test = X_test.shape[0]

    adj_r2_test = adjusted_r2(r2_test, n_test, p)

    # Sum of parameters of fittness
    results = {
        "Test MSE": mean_squared_error(y_test, y_test_pred),
        "Test R2": r2_test,
        "Test Adjusted R2": adj_r2_test
    }

    # print result
    print("\nLinear Regression Performance")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return model, data.scaler, data.selected_features,  results

def test_model(new_data_path=None, save_predictions=True):
    """
    Load trained model and test on dataset.
    If new_data_path is provided, test on that CSV.
    Otherwise, test on the original training CSV.
    """

    # Load saved model, scaler, and features
    model = joblib.load("linear_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selected_features = joblib.load("selected_features.pkl")

    # Default to original training dataset
    if new_data_path is None:
        new_data_path = "cancer_reg-1.csv"

    print(f"\nTesting model on: {new_data_path}")

    # Load raw dataset
    df_new = pd.read_csv(new_data_path, encoding="latin1", encoding_errors="replace")

    # Fill missing values
    for col in df_new.columns:
        if df_new[col].dtype == "object":
            df_new[col] = df_new[col].fillna(df_new[col].mode()[0])
        else:
            df_new[col] = df_new[col].fillna(df_new[col].mean())

    # Keep only selected features
    X_new = df_new[selected_features]
    X_new_scaled = scaler.transform(X_new)

    # Predict
    y_pred = model.predict(X_new_scaled)

    # If label exists → evaluate
    if "TARGET_deathRate" in df_new.columns:
        y_true = df_new["TARGET_deathRate"].values
        r2 = r2_score(y_true, y_pred)
        adj_r2 = adjusted_r2(r2, X_new.shape[0], X_new.shape[1])
        mse = mean_squared_error(y_true, y_pred)
        print("\n Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Adjusted R²: {adj_r2:.4f}")
    else:
        print("\n No TARGET_deathRate column found → only predictions will be saved.")

    # Save predictions
    if save_predictions:
        output_dir = os.path.dirname(new_data_path)
        output_file = os.path.join(output_dir, "predictions.csv")
        df_new["Predicted_TARGET_deathRate"] = y_pred
        df_new.to_csv(output_file, index=False)
        print(f"\n Predictions saved to: {output_file}")
        print(df_new[["Predicted_TARGET_deathRate"]].head())


    return y_pred


if __name__ == "__main__":
    model, scaler, selected_features, results = run_linear_regression()

    # Save model
    joblib.dump(model, "linear_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(selected_features, "selected_features.pkl")
    print("Model saved！")

    # Test on training dataset by default
    test_model()
