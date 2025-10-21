import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from .Data_preprocess import Data


def run_linear_regression():
    """
    Train Linear Regression on cancer_reg-1.csv,
    save model, scaler, selected features, and results into /outputs.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "cancer_reg-1.csv")

    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- Load and preprocess ----------------
    data = Data(data_path, corr_threshold=0.2)
    X, y = data.preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_data()

    # Merge train + val for linear regression (no early stopping)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # ---------------- Train model ----------------
    model = LinearRegression()
    model.fit(X_train_full, y_train_full)

    # ---------------- Evaluate ----------------
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n, p = X_test.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print("\n Linear Regression Performance (Test Set):")
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}, Adjusted R²: {adj_r2:.4f}")

    # ---------------- Save artifacts ----------------
    joblib.dump(model, os.path.join(output_dir, "linear_model.pkl"))
    joblib.dump(data.scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(data.selected_features, os.path.join(output_dir, "selected_features.pkl"))

    # ---------------- Save predictions ----------------
    pred_out = pd.DataFrame({"True": y_test, "Pred": y_pred})
    pred_out.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # ---------------- Save performance log ----------------
    with open(os.path.join(output_dir, "results_log.csv"), "w") as f:
        f.write("===== Linear Regression Results =====\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"R²: {r2:.4f}\n")
        f.write(f"Adjusted R²: {adj_r2:.4f}\n")

    # ---------------- Plot True vs Predicted ----------------
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression: True vs Predicted")
    plt.savefig(os.path.join(output_dir, "training_curve_linear.png"))
    plt.close()

    print(f"\n✅ Training complete. Results saved in: {output_dir}")


def test_model(new_data_path, device="cpu"):
    """
    Test trained Linear Regression model on a new dataset (with or without labels).
    - Loads model, scaler, and selected features from /outputs
    - Applies preprocessing (numeric coercion, fill NaN, standardize)
    - Saves predictions to outputs/linear_predictions.csv
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")

    model_path = os.path.join(output_dir, "linear_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    features_path = os.path.join(output_dir, "selected_features.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model not found. Please run training first.")

    print(f"\n✅ Running Linear Regression test_model on {new_data_path}")

    # Load artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selected_features = joblib.load(features_path)

    # Load new data
    data = pd.read_csv(new_data_path)
    available = [f for f in selected_features if f in data.columns]
    if not available:
        raise ValueError("❌ None of the selected features exist in the new dataset!")

    X_new = data[available].copy()
    for col in X_new.columns:
        X_new[col] = pd.to_numeric(X_new[col], errors="coerce").fillna(X_new[col].mean())

    X_new = scaler.transform(X_new)

    # Predict
    y_pred = model.predict(X_new)

    # Save predictions (prediction as first column)
    out_df = data.copy()
    out_df.insert(0, "Predicted_TARGET_deathRate", y_pred)
    out_path = os.path.join(output_dir, "linear_predictions.csv")
    out_df.to_csv(out_path, index=False)

    print(f"✅ Predictions saved to: {out_path}")

    # If ground truth exists
    if "TARGET_deathRate" in data.columns:
        y_true = pd.to_numeric(data["TARGET_deathRate"], errors="coerce").fillna(data["TARGET_deathRate"].mean()).values
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print("\n📊 Model Performance on New Data:")
        print(f"MSE: {mse:.4f}, R²: {r2:.4f}")


if __name__ == "__main__":
    run_linear_regression()


