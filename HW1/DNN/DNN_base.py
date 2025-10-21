import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from .Data_preprocess import Data
import itertools
import joblib
import os
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd


# ================= Helper Functions =================

def baseline_metrics(y):
    """
    Compute baseline metrics using a simple mean predictor.
    The baseline predictor always predicts the mean of y for all samples.

    Args:
        y (array-like): True target values.

    Returns:
        mse (float): Mean Squared Error of baseline predictor.
        r2 (float): R² score of baseline predictor.
    """
    y_mean = np.mean(y)
    mse = mean_squared_error(y, [y_mean] * len(y))
    r2 = r2_score(y, [y_mean] * len(y))
    return mse, r2


def get_model_dir(model_name):
    """
    Return the output directory for the given model.
    Ensures that each model has its own 'outputs/' folder.

    Example:
        HW1/DNN/DNN_30_16_8_4/outputs/

    Args:
        model_name (str): The name of the model class.

    Returns:
        model_dir (str): Path to the output directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, model_name, "outputs")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


# ================= Training Functions =================

def train_model(model, train_loader, val_loader, optimizer, criterion,
                epochs=20, patience=5, device="cpu"):
    """
    Train a PyTorch model with early stopping.
    Track and return the best model weights during training
    based on highest validation R² score.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer (e.g., SGD, Adam).
        criterion (loss function): Loss function (e.g., MSELoss).
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience (number of epochs without improvement).
        device (str): "cpu" or "cuda".

    Returns:
        history (dict): Training history including train_loss, val_loss, and val_r2.
        best_model_state (dict): Model state dict (weights) at the best R².
        best_r2 (float): Best validation R² score observed.
        best_val_mse (float): Validation MSE corresponding to best R².
    """
    history = {"train_loss": [], "val_loss": [], "val_r2": []}
    best_model_state = None
    best_r2_at_best = float("-inf")  # Track the best R² score
    best_val_mse = float("inf")      # Track MSE at the best R²
    wait = 0  # Patience counter

    for epoch in range(epochs):
        # ----- Training phase -----
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb).squeeze()
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Prevent exploding gradients
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ----- Validation phase -----
        model.eval()
        val_losses, val_true, val_pred = [], [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb).squeeze()
                val_losses.append(criterion(pred, yb).item())
                val_true.extend(yb.cpu().numpy())
                val_pred.extend(pred.cpu().numpy())

        val_mse = sum(val_losses) / len(val_losses)
        val_r2 = r2_score(val_true, val_pred) if len(set(val_true)) > 1 else float("nan")

        # Log results
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_mse)
        history["val_r2"].append(val_r2)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Val MSE: {val_mse:.4f}, Val R²: {val_r2:.4f}")

        # ----- Save best model by R² -----
        if val_r2 > best_r2_at_best:
            best_r2_at_best = val_r2
            best_val_mse = val_mse
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

    return history, best_model_state, best_r2_at_best, best_val_mse


def run_gridsearch(ModelClass, model_name="DNN_30_16_8", device="cpu"):
    """
    Run grid search over hyperparameters for the given model class.
    Save the best model, its parameters, training curves,
    and the selected feature list.

    Selection criterion: Highest validation R² (not lowest MSE).

    Args:
        ModelClass (class): PyTorch model class.
        model_name (str): Name of the model.
        device (str): "cpu" or "cuda".
    """
    # ----- Load data -----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "cancer_reg-1.csv")
    data = Data(path, corr_threshold=0.2)
    X, y = data.preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_data()

    # Baseline (mean predictor)
    base_mse, base_r2 = baseline_metrics(y_val)
    print(f"📊 Baseline (mean predictor) - MSE: {base_mse:.4f}, R²: {base_r2:.4f}")

    # Convert to TensorDatasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val.values, dtype=torch.float32))

    # ----- Hyperparameter grid -----
    dropout_probs = [0.5, 0.7]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    epochs_list = [50]
    batch_sizes = [32, 64]

    # Global best tracking (maximize R²)
    global_best_r2 = float("-inf")
    global_best_params, global_best_model_state, global_best_history = None, None, None

    results = []   # Store results of all configs
    lr_results = {}  # Track best per learning rate
    model_dir = get_model_dir(model_name)

    # ----- Grid search -----
    for lr in learning_rates:
        best_r2_lr = float("-inf")
        best_params_lr, best_history_lr = None, None

        for dropout, epochs, batch in itertools.product(dropout_probs, epochs_list, batch_sizes):
            print(f"\n⚡ Trying config: dropout={dropout}, lr={lr}, epochs={epochs}, batch={batch}")

            train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch)

            model = ModelClass(X.shape[1], dropout).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                  nesterov=True, weight_decay=1e-4)

            # Train and evaluate
            history, best_model_state, best_r2, best_val_mse = train_model(
                model, train_loader, val_loader, optimizer, criterion,
                epochs, patience=10, device=device
            )

            # Log results for this config
            results.append([dropout, lr, epochs, batch, best_val_mse, best_r2])

            # Update best for this learning rate
            if best_r2 > best_r2_lr:
                best_r2_lr = best_r2
                best_params_lr = (dropout, lr, epochs, batch)
                best_history_lr = history

            # Update global best
            if best_r2 > global_best_r2:
                global_best_r2 = best_r2
                global_best_params = (dropout, lr, epochs, batch)
                global_best_model_state = best_model_state
                global_best_history = history
                print(f"New GLOBAL best! Val R²={best_r2:.4f}, Val MSE={best_val_mse:.4f}")

        lr_results[lr] = (best_r2_lr, best_params_lr, best_history_lr["val_r2"][-1])

    # ----- Save results -----
    results_df = pd.DataFrame(
        results,
        columns=["dropout", "lr", "epochs", "batch", "val_mse", "val_r2"]
    )
    results_path = os.path.join(model_dir, "results_log.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    print("\n===== Best Results per Learning Rate =====")
    for lr, (r2, params, _) in lr_results.items():
        print(f"LR={lr} → Best Val R²={r2:.4f}, Params={params}")

    # Save best model and params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(global_best_model_state,
               os.path.join(model_dir, f"best_{model_name}_model_{timestamp}.pth"))
    joblib.dump(global_best_params,
                os.path.join(model_dir, f"best_{model_name}_params_{timestamp}.pkl"))

    # Save selected features for consistency in prediction
    joblib.dump(data.selected_features,
                os.path.join(model_dir, f"{model_name}_features.pkl"))

    # Print final best result (R² and corresponding MSE)
    best_val_r2 = global_best_r2
    best_val_mse = global_best_history["val_loss"][-1]
    print(f"\n Global best model saved: {global_best_params}")
    print(f"     Best Val R²: {best_val_r2:.4f}")
    print(f"     Corresponding Val MSE: {best_val_mse:.4f}")

    # Plot validation performance curve (MSE vs. epoch)
    plt.figure(figsize=(8, 5))
    plt.plot(global_best_history["train_loss"], label="Training MSE", color="red")
    plt.plot(global_best_history["val_loss"], label="Validation MSE", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error Loss")
    plt.legend()
    plt.title(f"{model_name} Training and Validation Performance (Best by R²)")
    plt.savefig(os.path.join(model_dir, f"training_curve_{model_name}_{timestamp}.png"))
    plt.close()


def test_model(ModelClass, model_name, new_data_path=None, model_path=None, params_path=None, device="cpu"):
    """
    Test the best saved model on a new dataset (with or without labels).
    If labels are present, evaluate performance (MSE, R²).
    If labels are absent, only generate predictions.

    This version:
      - aligns columns with saved selected_features.pkl (intersection),
      - coerces all selected columns to numeric and imputes NaNs with column means,
      - standardizes the inputs (z-score) to match training-time scaling style,
      - saves predictions as the FIRST column to ease grading,
      - robustly finds latest *_params_*.pkl and *.pth in outputs/.
    """
    import os, joblib, numpy as np, pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score
    import torch

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, model_name, "outputs")

    # 1) Resolve input CSV
    if new_data_path is None:
        new_data_path = os.path.join(base_dir, "cancer_reg-1.csv")
    print(f"\n Running test_model with {model_name} on {new_data_path}")
    print(f"\n Testing model on: {new_data_path}")

    # 2) Load raw data with the same helper used in training
    data = Data(new_data_path, corr_threshold=0.2)  # class in Data_preprocess.py  :contentReference[oaicite:2]{index=2}
    df = data.raw_data.copy()

    # 3) Load selected features (column list from training) and align by intersection
    features_path = os.path.join(model_dir, f"{model_name}_features.pkl")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features file: {features_path}")
    selected_features = joblib.load(features_path)

    # Keep only features that exist in the new CSV
    features = [c for c in selected_features if c in df.columns]
    if not features:
        raise ValueError(" None of the saved selected features are present in the new dataset!")

    # 4) Subset & CLEAN: coerce to numeric, impute NaNs with column means
    Xdf = df[features].copy()

    # Coerce every selected column to numeric; invalid parses become NaN
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
        # Fill NaNs with column mean (after coercion)
        col_mean = Xdf[c].mean()
        Xdf[c] = Xdf[c].fillna(col_mean)

    # 5) Standardize (z-score) — same effect as StandardScaler fit on new data
    X = Xdf.values.astype(np.float32)
    means = X.mean(axis=0, keepdims=True)
    stds = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - means) / stds

    # Quick sanity check (avoid silent NaNs)
    if np.isnan(X).any():
        n = np.isnan(X).sum()
        raise ValueError(f"Found {n} NaNs after cleaning — please check the input CSV.")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # 6) Locate params (*.pkl) & model (*.pth) — pick the latest by filename (timestamp at tail)
    if params_path is None:
        param_files = [f for f in os.listdir(model_dir) if "params" in f and f.endswith(".pkl")]
        if not param_files:
            raise FileNotFoundError(f"No params .pkl file found in {model_dir}")
        params_path = os.path.join(model_dir, sorted(param_files)[-1])
    params = joblib.load(params_path)

    if model_path is None:
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        if not model_files:
            raise FileNotFoundError(f"No model .pth file found in {model_dir}")
        model_path = os.path.join(model_dir, sorted(model_files)[-1])

    # 7) Build model with the same input dim & dropout, then load weights
    model = ModelClass(X.shape[1], params[0]).to(device)  # dropout from saved params  :contentReference[oaicite:3]{index=3}
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 8) Predict
    with torch.no_grad():
        y_pred = model(X_tensor).squeeze().cpu().numpy()

    # 9) Save predictions (put prediction as FIRST column; rows aligned 1:1)
    out_path = os.path.join(model_dir, f"{model_name}_predictions.csv")
    out_df = df.reset_index(drop=True).copy()
    out_df.insert(0, "Predicted_TARGET_deathRate", y_pred)  # first column
    out_df.to_csv(out_path, index=False)
    print(f" Predictions saved to: {out_path}")

    # 10) If ground truth exists, report metrics
    if "TARGET_deathRate" in df.columns:
        y_true = pd.to_numeric(df["TARGET_deathRate"], errors="coerce").fillna(df["TARGET_deathRate"].mean()).values
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        base_mse, base_r2 = baseline_metrics(y_true)  # from DNN_base.py  :contentReference[oaicite:4]{index=4}

        print("\n Model Performance:")
        print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
        print(f"Baseline MSE: {base_mse:.4f}, Baseline R²: {base_r2:.4f}")
    else:
        print("\n⚠ No TARGET_deathRate column found. Skipping evaluation (MSE/R²).")



