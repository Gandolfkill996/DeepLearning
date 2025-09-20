# DNN_base.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from Data_preprocess import Data
import itertools
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime


# ============ Training ============
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=20):
    history = {"train_loss": [], "val_loss": [], "val_r2": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch_idx, (xb, yb) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()

            # prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # prevent NaN crash
            if torch.isnan(pred).any():
                print(f"⚠️ NaN detected at Epoch {epoch+1}, Batch {batch_idx+1}")
                continue

            # batch metrics
            y_true = yb.detach().numpy()
            y_pred = pred.detach().numpy()
            batch_mse = mean_squared_error(y_true, y_pred)
            batch_r2 = r2_score(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")
            print(f"Epoch {epoch+1}, Batch {batch_idx+1} - MSE: {batch_mse:.4f}, R²: {batch_r2:.4f}")

            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses, val_true, val_pred = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).squeeze()
                val_losses.append(criterion(pred, yb).item())
                val_true.extend(yb.numpy())
                val_pred.extend(pred.numpy())

        val_mse = sum(val_losses) / len(val_losses)
        val_r2 = r2_score(val_true, val_pred) if len(set(val_true)) > 1 else float("nan")

        history["train_loss"].append(sum(train_losses) / len(train_losses))
        history["val_loss"].append(val_mse)
        history["val_r2"].append(val_r2)

        print(f"Epoch {epoch+1} Summary - Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val MSE: {val_mse:.4f}, Val R²: {val_r2:.4f}")

    return history


# ============ Grid Search ============
def run_gridsearch(ModelClass, model_name="DNN16"):
    path = "cancer_reg-1.csv"
    data = Data(path)
    X, y = data.preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_data()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val.values, dtype=torch.float32))

    # hyperparams
    dropout_probs = [0.5, 0.7, 0.9]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    epochs_list = [20]
    batch_sizes = [32, 64]

    # track best per-lr
    lr_results = {}
    global_best_score = float("inf")
    global_best_params, global_best_model, global_best_history = None, None, None

    for lr in learning_rates:
        best_score_lr = float("inf")
        best_params_lr, best_history_lr = None, None

        for dropout, epochs, batch in itertools.product(dropout_probs, epochs_list, batch_sizes):
            print(f"\n⚡ Trying config: dropout={dropout}, lr={lr}, epochs={epochs}, batch={batch}")
            train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch)

            model = ModelClass(X.shape[1], dropout)

            # Xavier initializer
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

            history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs)
            val_loss = history["val_loss"][-1]

            if val_loss < best_score_lr:
                best_score_lr = val_loss
                best_params_lr = (dropout, lr, epochs, batch)
                best_history_lr = history

                # update DNN_16 best model
                if val_loss < global_best_score:
                    global_best_score = val_loss
                    global_best_params = best_params_lr
                    global_best_model = model
                    global_best_history = history
                    print(f"🔥 New GLOBAL best! Val Loss={val_loss:.4f}, Val R²={history['val_r2'][-1]:.4f}")

        lr_results[lr] = (best_score_lr, best_params_lr, best_history_lr["val_r2"][-1])

    # print every lr model result
    print("\n===== Best Results per Learning Rate =====")
    for lr, (mse, params, r2) in lr_results.items():
        print(f"LR={lr} → Best Val MSE={mse:.4f}, R²={r2:.4f}, Params={params}")

    # Only save DNN16 best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_name == "DNN16":
        torch.save(global_best_model.state_dict(), f"best_{model_name}_model_{timestamp}.pth")
        joblib.dump(global_best_params, f"best_{model_name}_params_{timestamp}.pkl")

        # plot
        plt.figure(figsize=(8, 5))
        plt.plot(global_best_history["train_loss"], label="Train Loss")
        plt.plot(global_best_history["val_loss"], label="Val MSE")
        plt.plot(global_best_history["val_r2"], label="Val R²")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.title(f"{model_name} Training/Validation Metrics (Best)")
        plt.savefig(f"training_curve_{model_name}_{timestamp}.png")
        plt.close()

        print("\n✅ Global best model saved:", global_best_params)


# ============ Test Model ============
def test_model(ModelClass, model_name="DNN16", new_data_path="cancer_reg-1.csv", model_path=None, params_path=None):
    import pandas as pd
    path = new_data_path
    data = Data(path)
    X, y = data.preprocess()
    _, _, X_test, _, _, y_test = data.split_data()

    if model_path is None:
        model_path = sorted([f for f in os.listdir('.') if f.startswith(f"best_{model_name}_model")])[-1]
    if params_path is None:
        params_path = sorted([f for f in os.listdir('.') if f.startswith(f"best_{model_name}_params")])[-1]

    best_params = joblib.load(params_path)
    dropout, lr, epochs, batch = best_params

    model = ModelClass(X.shape[1], dropout)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()

    if "TARGET_deathRate" in data.raw_data.columns:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("Test MSE:", mse)
        print("Test R²:", r2)
    else:
        print("Predictions only.")
        out_path = os.path.join(os.path.dirname(new_data_path), f"{model_name}_predictions.csv")
        data.raw_data["Predicted_TARGET_deathRate"] = y_pred
        data.raw_data.to_csv(out_path, index=False)
        print("Predictions saved to:", out_path)
