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
import matplotlib.pyplot as plt
from datetime import datetime


# ============ Optimizer Selector ============
def get_optimizer(opt_name, model, lr=0.001):
    if opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    elif opt_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    elif opt_name == "Adagrad":
        return optim.Adagrad(model.parameters(), lr=lr)
    elif opt_name == "Adadelta":
        return optim.Adadelta(model.parameters(), lr=1.0)
    elif opt_name == "ASGD":
        return optim.ASGD(model.parameters(), lr=lr)
    elif opt_name == "Rprop":
        return optim.Rprop(model.parameters(), lr=lr)
    elif opt_name == "SparseAdam":
        return optim.SparseAdam(model.parameters(), lr=lr)
    elif opt_name == "LBFGS":
        return optim.LBFGS(model.parameters())
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ============ Training ============
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=20):
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            if isinstance(optimizer, optim.LBFGS):
                def closure():
                    optimizer.zero_grad()
                    pred2 = model(xb).squeeze()
                    loss2 = criterion(pred2, yb)
                    loss2.backward()
                    return loss2
                optimizer.step(closure)
            else:
                optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).squeeze()
                val_losses.append(criterion(pred, yb).item())

        history["train_loss"].append(sum(train_losses) / len(train_losses))
        history["val_loss"].append(sum(val_losses) / len(val_losses))

    return history


# ============ Grid Search ============
def run_gridsearch(ModelClass, model_name="DNN"):
    path = "cancer_reg-1.csv"
    data = Data(path)
    X, y = data.preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_data()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val.values, dtype=torch.float32))

    # hyperparams
    dropout_probs = [0.5, 0.6, 0.7, 0.8, 0.9]
    optimizers = ["SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta", "ASGD", "Rprop", "LBFGS"]

    epochs_list = [10, 20, 30, 40, 50]
    batch_sizes = [8, 16, 32, 64, 128]

    best_score = float("inf")
    best_params, best_model, best_history = None, None, None

    for dropout, opt_name, epochs, batch in itertools.product(dropout_probs, optimizers, epochs_list, batch_sizes):
        print(f"Trying config: dropout={dropout}, opt={opt_name}, epochs={epochs}, batch={batch}")
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch)

        model = ModelClass(X.shape[1], dropout)
        criterion = nn.MSELoss()
        optimizer = get_optimizer(opt_name, model)

        history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs)
        val_loss = history["val_loss"][-1]

        if val_loss < best_score:
            best_score = val_loss
            best_params = (dropout, opt_name, epochs, batch)
            best_model = model
            best_history = history
            print(f"New best model! Val Loss={val_loss:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(best_model.state_dict(), f"best_{model_name}_model_{timestamp}.pth")
    joblib.dump(best_params, f"best_{model_name}_params_{timestamp}.pkl")

    # plot
    plt.plot(best_history["train_loss"], label="Train Loss")
    plt.plot(best_history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title(f"{model_name} Training/Validation Loss")
    plt.savefig(f"training_curve_{model_name}_{timestamp}.png")
    plt.close()

    print("\nBest model saved:", best_params)


# ============ Test Model ============
def test_model(ModelClass, model_name="DNN", new_data_path="cancer_reg-1.csv", model_path=None, params_path=None):
    import pandas as pd
    path = new_data_path
    data = Data(path)
    X, y = data.preprocess()
    _, _, X_test, _, _, y_test = data.split_data()

    if model_path is None:
        model_path = sorted([f for f in os.listdir('..') if f.startswith(f"best_{model_name}_model")])[-1]
    if params_path is None:
        params_path = sorted([f for f in os.listdir('..') if f.startswith(f"best_{model_name}_params")])[-1]

    best_params = joblib.load(params_path)
    dropout, opt_name, epochs, batch = best_params

    model = ModelClass(X.shape[1], dropout)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()

    if "TARGET_deathRate" in data.raw_data.columns:
        print("Test MSE:", mean_squared_error(y_test, y_pred))
        print("Test R²:", r2_score(y_test, y_pred))
    else:
        print("Predictions only.")
        out_path = os.path.join(os.path.dirname(new_data_path), f"{model_name}_predictions.csv")
        data.raw_data["Predicted_TARGET_deathRate"] = y_pred
        data.raw_data.to_csv(out_path, index=False)
        print("Predictions saved to:", out_path)
