
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================
# 1. Dataset loading
# ==============================
def get_dataloaders(batch_size=128):
    """
    Load MNIST dataset from ../data and split into train/val/test loaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    # Split training set into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, val_loader, test_loader


# ==============================
# 2. DNN model
# ==============================
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x1 = self.relu(self.fc1(x))      # first layer features
        x1 = self.dropout(x1)
        x2 = self.relu(self.fc2(x1))     # second layer features
        x3 = self.relu(self.fc3(x2))
        x4 = self.fc4(x3)
        return x4, x1, x2


# ==============================
# 3. Evaluation function
# ==============================
def evaluate_model(model, test_loader, device='cpu', save_roc_path=None):
    """
    Evaluate model performance on test dataset and optionally save ROC curve.
    """
    model.eval()
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out, _, _ = model(xb)
            pred = torch.argmax(out, 1)
            prob = torch.softmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(yb.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    preds, labels, probs = np.array(preds), np.array(labels), np.array(probs)
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='weighted')
    labels_onehot = np.eye(10)[labels]
    auc = roc_auc_score(labels_onehot, probs, average='macro', multi_class='ovr')

    # Plot ROC if needed
    if save_roc_path:
        plt.figure()
        for i in range(10):
            fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
            plt.plot(fpr, tpr, lw=1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("AUC-ROC (Test Set)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(save_roc_path)
        plt.close()

    return acc, f1, auc


# ==============================
# 4. Feature visualization
# ==============================
def visualize_features(model, test_loader, save_dir, device='cpu'):
    """
    Visualize the first and second layer features of the model.
    """
    xb, _ = next(iter(test_loader))
    xb = xb.to(device)
    with torch.no_grad():
        _, x1, x2 = model(xb)
    x1, x2 = x1.cpu().numpy(), x2.cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(x1[:20, :].T, aspect='auto', cmap='viridis')
    plt.title("First Layer Feature Visualization (first 20 samples)")
    plt.colorbar()
    plt.savefig(f"{save_dir}/layer1_features.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.imshow(x2[:20, :].T, aspect='auto', cmap='plasma')
    plt.title("Second Layer Feature Visualization (first 20 samples)")
    plt.colorbar()
    plt.savefig(f"{save_dir}/layer2_features.png")
    plt.close()


# ==============================
# 5. Training loop
# ==============================
def train_and_validate(model, optimizer, criterion, train_loader, val_loader, epochs, device, save_dir):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out, _, _ = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (torch.argmax(out, 1) == yb).sum().item()
        train_losses.append(total_loss / len(train_loader))
        train_accs.append(correct / len(train_loader.dataset))

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out, _, _ = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()
                val_correct += (torch.argmax(out, 1) == yb).sum().item()
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_correct / len(val_loader.dataset))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")

    # Save loss and accuracy curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{save_dir}/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{save_dir}/accuracy_curve.png")
    plt.close()

def test_model(model_path, device, output_dir="outputs/test_eval"):
    """
    Load the trained DNN model and evaluate on 10% of MNIST test dataset.

    Parameters
    ----------
    model_path : str
        Path to the saved model (.pth file)
    device : torch.device
        Device to run evaluation (cuda, mps, or cpu)
    output_dir : str
        Directory to save ROC curve and metrics
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load test dataset (only 10%)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_test = datasets.MNIST(root="../data", train=False, download=True, transform=transform)

    subset_size = int(len(full_test) * 0.1)
    test_subset, _ = torch.utils.data.random_split(full_test, [subset_size, len(full_test) - subset_size])

    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

    # Define model structure (must match training)
    class DNNModel(nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Load trained model
    model = DNNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    try:
        auc = roc_auc_score(
            torch.nn.functional.one_hot(torch.tensor(y_true), num_classes=10),
            y_score,
            average="macro",
            multi_class="ovr"
        )
    except:
        auc = float('nan')

    print(f"✅ Test Results -> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # ROC curve for one-vs-rest of class 0
    fpr, tpr, _ = roc_curve(
        (torch.tensor(y_true) == 0).int(),
        [s[0] for s in y_score]
    )
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Class 0 vs Rest)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    print(f"📊 ROC curve saved to {os.path.join(output_dir, 'roc_curve.png')}")

# ==============================
# 6. Main function
# ==============================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()

    learning_rates = [0.1, 0.01, 0.001]
    results = []

    for lr in learning_rates:
        print(f"\n===== Training DNN with learning rate = {lr} =====")
        lr_dir = f"DNN/outputs/lr_{lr}"
        os.makedirs(lr_dir, exist_ok=True)

        model = DNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_and_validate(model, optimizer, criterion, train_loader, val_loader,
                           epochs=10, device=device, save_dir=lr_dir)

        # Save model weights
        model_path = f"{lr_dir}/best_dnn_lr{lr}.pth"
        torch.save(model.state_dict(), model_path)

        # Evaluate model
        roc_path = f"{lr_dir}/roc_curve.png"
        acc, f1, auc = evaluate_model(model, test_loader, device, save_roc_path=roc_path)
        results.append((lr, acc, f1, auc))
        print(f"LR={lr} → Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        # Feature visualization
        visualize_features(model, test_loader, lr_dir, device)

    print("\n===== Summary for all learning rates =====")
    for lr, acc, f1, auc in results:
        print(f"LR={lr:<6} | Accuracy={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")



