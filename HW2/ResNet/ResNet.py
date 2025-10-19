import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================
# 1. Dataset loading
# ==============================
def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize MNIST (28x28) to 224x224 for ResNet
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, val_loader, test_loader


# ==============================
# 2. ResNet18 model (modified for MNIST)
# ==============================
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # Load predefined ResNet18
        self.model = models.resnet18(weights=False)
        # Adjust first conv layer (MNIST has 1 channel)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)

    def forward(self, x):
        # Access early features for visualization
        x1 = self.model.conv1(x)       # first conv output
        x1 = self.model.bn1(x1)
        x1 = self.model.relu(x1)
        x2 = self.model.layer1(x1)     # second block output
        out = self.model(x)
        return out, x1, x2


# ==============================
# 3. Evaluate model
# ==============================
def evaluate_model(model, test_loader, device='cuda', save_roc_path=None):
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
    xb, _ = next(iter(test_loader))
    xb = xb.to(device)
    with torch.no_grad():
        _, x1, x2 = model(xb)

    x1, x2 = x1.cpu().numpy(), x2.cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.imshow(x1[0, :16].mean(axis=0), cmap='viridis')
    plt.title("First Conv Layer Feature Map (mean of 16 channels)")
    plt.colorbar()
    plt.savefig(f"{save_dir}/layer1_features.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.imshow(x2[0, :16].mean(axis=0), cmap='plasma')
    plt.title("Second Block Feature Map (mean of 16 channels)")
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

    # Save curves
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

def test_model(model_path="outputs/lr_0.001/best_resnet_lr0.001.pth", device=None, output_dir="outputs/test_eval"):
    """
    Load the trained ResNet18 model and evaluate on 10% of the MNIST test dataset.

    Parameters
    ----------
    model_path : str
        Path to the saved model (.pth file)
    device : torch.device
        Device to run evaluation (cuda, mps, or cpu)
    output_dir : str
        Directory to save ROC curve and metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    # ✅ Load test dataset (only 10%)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_test = datasets.MNIST(root="../data", train=False, download=True, transform=transform)

    subset_size = int(len(full_test) * 0.1)
    test_subset, _ = torch.utils.data.random_split(full_test, [subset_size, len(full_test) - subset_size])
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)

    # ✅ Define ResNet18 architecture (must match training)
    class ResNet18Modified(nn.Module):
        def __init__(self):
            super(ResNet18Modified, self).__init__()
            self.model = models.resnet18(weights=None)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 10)

        def forward(self, x):
            return self.model(x)

    # ✅ Load trained model
    model = ResNet18Modified().to(device)
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

    # ✅ Calculate metrics
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

    print(f"✅ Test Results (ResNet, lr=0.001) -> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # ✅ ROC curve for one-vs-rest of class 0
    fpr, tpr, _ = roc_curve(
        (torch.tensor(y_true) == 0).int(),
        [s[0] for s in y_score]
    )
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ResNet18 ROC Curve (Class 0 vs Rest, lr=0.001)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    print(f"ROC curve saved to {os.path.join(output_dir, 'roc_curve.png')}")

# ==============================
# 6. Main loop
# ==============================
if __name__ == "__main__":
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
    learning_rates = [0.1, 0.01, 0.001]
    results = []

    for lr in learning_rates:
        print(f"\n===== Training ResNet18 with learning rate = {lr} =====")
        lr_dir = f"outputs/lr_{lr}"
        os.makedirs(lr_dir, exist_ok=True)

        model = ResNet18().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_and_validate(model, optimizer, criterion, train_loader, val_loader,
                           epochs=10, device=device, save_dir=lr_dir)

        # Save model
        model_path = f"{lr_dir}/best_resnet_lr{lr}.pth"
        torch.save(model.state_dict(), model_path)

        # Evaluate
        roc_path = f"{lr_dir}/roc_curve.png"
        acc, f1, auc = evaluate_model(model, test_loader, device, save_roc_path=roc_path)
        results.append((lr, acc, f1, auc))
        print(f"LR={lr} → Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        # Visualize features
        visualize_features(model, test_loader, lr_dir, device)

    print("\n===== Summary for all learning rates =====")
    for lr, acc, f1, auc in results:
        print(f"LR={lr:<6} | Accuracy={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
