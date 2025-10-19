import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================
# 1. Dataset loading
# ==============================
def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize MNIST (28x28) to 224x224 for ResNet
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
def evaluate_model(model, test_loader, device='cpu', save_roc_path=None):
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


# ==============================
# 6. Main loop
# ==============================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()

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
