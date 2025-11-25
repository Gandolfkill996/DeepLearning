# gcn_1layer.py
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For evaluation metrics and AUC plot
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# 1. Load ENZYMES dataset
def load_enzymes(root_dir="ENZYMES", prefix="ENZYMES"):
    """
    Load ENZYMES dataset from TUDataset style text files.

    Returns:
        graphs: list of dicts, each containing:
                - "X": (num_nodes, feature_dim)
                - "A": (num_nodes, num_nodes) adjacency matrix
        labels: numpy array of shape (num_graphs,), graph labels (0-based)
    """
    path = lambda name: os.path.join(root_dir, f"{prefix}_{name}.txt")

    edges = np.loadtxt(path("A"), delimiter=",", dtype=int)
    graph_indicator = np.loadtxt(path("graph_indicator"), dtype=int)
    graph_labels = np.loadtxt(path("graph_labels"), dtype=int)
    node_attr = np.loadtxt(path("node_attributes"), delimiter=",")
    node_labels = np.loadtxt(path("node_labels"), dtype=int)

    num_nodes = node_attr.shape[0]
    num_graphs = graph_labels.shape[0]

    # List which graph each node belongs to
    nodes_of_graph = [[] for _ in range(num_graphs)]
    for node_id, g in enumerate(graph_indicator, start=1):
        nodes_of_graph[g - 1].append(node_id)

    # Build global -> local node id maps
    global_to_local = []
    for g in range(num_graphs):
        mapping = {
            global_id: local_id
            for local_id, global_id in enumerate(nodes_of_graph[g])
        }
        global_to_local.append(mapping)

    # Initialize graph adjacency matrices
    graphs = []
    for g in range(num_graphs):
        k = len(nodes_of_graph[g])
        A = np.zeros((k, k), dtype=np.float32)
        graphs.append({"A": A})

    # Fill adjacency matrices
    for u, v in edges:
        g_idx = graph_indicator[u - 1] - 1
        lu = global_to_local[g_idx][u]
        lv = global_to_local[g_idx][v]
        graphs[g_idx]["A"][lu, lv] = 1.0
        graphs[g_idx]["A"][lv, lu] = 1.0

    # Build features = node attributes + node labels (one-hot)
    num_node_classes = int(node_labels.max())
    onehot = np.eye(num_node_classes, dtype=np.float32)

    for g in range(num_graphs):
        idx = np.array(nodes_of_graph[g]) - 1
        attr = node_attr[idx]
        lbl = node_labels[idx]
        lbl_onehot = onehot[lbl - 1]
        X = np.concatenate([attr, lbl_onehot], axis=1)
        graphs[g]["X"] = X.astype(np.float32)

    # Convert to 0-based graph labels
    labels = (graph_labels - 1).astype(np.int64)
    return graphs, labels


# 2. GCN model implementations
class GCNLayer(nn.Module):
    """Standard GCN layer: ReLU( D^{-1/2} (A + I) D^{-1/2} X W )."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: (num_nodes, in_dim), adj: (num_nodes, num_nodes)
        device = x.device
        I = torch.eye(adj.size(0), device=device)
        A_hat = adj + I

        degree = A_hat.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))

        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        out = A_norm @ x
        out = self.linear(out)
        return F.relu(out)


class GCN1LayerNet(nn.Module):
    """1-layer GCN + global sum pooling + linear classifier."""

    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        # x: (num_nodes, in_dim), adj: (num_nodes, num_nodes)
        h = self.gcn1(x, adj)                    # (num_nodes, hidden_dim)
        g = h.sum(dim=0, keepdim=True)           # (1, hidden_dim) global sum pooling
        out = self.classifier(g)                 # (1, num_classes)
        return out


# 3. Train / Evaluate helpers
def split_indices(N, train_ratio=0.8, val_ratio=0.1, seed=42):
    idx = list(range(N))
    random.Random(seed).shuffle(idx)
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


# 4. Training function (with time logging)
def train_model(
    root_dir="ENZYMES",
    num_epochs=400,
    hidden_dim=64,
    weight_decay=5e-4,
    model_path="gcn1_model.pth",
    device=None,
):
    """
    Train a 1-layer GCN model and save it to disk.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    graphs, labels = load_enzymes(root_dir)
    num_graphs = len(graphs)
    num_classes = int(labels.max()) + 1
    in_dim = graphs[0]["X"].shape[1]

    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    model = GCN1LayerNet(in_dim, hidden_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    train_idx, val_idx, test_idx = split_indices(num_graphs)

    print(f"Training on device: {device}")
    print(f"#Graphs={num_graphs}, #Classes={num_classes}, InputDim={in_dim}")
    print(f"Train/Val/Test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    # ---------------- Training Loop ----------------
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        model.train()
        random.shuffle(train_idx)

        total_loss = 0.0
        correct = 0

        for i in train_idx:
            x = torch.tensor(graphs[i]["X"], device=device)
            A = torch.tensor(graphs[i]["A"], device=device)
            y = labels_t[i].unsqueeze(0)

            optimizer.zero_grad()
            pred = model(x, A)                  # (1, num_classes)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()

        train_acc = correct / len(train_idx)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for i in val_idx:
                x = torch.tensor(graphs[i]["X"], device=device)
                A = torch.tensor(graphs[i]["A"], device=device)
                y = labels_t[i].unsqueeze(0)
                pred = model(x, A)
                val_correct += (pred.argmax(dim=1) == y).sum().item()
        val_acc = val_correct / len(val_idx)

        print(
            f"[Epoch {epoch:03d}] "
            f"Time={time.time() - epoch_start:.2f}s | "
            f"Loss={total_loss / len(train_idx):.4f} | "
            f"TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f}"
        )

    # Save trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model_path, test_idx


# 5. test_model() - load trained model and evaluate (Accuracy, F1, AUC)
def test_model(
    model_path,
    test_idx,
    root_dir="ENZYMES",
    hidden_dim=64,
    device=None,
    auc_plot_path="gcn1_auc.png",
):
    """
    Load the trained model and evaluate on the test set.

    Returns:
        acc: test accuracy
        f1_macro: macro-averaged F1 score
        roc_auc_macro: macro-averaged ROC-AUC score (OvR)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    graphs, labels = load_enzymes(root_dir)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    in_dim = graphs[0]["X"].shape[1]

    # Rebuild the model and load weights
    model = GCN1LayerNet(in_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    true_labels = []
    pred_labels = []
    all_probs = []

    with torch.no_grad():
        for i in test_idx:
            x = torch.tensor(graphs[i]["X"], device=device)
            A = torch.tensor(graphs[i]["A"], device=device)

            out = model(x, A)                               # (1, num_classes)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]  # (num_classes,)
            pred = probs.argmax()

            all_probs.append(probs)
            pred_labels.append(pred)
            true_labels.append(labels[i])

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    all_probs = np.vstack(all_probs)  # (n_test, num_classes)

    # Accuracy
    acc = (pred_labels == true_labels).mean()

    # Macro F1-score
    f1_macro = f1_score(true_labels, pred_labels, average="macro")

    # ---------------- ROC-AUC (multiclass, one-vs-rest) ----------------
    # Binarize labels for each class
    y_true_bin = label_binarize(true_labels, classes=list(range(num_classes)))

    # Compute per-class ROC curve and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for c in range(num_classes):
        fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, c], all_probs[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), all_probs.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average ROC-AUC using sklearn helper
    roc_auc_macro = roc_auc_score(
        y_true_bin, all_probs, average="macro", multi_class="ovr"
    )

    # Plot ROC curves
    plt.figure(figsize=(6, 6))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        linestyle=":",
        label=f"micro-average ROC (AUC = {roc_auc['micro']:.2f})",
    )

    for c in range(num_classes):
        plt.plot(
            fpr[c],
            tpr[c],
            label=f"class {c} (AUC = {roc_auc[c]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("GCN-1 layer ROC curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(auc_plot_path, dpi=300)
    plt.close()
    print(f"AUC plot saved to {auc_plot_path}")

    # Print metrics
    print(
        f"Test Accuracy = {acc:.4f} | "
        f"Macro F1 = {f1_macro:.4f} | "
        f"Macro ROC-AUC = {roc_auc_macro:.4f}"
    )

    return acc, f1_macro, roc_auc_macro


if __name__ == "__main__":
    model_path, test_idx = train_model()
    test_model(model_path, test_idx)
