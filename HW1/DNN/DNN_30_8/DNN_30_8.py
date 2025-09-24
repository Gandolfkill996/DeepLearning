from ..DNN_base import run_gridsearch, test_model
import torch.nn as nn

class DNN_30_8(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super(DNN_30_8, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1)
        )
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    # Run training with hyperparameter grid search
    run_gridsearch(DNN_30_8, "DNN_30_8")
