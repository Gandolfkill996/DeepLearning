from ..DNN_base import run_gridsearch, test_model
import torch.nn as nn

class DNN_16(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super(DNN_16, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Run training with hyperparameter grid search
    run_gridsearch(DNN_16, "DNN_16")

