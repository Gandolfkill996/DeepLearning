import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from Data_preprocess import Data
import itertools
from DNN_base import run_gridsearch, test_model
import torch.nn as nn

class DNN_30_16_8(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.5):
        super(DNN_30_16_8, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(8, 1)
        )
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    run_gridsearch(DNN_30_16_8, "DNN_30_16_8")
    test_model(DNN_30_16_8, "DNN_30_16_8")
