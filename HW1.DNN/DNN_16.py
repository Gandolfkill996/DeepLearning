import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from Data_preprocess import Data
import itertools
from DNN_base import run_gridsearch, test_model
import torch.nn as nn

class DNN16(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.5):
        super(DNN16, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    run_gridsearch(DNN16, "DNN16")
    test_model(DNN16, "DNN16")
