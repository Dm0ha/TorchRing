import torch.nn as nn
import torch

class Model(nn.Module):
    """
    A simple PyTorch model.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.layer1(x)
        return x

    def loss(self):
        return nn.MSELoss()
    
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)