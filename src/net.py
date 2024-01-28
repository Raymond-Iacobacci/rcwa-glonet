import torch.nn as nn
from torch import tanh
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(25, 100, bias = True)
        self.fc2 = nn.Linear(100, 1200, bias = True)
    def forward(self, ipt, binary_amp):
        ipt = self.fc(ipt.view(-1))
        ipt = self.fc2(ipt)
        return tanh(ipt * binary_amp).view(20, 20, 3)