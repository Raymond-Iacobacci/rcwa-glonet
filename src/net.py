import torch.nn as nn
from torch import tanh
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10 ** 2, 10 ** 4, bias = True)
    def forward(self, ipt, binary_amp):
        ipt = self.fc(ipt.view(-1))
        return tanh(ipt * binary_amp).view(100, 1, 100)
