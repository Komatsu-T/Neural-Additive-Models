import torch.nn as nn
import torch.nn.functional as F

class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        output = self.linear(x)
        output = F.relu(output)
        return output