import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ExULayer(nn.Module):
    def __init__(self, in_features, out_features, ReLU_n = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.ReLU_n = ReLU_n
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.weights, mean = 4.0, std = 0.5)
        nn.init.normal_(self.bias, std = 0.5)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, Activation function: ReLU-{self.ReLU_n}'

    def forward(self, x):
        output = (x - self.bias).matmul(torch.exp(self.weights))
        output = F.relu(output)
        output = torch.clamp(output, 0, self.ReLU_n)
        return output