import torch
import torch.nn as nn
import itertools

class InteractionNN(nn.Module):
    def __init__(self, num_in_features, hidden_layers = [64, 128, 128, 128]):
        super().__init__()
        self.layers = []
        self.num_in_features = num_in_features
        self.hidden_layers = hidden_layers
        for input_num, output_num in zip([num_in_features] + hidden_layers + [hidden_layers[-1]], ([num_in_features] + hidden_layers + [1])[1:]):
            self.layers.append(nn.Linear(input_num, output_num))
            self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers[:-1])

    def forward(self, x):
        output = self.model(x)
        return output
