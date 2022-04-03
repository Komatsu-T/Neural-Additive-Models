import torch.nn as nn
from nam.model.relulayer import ReLULayer
from nam.model.exulayer import ExULayer

class FeatureNN(nn.Module):
    def __init__(self, config, in_features, num_first_layer_unit):
        super().__init__()
        self.dropout = nn.Dropout(p = 0.0)

        # Create neural network
        network_structure = [num_first_layer_unit] + config.hidden_layer_size
        layer_list = []
        for i, (input, output) in enumerate(zip([in_features] + network_structure, network_structure + [1])):
            # First layer
            if i == 0:
                if config.first_layer == 'exu':
                    layer = ExULayer(input, output)
                else:
                    layer = ReLULayer(input, output)
            # Last layer
            elif i == len(network_structure):
                layer = nn.Linear(input, output)
            # Intermediate layer
            else:
                layer = ReLULayer(input, output)
            # Create layer structure
            layer_list.append(layer)

        # Pytorch model
        self.model = nn.ModuleList(layer_list)

    def forward(self, x):
        output = x.unsqueeze(1)
        for layer in self.model:
            output = self.dropout(layer(output))
        return output