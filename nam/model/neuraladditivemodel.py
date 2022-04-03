import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from nam.model.featurenn import FeatureNN

class NeuralAdditiveModel(nn.Module):
    def __init__(self, config, num_in_features, num_first_layer_unit, intercept):
        super().__init__()
        self.num_in_features = num_in_features
        self.dropout = nn.Dropout(p = 0.0)
        self.num_first_layer_unit = num_first_layer_unit

        # Create FeatureNN of all input features
        FeatureNNs = [FeatureNN(config, 1, num_first_layer_unit[i]) for i in range(num_in_features)]
        FeatureNNs = nn.ModuleList(FeatureNNs)
        self.FeatureNNs = FeatureNNs

        # Intercept of sum of FeatureNNs output
        self.intercept = Parameter(data = torch.FloatTensor(np.array([intercept])), requires_grad = False)

    def forward(self, x):
        per_feature_output = [self.FeatureNNs[i](x[:, i]) for i in range(self.num_in_features)]
        per_feature_output = torch.cat(per_feature_output, dim = -1)
        per_feature_output = self.dropout(per_feature_output)
        linear_combination = self.intercept + torch.sum(per_feature_output, dim = -1)
        return linear_combination, per_feature_output