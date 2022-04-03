import torch
import torch.nn as nn

class OutputPenalty(nn.Module):
    def __init__(self, reg_param):
        super().__init__()
        self.reg_param = reg_param

    def extra_repr(self):
        return f'Regularization: {self.reg_param}'

    def forward(self, per_feature_output):
        penalty = [torch.mean(torch.square(outputs)) for outputs in per_feature_output]
        penalty = self.reg_param*(sum(penalty)/len(penalty))
        return penalty

class WeightDecay(nn.Module):
    def __init__(self, reg_param):
        super().__init__()
        self.reg_param = reg_param

    def extra_repr(self):
        return f'Regularization: {self.reg_param}'

    def forward(self, model):
        num_networks = len(model.FeatureNNs)
        penalty = [(x**2).sum() for x in model.parameters()]
        penalty = self.reg_param*(sum(penalty)/num_networks)
        return penalty

class LossFunction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MSELoss = nn.MSELoss()
        self.output_penalty = OutputPenalty(config.output_regularization)
        self.weight_decay = WeightDecay(config.l2_regularization)

    def forward(self, predicted, target, per_feature_output, model):
        loss = self.MSELoss(predicted, target.view(-1))
        loss += self.output_penalty(per_feature_output)
        loss += self.weight_decay(model)
        return loss