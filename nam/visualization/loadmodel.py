import json
import torch
from nam.model.neuraladditivemodel import NeuralAdditiveModel

def load_model(config):
    # Load model info
    with open(config.output_dir + '/model_info.json', 'r') as f:
        model_info = json.load(f)

    # Load model
    trained_model = NeuralAdditiveModel(config,
                                        num_in_features = len(model_info['feature_names']),
                                        num_first_layer_unit = model_info['num_first_layer_unit'],
                                        intercept = model_info['intercept'])
    trained_model.load_state_dict(torch.load(config.output_dir + '/model.pth', map_location = config.device))
    trained_model = trained_model.to(config.device)
    return trained_model