import pickle
import json
import torch
from nam.model.neuraladditivemodel import NeuralAdditiveModel
from nam.model.InteractionNN import InteractionNN

def load_model(path, device):

    # Load config
    with open(path + '/config.bin', 'rb') as f:
        config = pickle.load(f)

    # Load model information
    with open(path + '/model_info.json', 'r') as f:
        model_info = json.load(f)

    # Build model
    model = NeuralAdditiveModel(config,
                                num_in_features = len(model_info['feature_names']),
                                num_first_layer_unit = model_info['num_first_layer_unit'],
                                intercept = model_info['intercept'])

    # Load trained model parameters
    model.load_state_dict(torch.load(path + '/model.pth', map_location = device))
    model = model.to(device)

    return model

def load_interaction_model(path, device):

    # Load model information
    with open(path + '/interaction_model_info.json', 'r') as f:
        model_info = json.load(f)

    # Build model
    model = InteractionNN(model_info['num_in_features'], model_info['hidden_layers'])

    # Load trained model parameters
    model.load_state_dict(torch.load(path + '/interaction_model.pth', map_location=device))
    model = model.to(device)

    return model


def load_stscaler(path):

    # Load standard scaler
    with open(path + '/StandardScaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return scaler
