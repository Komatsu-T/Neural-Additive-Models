import torch
from torch.utils.data import Dataset

class NAMDataset(Dataset):
    def __init__(self, config, dataframe, feature_names, target_name):
        self.X = torch.from_numpy(dataframe[feature_names].values).float().to(config.device)
        self.y = torch.from_numpy(dataframe[target_name].values).view(-1, 1).float().to(config.device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]