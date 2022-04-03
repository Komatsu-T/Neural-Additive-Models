import pandas as pd
import random
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from nam.dataset.namdataset import NAMDataset

class AnalysisDataset:
    def __init__(self, config, dataframe, feature_names, target_name):

        # Input data information
        self.feature_names = list(feature_names)
        self.target_name = target_name
        self.data = dataframe[self.feature_names + [self.target_name]].copy()
        self.data.reset_index(inplace = True, drop = True)
        self.config = config

        # Data split
        self.data_split()
        # Standardization
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_data)
        self.scaled_train_data = pd.DataFrame(self.scaler.transform(self.train_data), columns = self.feature_names + [self.target_name])
        self.scaled_test_data = pd.DataFrame(self.scaler.transform(self.test_data), columns = self.feature_names + [self.target_name])
        self.scaled_val_data = pd.DataFrame(self.scaler.transform(self.val_data), columns = self.feature_names + [self.target_name])
        # Intercept
        self.intercept = self.scaled_train_data[target_name].mean()

    def data_split(self):
        # Data size
        test_size = int(len(self.data)*self.config.test_size)
        val_size = int((len(self.data)-test_size)*self.config.validation_size)
        train_size = len(self.data)-(test_size+val_size)

        # Split
        index = [i for i in range(len(self.data))]
        random.shuffle(index)
        self.train_data = self.data[self.data.index.isin(index[:train_size])]
        self.test_data = self.data[self.data.index.isin(index[train_size:train_size+test_size])]
        self.val_data = self.data[self.data.index.isin(index[train_size+test_size:])]

    def get_trainloader(self):
        train_dataset = NAMDataset(self.config, self.scaled_train_data, self.feature_names, self.target_name)
        trainloader = DataLoader(train_dataset, batch_size = self.config.batch_size, shuffle = True)
        return trainloader

    def get_valloader(self):
        val_dataset = NAMDataset(self.config, self.scaled_val_data, self.feature_names, self.target_name)
        valloader = DataLoader(val_dataset, batch_size = self.config.batch_size, shuffle = True)
        return valloader

    def get_testloader(self):
        test_dataset = NAMDataset(self.config, self.scaled_test_data, self.feature_names, self.target_name)
        testloader = DataLoader(test_dataset, batch_size = self.config.batch_size, shuffle = False)
        return testloader