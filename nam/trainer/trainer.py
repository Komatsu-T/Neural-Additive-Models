import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nam.trainer.lossfunction import LossFunction
from nam.model.neuraladditivemodel import NeuralAdditiveModel

class Trainer:
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.trainloader = self.dataset.get_trainloader()
        self.valloader = self.dataset.get_valloader()
        self.testloader = self.dataset.get_testloader()
        self.criterion = LossFunction(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma = self.config.gamma, step_size = self.config.step_size)

    def load_trained_model(self):
        # Load model info
        with open(self.config.output_dir + '/model_info.json', 'r') as f:
            model_info = json.load(f)

        # Load model
        trained_model = NeuralAdditiveModel(self.config,
                                            num_in_features = len(model_info['feature_names']),
                                            num_first_layer_unit = model_info['num_first_layer_unit'],
                                            intercept = model_info['intercept'])
        trained_model.load_state_dict(torch.load(self.config.output_dir + '/model.pth', map_location = self.config.device))
        trained_model = trained_model.to(self.config.device)
        return trained_model

    def plot_training_result(self):
        trained_model = self.load_trained_model()
        trained_model.eval()
        fig, ax = plt.subplots(1, 3, figsize = (25, 5), facecolor = 'w')

        # Loss plot
        ax[0].plot([i.detach().cpu() for i in self.train_loss_list], label = 'Training loss')
        ax[0].plot([i.detach().cpu() for i in self.val_loss_list], label = 'Validation loss')
        ax[0].set_xlabel('Epoch', fontsize = 15)
        ax[0].set_ylabel('Loss', fontsize = 15)
        ax[0].legend(fontsize = 15)
        ax[0].set_title('Training and Validation Loss', fontsize = 20)

        #  Actual vs Predicted of train dataset
        predicted_list = []
        actual_list = []
        with torch.no_grad():
            for X, y in self.trainloader:
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                predicted, _ = trained_model(X)
                predicted_list += [i.detach().cpu() for i in predicted]
                actual_list += [i[0].detach().cpu() for i in y]

        predicted_list = [(i*self.dataset.scaler.scale_[-1])+self.dataset.scaler.mean_[-1] for i in predicted_list]
        actual_list = [(i*self.dataset.scaler.scale_[-1])+self.dataset.scaler.mean_[-1] for i in actual_list]

        corr_coef = np.corrcoef(actual_list, predicted_list)[0][1]
        ax[1].scatter(actual_list, predicted_list, s = 10)
        ax[1].plot([min(actual_list), max(actual_list)], [min(actual_list), max(actual_list)], color = 'black')
        ax[1].text(x = min(actual_list), y = max(predicted_list)*0.95, s = r"$\rho = $" + f'{corr_coef:.4f}', fontsize = 15)
        ax[1].set_xlabel('Actual', fontsize = 15)
        ax[1].set_ylabel('Predicted', fontsize = 15)
        ax[1].set_title('Actual vs Predicted of Training Set', fontsize = 20)

        # Actual vs Predicted of test dataset
        predicted_list = []
        actual_list = []
        with torch.no_grad():
            for X, y in self.testloader:
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                predicted, _ = trained_model(X)
                predicted_list += [i.detach().cpu() for i in predicted]
                actual_list += [i[0].detach().cpu() for i in y]

        predicted_list = [(i*self.dataset.scaler.scale_[-1])+self.dataset.scaler.mean_[-1] for i in predicted_list]
        actual_list = [(i*self.dataset.scaler.scale_[-1])+self.dataset.scaler.mean_[-1] for i in actual_list]

        corr_coef = np.corrcoef(actual_list, predicted_list)[0][1]
        ax[2].scatter(actual_list, predicted_list, s = 10)
        ax[2].plot([min(actual_list), max(actual_list)], [min(actual_list), max(actual_list)], color = 'black')
        ax[2].text(x = min(actual_list), y = max(predicted_list)*0.95, s = r"$\rho = $" + f'{corr_coef:.4f}', fontsize = 15)
        ax[2].set_xlabel('Actual', fontsize = 15)
        ax[2].set_ylabel('Predicted', fontsize = 15)
        ax[2].set_title('Actual vs Predicted of Test Set', fontsize = 20)
        return fig

    def train(self):
        self.train_loss_list = []
        self.val_loss_list = []
        print("Start of training loop")

        for epoch in range(1, self.config.num_epochs+1):

            # Training step
            self.model.train()
            for X, y in self.trainloader:
                X = X.to(self.config.device)
                y = y.to(self.config.device)

                predicted_y, per_feature_outputs = self.model(X)
                loss = self.criterion(predicted_y, y, per_feature_outputs, self.model)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.train_loss_list.append(loss)

            # Validation step
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in self.valloader:
                    X = X.to(self.config.device)
                    y = y.to(self.config.device)

                    predicted_y, per_feature_outputs = self.model(X)
                    val_loss += self.criterion(predicted_y, y, per_feature_outputs, self.model)

                val_loss = val_loss/len(self.valloader)
                self.val_loss_list.append(val_loss)

            # Save model
            if epoch == 1:
                torch.save(self.model.to('cpu').state_dict(), self.config.output_dir + '/model.pth')
                best_val_loss = val_loss
                self.model.to(self.config.device)
            else:
                if best_val_loss > val_loss:
                    torch.save(self.model.to('cpu').state_dict(), self.config.output_dir + '/model.pth')
                    best_val_loss = val_loss
                    self.model.to(self.config.device)

            # Log
            if (epoch%100 == 0) or (epoch == 1):
                print('Epoch ' + str(epoch) + '/' + str(self.config.num_epochs))

        # Save standard scaler
        with open(self.config.output_dir + '/StandardScaler.pkl', 'wb') as f:
            pickle.dump(self.dataset.scaler, f)

        # Save model information
        info_dict = {'feature_names':self.dataset.feature_names,
                     'target_name':self.dataset.target_name,
                     'intercept':self.dataset.intercept,
                     'num_in_features':self.model.num_in_features,
                     'num_first_layer_unit':self.model.num_first_layer_unit}
        with open(self.config.output_dir + '/model_info.json', 'w') as f:
            json.dump(info_dict, f, indent = 4)

        # Log
        print('The trained model was saved in "' + self.config.output_dir + '"')

        # Plot training result
        self.fig = self.plot_training_result()










