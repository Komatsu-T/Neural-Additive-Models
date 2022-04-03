import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nam.model.InteractionNN import InteractionNN

class InteractionTrainer():
    def  __init__(self, config, trained_nam_model, interaction_model, dataset):
        self.config = config
        self.trained_nam_model = trained_nam_model
        self.interaction_model = interaction_model
        self.dataset = dataset
        self.trainloader = self.dataset.get_trainloader()
        self.valloader = self.dataset.get_valloader()
        self.testloader = self.dataset.get_testloader()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.interaction_model.parameters(), lr = self.config.lr, weight_decay = 0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma = self.config.gamma,step_size = self.config.step_size)

    def load_trained_model(self):
        # Load model info
        with open(self.config.output_dir + '/interaction_model_info.json', 'r') as f:
            model_info = json.load(f)

        # Load model
        trained_interaction_model = InteractionNN(self.trained_nam_model.num_in_features,
                                                  model_info['hidden_layers'])
        trained_interaction_model.load_state_dict(torch.load(self.config.output_dir + '/interaction_model.pth', map_location = self.config.device))
        trained_interaction_model = trained_interaction_model.to(self.config.device)
        return trained_interaction_model

    def plot_training_result(self):
        trained_interaction_model = self.load_trained_model()
        trained_interaction_model.eval()

        #fig, ax = plt.subplots(2, 3, figsize=(25, 12), facecolor='w')
        #ax = ax.ravel()
        fig = plt.figure(figsize=(25, 12), constrained_layout = True)
        spec = fig.add_gridspec(2, 3)
        ax = [ fig.add_subplot(spec[0,0]),  fig.add_subplot(spec[0,1]),  fig.add_subplot(spec[0,2]),  fig.add_subplot(spec[1,:2])]

        # Loss plot
        ax[0].plot([i.detach().cpu() for i in self.train_loss_list], label='Training loss')
        ax[0].plot([i.detach().cpu() for i in self.val_loss_list], label='Validation loss')
        ax[0].set_xlabel('Epoch', fontsize=15)
        ax[0].set_ylabel('Loss', fontsize=15)
        ax[0].legend(fontsize=15)
        ax[0].set_title('Training and Validation Loss', fontsize=20)

        #  Actual vs Predicted of train dataset
        predicted_list = []
        actual_list = []
        with torch.no_grad():
            for X, y in self.trainloader:
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                predicted_y_nam, _ = self.trained_nam_model(X)
                predicted_y_interaction = trained_interaction_model(X)
                predicted = predicted_y_nam + predicted_y_interaction.view(1, -1)[0]
                predicted_list += [i.detach().cpu() for i in predicted]
                actual_list += [i[0].detach().cpu() for i in y]

        predicted_list = [(i * self.dataset.scaler.scale_[-1]) + self.dataset.scaler.mean_[-1] for i in predicted_list]
        actual_list = [(i * self.dataset.scaler.scale_[-1]) + self.dataset.scaler.mean_[-1] for i in actual_list]

        corr_coef = np.corrcoef(actual_list, predicted_list)[0][1]
        ax[1].scatter(actual_list, predicted_list, s=10)
        ax[1].plot([min(actual_list), max(actual_list)], [min(actual_list), max(actual_list)], color='black')
        ax[1].text(x=min(actual_list), y=max(predicted_list) * 0.95, s=r"$\rho = $" + f'{corr_coef:.4f}', fontsize=15)
        ax[1].set_xlabel('Actual', fontsize=15)
        ax[1].set_ylabel('Predicted', fontsize=15)
        ax[1].set_title('Actual vs Predicted of Training Set', fontsize=20)

        # Actual vs Predicted of test dataset
        predicted_list = []
        actual_list = []
        with torch.no_grad():
            for X, y in self.testloader:
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                predicted_y_nam, _ = self.trained_nam_model(X)
                predicted_y_interaction = trained_interaction_model(X)
                predicted = predicted_y_nam + predicted_y_interaction.view(1, -1)[0]
                predicted_list += [i.detach().cpu() for i in predicted]
                actual_list += [i[0].detach().cpu() for i in y]

        predicted_list = [(i * self.dataset.scaler.scale_[-1]) + self.dataset.scaler.mean_[-1] for i in predicted_list]
        actual_list = [(i * self.dataset.scaler.scale_[-1]) + self.dataset.scaler.mean_[-1] for i in actual_list]

        corr_coef = np.corrcoef(actual_list, predicted_list)[0][1]
        ax[2].scatter(actual_list, predicted_list, s=10)
        ax[2].plot([min(actual_list), max(actual_list)], [min(actual_list), max(actual_list)], color='black')
        ax[2].text(x=min(actual_list), y=max(predicted_list) * 0.95, s=r"$\rho = $" + f'{corr_coef:.4f}', fontsize=15)
        ax[2].set_xlabel('Actual', fontsize=15)
        ax[2].set_ylabel('Predicted', fontsize=15)
        ax[2].set_title('Actual vs Predicted of Test Set', fontsize=20)

        # Accuracy of NAM with or without interaction
        predicted_list_with_interaction = []
        predicted_list_without_interaction = []
        actual_list = []
        with torch.no_grad():
            for X, y in self.testloader:
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                predicted_y_nam, _ = self.trained_nam_model(X)
                predicted_y_interaction = trained_interaction_model(X)
                predicted = predicted_y_nam + predicted_y_interaction.view(1, -1)[0]
                predicted_list_with_interaction += [i.detach().cpu() for i in predicted]
                predicted_list_without_interaction += [i.detach().cpu() for i in predicted_y_nam]
                actual_list += [i[0].detach().cpu() for i in y]

        predicted_list_with_interaction = [(i * self.dataset.scaler.scale_[-1]) + self.dataset.scaler.mean_[-1] for i in predicted_list_with_interaction]
        predicted_list_without_interaction = [(i * self.dataset.scaler.scale_[-1]) + self.dataset.scaler.mean_[-1] for i in predicted_list_without_interaction]
        actual_list = [(i * self.dataset.scaler.scale_[-1]) + self.dataset.scaler.mean_[-1] for i in actual_list]

        corr_coef_with_interaction = np.corrcoef(actual_list, predicted_list_with_interaction)[0][1]
        corr_coef_without_interaction = np.corrcoef(actual_list, predicted_list_without_interaction)[0][1]

        ax[3].barh(['NAM without interaction', 'NAM with interaction'], [corr_coef_without_interaction, corr_coef_with_interaction])
        ax[3].set_xlabel('Correlation coefficient of actual and predicted values of test set', fontsize = 15)
        ax[3].set_yticks([0, 1])
        ax[3].set_yticklabels(["NAM without \n interaction", "NAM with \n interaction"], fontsize = 15)
        ax[3].set_title('Accuracy of model with or without interaction', fontsize=20)
        ax[3].set_xlim(0, 1)

        return fig

    def train(self):
        self.train_loss_list = []
        self.val_loss_list = []
        print("Start of training loop")

        # Stop update of trained nam parameters
        self.trained_nam_model.eval()
        for param in self.trained_nam_model.parameters():
            param.requires_grad = False

        for epoch in range(1, self.config.num_epochs + 1):

            # Training step
            self.interaction_model.train()
            for X, y in self.trainloader:
                X = X.to(self.config.device)
                y = y.to(self.config.device)

                predicted_y_nam, _ = self.trained_nam_model(X)
                predicted_y_interaction = self.interaction_model(X)
                predicted_y = predicted_y_nam + predicted_y_interaction.view(1, -1)[0]
                loss = self.criterion(predicted_y, y.view(1, -1)[0])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.train_loss_list.append(loss)

            # Validation step
            self.interaction_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in self.valloader:
                    X = X.to(self.config.device)
                    y = y.to(self.config.device)

                    predicted_y_nam, _ = self.trained_nam_model(X)
                    predicted_y_interaction = self.interaction_model(X)
                    predicted_y = predicted_y_nam + predicted_y_interaction.view(1, -1)[0]
                    val_loss += self.criterion(predicted_y, y.view(1, -1)[0])

                val_loss = val_loss / len(self.valloader)
                self.val_loss_list.append(val_loss)

            # Save model
            if epoch == 1:
                torch.save(self.interaction_model.to('cpu').state_dict(), self.config.output_dir + '/interaction_model.pth')
                best_val_loss = val_loss
                self.interaction_model.to(self.config.device)
            else:
                if best_val_loss > val_loss:
                    torch.save(self.interaction_model.to('cpu').state_dict(), self.config.output_dir + '/interaction_model.pth')
                    best_val_loss = val_loss
                    self.interaction_model.to(self.config.device)

            # Log
            if (epoch % 100 == 0) or (epoch == 1):
                print('Epoch ' + str(epoch) + '/' + str(self.config.num_epochs))

        # Save model information
        info_dict = {'num_in_features':self.interaction_model.num_in_features,
                     'hidden_layers':self.interaction_model.hidden_layers}
        with open(self.config.output_dir + '/interaction_model_info.json', 'w') as f:
            json.dump(info_dict, f, indent = 4)

        # Log
        print('The trained model was saved in "' + self.config.output_dir + '"')

        # Plot training result
        self.fig = self.plot_training_result()
