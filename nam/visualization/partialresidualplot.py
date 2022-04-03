import torch
import numpy as np
import matplotlib.pyplot as plt

def partial_residual_plot(model, dataset):

    # Plot
    model = model.to('cpu')
    model.eval()
    with torch.no_grad():

        # Residuals from the full model
        data = dataset.scaler.transform(dataset.data[list(dataset.feature_names) + [dataset.target_name]])
        target = dataset.data[dataset.target_name].values
        input = torch.from_numpy(data[:,:-1]).float().to('cpu')
        output, output_per_feature = model(input)
        output = (output.detach().cpu().numpy()*dataset.scaler.scale_[-1]) + dataset.scaler.mean_[-1]
        full_residuals = target - output

        # Partial residuals
        partial_residuals = {}
        for i, feature in enumerate(dataset.feature_names):
            presid = (output_per_feature[:, i].detach().cpu().numpy()*dataset.scaler.scale_[-1]) + dataset.scaler.mean_[-1]
            presid += full_residuals
            partial_residuals[feature] = [dataset.data[feature].values, presid]

        # Regression curve
        num_point = max([len(np.unique(data[:, i])) for i in range(len(dataset.feature_names))])*10
        reg_curve_input = []
        for i in range(len(dataset.feature_names)):
            min_value = data[:,i].min()
            max_value = data[:,i].max()
            min_max = np.linspace(min_value, max_value, num_point)
            reg_curve_input.append(min_max)
        reg_curve_input = np.stack(reg_curve_input, 1)
        reg_curve_input = torch.from_numpy(reg_curve_input).float().to('cpu')
        _, output_per_feature = model(reg_curve_input)

        regression_curve = {}
        for i, feature in enumerate(dataset.feature_names):
            curve_x = (reg_curve_input[:,i].detach().cpu()*dataset.scaler.scale_[i]) + dataset.scaler.mean_[i]
            curve_y = (output_per_feature[:,i].detach().cpu()*dataset.scaler.scale_[-1]) + dataset.scaler.mean_[-1]
            regression_curve[feature] = [curve_x, curve_y]

        # Plot
        num_row = int((len(dataset.feature_names)-1)/3)+1
        target_mean = dataset.data[dataset.target_name].mean()
        fig, ax = plt.subplots(num_row, 3, figsize = (25, 5*num_row), facecolor = 'w')
        ax = ax.ravel()
        for i, feature in enumerate(dataset.feature_names):
            ax[i].scatter(partial_residuals[feature][0], partial_residuals[feature][1], s = 1)
            ax[i].axhline(target_mean, color='black', label='E[y]')
            ax[i].plot(regression_curve[feature][0], regression_curve[feature][1], color = 'orange', linewidth = 3, label = 'Regression curve')
            ax[i].legend(fontsize = 12)
            ax[i].set_xlabel(feature, fontsize = 15)
            ax[i].set_ylabel('Feature contribution', fontsize = 15)
        return fig