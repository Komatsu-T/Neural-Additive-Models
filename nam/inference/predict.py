import torch

def predict(nam_model, scaler, df, device, interaction_model = None):

    # Standardization of dataset and make tensor
    df[scaler.feature_names_in_[-1]] = 0
    scaled_data = scaler.transform(df)[:, :-1]
    scaled_data = torch.from_numpy(scaled_data).float().to(device)

    # Prediction
    nam_model.eval()
    with torch.no_grad():
        output, _ = nam_model(scaled_data)

    if interaction_model != None:
        interaction_model.eval()
        with torch.no_grad():
            pred_values = interaction_model(scaled_data)
            output = output + pred_values.view(1, -1)[0]

    # Inverse Standardization and make numpy array
    output = (output * scaler.scale_[-1]) + scaler.mean_[-1]
    output = output.numpy()
    return output
