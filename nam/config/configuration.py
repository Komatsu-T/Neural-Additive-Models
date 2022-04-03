import torch
import os
import pickle

class Configuration:
    def __init__(
        self,
        output_dir = './trained_model',
        batch_size = 1024,
        #dropout = 0.0,
        #feature_dropout = 0.0,
        hidden_layer_size = [64, 32],
        first_layer_size = [512],
        first_layer = 'ReLU',
        output_regularization = 10**-5,
        l2_regularization = 10**-6,
        lr = 0.01,
        num_epochs = 500,
        gamma = 0.995,
        step_size = 1,
        test_size = 0.2,
        validation_size = 0.1,
        ):

        self.batch_size = batch_size # Batch size
        #self.dropout = dropout # Dropout probability of each feature net.
        #self.feature_dropout = feature_dropout # Dropout probability of individual feature networks.
        self.hidden_layer_size = hidden_layer_size # The number of node and layer ([64, 32]: 64-node 2nd hidden layer, 32-node 3rd hidden layer).
        self.first_layer_size = first_layer_size # The number of node of 1st hidden layer (All feature nets have the same number of node).
        self.first_layer = first_layer # Type of 1st hidden layer. Set 'exu', if you use ExU layer.
        self.output_regularization = output_regularization # Regularization for output penalty
        self.l2_regularization = l2_regularization # Regularization for weight decay
        self.lr = lr # Learning rate
        self.num_epochs = num_epochs # The number of training epoch
        self.gamma = gamma # Learning rate decay
        self.step_size = step_size # Step size of Learning rate decay
        self.test_size = test_size # Proportion of test data
        self.validation_size = validation_size # Proportion of validation data. If ensemble = True, validation_size is ignored.

        # Device
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.device = device

        # Output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir)
        print('---------------------------------------------------------------------------------------')
        print('Directory "' + output_dir + '" was created for saving trained model and information.')
        print('---------------------------------------------------------------------------------------')

        # Save config
        self.save_config()

    def save_config(self):
        with open(self.output_dir + '/config.bin', 'wb') as f:
            pickle.dump(self, f)

    def show(self):
        print('')
        print('output_dir: ', self.output_dir)
        print('device: ', self.device)
        print('first_layer: ', self.first_layer)
        print('first_layer_size: ', self.first_layer_size)
        print('hidden_layer_size: ', self.hidden_layer_size)
        print('batch_size: ', self.batch_size)
        #print('dropout: ', self.dropout)
        #print('feature_dropout: ', self.feature_dropout)
        print('output_regularization: ', self.output_regularization)
        print('l2_regularization: ', self.l2_regularization)
        print('lr: ', self.lr)
        print('gamma: ', self.gamma)
        print('step_size: ', self.step_size)
        print('num_epochs: ', self.num_epochs)
        print('test_size: ', self.test_size)
        print('validation_size: ', self.validation_size)