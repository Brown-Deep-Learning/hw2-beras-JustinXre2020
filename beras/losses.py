import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return np.mean((y_pred - y_true)**2)

    def get_input_gradients(self) -> list[Tensor]:
        # partial derivative of MSE = (y_pred - 2 * y_true) / n
        y_pred, y_true = self.inputs
        grad_y_pred = 2 * (y_pred - y_true) / y_pred.size
        grad_y_true = np.zeros_like(y_true)
        return [grad_y_pred, grad_y_true]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Store inputs for gradient computation
        # self.input_dict = {'y_pred': y_pred}
        
        # Check if y_true is not one-hot encoded (i.e., it's a 1D array of class indices)
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # Convert class indices to one-hot
            num_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros_like(y_pred)
            
            # Handle both flat arrays and column vectors
            if y_true.ndim == 2:
                indices = y_true.flatten()
            else:
                indices = y_true
                
            for i in range(len(indices)):
                y_true_one_hot[i, indices[i]] = 1.0
                
            y_true = y_true_one_hot
        
        # Store the one-hot version for gradient computation
        # self.input_dict['y_true'] = y_true
        
        # Compute cross-entropy loss
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        return [-(self.input_dict['y_true'] / self.input_dict['y_pred']) / self.input_dict['y_pred'].shape[0], np.zeros_like(self.input_dict['y_true'])]
