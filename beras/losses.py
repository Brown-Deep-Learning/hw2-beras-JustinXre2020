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
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        return [-(self.input_dict['y_true'] / self.input_dict['y_pred']) / self.input_dict['y_pred'].shape[0], np.zeros_like(self.input_dict['y_true'])]
