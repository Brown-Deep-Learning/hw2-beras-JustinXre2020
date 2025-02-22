import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='
        pred_cls = np.argmax(probs, axis=1)
        true_cls = np.argmax(labels, axis=1)
        return np.mean(pred_cls == true_cls)
