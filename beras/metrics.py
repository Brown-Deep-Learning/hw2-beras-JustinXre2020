import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='

        # Get the predicted class indices (argmax of probabilities)
        pred_cls = np.argmax(probs, axis=1)
        
        # Convert one-hot encoded labels to class indices
        true_cls = np.argmax(labels, axis=-1)
        
        # Compare predictions with true labels and compute accuracy
        return np.mean(pred_cls == true_cls)
