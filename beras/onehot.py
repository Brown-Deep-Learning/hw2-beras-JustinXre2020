import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        self.classes = np.unique(data)
        self.n_classes = self.classes.shape[0]
        # create mappings from class to one_hot
        self.class_to_indexes = {classs: index for index, classs in enumerate(self.classes)}


    def forward(self, data):
        if not hasattr(self, "class_to_indexes"):
            self.fit(data)
        label_indexes = np.array([self.class_to_indexes[label] for label in data])
        return np.eye(self.n_classes)[label_indexes]

    def inverse(self, data):
        indexes = np.argmax(data, axis=1)
        return self.classes[indexes]
