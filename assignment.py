from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

def get_model():
    model = SequentialModel(
        [
           # Add in your layers here as elements of the list!
           # e.g. Dense(10, 10),
           Dense(784, 128, initializer='xavier'),
           ReLU(),
           Dense(128, 10, initializer='xavier'),
           Softmax()
        ]
    )
    return model

def get_optimizer():
    # choose an optimizer, initialize it and return it!
    return Adam(learning_rate=0.001)

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    return CategoricalCrossEntropy()

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    return CategoricalAccuracy()

if __name__ == '__main__':

    ### Use this area to test your implementation!

    # 1. Create a SequentialModel using get_model
    model = get_model()
    
    # 2. Compile the model with optimizer, loss function, and accuracy metric
    optimizer = get_optimizer()
    loss_fn = get_loss_fn()
    acc_fn = get_acc_fn()
    model.compile(optimizer, loss_fn, acc_fn)

    # 3. Load and preprocess the data
    train_inputs, train_labels, test_inputs, test_labels = load_and_preprocess_data()
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(train_labels)
    train_labels = one_hot_encoder.forward(train_labels)

    # 4. Train the model
    model.fit(train_inputs, train_labels, epochs=10, batch_size=32)

    # 5. Evaluate the model
    metrics = model.evaluate(test_inputs, test_labels, batch_size=32)
    print(f"Test matrics: {metrics}")
    
