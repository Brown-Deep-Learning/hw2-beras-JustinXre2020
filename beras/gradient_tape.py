from collections import defaultdict, deque

from beras.core import Diffable, Tensor
import numpy as np

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.

        queue = deque([target])                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}
        visited = set([id(target)])

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful
        while queue:
            current_layer = queue.popleft()
            current_gradients = grads[id(current_layer)]
            # Look up the layer that produced this tensor.
            prev_layers = self.previous_layers.get(id(current_layer))
            if prev_layers is None:
                continue


            # compose the gradients with respect to the layer's inputs
            input_grads = prev_layers.compose_input_gradients(current_gradients)
            len_inputs = min(len(prev_layers.inputs), len(input_grads))
            for i in range(len_inputs):
                input_val = prev_layers.inputs[i]
                grad = input_grads[i]
                grads[id(input_val)] = [grad]
                if id(input_val) in visited:
                    continue
                visited.add(id(input_val))
                queue.append(input_val)
            
            # Compose gradients with respect to the layer’s weights
            weight_grads = prev_layers.compose_weight_gradients(current_gradients)
            len_weights = min(len(prev_layers.weights), len(weight_grads))
            for i in range(len_weights):
                input_weight = prev_layers.weights[i]
                grad = weight_grads[i]
                grads[id(input_weight)] = [grad]

        # For each source, sum the gradient contributions.
        return [grads[id(src)][0] if grads[id(src)] else np.zeros_like(src) for src in sources]
        # for src in sources:
        #     # Sum gradients; if no gradient was collected, return an array of zeros matching src.
        #     if id(src) in grads:
        #         # Sum the list of gradient arrays.
        #         total_grad = sum(grads[id(src)], start=np.zeros_like(src))
        #     else:
        #         print(src)
        #         total_grad = np.zeros_like(src)
        #     result.append(total_grad)
        # return result
