import tensorflow as tf
import numpy as np
import utils
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def feedforward(self, x):
        return

class BasicLayer(Layer):
    def __init__(self, w, bias=True, nonlinearity=None):
        super().__init__()
        self.w = tf.Variable(w, dtype=tf.float32)
        if bias:
            self.b = tf.Variable(np.zeros([1, w.shape[1]]), dtype=tf.float32)
        else:
            self.b = None

        if nonlinearity is not None:
            self.nonlinearity = nonlinearity
        else:
            self.nonlinearity = BasicLayer._identity

    @staticmethod
    def _identity(x):
        return x

    def feedforward(self, x):
        return self.nonlinearity(utils.dense_linear(x, self.w, self.b))


# TODO: Add a softmax layer class.


# Recurrent layers 1: Elman
class ElmanLayer(Layer):
    def __init__(self, size_input, size_output, initial_context=None, nonlinearity=tf.nn.relu, initialization=utils.HE):
        super().__init__()
        self.layer = BasicLayer(self._make_weights(size_input, size_output, initialization), nonlinearity)
        if initial_context is None:
            self.h = np.zeros((1, size_output), dtype = np.float32)
        else:
            self.h = initial_context

        self.all_weights = [self.layer.w, self.layer.b]

    def _make_weights(self, size_input, size_output, initialization_mode=utils.HE):
        dimensions = [size_input + size_output, size_output]
        return utils.initialize(dimensions, initialization_mode)

    def feedforward(self, x):
        self.h = self.layer.feedforward(tf.concat([x, self.h], axis=1))
        return self.h


# Recurrent layers 2: GRU
class GRULayer(Layer):
    # TODO: identity matrix weight initialization for recurrent units.
    # TODO: ReLU instead of tanh, plus appropriate HE initialization.
    # Gru can be interpreted as consisting of 3 layers R (reset), Z (update), and H (history)
    # Layer size is equal to input size + number of units. So e.g. if we want 10 GRU units with 20 inputs,
    # then the weight matrices should be 30x10.
    def __init__(self, size_input, size_output, initial_context=None):
        super().__init__()
        weights = self._make_weights(size_input, size_output)
        self.resetgate_layer = BasicLayer(weights[0], nonlinearity=tf.sigmoid)
        self.updategate_layer = BasicLayer(weights[1], nonlinearity=tf.sigmoid)
        self.tanh_layer = BasicLayer(weights[2], nonlinearity=tf.tanh)
        # Activation
        if initial_context is None:
            self.h = np.zeros((1, size_output), dtype=np.float32)
        else:
            self.h = initial_context

        self.all_weights = [self.resetgate_layer.w, self.resetgate_layer.b,
                            self.updategate_layer.w, self.updategate_layer.b,
                            self.tanh_layer.w, self.tanh_layer.b]

    def _make_weights(self, size_input, size_output):
        dimensions = [size_input + size_output, size_output]
        # Note, everytime it's different weights.
        w_reset = utils.xavier_initialization(dimensions)
        w_update = utils.xavier_initialization(dimensions)
        w_tanh = utils.xavier_initialization(dimensions)
        return w_reset, w_update, w_tanh

    def feedforward(self, x):
        #reset gate output
        reset = self.resetgate_layer.feedforward(tf.concat([x, self.h], axis=1))
        #tanh output
        hreset = tf.math.multiply(reset, self.h)  # elemwise
        tanh_out = self.tanh_layer.feedforward(tf.concat([x, hreset], axis=1))
        # update gate output
        update = self.updategate_layer.feedforward(tf.concat([x, self.h], axis=1))
        self.h = tf.math.multiply(update, tanh_out) + tf.multiply(1-update, self.h)
        return self.h
