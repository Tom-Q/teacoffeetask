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
# TODO: Add a recurrent layer abstract class



class RecurrentLayer(Layer):
    @abstractmethod
    def reset(self, state=None):
        return


# Recurrent layers 1: Elman
class ElmanLayer(RecurrentLayer):
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

    def reset(self, state=None):
        if state is None:
            self.h *= 0.
        else:
            self.h = state

# Recurrent layers 2: GRU
class GRULayer(RecurrentLayer):
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

    def reset(self, state=None):
        if state is None:
            self.h *= 0.
        else:
            self.h = state

class LSTMLayer(RecurrentLayer):
    # TODO: identity matrix weight initialization for recurrent units (check whether that makes sense)
    # Gru can be interpreted as consisting of 3 layers R (reset), Z (update), and H (history)
    # Layer size is equal to input size + number of units. So e.g. if we want 10 GRU units with 20 inputs,
    # then the weight matrices should be 30x10.
    def __init__(self, size_input, size_output, initial_context=None):
        super().__init__()
        weights = self._make_weights(size_input, size_output)
        self.forgetcellgate_layer = BasicLayer(weights[0], nonlinearity=tf.sigmoid)
        self.updatecellgate_layer = BasicLayer(weights[1], nonlinearity=tf.sigmoid)
        self.updatecell_layer = BasicLayer(weights[2], nonlinearity=tf.tanh)
        self.output_layer = BasicLayer(weights[3], nonlinearity=tf.sigmoid)

        if initial_context is None:
            self.h = np.zeros((1, size_output), dtype=np.float32)
            self.c = np.zeros((1, size_output), dtype=np.float32)
        else:
            self.h = initial_context[0]
            self.c = initial_context[1]

        self.all_weights = [self.forgetcellgate_layer.w, self.forgetcellgate_layer.b,
                            self.updatecellgate_layer.w, self.updatecellgate_layer.b,
                            self.updatecell_layer.w, self.updatecell_layer.b,
                            self.output_layer.w, self.output_layer.b]

    def _make_weights(self, size_input, size_output):
        dimensions = [size_input + size_output, size_output]
        # Note, everytime it's different weights.
        w_forgetcellgate = utils.xavier_initialization(dimensions)
        w_updatecellgate = utils.xavier_initialization(dimensions)
        w_updatecell = utils.xavier_initialization(dimensions)
        w_output = utils.xavier_initialization(dimensions)
        return w_forgetcellgate, w_updatecellgate, w_updatecell, w_output

    def feedforward(self, x):
        # concatenate x and h
        combined_input = tf.concat([x, self.h], axis=1)

        # compute and update the cell
        c_forget_gated = tf.math.multiply(self.c, self.forgetcellgate_layer.feedforward(combined_input))
        gate_cell_updates = self.updatecellgate_layer.feedforward(combined_input)
        cell_updates = self.updatecell_layer.feedforward(combined_input)
        self.c = c_forget_gated + tf.math.multiply(gate_cell_updates, cell_updates)

        # Compute and update the hidden state / output
        output_updates = self.output_layer.feedforward(combined_input)
        output_updates_gated = tf.math.multiply(tf.tanh(self.c), output_updates)
        self.h = output_updates_gated
        return self.h

    def reset(self, state=None):
        if state is None:
            self.h *= 0.
            self.c *= 0.
        else:
            raise NotImplementedError("figure this out later")