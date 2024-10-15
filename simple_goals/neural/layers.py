import tensorflow as tf
import numpy as np
import utils
from abc import ABC, abstractmethod

ELMAN = "elman"
GRU = "gru"
LSTM = "lstm"

def winner_take_all(a):
    a = a.numpy()
    b = np.zeros_like(a)
    b[range(len(a)), a.argmax(1)] = 1
    return b

def rounding(a):
    return np.rint(a.numpy())

def make_layer(layer_type, size_input, size_hidden, nonlinearity, initialization=None):
    if layer_type == ELMAN:
        return ElmanLayer(size_input, size_hidden, initial_context=None, nonlinearity=nonlinearity, initialization=initialization)
    elif layer_type == GRU:
        return GRULayer(size_input, size_hidden)
    elif layer_type == LSTM:
        return LSTMLayer(size_input, size_hidden)
    else:
        raise NotImplementedError("unknown layer type")


class Layer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def feedforward(self, x):
        return

    @property
    @abstractmethod
    def parameters(self):
        return


class BasicLayer(Layer):
    def __init__(self, w = None, input_size=None, output_size=None, bias=True, nonlinearity=None):
        super().__init__()
        if w is None:
            if input_size is None or output_size is None:
                raise ValueError("either weights or dimensions must be specified")
            if nonlinearity == tf.nn.relu:
                w = utils.initialize([input_size, output_size], mode=utils.HE)
            elif nonlinearity in [tf.nn.tanh, tf.sigmoid]:
                w = utils.initialize([input_size, output_size], mode=utils.XAVIER)
            elif nonlinearity is None:
                w = utils.initialize([input_size, output_size], mode=utils.XAVIER)
            else:
                raise NotImplementedError("No default initialization for this nonlinearity")
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

    def feedforward(self, x, noise=0.):
        # TODO: remove this after recovering data
        pre_activation = utils.dense_linear(x, self.w, self.b)
        # adding noise before non-linearity.
        pre_activation += np.random.normal(loc=0.0, scale=noise, size=np.shape(pre_activation))
        return self.nonlinearity(pre_activation)

    @property
    def parameters(self):
        return [self.w, self.b]

    # Best defined here because it's incorrect to do this within ADAM
    def l1_reg(self):
        pass
    def l2_reg(self):
        pass

def group_normalization(x, group_size):
    """
    @param x: matrix/vector of size [1, n]
    @param group_size: size of each normalization group in units
    @return: group-normalized vector
    """
    # cut x into groups of size group size
    m, n = x.shape
    x = tf.reshape(x, [-1, group_size])  # This assumes x is divisible by the group size
    mean, var = tf.nn.moments(x, [1], keepdims=True)
    x = (x-mean) / tf.sqrt(var+0.0001)
    x = tf.reshape(x, [m, n])
    return x

#TODO: Add a softmax layer class. This is more complicated because of implementation tricks for backprop
#Current implementation is just using a dense_linear.


class RecurrentLayer(Layer):
    @abstractmethod
    def reset(self, state=None):
        return


# Recurrent layers 1: Elman
class ElmanLayer(RecurrentLayer):
    def __init__(self, size_input, size_output, initial_context=None, nonlinearity=tf.nn.relu, initialization=utils.HE):
        super().__init__()
        self.layer = BasicLayer(self._make_weights(size_input, size_output, initialization), nonlinearity=nonlinearity)
        if initial_context is None:
            self.h = np.zeros((1, size_output), dtype = np.float32)
        else:
            self.h = initial_context

    @staticmethod
    def _make_weights(size_input, size_output, initialization_mode=utils.HE):
        dimensions = [size_input + size_output, size_output]
        return utils.initialize(dimensions, initialization_mode)

    def feedforward(self, x, noise=0.):
        self.h = self.layer.feedforward(tf.concat([x, self.h], axis=1), noise)
        return self.h

    def reset(self, state=None):
        if state is None:
            self.h *= 0.
        else:
            self.h = state

    @property
    def parameters(self):
        return self.layer.parameters


# Recurrent layers 2: GRU
class GRULayer(RecurrentLayer):
    # TODO: identity matrix weight initialization for recurrent units.
    # TODO: ReLU instead of tanh, plus appropriate HE initialization.
    # Gru can be interpreted as consisting of 3 layers R (reset), Z (update), and H (history)
    # Layer size is equal to input size + number of units. So e.g. if we want 10 GRU units with 20 inputs,
    # then the weight matrices should be 30x10.
    def __init__(self, input_size, output_size, initial_context=None):
        super().__init__()
        self.resetgate_layer = BasicLayer(input_size=output_size, output_size=output_size, nonlinearity=tf.sigmoid)
        self.updategate_layer = BasicLayer(input_size=output_size, output_size=output_size, nonlinearity=tf.sigmoid)
        self.tanh_layer = BasicLayer(input_size=input_size+output_size, output_size=output_size, nonlinearity=tf.tanh)
        # Activation
        if initial_context is None:
            self.h = np.zeros((1, output_size), dtype=np.float32)
        else:
            self.h = initial_context

    def feedforward(self, x):
        #reset gate output
        reset = self.resetgate_layer.feedforward(self.h)
        #tanh output
        hreset = tf.math.multiply(reset, self.h)  # elemwise
        tanh_out = self.tanh_layer.feedforward(tf.concat([x, hreset], axis=1))
        # update gate output
        update_values = self.updategate_layer.feedforward(self.h)
        h = tf.math.multiply(update_values, tanh_out) + tf.multiply(1-update_values, self.h)
        self.h = h
        return h

    def reset(self, state=None):
        if state is None:
            self.h *= 0.
        else:
            self.h = state

    @property
    def parameters(self):
        return self.resetgate_layer.parameters + self.updategate_layer.parameters + self.tanh_layer.parameters


class LSTMLayer(RecurrentLayer):
    # TODO: identity matrix weight initialization for recurrent units (check whether that makes sense)
    # Gru can be interpreted as consisting of 3 layers R (reset), Z (update), and H (history)
    # Layer size is equal to input size + number of units. So e.g. if we want 10 GRU units with 20 inputs,
    # then the weight matrices should be 30x10.
    def __init__(self, input_size, output_size, initial_context=None):
        super().__init__()
        self.forgetcellgate_layer = BasicLayer(input_size=input_size+output_size, output_size=output_size, nonlinearity=tf.sigmoid)
        self.updatecellgate_layer = BasicLayer(input_size=input_size+output_size, output_size=output_size, nonlinearity=tf.sigmoid)
        self.updatecell_layer = BasicLayer(input_size=input_size+output_size, output_size=output_size, nonlinearity=tf.tanh)
        self.output_layer = BasicLayer(input_size=input_size+output_size, output_size=output_size, nonlinearity=tf.sigmoid)
        self.output_size = output_size

        if initial_context is None:
            self.h = np.zeros((1, output_size), dtype=np.float32)
            self.c = np.zeros((1, output_size), dtype=np.float32)
        else:
            self.h = np.reshape(initial_context[0], (1, -1))
            self.c = np.reshape(initial_context[1], (1, -1))

    def feedforward(self, x):
        # concatenate x and h
        combined_input = tf.concat([x, self.h], axis=1)

        # compute and update the cell
        c_forget_gated = tf.math.multiply(self.c, self.forgetcellgate_layer.feedforward(combined_input))
        gate_cell_updates = self.updatecellgate_layer.feedforward(combined_input)
        cell_updates = self.updatecell_layer.feedforward(combined_input)
        c = c_forget_gated + tf.math.multiply(gate_cell_updates, cell_updates)

        # Compute and update the hidden state / output
        output_updates = self.output_layer.feedforward(combined_input)
        output_updates_gated = tf.math.multiply(tf.tanh(c), output_updates)
        self.h = output_updates_gated
        self.c = c
        return output_updates_gated

    def reset(self, state=None):
        if state is None:
            self.h = np.zeros((1, self.output_size), dtype=np.float32)
            self.c = np.zeros((1, self.output_size), dtype=np.float32)
        else:
            self.h = np.reshape(state[0], (1, -1))
            self.c = np.reshape(state[1], (1, -1))

    @property
    def parameters(self):
        return self.forgetcellgate_layer.parameters + self.updatecellgate_layer.parameters +\
               self.updatecell_layer.parameters + self.output_layer.parameters


# A predictive coding layer as described in arXiv:1605.08104
# Note, the feedforward input consists of the
class PredLayerLotteEtAl2018(RecurrentLayer):
    def __init__(self, size_input, size_output, initial_context=None, nonlinearity=tf.nn.relu):
        super().__init__()
        self.representation_layer = LSTMLayer(size_input, size_output, initial_context)
        self.predictive_layer = BasicLayer(input_size=size_input, output_size=size_output, nonlinearity=nonlinearity)
        self.input_layer = BasicLayer(input_size=size_input, output_size=size_output, nonlinearity=nonlinearity)
        self.last_error_loss = None

    @property
    def representation(self):
        return self.representation_layer.h

    def feedforward(self, x):
        bottom_up = x[0]
        top_down = x[1]
        prediction = self.predictive_layer.feedforward(bottom_up)
        target = self.input_layer.feedforward(bottom_up)
        # use the LSTM's previous value as a predictor, and output only the resulting error.
        error = target - prediction
        self.last_error_loss = tf.math.square(error)

        # upade the LSTM
        self.representation_layer.feedforward(tf.concat([error, top_down], axis=1))

        # output the error
        return error

    # There's two kinds of weights, because they're not trained the same way.
    def predictive_parameters(self):
        return self.representation_layer.parameters + self.predictive_layer.parameters

    def optimizing_parameters(self):
        return self.input_layer.parameters

    def reset(self, state=None):
        self.representation_layer.reset(state)

    @property
    def parameters(self):
        return self.representation_layer.parameters + self.predictive_layer.parameters + self.input_layer.parameters


# Alternative to arXiv:1605.08104 where we predict the input, and only afterwards we process it.
# This is essentially the same (when stacked)
class PredLayerPredFirst(RecurrentLayer):
    def __init__(self, size_input, size_output, initial_context=None, nonlinearity=tf.nn.relu):
        super().__init__()
        self.representation_layer = LSTMLayer(size_input, size_output, initial_context)
        self.predictive_layer = BasicLayer(input_size=size_input, output_size=size_output, nonlinearity=nonlinearity)
        self.output_layer = BasicLayer(input_size=size_input, output_size=size_output, nonlinearity=nonlinearity)
        self.last_error_loss = None

    @property
    def representation(self):
        return self.representation_layer.h

    def feedforward(self, x):
        bottom_up = x[0]
        top_down = x[1]
        prediction = self.predictive_layer.feedforward(bottom_up)
        # use the LSTM's previous value as a predictor, and output only the resulting error.
        error = x - prediction
        self.last_error_loss = tf.math.square(error)

        # upade the LSTM
        self.representation_layer.feedforward(tf.concat([error, top_down], axis=1))

        # output the processed error
        return self.output_layer.feedforward(error)

    # There's two kinds of weights, because they're not trained the same way.
    def predictive_parameters(self):
        return self.representation_layer.parameters + self.predictive_layer.parameters

    def optimizing_parameters(self):
        return self.output_layer.parameters

    def reset(self, state=None):
        self.representation_layer.reset(state)

    @property
    def parameters(self):
        return self.representation_layer.parameters + self.predictive_layer.parameters + self.output_layer.parameters

class ConvLayer(BasicLayer):
    def __init__(self, observation, num_filters, filter_size):
        pass

class MaxPoolLayer(BasicLayer):
    def __init__(self):
        pass


# Reinforcement predlayer.
# This has the following structure:  (Plus some stop gradients)
#      top               up
#       |                ^
#       |              (relu)     - output layer: relu  (input = concat lstm out + prediction error out)
#       |    -->-------/ ^
#       v   /-->(sig)----|-\      - control gate: sigmoid
#      LSTM ------<-----(-) |     - representation layer: lstm
#       | \___(relu)_>__/| /      - predictive layer: relu
#       |                |/
#       v                ^
#     down             bottom
ALL = "all"
PREDICT_ONLY = "predict_only"
REWARD_ONLY = "reward_only"
CONTROL_ONLY = "control_only"

#Utility class to pass layer sizes
class RCPLayerSizes():
    def __init__(self, input_bottomup, output_bottomup, input_topdown, output_topdown):
        self.input_bottomup = input_bottomup
        self.output_bottomup = output_bottomup
        self.input_topdown = input_topdown
        self.output_topdown = output_topdown

class RewardControlPredLayer(RecurrentLayer):
    def __init__(self, size, initial_context=None, mode="all", predictive_nonlinearity=tf.nn.relu):
        super().__init__()
        self.size = size
        self.representation_layer = LSTMLayer(size.input_topdown + size.input_bottomup, size.output_topdown, initial_context)
        self.predictive_layer = BasicLayer(input_size=size.output_topdown, output_size=size.input_bottomup, nonlinearity=predictive_nonlinearity) # Sigmoid is logical if we're predicting a one hot output.
        self.output_layer = LSTMLayer(input_size=size.input_bottomup * 2 + size.output_topdown, output_size=size.output_bottomup)#, nonlinearity=output_nonlinearity)
        self.control_gate_layer = BasicLayer(input_size=size.output_topdown, output_size=1, nonlinearity=tf.sigmoid)
        # Bias the control gate layer to the left by one so that control starts at approx. 0 at the beginning of learning.
        self.control_gate_layer.b.assign_sub(np.ones([1, 1], dtype=np.float32))
        self.prediction_loss = None
        self.control_loss = None
        self.prediction_error = None
        self.mode = mode

    @property
    def representation(self):
        return self.representation_layer.h

    def feedforward(self, x):
        if self.mode == PREDICT_ONLY: x = tf.stop_gradient(x)
        # compute the prediction and prediction error
        prediction = self.predictive_layer.feedforward(self.representation_layer.h)
        if self.mode == REWARD_ONLY: prediction = tf.stop_gradient(prediction)
        self.prediction_error = tf.math.subtract(x, prediction)

        self.prediction_loss = tf.reduce_sum(tf.math.square(self.prediction_error))

        #compute the control level and apply it to the bottom-up and top-down information
        control_level = self.control_gate_layer.feedforward(self.representation_layer.h)  # This should start at zero.

        # Compute the layer's outputs, up
        representation = self.representation_layer.h   # Remove the input from the top-down component
        if self.mode == PREDICT_ONLY: representation = tf.stop_gradient(representation)
        gated_bottom_up = tf.concat([x * (1.-control_level), self.prediction_error*control_level, representation*control_level], axis=1)
        self.control_loss = tf.reduce_sum(1.-control_level)
        bottom_up_output = self.output_layer.feedforward(gated_bottom_up)

        # output the bottom-up  (top down can be accessed via the representation sub-layer)
        return bottom_up_output

    # This ensures that the LSTM layer gets updated with information from every future layer.
    def feedbackward(self, x):
        prediction_error = tf.stop_gradient(self.prediction_error) if self.mode == PREDICT_ONLY else self.prediction_error
        if x is not None:
            return self.representation_layer.feedforward(tf.concat([x, prediction_error], axis=1))
        else:
            return self.representation_layer.feedforward(prediction_error)

    # There's two kinds of weights, because they're not trained the same way.
    def predictive_parameters(self):
        return self.representation_layer.parameters + self.predictive_layer.parameters

    def optimizing_parameters(self):
        return self.output_layer.parameters + self.control_gate_layer.parameters

    def reset(self, state=None):
        self.representation_layer.reset(state)
        self.output_layer.reset(state)
        self.prediction = None

    @property
    def parameters(self):
        return self.representation_layer.parameters + self.predictive_layer.parameters +\
               self.output_layer.parameters + self.control_gate_layer.parameters


