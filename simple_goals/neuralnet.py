import tensorflow as tf
import numpy as np
import teacoffeeenv as tce
import utils
import timeit
from abc import ABC, abstractmethod
import copy

class Layer(object):
    def __init__(self, w, bias=True):
        self.w = tf.Variable(w, dtype=tf.float32)
        if bias:
            self.b = tf.Variable(np.zeros([1, w.shape[1]]), dtype=tf.float32)
        else:
            self.b = None

class Optimizer(ABC):
    def __init__(self, weights_list):
        self.weights_list = weights_list

    @abstractmethod
    def update_weights(self, gradients, learning_rate):
        pass


class MomentumSGDOptimizer(Optimizer):
    def __init__(self, weights_list, beta=0.9):
        super().__init__(weights_list)
        self.beta = beta
        self.momentum = [tf.zeros_like(weights) for weights in self.weights_list]
        self.t = 0

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights_list)):
            self.t+=1
            self.momentum[i] = self.beta * self.momentum[i] + (1 - self.beta) * gradients[i]
            unbiased_momentum = self.momentum[i] / (1 - self.beta ** self.t)
            self.weights_list[i].assign_sub(learning_rate * unbiased_momentum)


class SGDOptimizer(Optimizer):
    def __init__(self, weights_list):
        super().__init__(weights_list)

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights_list)):
            if gradients[i] is not None:
                self.weights_list[i].assign_sub(gradients[i] * learning_rate)
            else:
                Warning("None gradient")

class AdamOptimizer(Optimizer):
    def __init__(self, weights_list):
        super().__init__(weights_list)
        self.epsilon = 10 ** (-8)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.firstmoment = [tf.zeros_like(weights) for weights in self.weights_list]
        self.secondmoment = [tf.zeros_like(weights) for weights in self.weights_list]
        self.t = 0

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights_list)):
            self.t+=1
            self.firstmoment[i] = self.beta1 * self.firstmoment[i] + (1 - self.beta1) * gradients[i]
            self.secondmoment[i] = self.beta2 * self.secondmoment[i] + (1 - self.beta2) * tf.square(gradients[i])
            unbiased_firstmoment = self.firstmoment[i] / (1 - self.beta1 ** self.t)  # Probably useless for me but that's part of ADAM.
            unbiased_secondmoment = self.secondmoment[i] / (1 - self.beta2 ** self.t)
            self.weights_list[i].assign_sub(learning_rate * unbiased_firstmoment / (tf.sqrt(unbiased_secondmoment) + self.epsilon))

class RMSPropOptimizer(Optimizer):
    def __init__(self, weights_list):
        super().__init__(weights_list)
        self.beta = 0.9
        self.epsilon = 10 ** (-8)
        self.meansquares = [tf.zeros_like(weights) for weights in self.weights_list]

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights_list)):
            self.meansquares[i] = self.beta * self.meansquares[i] + (1 - self.beta) * tf.square(gradients[i])
            self.weights_list[i].assign_sub(learning_rate * gradients[i] / (tf.sqrt(self.meansquares[i]) + self.epsilon))

# Optimization techniques
SGD = "sgd"
RMSPROP = "rmsprop"
ADAM = "adam"

# Initialization types
SEMINORMAL = "seminormal"
NORMAL = "normal"  # for sigmoid, instead of a normal centered on 0.5, I use half a normal distribution, centered on 0.
UNIFORM = "uniform"
XAVIER = "xavier"
HE = "he" # for relu
ZEROS = "zeros"  # for context initialization

# Nonlinearity
RELU="relu"
SIGMOID="sigmoid"
TANH="tanh"


# convenience function for stopping parameters since I'll be using this in many places.
# Checks if it's time to stop training a network
class ParamsStopping(object):
    def __init__(self, max_iterations, min_iterations = None, check_frequency=None, stop_condition=None,
                 **stop_condition_kwargs):
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.check_frequency = check_frequency
        self.stop_condition = stop_condition
        self.stop_condition_kwargs = stop_condition_kwargs

    def is_time_to_stop(self, model, iteration):
        if iteration >= self.max_iterations:
            return True
        elif self.min_iterations is not None and iteration < self.min_iterations:
            return False
        elif self.check_frequency is None:
            return False
        elif iteration % self.check_frequency == 0:
            if self.stop_condition is None:
                raise ValueError("if check_frequency is not None then a stop condition must be provided")
            # Typically this will check whether the model has achieved perfect accuracy
            return self.stop_condition(model, **self.stop_condition_kwargs)
        else:
            return False


# Convenience classes for encapsulating neural net parameters. It seems this only adds complexity to the NN classes but
# it is useful e.g. when I want to train multiple neural nets while changing only one or two parameters.
# Also useful for storing a set of default parameters. Also neat for creating multiple nets from the same set of params
# Etc.
class ParamsNeuralNet(object):
    def __init__(self, size_observation, size_action, initialization, algorithm, nonlinearity, learning_rate, L1_reg, L2_reg):
        self.size_observation = size_observation
        self.size_action = size_action
        self.initialization = initialization
        self.algorithm = algorithm
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

class ParamsGoalNet(ParamsNeuralNet):
    def __init__(self, size_observation, size_action, size_hidden, initialization, algorithm, nonlinearity, learning_rate,
                 L1_reg=0, L2_reg=0, last_action_inputs=False, size_goal1=0, size_goal2=0):
        super().__init__(size_observation, size_action, initialization,
                         algorithm, nonlinearity, learning_rate, L1_reg, L2_reg)
        self.size_hidden = size_hidden
        self.size_goal1 = size_goal1
        self.size_goal2 = size_goal2
        self.last_action_inputs = last_action_inputs

# Abstract class for neural networks. Contains just the basic stuff.
class NeuralNet(ABC):
    def __init__(self, size_observation, size_action, initialization=UNIFORM, nonlinearity=SIGMOID,
                 algorithm=SGD, learning_rate=0.1,
                 L1_regularization=0., L2_regularization=0., params=None):
        if params is None:
            self.algorithm = algorithm
            self.size_observation = size_observation
            self.size_action = size_action
            self.learning_rate = learning_rate
            self.nonlinearity = nonlinearity
            self.initialization = initialization
            self.L1_regularization = L1_regularization
            self.L2_regularization = L2_regularization
        else:
            self.algorithm = params.algorithm
            self.size_observation = params.size_observation
            self.size_action = params.size_action
            self.learning_rate = params.learning_rate
            self.nonlinearity = params.nonlinearity
            self.initialization = params.initialization
            self.L1_regularization = params.L1_reg
            self.L2_regularization = params.L2_reg

    def get_params(self):
        return ParamsNeuralNet(size_observation=self.size_observation, size_action=self.size_action,
                               initialization=self.initialization, algorithm=self.algorithm,
                               nonlinearity=self.nonlinearity, learning_rate=self.learning_rate,
                               L1_reg=self.L1_regularization, L2_reg=self.L2_regularization)
    @abstractmethod
    def feedforward(self, observation):
        """
        Feedforward of the network. Should typically be run within the scope of a gradient tape so
        that train can be called afterwards.
        @return:
        """

    @abstractmethod
    def train(self, tape, parameters):
        """
        @param tape: the gradient tape
        @param parameters: typically, targets used to determine the loss
        @return:
        """

    @abstractmethod
    def new_episode(self):
        """
        Reset the network in view of moving on to a new episode
        @return:
        """

    @staticmethod
    def dense_linear(x, layer):
        """
        :param x: input
        :param layer: Layer object with weights matrix [+ bias]
        :return: densely connected layer with no activation function
        """
        preactivation = tf.matmul(x, layer.w)
        if layer.b is not None:
            preactivation += layer.b
        return preactivation

    @staticmethod
    def dense_softmax(x, layer):
        """
        :param x: input
        :param layer: Layer object with weights matrix [+ bias]
        :return: densely connected layer with softmax output
        WARNING: DONT USE THIS METHOD FOR A LAST LAYER. TRAINING ASSUMES YOURE WORKING WITH LOGITS INSTEAD.
        """
        return tf.nn.softmax(NeuralNet.dense_linear(x, layer))

    @staticmethod
    def dense_sigmoid(x, layer):
        """
        :param x: input
        :param layer: Layer object with weights matrix [+ bias]
        :return: densely connected layer with sigmoid output
        """
        return tf.nn.sigmoid(NeuralNet.dense_linear(x, layer))

    @staticmethod
    def dense_relu(x, layer):
        return tf.nn.relu(NeuralNet.dense_linear(x, layer))


    @staticmethod
    def dense_tanh(x, layer):
        return tf.nn.tanh(NeuralNet.dense_linear(x, layer))

    @staticmethod
    def winner_take_all(a):
        a = a.numpy()
        b = np.zeros_like(a)
        b[range(len(a)), a.argmax(1)] = 1
        return b


class ElmanGoalNet(NeuralNet):
    def __init__(self, size_hidden=15, algorithm=SGD, learning_rate=0.1,
                 size_observation=len(tce.TeaCoffeeData.observations_list),
                 size_goal1=len(tce.TeaCoffeeData.goals1_list), size_goal2=len(tce.TeaCoffeeData.goals2_list),
                 size_action=len(tce.TeaCoffeeData.actions_list), initialization=NORMAL,
                 L1_reg = 0., L2_reg = 0.,
                 last_action_inputs=False,
                 nonlinearity=SIGMOID, params=None,
                 gradient=False):
        super().__init__(size_observation, size_action, initialization=initialization,
                         algorithm=algorithm, learning_rate=learning_rate, params=params)
        if params is None:
            self.algorithm = algorithm
            self.size_hidden = size_hidden
            self.size_goal1 = size_goal1
            self.size_goal2 = size_goal2
            self.nonlinearity = nonlinearity
            self.initialization = initialization
            # Whether to feed the last action as another input
            self.last_action_inputs = last_action_inputs
            self.learning_rate = learning_rate
            self.L1_regularization = L1_reg
            self.L2_regularization = L2_reg
        else:
            self.size_action = params.size_action
            self.size_observation = params.size_observation
            self.algorithm = params.algorithm
            self.size_hidden = params.size_hidden
            self.size_goal1 = params.size_goal1
            self.size_goal2 = params.size_goal2
            self.nonlinearity = params.nonlinearity
            self.initialization = params.initialization
            # Whether to feed the last action as another input
            self.last_action_inputs = params.last_action_inputs
            self.learning_rate = params.learning_rate
            self.L1_regularization = params.L1_reg
            self.L2_regularization = params.L2_reg

        self.goal1 = self.goal2 = self.action = self.context = self.action_softmax =\
            self.goal1_softmax = self.goal2_softmax = None

        if initialization == NORMAL:
            self.hidden_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden + self.size_observation +
                                                                     self.size_action + self.size_goal1 + self.size_goal2,
                                                                     self.size_hidden]))

            self.goal1_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden, self.size_goal1]))
            self.goal2_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden, self.size_goal2]))
            self.action_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden, self.size_action]))

        elif initialization == UNIFORM:
            self.hidden_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden + self.size_observation +
                                                                     self.size_action + self.size_goal1 + self.size_goal2,
                                                                     self.size_hidden]))

            self.goal1_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden, self.size_goal1]))
            self.goal2_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden, self.size_goal2]))
            self.action_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden, self.size_action]))
        elif initialization == XAVIER or HE:
            init_const = 1 if initialization == XAVIER else 2
            # higher and lowre bound hidden layer
            hbh = init_const/np.sqrt(self.size_hidden + self.size_observation + self.size_action + self.size_goal1 + self.size_goal2)
            lbh = -hbh
            self.hidden_layer = Layer(np.random.uniform(lbh, hbh,
                                                        size=[self.size_hidden + self.size_observation +
                                                                     self.size_action + self.size_goal1 + self.size_goal2,
                                                                     self.size_hidden]))
            hbo = init_const/np.sqrt(self.size_hidden)
            lbo = -hbo
            self.goal1_layer = Layer(np.random.uniform(lbo, hbo, size=[self.size_hidden, self.size_goal1]))
            self.goal2_layer = Layer(np.random.uniform(lbo, hbo, size=[self.size_hidden, self.size_goal2]))
            self.action_layer = Layer(np.random.uniform(lbo, hbo, size=[self.size_hidden, self.size_action]))
        else:
            raise ValueError("Initialization should be normal or uniform")

        self.all_weights = [self.hidden_layer.w, self.hidden_layer.b,
                            self.action_layer.w, self.action_layer.b,
                            self.goal1_layer.w, self.goal1_layer.b,
                            self.goal2_layer.w, self.goal2_layer.b]


        self.h_action_softmax = []
        self.h_goal1_softmax = []
        self.h_goal2_softmax = []
        self.h_action_wta = []
        self.h_goal1_wta = []
        self.h_goal2_wta = []
        self.h_context = []
        self.history = [self.h_action_softmax, self.h_goal1_softmax, self.h_goal2_softmax,
                        self.h_action_wta, self.h_goal1_wta, self.h_goal2_wta, self.h_context]
        if algorithm == SGD:
            self.optimizer = SGDOptimizer(self.all_weights)
        elif algorithm == RMSPROP:
            self.optimizer = RMSPropOptimizer(self.all_weights)
        elif algorithm == ADAM:
            self.optimizer = AdamOptimizer(self.all_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")

    def get_params(self):
        return ParamsGoalNet(size_observation=self.size_observation, size_action=self.size_action,
                             size_hidden=self.size_hidden, initialization=self.initialization,
                             algorithm=self.algorithm, learning_rate=self.learning_rate,
                             L1_reg=self.L1_regularization, L2_reg=self.L2_regularization,
                             size_goal1=self.size_goal1, size_goal2=self.size_goal2,
                             last_action_inputs=self.last_action_inputs)

    def feedforward(self, observation):#, gain_multiplier=1., gain_multiplier_from=0, gain_multiplier_to=None):
        network_input = tf.concat([self.context, observation], 1)
        if not self.last_action_inputs: # cancel out actions
            self.action *= 0.
        for inputs in [self.action, self.goal2, self.goal1]:
            if inputs is not None:
                # Not sure if thatś necessary? In theory the winner take all has no gradient anyway.
                inputs = tf.stop_gradient(inputs)
                network_input = tf.concat([network_input, inputs], 1)

        # Backwards compatibility
        if not hasattr(self, 'nonlinearity'):
            self.nonlinearity = SIGMOID

        if self.nonlinearity == SIGMOID:
            hidden_activation = self.dense_sigmoid(network_input, self.hidden_layer)
        elif self.nonlinearity == RELU:
            hidden_activation = self.dense_relu(network_input, self.hidden_layer)
        elif self.nonlinearity == TANH:
            hidden_activation = self.dense_tanh(network_input, self.hidden_layer)
        else:
            raise(NotImplementedError("Nonlinearity " + self.nonlinearity + " not implemented"))

        #if gain_multiplier != 1:
        #    if gain_multiplier_to is None:
        #        gain_multiplier_to = self.size_hidden
        #    hidden_activation = hidden_activation.numpy()
        #    hidden_activation[0, gain_multiplier_from:gain_multiplier_to] *= gain_multiplier
        #    hidden_activation[0, gain_multiplier_to:] *= 1./gain_multiplier

        # Three separate softmaxes for the action and the goal
        self.action_softmax = self.dense_linear(hidden_activation, self.action_layer)
        self.action = self.winner_take_all(self.action_softmax)
        if self.size_goal1 > 0:
            self.goal1_softmax = self.dense_linear(hidden_activation, self.goal1_layer)
            self.goal1 = self.winner_take_all(self.goal1_softmax)
        if self.size_goal2 > 0:
            self.goal2_softmax = self.dense_linear(hidden_activation, self.goal2_layer)
            self.goal2 = self.winner_take_all(self.goal2_softmax)

        # Set up the next context.
        self.context = hidden_activation

        # The actual chosen action and goal
        self.save_history()

    def new_episode(self, initial_context=ZEROS):
        # Reinitialize the entire state of the network (anything that could affect the next episode.)
        self.clear_history()
        self.action = np.zeros((1, self.size_action), dtype=np.float32)
        if initial_context == ZEROS:
            self.context = np.zeros((1, self.size_hidden), dtype=np.float32)
        elif initial_context == UNIFORM:
            self.context = np.float32(np.random.uniform(0.01, 0.99, (1, self.size_hidden)))
        else:
            raise ValueError("Initial context value not implemented")
        if self.size_goal1 > 0:
            self.goal1 = np.zeros((1, self.size_goal1), dtype=np.float32)
        if self.size_goal2 > 0:
            self.goal2 = np.zeros((1, self.size_goal2), dtype=np.float32)

    def clear_history(self):
        for data in self.history:
            data.clear()

    def save_history(self):
        self.h_action_softmax.append(self.action_softmax)
        self.h_goal1_softmax.append(self.goal1_softmax)
        self.h_goal2_softmax.append(self.goal2_softmax)
        self.h_action_wta.append(copy.deepcopy(self.action))
        self.h_goal1_wta.append(self.goal1)
        self.h_goal2_wta.append(self.goal2)
        self.h_context.append(self.context)

    def train(self, tape, targets, extra_loss=0.):
        loss = 0
        for i, target in enumerate(targets):
            if target.action_one_hot is not None:
                loss += tf.nn.softmax_cross_entropy_with_logits(target.action_one_hot, self.h_action_softmax[i])
            if target.goal1_one_hot is not None:
                loss += tf.nn.softmax_cross_entropy_with_logits(target.goal1_one_hot, self.h_goal1_softmax[i])
            if target.goal2_one_hot is not None:
                loss += tf.nn.softmax_cross_entropy_with_logits(target.goal2_one_hot, self.h_goal2_softmax[i])
        loss += self.L2_regularization * sum([tf.reduce_sum(weights**2) for weights in self.all_weights])
        loss += extra_loss
        gradients = tape.gradient(loss, self.all_weights)
        self.optimizer.update_weights(gradients, self.learning_rate)
        self.clear_history()
        return loss

    def train_obsolete(self, targets_action, targets_goal1, targets_goal2, tape, extra_loss=0.):
        # Compute error + backprop.
        loss = 0.
        for i in range(len(targets_action)):
            loss += tf.nn.softmax_cross_entropy_with_logits(targets_action[i], self.h_action_softmax[i])
            if targets_goal1 is not None:
                loss += tf.nn.softmax_cross_entropy_with_logits(targets_goal1[i], self.h_goal1_softmax[i])
            if targets_goal2 is not None:
                loss += tf.nn.softmax_cross_entropy_with_logits(targets_goal2[i], self.h_goal2_softmax[i])
        loss += self.L2_regularization * sum([tf.reduce_sum(weights**2) for weights in self.all_weights])
        loss += extra_loss
        gradients = tape.gradient(loss, self.all_weights)
        self.optimizer.update_weights(gradients, self.learning_rate)
        self.clear_history()
        return loss, gradients

    def train_MSE(self, targets_action, targets_goal1, targets_goal2, tape):
        # Compute error + backprop.
        loss = 0.
        for i in range(len(targets_action)):
            loss += tf.reduce_sum((targets_action[i] - tf.nn.sigmoid(self.h_action_softmax[i]))**2)
            if targets_goal1 is not None:
                loss += tf.reduce_sum((targets_goal1[i] - tf.nn.sigmoid(self.h_goal1_softmax[i])) ** 2)
            if targets_goal2 is not None:
                loss += tf.reduce_sum((targets_goal2[i] - tf.nn.sigmoid(self.h_goal2_softmax[i])) ** 2)
        # I'm going to assume that "weight persistence 0.999999" means L1 regularization. Multiplying by
        # the learning rate too.
        loss += self.L2_regularization * sum([tf.reduce_sum(weights**2) for weights in self.all_weights])
        gradients = tape.gradient(loss, self.all_weights)
        self.optimizer.update_weights(gradients, self.learning_rate)
        self.clear_history()
        return loss, gradients


"""
Currently broken
class PredictiveNet(NeuralNet):
    def __init__(self, size_hidden=15, algorithm=SGD, learning_rate=0.1,
                 size_observation=len(tce.TeaCoffeeData.observations_list),
                 size_action=len(tce.TeaCoffeeData.actions_list)):
        @param layer_sizes: list of integers, the size of each hidden layer
        @param algorithm: nn.SGD, ADAM, or RMSPROP
        @param learning_rate:
        @param size_observation:
        @param size_action:
        super().__init__(size_observation=size_observation, size_action=size_action,
                         algorithm=algorithm, learning_rate=learning_rate)
        self.size_hidden = size_hidden

        self.action = self.context = self.action_linear = self.prediction_linear = None

        self.hidden_layers = []
        self.predictive_layers = []
        self.all_weights = []
        # wtf is that shit
        for i, layer_size in enumerate(layer_sizes):
            if i == 0:
                input_size = layer_size + self.size_observation + self.size_action
                predictive_output_size = self.size_observation
            else:
                input_size = layer_size + layer_sizes[i - 1]
                predictive_output_size = layer_sizes[i - 1]
                
            self.hidden_layers.append(Layer(np.random.normal(0., .1, size=[input_size, layer_size])))
            self.predictive_layers.append(Layer(np.random.normal(0., .1, size=[layer_size, predictive_output_size])))
            self.all_weights.append(self.hidden_layers[-1].w)
            self.all_weights.append(self.hidden_layers[-1].b)
            self.all_weights.append(self.predictive_layers[-1].w)
            self.all_weights.append(self.predictive_layers[-1].b)

        self.action_layer = Layer(np.random.normal(0., .1, size=[self.layer_sizes[-1], self.size_action]))

        self.all_weights.append(self.action_layer.w)
        self.all_weights.append(self.action_layer.b)

        self.learning_rate = learning_rate
        self.L2_regularization = 0.0001
        self.h_action_linear = []
        self.h_action_wta = []
        self.h_prediction_linear = []
        self.h_prediction_wta = []
        self.history = [self.h_action_linear, self.h_action_wta, self.h_prediction_linear, self.h_prediction_wta]
        if algorithm == SGD:
            self.optimizer = SGDOptimizer(self.all_weights)
        elif algorithm == RMSPROP:
            self.optimizer = RMSPropOptimizer(self.all_weights)
        elif algorithm == ADAM:
            self.optimizer = AdamOptimizer(self.all_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")

    def feedforward(self, observation):
        if type == 'sigmoid':
            prediction_error = observation - self.prediction_probability
        elif type == "wta":
            prediction_error = observation - self.prediction_wta

        network_input = tf.concat([self.context, prediction_error], 1)
        for inputs in [self.action]:
            if inputs is not None:
                # Not sure if that's necessary? In theory the winner take all has no gradient anyway.
                inputs = tf.stop_gradient(inputs)
                network_input = tf.concat([network_input, inputs], 1)
        hidden_activation = self.dense_sigmoid(network_input, self.hidden_layer)

        self.action_linear = self.dense_linear(hidden_activation, self.action_layer)
        # self.action_probability = tf.nn.softmax(self.action_linear)
        self.action = self.winner_take_all(self.action_linear)

        self.prediction_linear = self.dense_linear(hidden_activation, self.predictive_layer)
        self.prediction_probability = tf.stop_gradient(tf.nn.softmax(self.prediction_linear))
        self.prediction = self.prediction_probability  # tf.stop_gradient(self.prediction_probability)
        self.prediction_wta = self.winner_take_all(self.prediction_linear)
        # The actual chosen action and goal
        self.save_history()

        # Set up the next context.
        self.context = hidden_activation

    def clear_history(self):
        for data in self.history:
            data.clear()

    def save_history(self):
        self.h_action_linear.append(self.action_linear)
        self.h_action_wta.append(self.action)
        self.h_prediction_linear.append(self.prediction_linear)
        self.h_prediction_wta.append(self.prediction_wta)

    def train(self, tape, targets):
        targets_action = targets[0]
        targets_prediction = targets[1]
        # Compute error + backprop.

        loss = 0.
        # for i in range(len(targets_action)):
        # loss += tf.reduce_sum((targets_action[i] - tf.nn.sigmoid(self.h_action_linear[i])) ** 2)  # mse
        loss_action = tf.nn.softmax_cross_entropy_with_logits(targets_action, self.h_action_linear)  # cross entropy

        # for i in range(len(targets_prediction)):
        # loss += tf.reduce_sum((targets_prediction[i] - tf.nn.sigmoid(self.h_prediction_linear[i])) ** 2)
        loss_prediction = tf.nn.softmax_cross_entropy_with_logits(targets_prediction,
                                                                  self.h_prediction_linear)  # cross entropy

        # print(loss_action, loss_prediction)
        loss = tf.reduce_sum(loss_action + loss_prediction)
        # I'm going to assume that "weight persistence 0.999999" means L1 regularization. Multiplying by
        # the learning rate too.
        loss += self.L2_regularization * sum([tf.reduce_sum(weights ** 2) for weights in self.all_weights])
        gradients = tape.gradient(loss, self.all_weights)
        self.optimizer.update_weights(gradients, self.learning_rate)
        self.clear_history()
        return loss, gradients
"""

# Deep prednet architecture:
#           action
#             ^
#  pred2 <- hidden2 <-> context2
#    |         ^
#    >>>>>>   (-)
#              ^
#  pred1 <- hidden1 <-> context1
#    |         ^
#    >>>>>>   (-)
#              ^
#          observation
class DeepPredNet(NeuralNet):
    def __init__(self, size_hidden1=15, size_hidden2=15, algorithm=SGD, learning_rate=0.1,
                 size_observation=len(tce.TeaCoffeeData.observations_list),
                 size_action=len(tce.TeaCoffeeData.actions_list)):
        super().__init__(size_observation=size_observation, size_action=size_action,
                         algorithm=algorithm, learning_rate=learning_rate)
        self.algorithm = algorithm
        self.size_observation = size_observation
        self.size_hidden1 = size_hidden1
        self.size_hidden2 = size_hidden2
        self.size_action = size_action

        self.action = self.context = self.action_linear = self.prediction_linear = None

        self.hidden_layer1 = Layer(np.random.normal(0., .1, size=[self.size_hidden1 + self.size_observation +
                                                                    self.size_action,
                                                                    self.size_hidden1]))
        self.predictive_layer1 = Layer(np.random.normal(0., .1, size=[self.size_hidden1, self.size_observation]))

        self.hidden_layer2 = Layer(np.random.normal(0., .1, size=[self.size_hidden1 * 2,
                                                                    self.size_hidden2]))
        self.predictive_layer2 = Layer(np.random.normal(0., .1, size=[self.size_hidden2, self.size_hidden1]))

        self.action_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden2, self.size_action]))

        self.all_weights = [self.hidden_layer1.w, self.hidden_layer1.b,
                            self.hidden_layer2.w, self.hidden_layer2.b,
                            self.predictive_layer1.w, self.predictive_layer1.b,
                            self.predictive_layer2.w, self.predictive_layer2.b,
                            self.action_layer.w, self.action_layer.b]

        self.learning_rate = learning_rate
        self.L2_regularization = 0.0001
        self.h_action_linear = []
        self.h_action_wta = []
        self.h_prediction1 = []
        self.h_prediction2 = []
        self.history = [self.h_action_linear, self.h_action_wta, self.h_prediction1, self.h_prediction2]

        if algorithm == SGD:
            self.optimizer = SGDOptimizer(self.all_weights)
        elif algorithm == RMSPROP:
            self.optimizer = RMSPropOptimizer(self.all_weights)
        elif algorithm == ADAM:
            self.optimizer = AdamOptimizer(self.all_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")

    def feedforward(self, observation):
        prediction1_error = observation - self.prediction1

        # Concatenate all the inputs for the first hidden layer
        input_hidden1 = tf.concat([self.context1, prediction1_error], 1)
        for inputs in [self.action]:
            if inputs is not None:
                # Not sure if that's necessary? In theory the winner take all has no gradient anyway.
                inputs = tf.stop_gradient(inputs)
                input_hidden1 = tf.concat([input_hidden1, inputs], 1)

        hidden_activation1 = self.dense_sigmoid(input_hidden1, self.hidden_layer1)
        self.prediction1 = self.dense_sigmoid(hidden_activation1, self.predictive_layer1)
        #self.prediction1 = tf.stop_gradient(self.prediction1)

        prediction2_error = hidden_activation1 - self.prediction2
        # Concatenate all the inputs for the second hidden layer
        input_hidden2 = tf.concat([self.context2, prediction2_error], 1)

        hidden_activation2 = self.dense_sigmoid(input_hidden2, self.hidden_layer2)

        self.prediction2 = self.dense_sigmoid(hidden_activation2, self.predictive_layer2)
        #self.prediction2 = tf.stop_gradient(self.prediction2)

        self.action_linear = self.dense_linear(hidden_activation2, self.action_layer)
        self.action = self.winner_take_all(self.action_linear)

        # The actual chosen action and goal
        self.save_history()

        # Set up the next context.
        self.context1 = hidden_activation1
        self.context2 = hidden_activation2

    def train(self, tape, targets):
        targets_action = targets
        # Compute error + backprop.
        loss_action = 0
        for i in range(len(targets_action)):
            loss_action += tf.reduce_sum((targets_action[i] - tf.nn.sigmoid(self.h_action_linear[i])) ** 2)  # mse

        loss_prediction1 = 0
        for i in range(len(self.h_prediction1)-1):
            loss_prediction1 += tf.reduce_sum((self.h_prediction1[i] - self.h_prediction1[i+1]) ** 2)

        loss_prediction2 = 0
        for i in range(len(self.h_prediction2)-1):
            loss_prediction2 += tf.reduce_sum((self.h_prediction2[i] - self.h_prediction2[i+1]) ** 2)

        loss = tf.reduce_sum(loss_action + 0.5 * (loss_prediction1 + loss_prediction2))
        loss += self.L2_regularization * sum([tf.reduce_sum(weights**2) for weights in self.all_weights])
        gradients = tape.gradient(loss, self.all_weights)
        self.optimizer.update_weights(gradients, self.learning_rate)
        self.clear_history()
        return loss, gradients

    def clear_history(self):
        for data in self.history:
            data.clear()

    def save_history(self):
        self.h_action_linear.append(self.action_linear)
        self.h_action_wta.append(self.action)
        self.h_prediction1.append(self.prediction1)
        self.h_prediction2.append(self.prediction2)

    def new_episode(self):
        # Use at the beginning of an episode
        self.prediction1 = np.zeros((1, self.size_observation), dtype=np.float32)
        self.prediction2 = np.zeros((1, self.size_hidden1), dtype=np.float32)
        self.context1 = np.zeros((1, self.size_hidden1), dtype=np.float32)
        self.context2 = np.zeros((1, self.size_hidden2), dtype=np.float32)
        self.action = np.zeros((1, self.size_action), dtype=np.float32)
        self.clear_history()


# Deep prednet architecture but the top level is just doing more abstract stuff:
#  pred2 <- hidden2 <-> context2
#    |         ^   \
#    >>>>>>   (-)  v
#              ^   v
# action<--    ^   v
#          \   ^   /
#  pred1 <- hidden1 <-> context1
#    |         ^
#    >>>>>>   (-)
#              ^
#          observation
class DeepControlPredNet(NeuralNet):
    def __init__(self, size_hidden1=15, size_hidden2=15, algorithm=SGD, learning_rate=0.1,
                 size_observation=len(tce.TeaCoffeeData.observations_list),
                 size_action=len(tce.TeaCoffeeData.actions_list)):
        super().__init__(size_observation=size_observation, size_action=size_action,
                         algorithm=algorithm, learning_rate=learning_rate)
        self.algorithm = algorithm
        self.size_observation = size_observation
        self.size_hidden1 = size_hidden1
        self.size_hidden2 = size_hidden2
        self.size_action = size_action

        self.action = self.context = self.action_linear = self.prediction_linear = None

        self.hidden_layer1 = Layer(np.random.normal(0., .1, size=[self.size_hidden1 + self.size_observation +
                                                                    self.size_action + size_hidden2,
                                                                    self.size_hidden1]))
        self.predictive_layer1 = Layer(np.random.normal(0., .1, size=[self.size_hidden1, self.size_observation]))

        self.hidden_layer2 = Layer(np.random.normal(0., .1, size=[self.size_hidden1 * 2,
                                                                    self.size_hidden2]))
        self.predictive_layer2 = Layer(np.random.normal(0., .1, size=[self.size_hidden2, self.size_hidden1]))

        self.action_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden1, self.size_action]))

        self.all_weights =  [self.hidden_layer1.w, self.hidden_layer1.b,
                            self.hidden_layer2.w, self.hidden_layer2.b,
                            self.action_layer.w, self.action_layer.b,
                            self.predictive_layer1.w, self.predictive_layer1.b,
                            self.predictive_layer2.w, self.predictive_layer2.b]

        self.prediction_weights = [self.hidden_layer1.w, self.hidden_layer1.b,
                            self.hidden_layer2.w, self.hidden_layer2.b,
                            self.predictive_layer1.w, self.predictive_layer1.b,
                            self.predictive_layer2.w, self.predictive_layer2.b]

        self.action_weights = [self.hidden_layer1.w, self.hidden_layer1.b,
                            self.hidden_layer2.w, self.hidden_layer2.b,
                            self.action_layer.w, self.action_layer.b]

        self.learning_rate = learning_rate
        self.L2_regularization = 0.0001
        self.h_action_linear = []
        self.h_action_wta = []
        self.h_prediction1 = []
        self.h_prediction2 = []
        self.history = [self.h_action_linear, self.h_action_wta, self.h_prediction1, self.h_prediction2]

        if algorithm == SGD:
            self.optimizer_prediction = SGDOptimizer(self.prediction_weights)
            self.optimizer_action = SGDOptimizer(self.action_weights)
        elif algorithm == RMSPROP:
            self.optimizer_prediction = RMSPropOptimizer(self.prediction_weights)
            self.optimizer_action = RMSPropOptimizer(self.action_weights)
        elif algorithm == ADAM:
            self.optimizer_prediction = AdamOptimizer(self.prediction_weights)
            self.optimizer_action = AdamOptimizer(self.action_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")

    def feedforward(self, observation):
        prediction1_error = observation - self.prediction1

        # Concatenate all the inputs for the first hidden layer
        input_hidden1 = tf.concat([self.context1, prediction1_error, self.context2], 1)
        for inputs in [self.action]:
            if inputs is not None:
                # Not sure if that's necessary? In theory the winner take all has no gradient anyway.
                inputs = tf.stop_gradient(inputs)
                input_hidden1 = tf.concat([input_hidden1, inputs], 1)

        hidden_activation1 = self.dense_sigmoid(input_hidden1, self.hidden_layer1)
        self.prediction1 = self.dense_sigmoid(hidden_activation1, self.predictive_layer1)


        prediction2_error = hidden_activation1 - self.prediction2
        # Concatenate all the inputs for the second hidden layer
        input_hidden2 = tf.concat([self.context2, prediction2_error], 1)

        hidden_activation2 = self.dense_sigmoid(input_hidden2, self.hidden_layer2)

        self.prediction2 = self.dense_sigmoid(hidden_activation2, self.predictive_layer2)
        #self.prediction2 = tf.stop_gradient(self.prediction2)

        self.action_linear = self.dense_linear(hidden_activation1, self.action_layer)
        self.action = self.winner_take_all(self.action_linear)

        # The actual chosen action and goal
        self.save_history()

        # Set up the next context.
        self.context1 = hidden_activation1
        self.context2 = hidden_activation2

    def train(self, tape, targets):
        targets_action = targets
        # Compute error + backprop.
        loss_action = 0
        for i in range(len(targets_action)):
            loss_action += tf.reduce_sum((targets_action[i] - tf.nn.sigmoid(self.h_action_linear[i])) ** 2)  # mse

        loss_prediction = 0
        for i in range(len(self.h_prediction1)-1):
            loss_prediction += tf.reduce_sum((self.h_prediction1[i] - self.h_prediction1[i+1]) ** 2)

        for i in range(len(self.h_prediction2)-1):
            loss_prediction += tf.reduce_sum((self.h_prediction2[i] - self.h_prediction2[i+1]) ** 2)

        loss_prediction *= 0.5
        loss_prediction += self.L2_regularization *\
                           sum([tf.reduce_sum(weights**2) for weights in self.prediction_weights])
        gradients_prediction = tape.gradient(loss_prediction, self.prediction_weights)
        gradients_action = tape.gradient(loss_action, self.action_weights)
        self.optimizer_prediction.update_weights(gradients_prediction, self.learning_rate)
        self.optimizer_action.update_weights(gradients_action, self.learning_rate)
        self.clear_history()
        return loss_prediction, gradients_prediction

    def clear_history(self):
        for data in self.history:
            data.clear()

    def save_history(self):
        self.h_action_linear.append(self.action_linear)
        self.h_action_wta.append(self.action)
        self.h_prediction1.append(self.prediction1)
        self.h_prediction2.append(self.prediction2)

    def new_episode(self):
        # Use at the beginning of an episode
        self.prediction1 = np.zeros((1, self.size_observation), dtype=np.float32)
        self.prediction2 = np.zeros((1, self.size_hidden1), dtype=np.float32)
        self.context1 = np.zeros((1, self.size_hidden1), dtype=np.float32)
        self.context2 = np.zeros((1, self.size_hidden2), dtype=np.float32)
        self.action = np.zeros((1, self.size_action), dtype=np.float32)
        self.clear_history()


# First test: just doing subtraction
class BasicSubtractNet(NeuralNet):
    def __init__(self, size_hidden=15, algorithm=SGD, learning_rate=0.1,
                 size_observation=2,
                 size_goal1=0, size_goal2=0,
                 size_action=1, initialization=NORMAL):
        super().__init__(size_observation, size_action, initialization=initialization,
                         algorithm=algorithm, learning_rate=learning_rate)
        self.size_hidden = size_hidden
        if initialization == NORMAL:
            self.hidden_layer = Layer(np.random.normal(0., .1, size=[self.size_observation,
                                                                     self.size_hidden]))
            self.action_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden, self.size_action]))

        elif initialization == UNIFORM:
            self.hidden_layer = Layer(np.random.uniform(-1, 1., size=[self.size_observation,
                                                                     self.size_hidden]))
            self.action_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden, self.size_action]))
        else:
            raise ValueError("Initialization should be normal or uniform")

        self.all_weights = [self.hidden_layer.w, self.hidden_layer.b,
                            self.action_layer.w, self.action_layer.b]

        self.learning_rate = learning_rate
        self.L2_regularization = 0.0
        self.h_action = []
        self.history = [self.h_action]
        if algorithm == SGD:
            self.optimizer = SGDOptimizer(self.all_weights)
        elif algorithm == RMSPROP:
            self.optimizer = RMSPropOptimizer(self.all_weights)
        elif algorithm == ADAM:
            self.optimizer = AdamOptimizer(self.all_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")


    def feedforward(self, observation):
        network_input = observation
        hidden_activation = self.dense_sigmoid(network_input, self.hidden_layer)
        result = self.dense_linear(hidden_activation, self.action_layer)
        self.h_action.append(result)
        return result

    def train(self, tape, target):
            # Compute error + backprop.
            loss = (target - self.h_action[-1])**2
            gradients = tape.gradient(loss, self.all_weights)
            self.optimizer.update_weights(gradients, self.learning_rate)
            self.clear_history()
            return loss, gradients

    def clear_history(self):
        for data in self.history:
            data.clear()

    def new_episode(self):
        self.clear_history()


class BasicSubtractNetSymbolic(NeuralNet):
    def __init__(self, size_hidden=15, algorithm=SGD, learning_rate=0.1,
                 size_observation=12, # 12 symbols 0-9, +-
                 size_goal1=0, size_goal2=0,
                 size_action=12, # +/-, 1/0, 0-9
                 initialization=NORMAL):
        super().__init__(size_observation, size_action, initialization=initialization,
                         algorithm=algorithm, learning_rate=learning_rate)
        self.size_digit = 10
        self.size_hidden = size_hidden
        if initialization == NORMAL:
            self.hidden_layer = Layer(np.random.normal(0., .1, size=[self.size_observation,
                                                                     self.size_hidden]))
            self.sign_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden, 1]))
            self.tens_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden, 1]))
            self.unit_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden, 10]))

        elif initialization == UNIFORM:
            self.hidden_layer = Layer(np.random.uniform(-1, 1., size=[self.size_observation,
                                                                     self.size_hidden]))
            self.sign_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden, 1]))
            self.tens_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden, 1]))
            self.unit_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden, 10]))
        else:
            raise ValueError("Initialization should be normal or uniform")

        self.all_weights = [self.hidden_layer.w, self.hidden_layer.b,
                            self.sign_layer.w, self.sign_layer.b,
                            self.tens_layer.w, self.tens_layer.b,
                            self.unit_layer.w, self.unit_layer.b]

        self.learning_rate = learning_rate
        self.L2_regularization = 0.0
        self.h_results = []
        self.history = [self.h_action]
        if algorithm == SGD:
            self.optimizer = SGDOptimizer(self.all_weights)
        elif algorithm == RMSPROP:
            self.optimizer = RMSPropOptimizer(self.all_weights)
        elif algorithm == ADAM:
            self.optimizer = AdamOptimizer(self.all_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")

    def feedforward(self, observation):
        network_input = observation
        hidden_activation = self.dense_sigmoid(network_input, self.hidden_layer)
        result_sign = self.dense_sigmoid(hidden_activation, self.sign_layer)
        result_tens = self.dense_sigmoid(hidden_activation, self.tens_layer)
        result_unit = self.dense_linear(hidden_activation, self.unit_layer)
        result = [result_sign, result_tens, result_unit]
        self.h_results.append(result)
        return result

    def train(self, tape, target):
            # Compute error + backprop.
            loss = 0

            loss += tf.nn.softmax_cross_entropy_with_logits(target[2], self.h_results[-1][2])
            loss = (target - self.h_action[-1])**2
            gradients = tape.gradient(loss, self.all_weights)
            self.optimizer.update_weights(gradients, self.learning_rate)
            self.clear_history()
            return loss, gradients

    def clear_history(self):
        for data in self.history:
            data.clear()

    def new_episode(self):
        self.clear_history()

# Second test: subtraction with recurrence - num 1 - operator - num 2
class RecurrentArithmeticNet(NeuralNet):
    def __init__(self, size_hidden=15, algorithm=SGD, learning_rate=0.1,
                 size_observation=2,  # one unit for operator, one unit for numbers.
                 size_goal1=0, size_goal2=0,
                 size_action=1, initialization=NORMAL):
        super().__init__(size_observation, size_action, initialization=initialization,
                         algorithm=algorithm, learning_rate=learning_rate)
        self.context = None
        self.size_hidden = size_hidden
        if initialization == NORMAL:
            self.hidden_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden + self.size_observation,
                                                                     self.size_hidden]))
            self.hidden_layer2 = Layer(np.random.normal(0., .1, size=[self.size_hidden,
                                                                     self.size_hidden]))
            self.action_layer = Layer(np.random.normal(0., .1, size=[self.size_hidden, self.size_action]))

        elif initialization == UNIFORM:
            self.hidden_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden + self.size_observation,
                                                                     self.size_hidden]))

            self.hidden_layer2 = Layer(np.random.uniform(-1., .1, size=[self.size_hidden,
                                                                     self.size_hidden]))
            self.action_layer = Layer(np.random.uniform(-1, 1., size=[self.size_hidden, self.size_action]))
        else:
            raise ValueError("Initialization should be normal or uniform")

        self.all_weights = [self.hidden_layer.w, self.hidden_layer.b,
                            self.hidden_layer2.w, self.hidden_layer2.b,
                            self.action_layer.w, self.action_layer.b]

        self.learning_rate = learning_rate
        self.L2_regularization = 0.0
        self.h_action = []
        self.history = [self.h_action]
        if algorithm == SGD:
            self.optimizer = SGDOptimizer(self.all_weights)
        elif algorithm == RMSPROP:
            self.optimizer = RMSPropOptimizer(self.all_weights)
        elif algorithm == ADAM:
            self.optimizer = AdamOptimizer(self.all_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")


    def feedforward(self, observation):
        network_input = tf.concat([self.context, observation], 1)
        hidden_activation = self.dense_sigmoid(network_input, self.hidden_layer)
        hidden_activation2 = self.dense_sigmoid(hidden_activation, self.hidden_layer2)
        result = self.dense_linear(hidden_activation2, self.action_layer)
        self.h_action.append(result)
        return result

    def train(self, tape, target):
        # Compute error + backprop.
        loss = (target - self.h_action[-1])**2
        gradients = tape.gradient(loss, self.all_weights)
        self.optimizer.update_weights(gradients, self.learning_rate)
        return loss, gradients

    def clear_history(self):
        for data in self.history:
            data.clear()

    def new_episode(self):
        self.context = np.zeros((1, self.size_hidden), dtype=np.float32)
        self.clear_history()


