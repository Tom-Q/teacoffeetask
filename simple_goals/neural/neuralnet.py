import tensorflow as tf
import numpy as np
import teacoffeeenv as tce
import utils
import timeit
from abc import ABC, abstractmethod
import copy
from neural import layers, optimizers


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

RELU = "relu"
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
    def __init__(self, size_observation, size_action, initialization=utils.UNIFORM, nonlinearity=tf.sigmoid,
                 algorithm=optimizers.SGD, learning_rate=0.1,
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



ZEROS = "zeros"
UNIFORM = "uniform"
ELMAN = "elman"
GRU = "gru"
LSTM = "lstm"
class GoalNet(NeuralNet):
    def __init__(self, size_hidden=15, algorithm=optimizers.SGD, learning_rate=0.1,
                 size_observation=len(tce.TeaCoffeeData.observations_list),
                 size_goal1=len(tce.TeaCoffeeData.goals1_list), size_goal2=len(tce.TeaCoffeeData.goals2_list),
                 size_action=len(tce.TeaCoffeeData.actions_list), initialization=utils.NORMAL,
                 L1_reg = 0., L2_reg = 0.,
                 last_action_inputs=False,
                 nonlinearity=tf.sigmoid, params=None,
                 recurrent_layer=ELMAN):
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

        self.goal1 = self.goal2 = self.action = self.action_softmax =\
            self.goal1_softmax = self.goal2_softmax = None

        size_hidden_input = self.size_observation + self.size_action + self.size_goal1 + self.size_goal2
        if recurrent_layer == ELMAN:
            self.hidden_layer = layers.ElmanLayer(size_hidden_input, self.size_hidden, initial_context=None,
                                                  nonlinearity=self.nonlinearity, initialization=initialization)
        elif recurrent_layer == GRU:
            self.hidden_layer = layers.GRULayer(size_hidden_input, self.size_hidden)
        elif recurrent_layer == LSTM:
            self.hidden_layer = layers.LSTMLayer(size_hidden_input, self.size_hidden)
        else:
            raise NotImplementedError("unknown layer type")
        self.goal1_layer = layers.BasicLayer(utils.initialize([self.size_hidden, self.size_goal1], initialization))
        self.goal2_layer = layers.BasicLayer(utils.initialize([self.size_hidden, self.size_goal2], initialization))
        self.action_layer = layers.BasicLayer(utils.initialize([self.size_hidden, self.size_action], initialization))

        self.all_weights = self.hidden_layer.parameters + self.action_layer.parameters +\
                           self.goal1_layer.parameters + self.goal2_layer.parameters

        self.h_action_softmax = []
        self.h_goal1_softmax = []
        self.h_goal2_softmax = []
        self.h_action_wta = []
        self.h_goal1_wta = []
        self.h_goal2_wta = []
        self.h_context = []
        self.history = [self.h_action_softmax, self.h_goal1_softmax, self.h_goal2_softmax,
                        self.h_action_wta, self.h_goal1_wta, self.h_goal2_wta, self.h_context]

        if algorithm == optimizers.SGD: self.optimizer = optimizers.SGDOptimizer(self.all_weights)
        elif algorithm == optimizers.RMSPROP: self.optimizer = optimizers.RMSPropOptimizer(self.all_weights)
        elif algorithm == optimizers.ADAM: self.optimizer = optimizers.AdamOptimizer(self.all_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")

    def get_params(self):
        return ParamsGoalNet(size_observation=self.size_observation, size_action=self.size_action,
                             size_hidden=self.size_hidden, initialization=self.initialization,
                             algorithm=self.algorithm, learning_rate=self.learning_rate,
                             L1_reg=self.L1_regularization, L2_reg=self.L2_regularization,
                             size_goal1=self.size_goal1, size_goal2=self.size_goal2,
                             last_action_inputs=self.last_action_inputs)

    def feedforward(self, observation): #, gain_multiplier=1., gain_multiplier_from=0, gain_multiplier_to=None):
        network_input = observation
        if not self.last_action_inputs: # cancel out actions
            self.action *= 0.
        for inputs in [self.action, self.goal2, self.goal1]:
            if inputs is not None:
                # Not sure if thatś necessary? In theory the winner take all has no gradient anyway.
                inputs = tf.stop_gradient(inputs)
                network_input = tf.concat([network_input, inputs], 1)

        hidden_activation = self.hidden_layer.feedforward(network_input)

        #if gain_multiplier != 1:
        #    if gain_multiplier_to is None:
        #        gain_multiplier_to = self.size_hidden
        #    hidden_activation = hidden_activation.numpy()
        #    hidden_activation[0, gain_multiplier_from:gain_multiplier_to] *= gain_multiplier
        #    hidden_activation[0, gain_multiplier_to:] *= 1./gain_multiplier

        # Three separate softmaxes for the action and the goal
        self.action_softmax = self.action_layer.feedforward(hidden_activation)
        self.action = layers.winner_take_all(self.action_softmax)
        if self.size_goal1 > 0:
            self.goal1_softmax = self.goal1_layer.feedforward(hidden_activation)
            self.goal1 = layers.winner_take_all(self.goal1_softmax)
        if self.size_goal2 > 0:
            self.goal2_softmax = self.goal2_layer.feedforward(hidden_activation)
            self.goal2 = layers.winner_take_all(self.goal2_softmax)


        # The actual chosen action and goal
        self.save_history()

    def new_episode(self, initial_context=ZEROS):
        # Reinitialize the entire state of the network (anything that could affect the next episode.)
        self.clear_history()
        self.action = np.zeros((1, self.size_action), dtype=np.float32)
        if initial_context == ZEROS:
            self.hidden_layer.reset()
        elif initial_context == UNIFORM:
            state = np.float32(np.random.uniform(0.01, 0.99, (1, self.size_hidden)))
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
        self.h_context.append(self.hidden_layer.h)

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

    @property
    def context(self):
        return self.hidden_layer.h

    @context.setter
    def context(self, val):
        self.hidden_layer.h = val


# Convenience class that contains and manages three ACC Nets
class TripleACCNet(NeuralNet):
    def __init__(self, size_observation, size_action, size_hidden1, size_hidden2):
        super().__init__(size_observation, size_action)
        self.ACCNetPrediction = ACCNet(size_observation, size_action, size_hidden1, size_hidden2)
        self.ACCNetReward = self.ACCNetPrediction.copy_with_same_weights()
        self.ACCNetControl = self.ACCNetPrediction.copy_with_same_weights()
        self.ACCNetPrediction._switch_mode(layers.PREDICT_ONLY)
        self.ACCNetReward._switch_mode(layers.REWARD_ONLY)
        self.ACCNetControl._switch_mode(layers.CONTROL_ONLY)
        self.PredictionTape = tf.GradientTape()
        self.RewardTape = tf.GradientTape()
        self.ControlTape = tf.GradientTape()
        self.nets = [(self.ACCNetPrediction, self.PredictionTape),
                     (self.ACCNetReward, self.RewardTape),
                     (self.ACCNetControl, self.ControlTape)]

    def feedforward(self, observation):
        for net, tape in self.nets:
            with tape:
                net.feedforward(observation)

    def new_episode(self):
        # Note, there should be no stochasticity here.
        for net, _ in self.nets:
            net.new_episode()

    def save_history(self):
        for net, _ in self.nets:
            net.save_history()

    def clear_history(self):
        for net, _ in self.nets:
            net.clear_history()

    def train(self, dismissed_tape, targets):
        gradients = []
        total_loss = 0.
        for net, tape in self.nets:
            gradient, loss = net.compute_gradients(tape, targets) #TODO: need 3 different tapes, one for each network.
            gradients.append(gradient)
            total_loss += loss
        for net, _ in self.nets:
            net.update_weights(gradients)
        return total_loss


class ACCNet(NeuralNet):
    def __init__(self, size_observation, size_action, size_hidden1, size_hidden2):
        super().__init__(size_observation, size_action)
        self.size_hidden1 = size_hidden1
        self.size_hidden2 = size_hidden2
        self.Layer1 = layers.RewardControlPredLayer(size_input=size_observation, size_output=size_hidden1, size_next=size_hidden2)
        self.Layer2 = layers.RewardControlPredLayer(size_input=size_hidden1, size_output=size_action, size_next=0,
                                                    output_nonlinearity=None)
        self.parameters = self.Layer1.parameters + self.Layer2.parameters

        self.h_action_logits = []
        self.h_action_wta = []
        self.h_context1 = []
        self.h_context2 = []
        self.history = [self.h_action_logits, self.h_action_wta, self.h_context1, self.h_context2]
        self.optimizer = optimizers.AdamOptimizer(self.parameters)

    def feedforward(self, observation):
        #1. Feedforward
        bottom_up = self.Layer1.feedforward(observation)
        self.h_action_logits = self.Layer2.feedforward(bottom_up)
        self.h_action_wta = layers.winner_take_all(self.h_action_logits)

        #2. Feedbackward
        top_down = self.Layer2.feedbackward(None)
        self.Layer1.feedbackward(top_down)

    def new_episode(self):
        # Note, there should be no stochasticity here.
        self.Layer1.reset()
        self.Layer2.reset()
        self.clear_history()

    def save_history(self):
        self.h_action_logits.append(self.h_action_logits)
        self.h_action_wta.append(copy.deepcopy(self.h_action_wta))
        self.h_context1.append(self.Layer1.representation_layer.h)
        self.h_context2.append(self.Layer2.representation_layer.h)

    def clear_history(self):
        for data in self.history:
            data.clear()

    def _switch_mode(self, mode):
        self.Layer1.mode = mode
        self.Layer2.mode = mode

    def compute_gradients(self, tape, targets):
        loss = 0
        for i, target in enumerate(targets):
            if target.action_one_hot is not None:
                loss += tf.nn.softmax_cross_entropy_with_logits(target.action_one_hot, self.h_action_logits[i])

        gradients = tape.gradient(loss, self.parameters)
        return gradients, loss

    def update_weights(self, gradients):
        for gradient in gradients:
            self.optimizer.update_weights(gradient, self.learning_rate)
        self.clear_history()

    def copy_with_same_weights(self):
        # Used to have different gradient updates for the same network
        # Warning, this doesn't copy the hidden state of the recurrent layers (LSTM/GRU).
        copy = ACCNet(self.size_observation, self.size_action, self.size_hidden1, self.size_hidden2)
        # Make a deep copy of all network parameters
        for i in range(len(self.parameters)):
            copy.parameters[i].assign(self.parameters[i])
        return copy