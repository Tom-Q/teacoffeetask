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
                                                  nonlinearity=self.nonlinearity, initialization=self.initialization)
        elif recurrent_layer == GRU:
            self.hidden_layer = layers.GRULayer(size_hidden_input, self.size_hidden)
        elif recurrent_layer == LSTM:
            self.hidden_layer = layers.LSTMLayer(size_hidden_input, self.size_hidden)
        else:
            raise NotImplementedError("unknown layer type")
        self.goal1_layer = layers.BasicLayer(utils.initialize([self.size_hidden, self.size_goal1], self.initialization))
        self.goal2_layer = layers.BasicLayer(utils.initialize([self.size_hidden, self.size_goal2], self.initialization))
        self.action_layer = layers.BasicLayer(utils.initialize([self.size_hidden, self.size_action], self.initialization))

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

        if self.algorithm == optimizers.SGD: self.optimizer = optimizers.SGDOptimizer(self.all_weights)
        elif self.algorithm == optimizers.RMSPROP: self.optimizer = optimizers.RMSPropOptimizer(self.all_weights)
        elif self.algorithm == optimizers.ADAM: self.optimizer = optimizers.AdamOptimizer(self.all_weights)
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

        self.save_history()

    def new_episode(self, initial_context=ZEROS):
        # Reinitialize the entire state of the network (anything that could affect the next episode.)
        self.clear_history()
        self.action = np.zeros((1, self.size_action), dtype=np.float32)
        if initial_context == ZEROS:
            self.hidden_layer.reset()
        elif initial_context == UNIFORM:
            self.hidden_layer.reset(state=np.float32(np.random.uniform(0.01, 0.99, (1, self.size_hidden))))
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
        self.clear_history()  # History will be used for various other stuff
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
    def __init__(self, size_observation, size_action, layer_sizes, output_layer=True):
        super().__init__(size_observation, size_action)
        self.ACCNetPrediction = ACCNet(size_observation, size_action, layer_sizes, output_layer)
        self.ACCNetReward = self.ACCNetPrediction.copy_with_same_weights()
        self.ACCNetControl = self.ACCNetPrediction.copy_with_same_weights()
        self.ACCNetPrediction._switch_mode(layers.PREDICT_ONLY)
        self.ACCNetReward._switch_mode(layers.REWARD_ONLY)
        self.ACCNetControl._switch_mode(layers.CONTROL_ONLY)
        # Tapes must be initialized in the right context
        self.PredictionTape = None
        self.RewardTape = None
        self.ControlTape = None
        self.nets = [self.ACCNetPrediction, self.ACCNetReward, self.ACCNetControl]

    def feedforward(self, observation):
        # Execute each net while only recording with one tape at a time.
        with self.ControlTape.stop_recording(), self.RewardTape.stop_recording():
            self.ACCNetPrediction.feedforward(observation)
        with self.PredictionTape.stop_recording(), self.ControlTape.stop_recording():
            self.ACCNetReward.feedforward(observation)
        with self.PredictionTape.stop_recording(), self.RewardTape.stop_recording():
            self.ACCNetControl.feedforward(observation)

    def new_episode(self, initial_context):
        # Note, there should be no stochasticity here.
        for net in self.nets:
            net.new_episode(initial_context)

    def save_history(self):
        for net in self.nets:
            net.save_history()

    def clear_history(self):
        for net in self.nets:
            net.clear_history()

    def train(self, dismissed_tape, targets):
        gradients = []
        ol_total, pl_total, cl_total, l2l_total = 0, 0, 0, 0
        for net, tape in [(self.ACCNetPrediction, self.PredictionTape),
                          (self.ACCNetReward, self.RewardTape),
                          (self.ACCNetControl, self.ControlTape)]:
            gradient, ol, pl, cl, l2l = net.compute_gradients(tape, targets)
            ol_total += ol
            pl_total += pl
            cl_total += cl
            l2l_total += l2l
            gradients.append(gradient)
        for net in self.nets:
            net.update_weights(gradients)
        return ol_total, pl_total, cl_total, l2l_total

    @property
    def h_action_wta(self):
        return self.ACCNetPrediction.h_action_wta  # doesn't matter which, all 3 networks should be identical.

    @property
    def context(self):
        context = self.ACCNetPrediction.layers[0].representation_layer.h
        for i in range(len(self.ACCNetPrediction.layers)-1):
            context = tf.concat((context, self.ACCNetPrediction.layers[i+1].representation_layer.h), axis=1)
        return context

    def delete_tapes(self):
        #del self.PredictionTape
        #del self.RewardTape
        #del self.ControlTape
        self.PredictionTape = None
        self.RewardTape = None
        self.ControlTape = None

class ACCNet(NeuralNet):
    def __init__(self, size_observation, size_action, layer_sizes, has_output_layer=False):
        super().__init__(size_observation, size_action)
        self.mode = None
        self.layers = []
        self.parameters = []
        self.layer_sizes = layer_sizes
        self.has_output_layer = True
        for i, size in enumerate(layer_sizes):
            predictive_nonlinearity = tf.nn.sigmoid if i == 0 else tf.nn.tanh  # Corresponds to the LSTM output which is tanh
            layer = layers.RewardControlPredLayer(size, predictive_nonlinearity=predictive_nonlinearity)
            self.layers.append(layer)
            self.parameters += self.layers[-1].parameters
        if has_output_layer:
            self.output_layer = layers.BasicLayer(input_size=layer_sizes[-1].output_bottomup,
                                                  output_size=size_action,
                                                  nonlinearity=None)
            self.parameters += self.output_layer.parameters

        self.h_pred_losses = []
        self.h_ctrl_losses = []
        self.h_action_logits = []
        self.h_action_wta = []
        self.h_contexts = []
        #TODO: Find a more elegant solution
        for layer in self.layers:
            self.h_contexts.append([]) # we save the contexts for each layer
            self.h_pred_losses.append([])
            self.h_ctrl_losses.append([])

        self.action_logits = None
        self.action_wta = None
        self.history = [self.h_action_logits, self.h_action_wta, self.h_pred_losses, self.h_ctrl_losses, self.h_contexts]
        self.optimizer = optimizers.AdamOptimizer(self.parameters)

    def feedforward(self, observation):
        # 1. Feedforward
        bottomup_activations = observation
        for layer in self.layers:
            bottomup_activations = layer.feedforward(bottomup_activations)
            bottomup_activations = layers.group_normalization(bottomup_activations, 32)  # This does more harm than good

        self.action_logits = self.output_layer.feedforward(bottomup_activations)
        self.action_wta = layers.winner_take_all(self.action_logits)

        #2. Feedbackward
        topdown_activations = self.action_wta
        for layer in reversed(self.layers):
            topdown_activations = layer.feedbackward(topdown_activations)
            topdown_activations = layers.group_normalization(topdown_activations, 32)

        # Keep track of values.
        self.save_history()

    def new_episode(self, initial_context=None):
        # Note, there should be no stochasticity here.
        for layer in self.layers:
            layer.reset()
        self.clear_history()

    def save_history(self):
        self.h_action_logits.append(self.action_logits)
        self.h_action_wta.append(copy.deepcopy(self.action_wta))
        for i, layer in enumerate(self.layers):
            self.h_contexts[i].append(layer.representation_layer.h)
            self.h_pred_losses[i].append(layer.prediction_loss)
            self.h_ctrl_losses[i].append(layer.control_loss)

    def clear_history(self):
        for data in self.history:
            data.clear()
        for layer in self.layers: # reinitalize context lists
            self.h_contexts.append([])
            self.h_pred_losses.append([])
            self.h_ctrl_losses.append([])

    def _switch_mode(self, mode):
        for layer in self.layers:
            layer.mode = mode
        self.mode = mode

    def compute_gradients(self, tape, targets):
        output_loss = 0.
        prediction_loss = 0.
        control_loss = 0.
        for i, target in enumerate(targets):
            if self.mode == None or self.mode == layers.REWARD_ONLY:
                output_loss += tf.nn.softmax_cross_entropy_with_logits(target, self.h_action_logits[i])  # decision loss
            for j, _ in enumerate(self.layers):
                if self.mode == None or self.mode == layers.PREDICT_ONLY:
                    prediction_loss += self.h_pred_losses[j][i]
                if self.mode == None or self.mode == layers.CONTROL_ONLY:
                    control_loss += self.h_ctrl_losses[j][i]

        l2_loss = tf.math.add_n([tf.reduce_sum(tf.abs(parameter)) for parameter in self.parameters]) * 0.00001 # L2 loss.
        total_loss = output_loss + prediction_loss + control_loss * 0.01 + l2_loss
        # Can't have L2 loss equal control loss or they cancel each other out
        gradients = tape.gradient(total_loss, self.parameters, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return gradients, output_loss, prediction_loss, control_loss, l2_loss

    def update_weights(self, gradients):
        for gradient in gradients:
            self.optimizer.update_weights(gradient, self.learning_rate)

    def train(self, tape, targets):
        raise NotImplementedError("Use compute gradients and update weights separately")

    def copy_with_same_weights(self):
        # Used to have different gradient updates for the same network
        # Warning, this doesn't copy the hidden state of the recurrent layers (LSTM/GRU).
        copy = ACCNet(self.size_observation, self.size_action, self.layer_sizes, self.has_output_layer)
        # Make a deep copy of all network parameters
        for i in range(len(self.parameters)):
            copy.parameters[i].assign(self.parameters[i])
        return copy
