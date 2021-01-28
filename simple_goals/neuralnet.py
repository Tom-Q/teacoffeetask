import tensorflow as tf
import numpy as np
import teacoffeeenv as tce
import utils
import timeit
from abc import ABC, abstractmethod

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
NORMAL = "normal"
UNIFORM = "uniform"

class NeuralNet(object):
    def __init__(self, size_hidden=15, algorithm=SGD, learning_rate=0.1,
                 size_observation=len(tce.TeaCoffeeData.observations_list),
                 size_goal1 = len(tce.TeaCoffeeData.goals1_list),
                 size_goal2 = len(tce.TeaCoffeeData.goals2_list),
                 size_action = len(tce.TeaCoffeeData.actions_list),
                 initialization=NORMAL):
        self.algorithm = algorithm
        self.size_observation = size_observation
        self.size_hidden = size_hidden
        self.size_goal1 = size_goal1
        self.size_goal2 = size_goal2
        self.size_action = size_action

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
        else:
            raise ValueError("Initialization should be normal or uniform")

        self.all_weights = [self.hidden_layer.w, self.hidden_layer.b,
                            self.action_layer.w, self.action_layer.b,
                            self.goal1_layer.w, self.goal1_layer.b,
                            self.goal2_layer.w, self.goal2_layer.b]

        self.learning_rate = learning_rate
        self.L2_regularization = 0.0001
        self.h_action_softmax = []
        self.h_goal1_softmax = []
        self.h_goal2_softmax = []
        self.h_action_wta = []
        self.h_goal1_wta = []
        self.h_goal2_wta = []
        self.history = [self.h_action_softmax, self.h_goal1_softmax, self.h_goal2_softmax,
                        self.h_action_wta, self.h_goal1_wta, self.h_goal2_wta]
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
        for inputs in [self.action, self.goal1, self.goal2]:
            if inputs is not None:
                # Not sure if thatś necessary? In theory the winner take all has no gradient anyway.
                inputs = tf.stop_gradient(inputs)
                network_input = tf.concat([network_input, inputs], 1)
        hidden_activation = self.dense_sigmoid(network_input, self.hidden_layer)

        # Three separate softmaxes for the action and the goal
        self.action_softmax = self.dense_linear(hidden_activation, self.action_layer)
        self.action = self.winner_take_all(self.action_softmax)
        if self.size_goal1 > 0:
            self.goal1_softmax = self.dense_linear(hidden_activation, self.goal1_layer)
            self.goal1 = self.winner_take_all(self.goal1_softmax)
        if self.size_goal2 > 0:
            self.goal2_softmax = self.dense_linear(hidden_activation, self.goal2_layer)
            self.goal2 = self.winner_take_all(self.goal2_softmax)

        # The actual chosen action and goal
        self.save_history()

        # Set up the next context.
        self.context = hidden_activation

    def clear_history(self):
        for data in self.history:
            data.clear()

    def save_history(self):
        self.h_action_softmax.append(self.action_softmax)
        self.h_goal1_softmax.append(self.goal1_softmax)
        self.h_goal2_softmax.append(self.goal2_softmax)
        self.h_action_wta.append(self.action)
        self.h_goal1_wta.append(self.goal1)
        self.h_goal2_wta.append(self.goal2)

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
    def winner_take_all(a):
        a = a.numpy()
        b = np.zeros_like(a)
        b[range(len(a)), a.argmax(1)] = 1
        return b

    def train(self, targets, tape):
        loss = 0
        for i, target in enumerate(targets):
            loss += tf.nn.softmax_cross_entropy_with_logits(target.action_one_hot, self.h_action_softmax[i])
            if target.goal1_one_hot is not None:
                loss += tf.nn.softmax_cross_entropy_with_logits(target.goal1_one_hot, self.h_goal1_softmax[i])
            if target.goal2_one_hot is not None:
                loss += tf.nn.softmax_cross_entropy_with_logits(target.goal2_one_hot, self.h_goal2_softmax[i])
        loss += self.L2_regularization * sum([tf.reduce_sum(weights**2) for weights in self.all_weights])
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
