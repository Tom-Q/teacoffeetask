import tensorflow as tf
from abc import ABC, abstractmethod


# Implemented optimization techniques
SGD = "sgd"
RMSPROP = "rmsprop"
ADAM = "adam"

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


# Reinitialize dead units = units with no activation or with low OUTGOING weights (below some threshold).
# These units should get new INGOING weights.
# Problem 1: how to identify units with low outgoing weights? one activation may have multiple output directions.
# Problem 2: how do we avoid resetting the same units every time
# Problem 3: how to do this in a layer based way (each layer should do this separately).
# Problem 4: complex layers (LSTM, GRU)
# --> each layer keeps track of outgoing layers (layer-local functions based on layer input)
# --> For each layer, compute sum of outgoing weights in all subsequent matrices.
# --> But those are not necessarily comparable. A high weight for ReLu is a lot more meaningful than a high weight for sigmoid....
# urgh this is a nightmare
