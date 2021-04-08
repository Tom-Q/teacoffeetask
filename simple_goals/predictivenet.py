import tensorflow as tf
import numpy as np
import teacoffeeenv as tce
import neuralnet as nn

class PredictiveNet(object):
    def __init__(self, layer_sizes, algorithm=nn.SGD, learning_rate=0.1,
                 size_observation=len(tce.TeaCoffeeData.observations_list),
                 size_action = len(tce.TeaCoffeeData.actions_list)):
        """
        @param layer_sizes: list of integers, the size of each hidden layer
        @param algorithm: nn.SGD, ADAM, or RMSPROP
        @param learning_rate:
        @param size_observation:
        @param size_action:
        """

        self.algorithm = algorithm
        self.size_observation = size_observation
        self.layer_sizes = layer_sizes
        self.size_action = size_action

        self.action = self.context = self.action_linear = self.prediction_linear = None

        self.hidden_layers = []
        self.predictive_layers = []
        self.all_weights = []
        for i, layer_size in enumerate(layer_sizes):
            if i == 0:
                input_size = layer_size + self.size_observation + self.size_action
                predictive_output_size = self.size_observation
            else:
                input_size = layer_size + layer_sizes[i-1]
                predictive_output_size = layer_sizes[i-1]
            self.hidden_layers.append(nn.Layer(np.random.normal(0., .1, size=[input_size, layer_size])))
            self.predictive_layers.append(nn.Layer(np.random.normal(0., .1, size=[layer_size, predictive_output_size])))
            self.all_weights.append(self.hidden_layers[-1].w)
            self.all_weights.append(self.hidden_layers[-1].b)
            self.all_weights.append(self.predictive_layers[-1].w)
            self.all_weights.append(self.predictive_layers[-1].b)

        self.action_layer = nn.Layer(np.random.normal(0., .1, size=[self.layer_sizes[-1], self.size_action]))

        self.all_weights.append(self.action_layer.w)
        self.all_weights.append(self.action_layer.b)

        self.learning_rate = learning_rate
        self.L2_regularization = 0.0001
        self.h_action_linear = []
        self.h_action_wta = []
        self.h_prediction_linear = []
        self.h_prediction_wta = []
        self.history = [self.h_action_linear, self.h_action_wta, self.h_prediction_linear, self.h_prediction_wta]
        if algorithm == nn.SGD:
            self.optimizer = nn.SGDOptimizer(self.all_weights)
        elif algorithm == nn.RMSPROP:
            self.optimizer = nn.RMSPropOptimizer(self.all_weights)
        elif algorithm == nn.ADAM:
            self.optimizer = nn.AdamOptimizer(self.all_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")

    def feedforward(self, observation, type):
        if type=='sigmoid':
            prediction_error = observation - self.prediction_probability
        elif type=="wta":
            prediction_error = observation - self.prediction_wta

        network_input = tf.concat([self.context, prediction_error], 1)
        for inputs in [self.action]:
            if inputs is not None:
                # Not sure if that's necessary? In theory the winner take all has no gradient anyway.
                inputs = tf.stop_gradient(inputs)
                network_input = tf.concat([network_input, inputs], 1)
        hidden_activation = self.dense_sigmoid(network_input, self.hidden_layer)

        self.action_linear = self.dense_linear(hidden_activation, self.action_layer)
        #self.action_probability = tf.nn.softmax(self.action_linear)
        self.action = self.winner_take_all(self.action_linear)

        self.prediction_linear = self.dense_linear(hidden_activation, self.predictive_layer)
        self.prediction_probability = tf.stop_gradient(tf.nn.softmax(self.prediction_linear))
        self.prediction = self.prediction_probability #tf.stop_gradient(self.prediction_probability)
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
        return tf.nn.softmax(PredictiveNet.dense_linear(x, layer))

    @staticmethod
    def dense_sigmoid(x, layer):
        """
        :param x: input
        :param layer: Layer object with weights matrix [+ bias]
        :return: densely connected layer with sigmoid output
        """
        return tf.nn.sigmoid(PredictiveNet.dense_linear(x, layer))

    @staticmethod
    def winner_take_all(a):
        a = a.numpy()
        b = np.zeros_like(a)
        b[range(len(a)), a.argmax(1)] = 1
        return b

    def train_MSE(self, targets_action, targets_prediction, tape):
        # Compute error + backprop.
        loss = 0.
        #for i in range(len(targets_action)):
            #loss += tf.reduce_sum((targets_action[i] - tf.nn.sigmoid(self.h_action_linear[i])) ** 2)  # mse
        loss_action = tf.nn.softmax_cross_entropy_with_logits(targets_action, self.h_action_linear)  # cross entropy

        #for i in range(len(targets_prediction)):
            #loss += tf.reduce_sum((targets_prediction[i] - tf.nn.sigmoid(self.h_prediction_linear[i])) ** 2)
        loss_prediction = tf.nn.softmax_cross_entropy_with_logits(targets_prediction, self.h_prediction_linear) # cross entropy

        #print(loss_action, loss_prediction)
        loss = tf.reduce_sum(loss_action + loss_prediction)
        # I'm going to assume that "weight persistence 0.999999" means L1 regularization. Multiplying by
        # the learning rate too.
        loss += self.L2_regularization * sum([tf.reduce_sum(weights**2) for weights in self.all_weights])
        gradients = tape.gradient(loss, self.all_weights)
        self.optimizer.update_weights(gradients, self.learning_rate)
        self.clear_history()
        return loss, gradients


class DeepPredNet(object):
    def __init__(self, size_hidden=15, algorithm=nn.SGD, learning_rate=0.1,
                 size_observation=len(tce.TeaCoffeeData.observations_list),
                 size_action = len(tce.TeaCoffeeData.actions_list)):
        self.algorithm = algorithm
        self.size_observation = size_observation
        self.size_hidden = size_hidden
        self.size_action = size_action

        self.action = self.context = self.action_linear = self.prediction_linear = None

        self.hidden_layer = nn.Layer(np.random.normal(0., .1, size=[self.size_hidden + self.size_observation +
                                                                    self.size_action,
                                                                    self.size_hidden]))

        self.action_layer = nn.Layer(np.random.normal(0., .1, size=[self.size_hidden, self.size_action]))
        self.predictive_layer = nn.Layer(np.random.normal(0., .1, size=[self.size_hidden, self.size_observation]))

        self.all_weights = [self.hidden_layer.w, self.hidden_layer.b,
                            self.predictive_layer.w, self.predictive_layer.b,
                            self.action_layer.w, self.action_layer.b]

        self.learning_rate = learning_rate
        self.L2_regularization = 0.0001
        self.h_action_linear = []
        self.h_action_wta = []
        self.h_prediction_linear = []
        self.h_prediction_wta = []
        self.history = [self.h_action_linear, self.h_action_wta, self.h_prediction_linear, self.h_prediction_wta]
        if algorithm == nn.SGD:
            self.optimizer = nn.SGDOptimizer(self.all_weights)
        elif algorithm == nn.RMSPROP:
            self.optimizer = nn.RMSPropOptimizer(self.all_weights)
        elif algorithm == nn.ADAM:
            self.optimizer = nn.AdamOptimizer(self.all_weights)
        else:
            raise ValueError("Algorithm must be SGD, RMSPROP, or ADAM. Nothing else implemented ATM.")

    def feedforward(self, observation, type):
        if type=='sigmoid':
            prediction_error = observation - self.prediction_probability
        elif type=="wta":
            prediction_error = observation - self.prediction_wta

        network_input = tf.concat([self.context, prediction_error], 1)
        for inputs in [self.action]:
            if inputs is not None:
                # Not sure if that's necessary? In theory the winner take all has no gradient anyway.
                inputs = tf.stop_gradient(inputs)
                network_input = tf.concat([network_input, inputs], 1)
        hidden_activation = self.dense_sigmoid(network_input, self.hidden_layer)

        self.action_linear = self.dense_linear(hidden_activation, self.action_layer)
        #self.action_probability = tf.nn.softmax(self.action_linear)
        self.action = self.winner_take_all(self.action_linear)

        self.prediction_linear = self.dense_linear(hidden_activation, self.predictive_layer)
        self.prediction_probability = tf.stop_gradient(tf.nn.softmax(self.prediction_linear))
        self.prediction = self.prediction_probability #tf.stop_gradient(self.prediction_probability)
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
        return tf.nn.softmax(PredictiveNet.dense_linear(x, layer))

    @staticmethod
    def dense_sigmoid(x, layer):
        """
        :param x: input
        :param layer: Layer object with weights matrix [+ bias]
        :return: densely connected layer with sigmoid output
        """
        return tf.nn.sigmoid(PredictiveNet.dense_linear(x, layer))

    @staticmethod
    def winner_take_all(a):
        a = a.numpy()
        b = np.zeros_like(a)
        b[range(len(a)), a.argmax(1)] = 1
        return b

    def train_MSE(self, targets_action, targets_prediction, tape):
        # Compute error + backprop.
        loss = 0.
        #for i in range(len(targets_action)):
            #loss += tf.reduce_sum((targets_action[i] - tf.nn.sigmoid(self.h_action_linear[i])) ** 2)  # mse
        loss_action = tf.nn.softmax_cross_entropy_with_logits(targets_action, self.h_action_linear)  # cross entropy

        #for i in range(len(targets_prediction)):
            #loss += tf.reduce_sum((targets_prediction[i] - tf.nn.sigmoid(self.h_prediction_linear[i])) ** 2)
        loss_prediction = tf.nn.softmax_cross_entropy_with_logits(targets_prediction, self.h_prediction_linear) # cross entropy

        #print(loss_action, loss_prediction)
        loss = tf.reduce_sum(loss_action + loss_prediction)
        # I'm going to assume that "weight persistence 0.999999" means L1 regularization. Multiplying by
        # the learning rate too.
        loss += self.L2_regularization * sum([tf.reduce_sum(weights**2) for weights in self.all_weights])
        gradients = tape.gradient(loss, self.all_weights)
        self.optimizer.update_weights(gradients, self.learning_rate)
        self.clear_history()
        return loss, gradients