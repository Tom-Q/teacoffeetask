import math
import utils
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

class holroyd2018network():
    """
    This is more or less the network implemented by Holroyd et al 2018.
    """
    def __init__(self):
        # Make input one-hot encodings
        # Categories used
        input_categories = ["start", "coffee", "milk",  "stir", "sugar", "tea",    "water"]
        # Sequences used
        self.input_sequences_strings = [["start", "coffee", "milk", "stir", "water", "stir"],
                                    ["start", "coffee", "water", "stir", "milk", "stir"],
                                    ["start", "tea", "sugar", "stir", "water", "stir"],
                                    ["start", "tea", "water", "stir", "sugar", "stir"]]
        # Convert to one hot encoding
        self.input_sequences_one_hot = [utils.one_hot_encoding(input_categories, sequence)
                                        for sequence in self.input_sequences_strings]

        # Same thing for outputs
        output_categories = ["milk", "stir", "sugar", "tea", "water", "coffee", "serve_tea", "serve_coffee"]
        self.output_sequences_strings = [["coffee", "milk", "stir", "water", "stir", "serve_coffee"],
                                    ["coffee", "water", "stir", "milk", "stir", "serve_coffee"],
                                    ["tea", "sugar", "stir", "water", "stir", "serve_tea"],
                                    ["tea", "water", "stir", "sugar", "stir", "serve_tea"]]
        self.output_sequences_one_hot = [utils.one_hot_encoding(output_categories, sequence)
                                         for sequence in self.output_sequences_strings]

        # Set up the network : layer sizes, initialization (b stands for bias weights, w for non-bias weights)
        self.learning_rate = 0.5
        self._hidden_size = 15
        self._input_size = 7
        self._output_size = 8
        self._w_hidden = tf.Variable(np.random.normal(0., 1.,
                                                      size=[self._input_size + self._hidden_size, self._hidden_size]))
        self._w_output = tf.Variable(np.random.normal(0., 1., size=[self._hidden_size, self._output_size]))
        self._b_hidden = tf.Variable(np.zeros([1, self._hidden_size]))
        self._b_output = tf.Variable(np.zeros([1, self._output_size]))
        self._all_weights = [self._w_hidden, self._b_hidden, self._w_output, self._b_output]

    def _update_weights(self, gradients):
        for i in range(len(self._all_weights)):
            self._all_weights[i].assign_sub(gradients[i] * self.learning_rate)

    def forward_one_action(self, network_input, previous_hidden, activation=None):
        """
        :param network_input: input (previous action or start)
        :param previous_hidden: context (0 at start, otherwise previous context)
        :return: logits for the action, context layer activations
        """
        input_vec = tf.concat([network_input, previous_hidden], 1)
        hidden_activation = utils.dense_sigmoid(input_vec, self._w_hidden, self._b_hidden)
        output_activation = None
        if activation is None:
            output_activation = utils.dense_linear(hidden_activation, self._w_output, self._b_output)
        elif activation == "sigmoid":
            output_activation = utils.dense_sigmoid(hidden_activation, self._w_output, self._b_output)
        return output_activation, hidden_activation

    def full_sequence(self, input_sequence, output_sequence, error_type):
        if error_type == "mse":
            return self.full_sequence_mse(input_sequence, output_sequence)
        elif error_type == "cross_entropy":
            return self.full_sequence_cross_entropy(input_sequence, output_sequence)
        else:
            raise ValueError("error type is unknown/not implemented")

    def full_sequence_cross_entropy(self, input_sequence, output_sequence):
        """
        :param input_sequence: one-hot encoded
        :param output_sequence: one-hot encoded
        :return: loss, context activations, action probabilities
        """
        with tf.GradientTape() as tape:
            context_activation = np.zeros([1, self._hidden_size])
            outputs = []
            contexts = []
            # Do the forward pass
            for step in range(len(input_sequence)):
                logits, context_activation = self.forward_one_action(input_sequence[step], context_activation)
                outputs.append(logits)
                contexts.append(context_activation.numpy())

            # Compute the loss
            loss = sum([tf.nn.softmax_cross_entropy_with_logits(output_sequence[i], outputs[i]) for i in range(len(outputs))])
            self._update_weights(tape.gradient(loss, self._all_weights))

            # return the activation probabilities, not the logits
            for idx, logits in enumerate(outputs):
                outputs[idx] = tf.exp(logits) / tf.reduce_sum(tf.exp(logits))

            return loss.numpy()[0], contexts, [output.numpy() for output in outputs]

    def full_sequence_mse(self, input_sequence, output_sequence):
        """
        :param input_sequence: one-hot encoded
        :param output_sequence: one-hot encoded
        :return: loss, action probabilities
        """
        with tf.GradientTape() as tape:
            context_activation = np.zeros([1, self._hidden_size])
            outputs = []
            contexts = []
            # Do the forward pass
            for step in range(len(input_sequence)):
                network_output, context_activation = self.forward_one_action(input_sequence[step], context_activation,
                                                                             activation="sigmoid")
                outputs.append(network_output)
                contexts.append(context_activation.numpy())

            # Compute the loss
            loss = sum([tf.reduce_sum(tf.square(output_sequence[i]-outputs[i])) * 0.5 for i in range(len(outputs))])

            self._update_weights(tape.gradient(loss, self._all_weights))
            return loss.numpy(), contexts, [output.numpy() for output in outputs]

    def show_results(self, error_type, verbose=False):
        # Now show what happens.
        self.learning_rate = 0.  # Freeze the network
        accuracy_per_sequence = []
        for i in range(len(self.input_sequences_one_hot)):
            print("\n##############\n# SEQUENCE {0} #\n##############".format(i + 1))
            if error_type == "cross_entropy":
                _, _, probas = self.full_sequence_cross_entropy(self.input_sequences_one_hot[i],
                                                                self.output_sequences_one_hot[i])
            elif error_type == "mse":
                _, _, probas = self.full_sequence_mse(self.input_sequences_one_hot[i],
                                                      self.output_sequences_one_hot[i])
            else:
                raise ValueError("error type doesn't match anything")

            # Probability of activation only for target actions
            probas_targets = []
            for j in range(len(probas)):
                index = int(np.argmax(self.output_sequences_one_hot[i][j]))
                probas_targets.append(probas[j][0][index])
            # bit of an ugly line. Just prints each target action with the corresponding output probability
            print("Probability for each successive target action: {0}".format(
                [list(a) for a in zip(self.output_sequences_strings[i],
                                      ["{0:.4f}".format(proba) for proba in probas_targets])]))
            # Everything
            if verbose:
                print("Detailed results:")
                for j in range(len(probas)):
                    print("target:{0}".format(self.output_sequences_one_hot[i][j]))
                    print("output:{0}".format(probas[j]))

            total_error = 0.
            for j in range(len(probas)):
                total_error += sum(np.abs(self.output_sequences_one_hot[i][j][0] - probas[j][0]))
            accuracy = (48-total_error)/48.
            accuracy_per_sequence.append(accuracy)
            print("Total loss (absolute):{0:.4f} (% accuracy: {1:.4f})".format(total_error, accuracy))
        # There's 48 outputs in total for a sequence (6 * 8). Of these, two incur a total error of at least 1.
        # (In expectation, because 2 options are equally likely to be right)
        print("\nAverage accuracy (all seqs.): {0:.3f} (Max accuracy: {1:.4f})".format(sum(accuracy_per_sequence)/4.,
                                                                                           (48.-2)/48.))

    def generate_euclidian_distance_matrix(self, error_type):
        #1. For each sequence, get the context values.
        self.learning_rate = 0.
        all_contexts = []
        for i in range(len(self.input_sequences_one_hot)):
            if error_type == "mse":
                _, sequence_contexts, _ = self.full_sequence_mse(self.input_sequences_one_hot[i],
                                                        self.output_sequences_one_hot[i])
            elif error_type == "cross_entropy":
                _, sequence_contexts, _ = self.full_sequence_mse(self.input_sequences_one_hot[i],
                                                        self.output_sequences_one_hot[i])
            else: raise ValueError("Undefined error type")
            all_contexts = all_contexts + [context[0] for context in sequence_contexts]

        # 2. For each context, measure the distance with each other context
        distance_matrix = np.zeros(shape=[24, 24])
        for i, context1 in enumerate(all_contexts):
            for j, context2 in enumerate(all_contexts):
                distance_matrix[i, j] = np.linalg.norm(context1-context2) # euclidian distance

        return distance_matrix

def main():
    verbose = False
    error_type="cross_entropy"  #mse or cross_entropy
    holroyd_net = holroyd2018network()
    running_avg_loss = 0.
    print("Iteration\tLoss")
    for i in range(5000):
        idx = np.random.randint(4)
        loss, _, _ = holroyd_net.full_sequence(holroyd_net.input_sequences_one_hot[idx],
                                               holroyd_net.output_sequences_one_hot[idx],
                                               error_type)
        if i == 0:
            running_avg_loss = loss
        elif i < 10:
            running_avg_loss = running_avg_loss * 0.5 + 0.5 * loss
        elif i < 100:
            running_avg_loss = running_avg_loss * 0.8 + 0.2 * loss
        else:
            running_avg_loss = running_avg_loss * 0.99 + 0.01 * loss

        if i == 0 or (i+1) % 1000 == 0 or (i < 1000 and math.log(i+1, 3) % 1 == 0):
            print("{0:5d}:\t\t{1:8.4f}".format(i+1, running_avg_loss))

    holroyd_net.show_results(error_type, verbose=False)
    matrix = holroyd_net.generate_euclidian_distance_matrix(error_type)

    x_labels = [input_string for list_of_strings in holroyd_net.input_sequences_strings for input_string in list_of_strings]  # flatten the list of inputs
    y_labels = x_labels  # labels for y-axis
    sns.heatmap(matrix, cbar=True, square=True, xticklabels=x_labels, yticklabels=y_labels)
    plt.show()

    # export data to csv
    np.savetxt("euclidian_dist_matrix.csv", matrix, delimiter=",")

if __name__ == "__main__":
    main()
