import math
import utils
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from termcolor import colored


class Holroyd2018Network(object):
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
        # The paper reports initialization from a standard  normal, but it seems 0.1 * normal was used in practice
        self._w_hidden = tf.Variable(np.random.normal(0., .1,
                                                      size=[self._input_size + self._hidden_size, self._hidden_size]))
        self._w_output = tf.Variable(np.random.normal(0., .1, size=[self._hidden_size, self._output_size]))
        # Normally bias weights should be initialized to 0
        self._b_hidden = tf.Variable(np.random.normal(0., 1., size=[1, self._hidden_size]))
        self._b_output = tf.Variable(np.random.normal(0., .1, size=[1, self._output_size]))
        # This makes it easier to perform updates
        self._all_weights = [self._w_hidden, self._b_hidden, self._w_output, self._b_output]

    def _update_weights(self, gradients):
        for i in range(len(self._all_weights)):
            self._all_weights[i].assign_sub(gradients[i] * self.learning_rate)

    def forward_one_action(self, network_input, previous_hidden, activation=None):
        """
        :param network_input: input (previous action or start)
        :param previous_hidden: context (0 at start, otherwise previous context)
        :param activation: none, or a string for the desired activation function (atm only "sigmoid" works)
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
            loss = sum([tf.nn.softmax_cross_entropy_with_logits(output_sequence[i], outputs[i])
                        for i in range(len(outputs))])
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

    def learning_rate_warning(self):
        if self.learning_rate != 0.:
            warnings.warn("If the learning rate is not 0., this function may not work as expected")

    def run_on_all_sequences(self, error_type):
        self.learning_rate_warning()
        losses = []
        contexts = []
        network_outputs = []
        for i in range(len(self.input_sequences_one_hot)):
            if error_type == "cross_entropy":
                loss, context, outputs = self.full_sequence_cross_entropy(self.input_sequences_one_hot[i],
                                                                          self.output_sequences_one_hot[i])
            elif error_type == "mse":
                loss, context, outputs = self.full_sequence_mse(self.input_sequences_one_hot[i],
                                                                self.output_sequences_one_hot[i])
            else:
                raise ValueError("error type doesn't match anything")
            losses.append(loss)
            contexts.append(context)
            network_outputs.append(outputs)
        return losses, contexts, network_outputs

    def test_accuracy(self, error_type):
        return np.array_equal(self.holroyd_accuracy(error_type), np.asarray([0.5, 0.5, 0., 0., 0., 0.]))

    def holroyd_accuracy(self, error_type):
        self.learning_rate_warning()
        _, _, outputs = self.run_on_all_sequences(error_type)
        # Flatten the outputs. They're returned as a list of list of 2D-ndarrays. Let's make a list of 1D-ndarrays.
        all_outputs = self.flatten_holroyd_data(outputs)

        # Compute the winner take all result for all outputs
        wtas = [utils.numpy_winner_take_all(output) for output in all_outputs]

        # Flatten targets [this needs to be a separate util function since I'm using this in 3 different places]
        all_targets = self.flatten_holroyd_data(self.output_sequences_one_hot)

        # Compute the average accuracy for each time-step number in the sequence
        accuracy_per_step = np.zeros(6)  # A sequence has 6 time-steps
        for i in range(len(all_targets)):
            # Division per 2 required so bad outputs sum to 1, rather than 2
            accuracy_per_step[i % 6] += np.sum(np.abs(all_targets[i] - wtas[i]))/2.
        accuracy_per_step /= 4.  # There are 4 sequences
        return accuracy_per_step

    @staticmethod
    def flatten_holroyd_data(not_flat_list):
        """
        :param not_flat_list: data structure used for data in this class: a list of 4 lists (1 per sequence)
        of 2D-ndarrays
        :return: a flattened list of 1D-ndarrays
        """
        flat_list = utils.flatten_list_of_lists(not_flat_list)
        return [element[0] for element in flat_list]

    def show_results(self, error_type, verbose=False):
        self.learning_rate_warning()

        # Now show what happens.
        accuracy_per_sequence = []
        _, _, network_outputs = self.run_on_all_sequences(error_type)
        for i, outputs in enumerate(network_outputs):
            print("\n##############\n# SEQUENCE {0} #\n##############".format(i + 1))
            # Probability of activation only for target actions
            probas_targets = []
            for j in range(len(outputs)):
                index = int(np.argmax(self.output_sequences_one_hot[i][j]))
                probas_targets.append(outputs[j][0][index])
            # bit of an ugly line. Just prints each target action with the corresponding output probability
            print("Probability for each successive target action: {0}".format(
                [list(a) for a in zip(self.output_sequences_strings[i],
                                      ["{0:.4f}".format(proba) for proba in probas_targets])]))
            # Everything
            if verbose:
                print("Detailed results:")
                for j in range(len(outputs)):
                    print("target:{0}".format(self.output_sequences_one_hot[i][j]))
                    print("output:{0}".format(outputs[j]))

            total_error = 0.
            for j in range(len(outputs)):
                total_error += sum(np.abs(self.output_sequences_one_hot[i][j][0] - outputs[j][0]))
            accuracy = (48-total_error)/48.
            accuracy_per_sequence.append(accuracy)
            print("Total loss (absolute):{0:.4f} (% accuracy: {1:.4f})".format(total_error, accuracy))
        # There's 48 outputs in total for a sequence (6 * 8). Of these, two incur a total error of at least 1.
        # (In expectation, because 2 options are equally likely to be right)
        print("\nAverage accuracy (all seqs.): {0:.4f} (Max accuracy: {1:.4f})".format(sum(accuracy_per_sequence)/4.,
                                                                                           (48.-2)/48.))

    def generate_rdm(self, error_type, distance_type):
        # 1. For each sequence, get the context values.
        self.learning_rate = 0.
        _, contexts, _ = self.run_on_all_sequences(error_type)
        # Now flatten the contexts list. It's a list of lists of 2D-ndarrays; we want a flat list of 1D-ndarrays
        all_contexts = self.flatten_holroyd_data(contexts)

        # 2. For each context, measure the distance with each other context
        distance_matrix = np.zeros(shape=[24, 24])
        for i, context1 in enumerate(all_contexts):
            for j, context2 in enumerate(all_contexts):
                if distance_type == "euclidian":
                    distance_matrix[i, j] = np.linalg.norm(context1-context2)  # euclidian distance
                elif distance_type == "spearman_rho":
                    rho, _ = stats.spearmanr(context1, context2)
                    distance_matrix[i, j] = 1. - rho
        return distance_matrix


def main():
    verbose = False
    error_type = "mse"  # mse or cross_entropy
    num_networks = 10
    num_accurate_networks = 0
    training_steps_per_network = 10000

    spearman_matrix = np.zeros([24, 24])
    euclidian_matrix = np.zeros([24, 24])
    spearman_matrix_s = np.zeros([24, 24])
    euclidian_matrix_s = np.zeros([24, 24])
    for i in range(num_networks):
        print("\nTraining starts for network {}".format(i+1))
        running_avg_loss = 0.
        holroyd_net = Holroyd2018Network()
        print("Iteration\tLoss")
        for j in range(training_steps_per_network):
            idx = np.random.randint(4)
            loss, _, _ = holroyd_net.full_sequence(holroyd_net.input_sequences_one_hot[idx],
                                                   holroyd_net.output_sequences_one_hot[idx],
                                                   error_type)
            if j == 0:
                running_avg_loss = loss
            elif j < 10:
                running_avg_loss = running_avg_loss * 0.5 + 0.5 * loss
            elif j < 100:
                running_avg_loss = running_avg_loss * 0.8 + 0.2 * loss
            else:
                running_avg_loss = running_avg_loss * 0.99 + 0.01 * loss

            if j == 0 or (j+1) % 1000 == 0 or (j < 1000 and math.log(j+1, 4) % 1 == 0):
                print("{0:5d}:\t\t{1:8.4f}".format(j+1, running_avg_loss))

        # Freeze the network
        holroyd_net.learning_rate = 0.
        # holroyd_net.show_results(error_type, verbose=False)
        spearman_matrix += holroyd_net.generate_rdm(error_type, distance_type="spearman_rho")
        euclidian_matrix += holroyd_net.generate_rdm(error_type, distance_type="euclidian")
        if holroyd_net.test_accuracy(error_type):
            print(colored("Accuracy test for network {}: PASSED".format(i+1), 'green'))
            spearman_matrix_s += holroyd_net.generate_rdm(error_type, distance_type="spearman_rho")
            euclidian_matrix_s += holroyd_net.generate_rdm(error_type, distance_type="euclidian")
            num_accurate_networks += 1
        else:
            print(colored("Accuracy test for network {}: FAILED".format(i+1), 'red'))

    if num_accurate_networks == 0:
        print("Not a single network was accurate... Exiting")
        return
    else:
        print("Number of accurate networks out of 100: {0}".format(num_accurate_networks))

    # Generate Spearman and Euclidian matrices, one for all networks, one for only the accurate ones
    spearman_matrix = spearman_matrix / num_networks
    euclidian_matrix = euclidian_matrix / num_networks
    spearman_matrix_s = spearman_matrix_s / num_accurate_networks
    euclidian_matrix_s = euclidian_matrix_s / num_accurate_networks

    net = Holroyd2018Network() # This is just to have access to the input sequences strings. Not elegant but... oh well
    x_labels = utils.flatten_list_of_lists(net.input_sequences_strings)
    y_labels = x_labels  # labels for y-axis

    sns.heatmap(euclidian_matrix, cbar=True, square=True, xticklabels=x_labels, yticklabels=y_labels)
    plt.title("Euclidian RDM, all nets")
    plt.show()

    sns.heatmap(euclidian_matrix_s, cbar=True, square=True, xticklabels=x_labels, yticklabels=y_labels)
    plt.title("Euclidian RDM, only accurate nets")
    plt.show()

    sns.heatmap(spearman_matrix, cbar=True, square=True, xticklabels=x_labels, yticklabels=y_labels, vmin=0., vmax=1.5)
    plt.title("Spearman RDM, all nets")
    plt.show()

    sns.heatmap(spearman_matrix_s, cbar=True, square=True, xticklabels=x_labels, yticklabels=y_labels,
                vmin=0., vmax=1.5)
    plt.title("Spearman RDM, only accurate nets")
    plt.show()

    # export data to csv
    np.savetxt("euclidian_dist_matrix.csv", euclidian_matrix, delimiter=",")
    np.savetxt("spearmanrho_dist_matrix.csv", spearman_matrix, delimiter=",")
    np.savetxt("euclidian_dist_matrix_s.csv", euclidian_matrix_s, delimiter=",")
    np.savetxt("spearmanrho_dist_matrix_s.csv", spearman_matrix_s, delimiter=",")


if __name__ == "__main__":
    main()
