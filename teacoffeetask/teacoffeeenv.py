from abc import ABC
import utils
import tensorflow as tf
import numpy as np


class holroyd2018():
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
        self._w_hidden = tf.Variable(np.random.normal(0., 1., size=[self._input_size + self._hidden_size, self._hidden_size]))
        self._w_output = tf.Variable(np.random.normal(0., 1., size=[self._hidden_size, self._output_size]))
        self._b_hidden = tf.Variable(np.zeros([self._hidden_size, 1]))
        self._b_output = tf.Variable(np.zeros([self._output_size, 1]))
        self._all_weights = [self._w_hidden, self._b_hidden, self._w_output, self._b_output]



    def forward_one_action(self, network_input, previous_hidden):
        """
        :param network_input: input (previous action or start)
        :param previous_hidden: context (0 at start, otherwise previous context)
        :return: logits for the action, context layer activations
        """
        input_vec = tf.concat([network_input, previous_hidden], 1)
        hidden_activation = utils.dense_sigmoid(input_vec,
                                               self._w_hidden,
                                               tf.transpose(self._b_hidden))
        logits = utils.dense_linear(hidden_activation,
                                   self._w_output,
                                   tf.transpose(self._b_output))
        return logits, hidden_activation

    def full_sequence(self, input_sequence, output_sequence):
        """
        :param input_sequence: one-hot encoded
        :param output_sequence: one-hot encoded
        :return: loss, action probabilities
        """
        with tf.GradientTape() as tape:
            context_activation = np.zeros([1, self._hidden_size])
            outputs = []
            # Do the forward pass
            for step in range(len(input_sequence)):
                logits, context_activation = self.forward_one_action(input_sequence[step], context_activation)
                outputs.append(logits)

            # Compute the error
            loss = 0.
            for i in range(len(outputs)):
                loss += tf.nn.softmax_cross_entropy_with_logits(output_sequence[i], outputs[i])

            gradients = tape.gradient(loss, self._all_weights)
            for i in range(len(self._all_weights)):
                self._all_weights[i].assign_sub(gradients[i] * self.learning_rate)

            # return the activation probabilities, not the logits
            for idx, logits in enumerate(outputs):
                outputs[idx] = tf.exp(logits) / tf.reduce_sum(tf.exp(logits))

            return loss, outputs


def main():
    verbose = False
    holroyd_run = holroyd2018()
    running_avg_loss = 0.
    for i in range(5000):
        idx = np.random.randint(4)
        loss, _ = holroyd_run.full_sequence(holroyd_run.input_sequences_one_hot[idx],
                                            holroyd_run.output_sequences_one_hot[idx])
        if i == 0:
            running_avg_loss = loss
        elif i < 50:
            running_avg_loss = running_avg_loss * 0.8 + 0.2 * loss
        else:
            running_avg_loss = running_avg_loss * 0.99 + 0.01 * loss

        if i == 0 or (i+1) % 1000 == 0 or np.log2(i+1) % 1 == 0:
            tf.print("Loss (running average) at iteration {0}: {1}".format(i+1, running_avg_loss))

    # Now show what happens.
    holroyd_run.learning_rate = 0.
    for i in range(len(holroyd_run.input_sequences_one_hot)):
        print("\n##############\n# SEQUENCE {0} #\n##############".format(i+1))
        _, probas = holroyd_run.full_sequence(holroyd_run.input_sequences_one_hot[i],
                                              holroyd_run.output_sequences_one_hot[i])
        # Probability of activation only for target actions
        probas_targets = []
        for j in range(len(probas)):
            index = int(np.argmax(holroyd_run.output_sequences_one_hot[i][j]))
            probas_targets.append(probas[j].numpy()[0][index])
        # bit of an ugly line. Just prints each target action with the corresponding output probability
        print("Probability for each successive target action: {0}".format([list(a) for a in zip(holroyd_run.output_sequences_strings[i],
                                                                              ["{0:.2f}".format(proba) for proba in probas_targets])]))

        # Everything
        if verbose:
            print("Detailed results:")
            for j in range(len(probas)):
                print("target:{0}".format(holroyd_run.output_sequences_one_hot[i][j]))
                print("output:{0}".format(probas[j]))


if __name__ == "__main__":
    main()
