import tensorflow as tf
import numpy as np

# Inefficient, but will be fast enough so long as we're not dealing with massive amounts of data
def one_hot_encoding(categories, sequence):
    """
    :param categories: categories to code into
    :param sequence: sequence. Each element in the sequence must match one of the categories
    :return: one-hot encoding as a row vector (shape [1, num_categories])
    """
    num_categories = len(categories)
    idx_of_cat = dict(zip(categories, range(num_categories)))
    one_hot = []
    for element in sequence:
        encoded = np.zeros((1, num_categories))
        encoded[0, idx_of_cat[element]] = 1.
        one_hot.append(encoded)
    return one_hot


def numpy_winner_take_all(x):
    """
    :param x: a vector or matrix
    :return: a one hot vector or matrix where the maximum value for every row is 1.
    """
    y = np.zeros_like(x)
    y[np.argmax(x)] = 1.
    return y

def flatten_list_of_lists(list_of_lists):
    """
    :param list_of_lists: a list of lists
    :return: a single flat list
    """
    return [item for sublist in list_of_lists for item in sublist]

def dense_linear(x, W, b=None):
    """
     :param x: input
     :param W: weights
     :param b: bias
     :return: densely connected layer with no activation function
     """
    preactivation = tf.matmul(x, W)
    if b is not None:
        preactivation += b
    return preactivation


def dense_softmax(x, W, b=None):
    """
    :param x: input
    :param W: weights matrix
    :param b: bias weights
    :return: densely connected layer with softmax output
    """
    return tf.nn.softmax(dense_linear(x, W, b))

def dense_sigmoid(x, W, b=None):
    """
    :param x: input
    :param W: weights matrix
    :param b: bias weights
    :return: densely connected layer with sigmoid output
    """
    return tf.nn.sigmoid(dense_linear(x, W, b))
