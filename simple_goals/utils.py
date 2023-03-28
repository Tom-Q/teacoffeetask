# A collection of useful snippets of code
import numpy as np
import random
from datetime import datetime
import pickle
import os
import re
import tensorflow as tf
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import analysis
import copy

def initialize_random_seeds(seed):
    np.random.seed(seed)  # Note: This must be done before any other random calls are made.
    random.seed(seed)

SAVE_FOLDER='models'

def idx_from_probabilities(probabilities):
    if sum(probabilities) != 1:
        raise Warning("Probabilities don't add up to 1")
    decider = np.random.uniform()
    cumsum = 0
    for idx, probability in enumerate(probabilities):
        cumsum += probability
        if cumsum >= decider:
            return idx

def dic_to_vec(dic, keys):
    vec = np.zeros((1, len(keys)), dtype=np.float32)
    for idx, key in enumerate(keys):
        vec[0, idx] = dic[key]
    return vec


def str_to_onehot(str_key, keys):
    onehot = np.zeros((1, len(keys)), dtype=np.float32)
    onehot[0, keys.index(str_key)] = 1.
    return onehot


def liststr_to_onehot(list, keys):
    onehot = np.zeros((len(list), len(keys)), dtype=np.float32)
    for idx, key in enumerate(list):
        onehot[idx, keys.index(key)] = 1.
    return onehot


def onehot_to_str(onehot, ordered_keys):
    index = np.argmax(onehot)
    return ordered_keys[index]

def str_to_multihot(str_keys, ordered_keys):
    multihot = np.zeros((1, len(ordered_keys)), dtype=np.float32)
    for key in str_keys:
        for key in str_keys:
            multihot[0, ordered_keys.index(key)] = 1.
    return multihot


def rolling_avg(prev_avg, new_val, speed, num=None):
    if not (speed > 0. and speed <= 1.):
        raise ValueError("Should have 0 < speed < 1 for a rolling average, you set speed={0}".format(speed))
    if num is None:
        return prev_avg * (1-speed) + new_val * speed
    else:  # average of values so far with more weight on latest data.
        if num == 0:
            return new_val
        elif 1./num < speed:
            return prev_avg * (1-1./num) + new_val * 1./num
        else:
            return prev_avg * (1-speed) + new_val * speed


def save_object(name, object):
    """
    :param name: the file will be named [name]timestamp
    :param object: the object to save
    """
    if not os.path.exists('models'):
        os.makedirs('models')
    filehandler = open(SAVE_FOLDER + '/' + name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"), 'wb')
    pickle.dump(object, filehandler)
    filehandler.close()


def load_object(name, latest=0):
    """
    :param name: will load a file with a name like [name]_timestamp where the timestamp is in the format YYYYMMDD_HHMMSS
    :param latest: 0 for the latest file, 1 for 1 before last, etc.
    :return: the object (=typically a neural net model) encoded in that file
    """
    if not os.path.exists(SAVE_FOLDER):
        raise Exception('Folder ' + SAVE_FOLDER + 'does not exist; no files found.')
    # Load all files that match the name

    file_names = [fn for fn in os.listdir(SAVE_FOLDER) if re.search('^'+name+'_[0-9]{8}_[0-9]{6}', fn) is not None]
    # Since the timestamps are in the format YYYYMMDD_HHMMSS, an alphabetical sort will sort them chronologically
    files = sorted(file_names)
    if latest >= len(files):
        raise Exception("You asked for file # {0} ({1}) but there are only {2} matching files".format(latest+1, name, len(files)))

    filehandler = open(SAVE_FOLDER + '/' + files[-1 - latest], 'rb')
    object = pickle.load(filehandler)
    filehandler.close()
    return object


def dense_linear(x, w, b):
    """
    :param x: input
    :param layer: Layer object with weights matrix [+ bias]
    :return: densely connected layer with no activation function
    """
    preactivation = tf.matmul(x, w)
    if b is not None:
        preactivation += b
    return preactivation


UNIFORM = "uniform"
NORMAL = "normal"
XAVIER = "xavier"
HE = "he"
SEMINORMAL = "seminormal"
ZERO_INIT = "zero_init"

def initialize(dimensions, mode):
    if mode == UNIFORM:
        return uniform_initialization(dimensions)
    elif mode == NORMAL:
        return normal_initialization(dimensions)
    elif mode == XAVIER:
        return xavier_initialization(dimensions)
    elif mode == HE:
        return he_initialization(dimensions)
    elif mode == SEMINORMAL:
        return seminormal_initialization(dimensions)
    elif mode == ZERO_INIT:
        return zero_initialization(dimensions)
    else:
        return NotImplementedError("unknown/not implemented initialization mode")

def zero_initialization(dimensions):
    return np.zeros(dimensions)

def seminormal_initialization(dimensions):
    return np.abs(np.random.normal(0., .1, size=dimensions))

def normal_initialization(dimensions):
    return np.random.normal(0., .1, size=dimensions)

def uniform_initialization(dimensions):
    return np.random.uniform(-1, .1, size=dimensions)

# normalized
def xavier_initialization(dimensions):
    # higher and lowre bound hidden layer
    hb = np.sqrt(6 / np.sqrt(dimensions[0] + dimensions[1]))
    lb = -hb
    return np.random.uniform(lb, hb, size=dimensions)

def he_initialization(dimensions):
    # higher and lowre bound hidden layer
    if dimensions[0] * dimensions[1] != 0:
        hb = 2. / np.sqrt(dimensions[1])
        lb = -hb
    else:  # a zero-sized layer is easier to manipulate than dealing with special cases
        hb = lb = 0.
    return np.random.uniform(lb, hb, size=dimensions)


def load_objects(name, number):
    objects = []
    for i in range(number):
        objects.append(load_object(name, latest=i))
    return objects


def flatten_onelevel(list):
    # Flattens the top 2 levels
    return [item for sublist in list for item in sublist]


def datestr():
    return '_' + datetime.today().strftime('%Y%m%d')


# Single use progress bar
class ProgressBar(object):
    def __init__(self):
        # percentages at which to update progress
        self.updates = [0., 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98, 99, 100]
        self.update_counter = 0
        self.start_time = None

    # Print iterations progress (from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console)
    def updateProgress (self, iteration, total, prefix='Progress', suffix='complete', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        if iteration == 0:
            self.start_time = timer()
        ETC = '' #Estimated Time to Completion
        if (iteration/total)*100 >= self.updates[self.update_counter]:
            elapsed = timer() - self.start_time
            if iteration != 0:
                minutes = int((elapsed * total/iteration - elapsed)//60)
                seconds = int((elapsed * total/iteration - elapsed)%60)
                ETC = "(~{:d} mins {:d}s left)".format(minutes, seconds)
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            # Unfortunately \r doesn't work in the pycharm console, so we have to reprint the whole bar everytime,
            # clogging the console.
            #print(f'\r{prefix} |{bar}| {percent}% {suffix} {ETC}', end = printEnd)
            print(f'{prefix} |{bar}| {percent}% {suffix} {ETC}')
            # Print New Line on Complete
            if iteration == total:
                print()
            self.update_counter += 1



# Changes both rows and columns according to the new order.
# This assumes we're working with an RDM (symmetric etc.)
# Example:
# Matrix:
# 1 2 3 4
# 2 1 2 3
# 3 2 1 2
# 4 3 2 1
# new order format: [0, 1, 3, 2]
# Reordered matrix:
# 1 2 4 3
# 2 1 3 2
# 4 3 1 2
# 3 2 2 1

def reorder_matrix(matrix, new_order):
    copy_mat = copy.deepcopy(matrix)
    for i, j in enumerate(new_order):
        copy_mat[i, :] = matrix[j, :]
    copy_mat2 = copy.deepcopy(copy_mat)
    for i, j in enumerate(new_order):
        copy_mat2[:, i] = copy_mat[:, j]
    return copy_mat2

def reorder_list(list, new_order):
    copy_list = copy.deepcopy(list)
    for i, j in enumerate(new_order):
        copy_list[i] = list[j]
    return copy_list


def weight_regularization_calculator(weight_matrix, index_in, index_out, reg_const, reg_type="step",
                                     reg_increase="linear", middle=.5):
    # The section of the weights that we care about, specifically
    weights = weight_matrix[index_in[0]: index_in[1], index_out[0]: index_out[1]]
    mat = np.zeros_like(weights)
    num_rows, num_cols = mat.shape
    if reg_type == "step":
        # Extract the relevant area of the weight matrix:
        return tf.reduce_sum(
            tf.abs(weight_matrix[index_in[0]: index_in[1], index_out[0]: index_out[1]])) * reg_const
    elif reg_type == "recurrent":
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = i - j

    elif reg_type == "input_left":
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = j

    elif reg_type == "input_right":
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = num_cols - j - 1

    elif reg_type == "output_left":
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = i

    elif reg_type == "output_right":
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = (num_rows - i - 1)

    elif reg_type == "output_middle":
        for i in range(0, int(num_rows * middle)):
            for j in range(num_cols):
                mat[i, j] = num_rows*middle - i - 1
        for i in range(int(num_rows * middle), num_rows):
            for j in range(num_cols):
                mat[i, j] = i - num_rows * middle

    elif reg_type == "input_middle":
        for i in range(0, num_rows):
            for j in range(0, int(num_cols * middle)):
                mat[i, j] = num_cols * middle - j - 1
            for j in range(int(num_cols * middle), num_cols):
                mat[i, j] = j - num_cols * middle
    else:
        raise ValueError("reg_type not implemented")
    if reg_increase == "linear":
        mat = np.abs(mat)
    elif reg_increase == "square":
        mat = mat ** 2
    else:
        raise ValueError("reg_increase must be either linear or square")

    #print(reg_type)
    #print(mat)
    return tf.reduce_sum(tf.abs(weights) * mat * reg_const)


#mat = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape([3,3])
#vector = np.asarray([0, 1 ,2])
#print(np.matmul(vector, mat))
def test_weight_regularization(regtype):
    matrix = np.round(np.random.uniform(-1, 1, [5, 7])) * 2
    print(matrix)
    print(weight_regularization_calculator(matrix, [0, 3], [0, 2], 1, regtype))


# Writes data (in list format) to file (in csv format)
def write_lists_to_csv(file_name, listoflists, labels=None):
    with open(file_name, 'a') as the_file:
        if labels is not None:
            listoflists = [labels] + listoflists
        for datarow in listoflists:
            _write_line(the_file, datarow)

def write_line_to_csv(file_name, datalist):
    with open(file_name, 'a') as the_file:
        _write_line(the_file, datalist)

def _write_line(the_file, datalist):
    for element in datalist:
        the_file.write(str(element) + ';')
    the_file.write('\n')