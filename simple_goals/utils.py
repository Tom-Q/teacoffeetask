# A collection of useful snippets of code
import numpy as np
from datetime import datetime
import pickle
import os
import re
import tensorflow as tf
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import analysis

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


def rolling_avg(prev_avg, new_val, speed):
    if not (speed > 0. and speed <= 1.):
        raise ValueError("Should have 0 < speed < 1 for a rolling average, you set speed={0}".format(speed))
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
        raise Exception("You asked for file # {0} ("+name+") but there are only {1} matching files".format(latest+1, len(files)))

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

def dense_softmax(x, w, b):
    """
    :param x: input
    :param layer: Layer object with weights matrix [+ bias]
    :return: densely connected layer with softmax output
    WARNING: DONT USE THIS METHOD FOR A LAST LAYER. TRAINING ASSUMES YOURE WORKING WITH LOGITS INSTEAD.
    """
    return tf.nn.softmax(dense_linear(x, w, b))


# Pretty useless
def dense_sigmoid(x, w, b):
    """
    :param x: input
    :param layer: Layer object with weights matrix [+ bias]
    :return: densely connected layer with sigmoid output
    """
    return tf.nn.sigmoid(dense_linear(x, w, b))


def winner_take_all(a):
    # Tensorflow implementation. Actually slower, as is
    cond = tf.equal(a, tf.reduce_max(a))
    a_wta = tf.where(cond, tf.ones_like(a), tf.zeros_like(a))
    return a_wta


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

def save_rdm(rdm, name, labels, title):
    np.savetxt(name+"_rdm_mat.txt", rdm, delimiter="\t", fmt='%.2e')
    analysis.plot_rdm(rdm, labels, title + " spearman rho matrix", figsize=30, fontsize=0.6)
    plt.savefig(name+'_rdm.png', dpi=300, bbox_inches='tight')
    plt.clf()

def weight_regularization_calculator(weight_matrix, index_in, index_out, reg_const, reg_type="step",
                                     reg_increase="linear"):
    if reg_type == "step":
        # Extract the relevant area of the weight matrix:
        return tf.reduce_sum(
            tf.abs(weight_matrix[index_in[0]: index_in[1], index_out[0]: index_out[1]])) * reg_const
    elif reg_type == "recurrent":
        # 1 Make matrix:
        weights = weight_matrix[index_in[0]: index_in[1], index_out[0]: index_out[1]]
        mat = np.zeros_like(weights)
        num_rows, num_cols = mat.shape
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = i - j
        # print(mat)
    elif reg_type == "input_left":
        weights = weight_matrix[index_in[0]: index_in[1], index_out[0]: index_out[1]]
        # print(weights)
        mat = np.zeros_like(weights)
        num_rows, num_cols = mat.shape
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = j
        # print(mat)
    elif reg_type == "input_right":
        weights = weight_matrix[index_in[0]: index_in[1], index_out[0]: index_out[1]]
        # print(weights)
        mat = np.zeros_like(weights)
        num_rows, num_cols = mat.shape
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = num_cols - j - 1
        # print(mat)
    elif reg_type == "output_left":
        weights = weight_matrix[index_in[0]: index_in[1], index_out[0]: index_out[1]]
        # print(weights)
        mat = np.zeros_like(weights)
        num_rows, num_cols = mat.shape
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = i
        # print(mat)
    elif reg_type == "output_right":
        weights = weight_matrix[index_in[0]: index_in[1], index_out[0]: index_out[1]]
        # print(weights)
        mat = np.zeros_like(weights)
        num_rows, num_cols = mat.shape
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = (num_rows - i - 1)
        # print(mat)
    else:
        raise ValueError("reg_type not implemented")
    if reg_increase == "linear":
        mat = np.abs(mat)
    elif reg_increase == "square":
        mat = mat ** 2
    else:
        raise ValueError("reg_increase must be either linear or square")
    return tf.reduce_sum(tf.abs(weights) * mat * reg_const)


#mat = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape([3,3])
#vector = np.asarray([0, 1 ,2])
#print(np.matmul(vector, mat))
def test_weight_regularization(regtype):
    matrix = np.round(np.random.uniform(-1, 1, [5, 7])) * 2
    print(matrix)
    print(weight_regularization_calculator(matrix, [0, 3], [0, 2], 1, regtype))

