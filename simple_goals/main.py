# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import utils
import sys
from neural import neuralnet as nn, optimizers, layers
import analysis
import goalenv
from goalenv import environment as env, goalenv2020
import tensorflow as tf
import scipy
import numpy as np

utils.initialize_random_seeds(1)



