# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import teacoffeeenv as tce
import scripts
import neuralnet as nn
import utils
import pnas2018
import sys
import goalenv
import prederror_task
import timeit
import tensorflow as tf
import numpy as np
import analysis

"""
mat1 = pnas2018.make_rdm_multiple_predictive("pnas_predictivenet_rmsprop005_h25", 50,
                                                   title="pnas_predictivenet_rmsprop005_h25_20201015", save_files=True)

mat2 = prederror_task.make_rdm_multiple_predictive("prederorr_predictivenet_rmsprop001", 50,
                                                   title="prederorr_predictivenet_rmsprop005_h25_20201015", save_files=True)
"""

#sys.exit("DONE THANK U")
#start = timeit.default_timer()


with tf.device('/cpu:0'):
    for i in range(100):
        print(i+1)
        model = prederror_task.train_predictive_net(iterations=15000, algorithm=nn.RMSPROP, learning_rate=0.005, hidden_units=25)
        prederror_task.accuracy_test_predictive(model)
        utils.save_object("prederorr_predictivenet_rmsprop005_h25", model)
    for i in range(50):
        print(i + 1)
        model = pnas2018.train_predictive_net(iterations=15000, algorithm=nn.RMSPROP, learning_rate=0.005, hidden_units=25)
        pnas2018.accuracy_test_predictive(model)
        utils.save_object("pnas_predictivenet_rmsprop005_h25", model)

with tf.device('/gpu:0'):

    goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.00001, sequences=[0])
    goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.00001, sequences=[0, 1])
    goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.00001, sequences=[0, 1, 2, 3])
    goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.00001, sequences=[0, 1, 2, 3, 4, 5])

    stop = timeit.default_timer()
    #print('Time: ', stop - start)

    sys.exit("DONE THANK U")

env = goalenv.GoalEnv()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.001, L2_reg=0.00001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

env = goalenv.GoalEnv()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0001, L2_reg=0.000001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

