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

start = timeit.default_timer()

with tf.device('/cpu:0'):
    """
    #env = goalenv.GoalEnv()
    goalenv.train(goals=False, num_iterations=200000, learning_rate=0.001, L2_reg=0.00001,
                  sequences=[5])
    #env = goalenv.GoalEnv()
    goalenv.train(goals=False, num_iterations=200000, learning_rate=0.001, L2_reg=0.00001,
                  sequences=[0, 1, 2, 3, 4, 5])
    env = goalenv.GoalEnv()
    goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0001, L2_reg=0.00001,
                  sequences=[0, 1, 2, 3, 4, 5])
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    sys.exit("PROGRAM END")
    """

    for i in range(1):
        print(i+1)
        model = prederror_task.train(iterations=5000, algorithm=nn.RMSPROP, learning_rate=0.005, l2reg=0.0, hidden_units=15)
        prederror_task.accuracy_test(model)
        utils.save_object("prederror_rmsprop001_nogoals", model)

    for i in range(1):
        print(i+1)
        model = prederror_task.train_with_goals(iterations=5000, algorithm=nn.RMSPROP, learning_rate=0.005, l2reg=0.0, hidden_units=15)
        prederror_task.accuracy_test_with_goals(model)
        utils.save_object("prederror_rmsprop001_goals", model)

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    mat1 = prederror_task.make_rdm_multiple("prederror_rmsprop001_nogoals", 1, with_goals=False, title="cerms_prederror_2020-10-09", save_files=True)
    mat2 = prederror_task.make_rdm_multiple("prederror_rmsprop001_goals", 1, with_goals=True, title="cerms_prederror_goals_2020-10-09", save_files=True)
    #print(analysis.compare_matrices(mat1, mat2))
    sys.exit("DONE THANK U")

env = goalenv.GoalEnv()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.001, L2_reg=0.000001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

env = goalenv.GoalEnv()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0001, L2_reg=0.0000001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

