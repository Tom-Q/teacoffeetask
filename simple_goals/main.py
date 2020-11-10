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
mat1 = pnas2018.make_rdm_multiple_predictive("pnas_predictivenet_rmsprop005_h25", 100,
                                                   title="pnas_predictivenet_rmsprop005_h25_20201015", save_files=True)

mat2 = prederror_task.make_rdm_multiple_predictive("prederorr_predictivenet_rmsprop005_h25", 100,
                                                   title="prederorr_predictivenet_rmsprop005_h25_20201015", save_files=True)

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
"""
#pnas2018.train()

#for i in range(25):
#    print(i + 1)
#    model = pnas2018.train_with_goals(iterations=5000, reg_strength=0.01)
#    pnas2018.accuracy_test_with_goals(model)
#    utils.save_object("pnas_regularization_regstr_0", model)
#mat2 = pnas2018.make_rdm_multiple("pnas_regularization_regstr_0", 25, with_goals=True, title="pnas_regularization_test_noreg")


for i in range(25):
    print(i + 1)
    model = pnas2018.train_hierarchical(iterations=8000, reg_strength=0.001, reg_increase="linear")
    pnas2018.accuracy_test_reg_hierarchy(model)
    utils.save_object("pnas_regstr_0_001_lnr", model)

mat2 = pnas2018.make_rdm_multiple_hierarchy("pnas_regstr_0_001_sqr", 25, title="pnas_regstr_0_0001_sqr")


#print(model.hidden_layer.w)

#utils.test_weight_regularization()
#sys.exit()

#env = goalenv.GoalEnv()
#env.test_environment()

#goalenv.train(goals=False, num_iterations=100000, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1])
#goalenv.train(goals=False, num_iterations=50000, learning_rate=0.001, L2_reg=0.00001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
#goalenv.train(goals=False, num_iterations=50000, learning_rate=0.01, L2_reg=0.00001, noise=0., sequences=[0, 1])


#goalenv.train(goals=False, num_iterations=100000, learning_rate=0.0003, L2_reg=0.0001, noise=0., sequences=[0, 1, 2, 3])
#goalenv.train(goals=False, num_iterations=100000, learning_rate=0.0003, L2_reg=0.0001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
#goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.0001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
#goalenv.train(goals=False, num_iterations=50000, learning_rate=0.01, L2_reg=0.0001, noise=0., sequences=[0, 1])
#goalenv.train(goals=False, num_iterations=50000, learning_rate=0.1, L2_reg=0.0001, noise=0., sequences=[0, 1])
#goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0001, L2_reg=0.000001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

