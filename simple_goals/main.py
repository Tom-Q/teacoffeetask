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


# Simulation 7
for i in range(10):
    print(i + 1)
    model = pnas2018.train(iterations=5000)
    pnas2018.accuracy_test(model)
    utils.save_object("pnas_distribution_test", model)
pnas2018.make_rdm_multiple("pnas_distribution_test", 10)
"""
mat2 = pnas2018.make_rdm_multiple_hierarchy_nogoals("pnas_regstr_0_001_sqr_nogoals_2", 25, title="pnas_regstr_0_001_sqr_nogoals", file_save_name="mid2")
mat2 = pnas2018.make_rdm_multiple_hierarchy_nogoals("pnas_regstr_0_001_sqr_nogoals_2", 25, title="pnas_regstr_0_001_sqr_nogoals", cutoff=3, file_save_name="32")
mat2 = pnas2018.make_rdm_multiple_hierarchy_nogoals("pnas_regstr_0_001_sqr_nogoals_2", 25, title="pnas_regstr_0_001_sqr_nogoals", cutoff=5, file_save_name="52")
mat2 = pnas2018.make_rdm_multiple_hierarchy_nogoals("pnas_regstr_0_001_sqr_nogoals_2", 25, title="pnas_regstr_0_001_sqr_nogoals", cutoff=9, file_save_name="92")
mat2 = pnas2018.make_rdm_multiple_hierarchy_nogoals("pnas_regstr_0_001_sqr_nogoals_2", 25, title="pnas_regstr_0_001_sqr_nogoals", cutoff=12, file_save_name="122")
"""
"""
# Simulation 6
for i in range(25):
    print(i + 1)
    model = pnas2018.train_hierarchical(iterations=8000, reg_strength=0.001, reg_increase="lnr")
    pnas2018.accuracy_test_reg_hierarchy(model)
    utils.save_object("pnas_regstr_0_001_lnr_obs", model)
mat2 = pnas2018.make_rdm_multiple_hierarchy("pnas_regstr_0_001_lnr_obs", 25, title="pnas_regstr_0_001_lnr_obs")

# Simulation 1
for i in range(0):
    print(i + 1)
    model = pnas2018.train_hierarchical(iterations=8000, reg_strength=0.003, reg_increase="linear")
    pnas2018.accuracy_test_reg_hierarchy(model)
    utils.save_object("pnas_regstr_0_01_lnr", model)
mat2 = pnas2018.make_rdm_multiple_hierarchy("pnas_regstr_0_01_lnr", 101, title="pnas_regstr_0_003_lnr",
                                            file_save_name="pnas_regstr_0_03_lnr")

# Simulation 2
for i in range(0):
    print(i + 1)
    model = pnas2018.train_hierarchical(iterations=8000, reg_strength=0.001, reg_increase="linear")
    pnas2018.accuracy_test_reg_hierarchy(model)
    utils.save_object("pnas_regstr_0_001_lnr", model)
mat2 = pnas2018.make_rdm_multiple_hierarchy("pnas_regstr_0_001_lnr", 100, title="pnas_regstr_0_001_lnr")

#Simulation 3
for i in range(0):
    print(i + 1)
    model = pnas2018.train_hierarchical(iterations=8000, reg_strength=0.0001, reg_increase="linear")
    pnas2018.accuracy_test_reg_hierarchy(model)
    utils.save_object("pnas_regstr_0_0001_lnr", model)
mat2 = pnas2018.make_rdm_multiple_hierarchy("pnas_regstr_0_0001_lnr", 100, title="pnas_regstr_0_0001_lnr")

# Simulation 4
for i in range(0):
    print(i + 1)
    model = pnas2018.train_hierarchical(iterations=8000, reg_strength=0.001, reg_increase="square")
    pnas2018.accuracy_test_reg_hierarchy(model)
    utils.save_object("pnas_regstr_0_001_sqr", model)
mat2 = pnas2018.make_rdm_multiple_hierarchy("pnas_regstr_0_001_sqr", 100, title="pnas_regstr_0_001_sqr")
sys.exit("done")
# Simulation 5
for i in range(0):
    print(i + 1)
    model = pnas2018.train_hierarchical(iterations=8000, reg_strength=0.0001, reg_increase="square")
    pnas2018.accuracy_test_reg_hierarchy(model)
    utils.save_object("pnas_regstr_0_0001_sqr", model)
mat2 = pnas2018.make_rdm_multiple_hierarchy("pnas_regstr_0_0001_sqr", 100, title="pnas_regstr_0_0001_sqr")

"""



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

