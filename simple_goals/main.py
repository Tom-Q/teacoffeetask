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

start = timeit.default_timer()
with tf.device('/cpu:0'):
    for i in range(100):
        print(i+1)
        model = prederror_task.train(iterations=3000, algorithm=nn.RMSPROP, learning_rate=0.03, l2reg=0.0, hidden_units=15)
        prederror_task.accuracy_test(model)
        #utils.save_object("prederror_mse_goalsa", model)


    prederror_task.make_rdm_multiple("prederror_cenogoals003reg", 100, with_goals=False, title="ce_prederror_2020-10-06")
    #prederror_task.make_rdm_multiple("prederror_cerms_goalsa", 100, with_goals=True, title="cerms_prederror_goals_2020-10-06")


stop = timeit.default_timer()
print('Time: ', stop - start)
sys.exit("end of program")

env = goalenv.GoalEnv()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0005, L2_reg=0.0000005, noise=0., sequences=[0, 1, 2, 3, 4, 5])

env = goalenv.GoalEnv()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.001, L2_reg=0.000001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

env = goalenv.GoalEnv()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0001, L2_reg=0.0000001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

