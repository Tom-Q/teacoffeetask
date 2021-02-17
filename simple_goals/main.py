# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
from goalenv import goalenv2020, environment, task
from pnas import pnas2018
from pnas import pnashierarchy
import neuralnet
import utils
import sys

import hyperparamstest

hyperparamstest.hyperparameter_analysis()
sys.exit()
"""
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

"""import analysis
for i in range(5):
    print(i)
    model, _ = pnas2018.train(noise=0.15)
    utils.save_object("rdm_test015", model)
for i in range(5):
    print(i)
    model, _ = pnas2018.train(noise=0.2)
    utils.save_object("rdm_test02", model)

#for i in range(78):
for i in range(0):
    print(i)
    model, _ = pnas2018.train(iterations=8000, noise=0.09)
    utils.save_object("rdm_test015", model)

pnas2018.make_rdm_noisy("rdm_test015", 40, 0.09, num_runs_per_network=100)
sys.exit()

pnas2018.make_rdm_multiple("rdm_test", 100, with_goals=False, rdm_type=analysis.MAHALANOBIS)
pnas2018.make_rdm_multiple("rdm_test", 100, with_goals=False, rdm_type=analysis.SPEARMAN)
pnas2018.make_rdm_multiple("rdm_test", 100, with_goals=False, rdm_type=analysis.EUCLIDIAN)
sys.exit()
"""



env = environment.GoalEnv()
#env.test_environment(task.sequences_list)
#model = goalenv2020.train(goals=False, num_iterations=100000, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

#model = utils.load_object("bigmodel1")
#goalenv2020.accuracy_test_botvinick(model, num_tests=100, sequences=[0, 1, 2, 3, 4, 5])
#sys.exit()

for i in range(0):
    model = goalenv2020.train(goals=True, num_iterations=50000, learning_rate=0.003, L2_reg=0.0001, noise=0.05, sequences=range(21))
    utils.save_object("bigmodel1", model)
    #model = utils.load_object("bigmodel")
    goalenv2020.accuracy_test_botvinick(model, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.3, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.5, noise_step=5, goals=True, sequence_ids=range(21))

#model = utils.load_object("bigmodel1")
#goalenv2020.accuracy_test_botvinick(model, noise=0.15, goals=True, sequence_ids=range(21))
#goalenv2020.accuracy_test_botvinick(model, noise=0.2, goals=True, sequence_ids=range(21))
#goalenv2020.accuracy_test_botvinick(model, noise=0.25, goals=True, sequence_ids=range(21))
#goalenv2020.accuracy_test_botvinick(model, noise=0.35, goals=True, sequence_ids=range(21))
"""
for i in range(1):
    model = goalenv2020.train(goals=True, num_iterations=50000, learning_rate=0.001, L2_reg=0.0001, noise=0.05, sequences=range(21))
    utils.save_object("bigmodel2", model)
    #model = utils.load_object("bigmodel")
    goalenv2020.accuracy_test_botvinick(model, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.15, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.3, noise_step=5, goals=True, sequence_ids=range(21))

for i in range(1):
    model = goalenv2020.train(goals=True, num_iterations=50000, learning_rate=0.001, L2_reg=0.001, noise=0.05, sequences=range(21))
    utils.save_object("bigmodel2.5", model)
    #model = utils.load_object("bigmodel")
    goalenv2020.accuracy_test_botvinick(model, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.15, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.3, noise_step=5, goals=True, sequence_ids=range(21))
"""
#for i in range(1):
model = goalenv2020.train(goals=True, num_iterations=15000, learning_rate=0.0003, L2_reg=0.0001, noise=0., sequences=range(21))
utils.save_object("bigmodel3", model)
#model = utils.load_object("bigmodel")
goalenv2020.accuracy_test_botvinick(model, goals=True, sequence_ids=range(21))
goalenv2020.accuracy_test_botvinick(model, noise=0.05, goals=True, sequence_ids=range(21))
goalenv2020.accuracy_test_botvinick(model, noise=0.1, goals=True, sequence_ids=range(21))
goalenv2020.accuracy_test_botvinick(model, noise=0.2, goals=True, sequence_ids=range(21))
goalenv2020.accuracy_test_botvinick(model, noise=0.3, noise_step=5, goals=True, sequence_ids=range(21))

for i in range(0):
    print("Another 10,000 steps of training to reach " + str(i) + ",000 steps")
    goalenv2020.train(model, goals=True, num_iterations=10000, learning_rate=0.0003, L2_reg=0.0001, noise=0., sequences=range(21))
    utils.save_object("bigmodel3"+str(i), model)
    goalenv2020.accuracy_test_botvinick(model, goals=True, sequence_ids=range(21))
    print("At noise 0.05:")
    goalenv2020.accuracy_test_botvinick(model, noise=0.05, goals=True, sequence_ids=range(21))
    print("At noise 0.15:")
    goalenv2020.accuracy_test_botvinick(model, noise=0.15, goals=True, sequence_ids=range(21))
    print("At noise 0.25:")
    goalenv2020.accuracy_test_botvinick(model, noise=0.25, goals=True, sequence_ids=range(21))
    print("At noise 0.4 on step 5:")
    goalenv2020.accuracy_test_botvinick(model, noise=0.4, noise_step=5, goals=True, sequence_ids=range(21))

"""
model = goalenv2020.train(model=model, goals=False, num_iterations=1, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1])
goalenv2020.accuracy_test(model, sequences=[0, 1])
model = goalenv2020.train(model=model, goals=False, num_iterations=1, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1])
goalenv2020.accuracy_test(model, sequences=[0, 1])
model = goalenv2020.train(model=model, goals=False, num_iterations=1, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1])
goalenv2020.accuracy_test(model, sequences=[0, 1])
model = goalenv2020.train(model=model, goals=False, num_iterations=1, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1])
goalenv2020.accuracy_test(model, sequences=[0, 1])
model = goalenv2020.train(model=model, goals=False, num_iterations=1, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1])
goalenv2020.accuracy_test(model, sequences=[0, 1])
model = goalenv2020.train(model=model, goals=False, num_iterations=1, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1])
goalenv2020.accuracy_test(model, sequences=[0, 1])
model = goalenv2020.train(model=model, goals=False, num_iterations=1, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1])
goalenv2020.accuracy_test(model, sequences=[0, 1])
#goalenv.train(goals=False, num_iterations=50000, learning_rate=0.001, L2_reg=0.00001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
#goalenv.train(goals=False, num_iterations=50000, learning_rate=0.01, L2_reg=0.00001, noise=0., sequences=[0, 1])
"""

#goalenv.train(goals=False, num_iterations=100000, learning_rate=0.0003, L2_reg=0.0001, noise=0., sequences=[0, 1, 2, 3])
#goalenv.train(goals=False, num_iterations=100000, learning_rate=0.0003, L2_reg=0.0001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
#goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.0001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
#goalenv.train(goals=False, num_iterations=50000, learning_rate=0.01, L2_reg=0.0001, noise=0., sequences=[0, 1])
#goalenv.train(goals=False, num_iterations=50000, learning_rate=0.1, L2_reg=0.0001, noise=0., sequences=[0, 1])
#goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0001, L2_reg=0.000001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
