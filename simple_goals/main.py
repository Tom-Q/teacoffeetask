# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import teacoffeeenv as tce
import scripts
import neuralnet as nn
import utils
import pnas2018
import sys
import goalenv
import prederror_task

"""
for i in range(2):
    print(i+1)
    model = prederror_task.train(iterations=5000, simulated_annealing=False)
    prederror_task.accuracy_test(model)
    utils.save_object("prederror_no_goals", model)

for i in range(2):
    print(i+1)
    model = utils.load_object("prederror_no_goals", i)
    model = prederror_task.train(model, iterations=1000, simulated_annealing=True)
    prederror_task.accuracy_test(model)
    utils.save_object("prederror_no_goals_simulated_annealing", model)

for i in range(2):
    print(i+1)
    model = prederror_task.train_with_goals(iterations=5000)
    prederror_task.accuracy_test_with_goals(model)
    utils.save_object("prederror_goals", model)

for i in range(2):
    print(i + 1)
    model = utils.load_object("prederror_goals", i)
    model = prederror_task.train_with_goals(model, iterations=1000, simulated_annealing=True)
    prederror_task.accuracy_test_with_goals(model)
    utils.save_object("prederror_goals_simulated_annealing", model)
"""

prederror_task.make_rdm_multiple("prederror_no_goals", 2, with_goals=False, title="ce_prederror_2020-09-24")
prederror_task.make_rdm_multiple("prederror_goals", 2, with_goals=False, title="ce_goals_prederror_2020-09-24")
prederror_task.make_rdm_multiple("prederror_no_goals_simulated_annealing", 2, with_goals=False, title="ce_prederror_annealing2020-09-24")
prederror_task.make_rdm_multiple("prederror_goals_simulated_annealing", 2, with_goals=False, title="ce_prederror_goals_annealing2020-09-24")


sys.exit("end of program")
env = goalenv.GoalEnv()
#env.test_environment()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0005, L2_reg=0.0000005, noise=0., sequences=[0, 1 ,2 ,3 , 4, 5])

env = goalenv.GoalEnv()
#env.test_environment()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.001, L2_reg=0.000001, noise=0., sequences=[0, 1 ,2 ,3 , 4, 5])

env = goalenv.GoalEnv()
#env.test_environment()
goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0001, L2_reg=0.0000001, noise=0., sequences=[0, 1 ,2 ,3 , 4, 5])
#peng.make_rdm_multiple_peng('peng_with_goals10000', 100, with_goals=True, title="ce_goals_predictionerror_2020-09-24")  # reuse that 'pnas_goals_ce_5000' for probas with goals
#pnas2018.make_rdm_and_mds_pnas('pnas2')


for i in range(100):
    model = pnas2018.train_pnas(noise=0., iterations=5000)
    pnas2018.accuracy_test_pnas(model)
    print(i)
    utils.save_object("pnas_ce_CCCWTSTW", model)  # rerun. Rename this to reg001

for i in range(100):
    model = pnas2018.train_pnas_with_goals(noise=0., iterations=5000)
    pnas2018.accuracy_test_pnas_with_goals(model)
    print(i)
    utils.save_object("pnas_cewgoals_CWCSTWTS", model)

