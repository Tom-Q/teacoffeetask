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
for i in range(1):
    print(i+1)
    model = peng.train_peng(iterations=15000)
    peng.accuracy_test_peng(model)
    utils.save_object("peng_no_goals10000", model)

for i in range(100):
    model = peng.train_peng_with_goals(iterations=10000)
    peng.accuracy_test_peng_with_goals(model)
    utils.save_object("peng_with_goals10000", model)


for network_number in [7, 26, 62, 83, 66]:
    model = utils.load_object("peng_no_goals10000", network_number)
    peng.accuracy_test_peng(model)
    model = peng.train_peng(model=model, iterations=10)
    peng.accuracy_test_peng(model)

sys.exit("end of program")
"""

#env = goalenv.GoalEnv()
#env.test_environment()
#goalenv.train(goals=False, num_iterations=100000, learning_rate=0.003, L2_reg=0.000003, noise=0., sequences=[0, 1 ,2 ,3 , 4, 5])

prederror_task.make_rdm_multiple('peng_no_goals10000', 100, with_goals=False, title="ce_predictionerror_2020-09-24")
#peng.make_rdm_multiple_peng('peng_with_goals10000', 100, with_goals=True, title="ce_goals_predictionerror_2020-09-24")  # reuse that 'pnas_goals_ce_5000' for probas with goals
#pnas2018.make_rdm_and_mds_pnas('pnas2')

sys.exit("end of program")

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

