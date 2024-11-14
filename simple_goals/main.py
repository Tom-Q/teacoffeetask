# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import utils
import sys
#from neural import neuralnet as nn, optimizers, layers
import analysis
import goalenv
from goalenv import environment as env, goalenv2020, scripts
import tensorflow as tf
import scipy
import numpy as np

from pnas import pnas2018, pnashierarchy
import rdm
#goalenv.scripts.bargraph_with_without_goalunits_redo("figure4_100units")
#sys.exit()

utils.initialize_random_seeds(525685)
# SVM.
#model = utils.load_object("bigmodel1_0goals_relu_adam_nonoise")  # no goal model.
# extract activations.
import statistics as stats
import scipy.stats as scistats
from copy import deepcopy
if False:  # SVM classification
    scripts.basic_analysis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                           "bigmodel1_0goals_relu_adam_nonoise", False, constant_noise=0)
    goals_accuracies1 = deepcopy(goalenv2020.list_svms_accuracies_goals)
    subgoals_accuracies1 = deepcopy(goalenv2020.list_svms_accuracies_subgoals)
    print(stats.mean(goals_accuracies1), stats.variance(goals_accuracies1))
    print(stats.mean(subgoals_accuracies1), stats.variance(subgoals_accuracies1))
    sys.exit()
    goalenv2020.list_svms_accuracies_goals = []
    goalenv2020.list_svms_accuracies_subgoals = []
    scripts.basic_analysis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                           "bigmodel1_1goals_relu_adam_nonoise", True, constant_noise=0)
    goals_accuracies2 = deepcopy(goalenv2020.list_svms_accuracies_goals)
    subgoals_accuracies2 = deepcopy(goalenv2020.list_svms_accuracies_subgoals)
    print(stats.mean( goals_accuracies2), stats.variance( goals_accuracies2))
    print(stats.mean(subgoals_accuracies2), stats.variance(subgoals_accuracies2))
    goalenv2020.list_svms_accuracies_goals = []
    goalenv2020.list_svms_accuracies_subgoals = []
    print("ttests goals / subgoals:")
    print(scistats.ttest_rel(goals_accuracies1, goals_accuracies2))
    print(scistats.ttest_rel(subgoals_accuracies1, subgoals_accuracies2))
    scripts.basic_analysis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                           "bigmodel1_0goals_relu_rmsprop_nonoise_100_units", False, constant_noise=0)
    goals_accuracies1 = deepcopy(goalenv2020.list_svms_accuracies_goals)
    subgoals_accuracies1 = deepcopy(goalenv2020.list_svms_accuracies_subgoals)
    print(stats.mean(goals_accuracies1), stats.variance(goals_accuracies1))
    print(stats.mean(subgoals_accuracies1), stats.variance(subgoals_accuracies1))

    goalenv2020.list_svms_accuracies_goals = []
    goalenv2020.list_svms_accuracies_subgoals = []
    scripts.basic_analysis([1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25],
                           "bigmodel1_1goals_relu_rmsprop_nonoise_100_units", True, constant_noise=0)
    goals_accuracies2 = deepcopy(goalenv2020.list_svms_accuracies_goals)
    subgoals_accuracies2 = deepcopy(goalenv2020.list_svms_accuracies_subgoals)
    print(stats.mean( goals_accuracies2), stats.variance( goals_accuracies2))
    print(stats.mean(subgoals_accuracies2), stats.variance(subgoals_accuracies2))
    goalenv2020.list_svms_accuracies_goals = []
    goalenv2020.list_svms_accuracies_subgoals = []

    print("ttests goals / subgoals:")
    print(scistats.ttest_rel(goals_accuracies1, goals_accuracies2))
    print(scistats.ttest_rel(subgoals_accuracies1, subgoals_accuracies2))
    sys.exit()

if True:
    # LARGE MODELS PNAS
    if False: # train models
        models = []
        for i in range(0):
            model, _ = pnas2018.train(size_hidden=50, iterations=4000)
            utils.save_object("ctt_nogoals_50units_4000i", model)
        for i in range(1):
            model = pnashierarchy.train_with_goals(size_hidden=50, iterations=4500)
            utils.save_object("ctt_goals_50units_4500i", model)
        sys.exit()
        for i in range(20):
            model, _ = pnas2018.train(size_hidden=100, iterations=3500)
            utils.save_object("ctt_nogoals_100units_3500i", model)

        for i in range(20):
            model, _ = pnas2018.train(size_hidden=1000, iterations=2000)
            utils.save_object("ctt_nogoals_1000units_2000i", model)

        for i in range(20):
            model = pnashierarchy.train_with_goals(size_hidden=100, iterations=4000)

            utils.save_object("ctt_goals_100units_4000i", model)

        for i in range(20):
            model = pnashierarchy.train_with_goals(size_hidden=1000, iterations=2500)
            utils.save_object("ctt_goals_1000units_2500i", model)

    if False: # Hierarchy models
        for i in range(0):
            model = pnashierarchy.train_hierarchical(iterations=6500, size_hidden=50, reg_strength=0.0003)
            utils.save_object("pnashierarchy50_6500_0003", model)
        for i in range(0):
            model = pnashierarchy.train_hierarchical(iterations=6500, size_hidden=100, reg_strength=0.00015)
            utils.save_object("pnashierarchy100_6500_00015", model) # should be 0003
        for i in range(0):
            model = pnashierarchy.train_hierarchical(iterations=5500, size_hidden=1000, reg_strength=0.000015)
            utils.save_object("pnashierarchy1000_5500_000015", model) # should be 0003
        sys.exit()
    pnas2018.make_rdm_multiple("ctt_nogoals_50units_4000i", 20, with_goals=False, rdm_type=rdm.SPEARMAN, title="ctt_nogoals_50units_4000i", save_name="ctt_nogoals_50units_4000i")
    pnas2018.make_rdm_multiple("ctt_goals_50units_4500i", 20, with_goals=True, rdm_type=rdm.SPEARMAN, title="ctt_goals_50units_4500i", save_name="ctt_goals_50units_4500i")
    sys.exit()
    #pnas2018.make_rdm_multiple("ctt_nogoals_100units_3500i", 20, with_goals=False, rdm_type=rdm.SPEARMAN,  title="ctt_nogoals_100units_3500i", save_name="ctt_nogoals_100units_3500i")
    #pnas2018.make_rdm_multiple("ctt_nogoals_1000units_2000i", 20, with_goals=False, rdm_type=rdm.SPEARMAN, title="ctt_nogoals_1000units_2000i", save_name="ctt_nogoals_1000units_2000i")
    #pnas2018.make_rdm_multiple("ctt_goals_100units_4000i", 20, with_goals=True, rdm_type=rdm.SPEARMAN, title="ctt_goals_100units_4000i", save_name="ctt_goals_100units_4000i")
    #pnas2018.make_rdm_multiple("ctt_goals_1000units_2500i", 20, with_goals=True, rdm_type=rdm.SPEARMAN, title="ctt_goals_1000units_2500i", save_name="ctt_goals_1000units_2500i")

    pnashierarchy.make_rdm_multiple_hierarchy("pnashierarchy50_6500_0003", 20, skips=[], title="pnashierarchy50_6500_0003")
    pnashierarchy.make_rdm_multiple_hierarchy("pnashierarchy100_6500_00015", 20, skips=[1, 5, 6, 7, 14, 19, 23, 24, 25, 27],
                                              title="pnashierarchy100_6500_00015")
    pnashierarchy.make_rdm_multiple_hierarchy("pnashierarchy1000_5500_000015", 20, skips=[],
                                              title="pnashierarchy1000_5500_000015")

    #pnashierarchy.make_rdm_multiple("ctt_goals_1000units_2500i", 20, with_goals=True, rdm_type=rdm.SPEARMAN, title="ctt_goals_1000units_2500i", save_name="ctt_goals_1000units_2500i")

    sys.exit(0)





#scripts.basic_analysis(range(20), "bigmodel1_1goals_relu_adam_nonoise", True)
#scripts.basic_analysis(range(20), "bigmodel1_0goals_relu_adam_nonoise", False)
#scripts.plot_pca()
#sys.exit()
"""
#rdm_sum_100_units_goals = scripts.basic_analysis([0, 4, 5, 6, 7], "bigmodel1_1goals_relu_adam_nonoise_100_units", True)
#rdm_sum_100_units_no_goals = scripts.basic_analysis([0, 1, 2, 5, 6], "bigmodel1_0goals_relu_adam_nonoise_100_units", False)
#rdm_sum_goals = scripts.basic_analysis(range(5), "bigmodel1_1goals_relu_adam_nonoise", True)
# rdm_sum_no_goals = scripts.basic_analysis(range(7), "bigmodel1_0goals_relu_adam_nonoise", False)
#print("bigmodel1_1goals_relu_adam_nonoise_100_units")
#rdm_sum_100_units_goals = scripts.basic_analysis([0, 4, 5, 6, 7], "bigmodel1_1goals_relu_adam_nonoise_100_units", True)
#print("bigmodel1_0goals_relu_adam_nonoise_100_units")
rdm_sum_100_units_no_goals = scripts.basic_analysis([0, 1, 2, 4, 6, 9, 10], "bigmodel1_0goals_relu_adam_nonoise_100_units", False) # all those are correct
rdm_sum_no_goals = scripts.basic_analysis(range(7), "bigmodel1_0goals_relu_adam_nonoise", False)
"""
"""
print("bigmodel1_1goals_relu_adam_nonoise")
rdm_sum_goals = scripts.basic_analysis(range(10), "bigmodel1_1goals_relu_adam_nonoise", True) # all correct
print("bigmodel1_0goals_relu_adam_nonoise")
rdm_sum_no_goals = scripts.basic_analysis(range(10), "bigmodel1_0goals_relu_adam_nonoise", False) #

# Save each RDM png
"""
"""
rdm_list = [#("100 units goals", rdm_sum_100_units_goals),
            ("100 units no goals", rdm_sum_100_units_no_goals),
            #("goals", rdm_sum_goals),
            ("no goals", rdm_sum_no_goals)]

# Now correlate those rdms
for i in rdm_list:
    for j in rdm_list:
        print(i[0] + " vs "+ j[0] + "(1. pearson, 2. spearman)")
        print(rdm.compare_matrices_pearson(i[1].matrix, j[1].matrix))
        print(rdm.compare_matrices_spearman(i[1].matrix, j[1].matrix))

for i in rdm_list:
    i[1].save(i[0])
sys.exit()

"""
#for i in range(20):
#    model = pnashierarchy.train_with_goals()
#    utils.save_object("model_with_goals100units_ctt", model)


#goalenv.scripts.train_networks_100_units(10)
#nets = utils.load_objects("bigmodel1_0goals_relu_rmsprop_nonoise_100_units", 10)
# Test the models (try to train 20 of each).\


for noise in [0.01, 0.1, 0.2, 0.5, 1.]:
    error_testing_results = goalenv.scripts.basic_analysis([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                                   "bigmodel1_0goals_relu_rmsprop_nonoise_100_units", False, constant_noise=noise)
    action_errors = sequence_errors = correct = 0
    for net in error_testing_results:
        correct += net[3][0]
        sequence_errors += net[3][2]
        action_errors += net[3][1]

    num_nets = len(error_testing_results)
    print("\n\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%" +\
          "noise="+str(noise)+"\ncorrect:"+ str(correct/num_nets) + "\nseq:" + str(sequence_errors/num_nets), "\nact:"+str(action_errors/num_nets))

for noise in [0.01, 0.1, 0.2, 0.5, 1.]:
    error_testing_results = goalenv.scripts.basic_analysis([1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25],
                                   "bigmodel1_1goals_relu_rmsprop_nonoise_100_units", True, constant_noise=noise)
    action_errors = sequence_errors = correct = 0
    for net in error_testing_results:
        correct += net[3][0]
        sequence_errors += net[3][2]
        action_errors += net[3][1]

    num_nets = len(error_testing_results)
    print("\n\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%" +\
          "noise="+str(noise)+"\ncorrect:"+ str(correct/num_nets) + "\nseq:" + str(sequence_errors/num_nets), "\nact:"+str(action_errors/num_nets))

# correct: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
# failed: 1, 6, 7, 9, 21, 22, 27,
# VALID GOAL NETWORKS: [1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25] "bigmodel1_1goals_relu_rmsprop_nonoise_100_units"
# VALID FLAT NETWORKS: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] (all of them)
sys.exit(0)
#pnas2018.make_rdm_multiple("model_with_goals100units_ctt", 10, with_goals=True, rdm_type=rdm.SPEARMAN)
pnashierarchy.make_rdm_multiple_goals_actions("model_with_goals", 10)
sys.exit(0)
#sys.exit(0)
# Leave-one-out-analysis:
# Still have to train 11 networks (no goals)
#for i in range(2):
goalenv.scripts.train_networks_100_units(6)
#goalenv.scripts.leave_one_out_analysis()


sys.exit(0)
"""
from pnas import pnas2018
l1reg = [0., 0.0001]
l2reg = [0, 0.0001]
algo = [optimizers.SGD, optimizers.RMSPROP, optimizers.ADAM]
size_hidden = [5, 15, 100]
learning_rate = [0.05, 0.1]
loss_type = ["cross_entropy", "mse"]

for l1 in l1reg:
    for l2 in l2reg:
        for alg in algo:
            for size in size_hidden:
                for lr in learning_rate:
                    for loss in ["cross_entropy"]:
                        model, perf = pnas2018.train(iterations=500, l1reg=l1, l2reg=l2, algorithm=alg,
                                               size_hidden=size, learning_rate=lr, loss_type=loss)
                        #_, accuracy_totals = pnas2018.accuracy_test(model)
                        model_desc = "l1{0}_l2{1}_alg{2}_hl{3}_lr{4}_loss{5}_perf{6}".format(str(l1),
                                                                                             str(l2),
                                                                                             alg,
                                                                                             str(size),
                                                                                             str(lr),
                                                                                             loss,
                                                                                             str(perf))
                        utils.save_object(model_desc, model)
                        rdm = pnas2018.make_rdm_multiple(model_desc, num_networks=1, save_name=model_desc)
                        #sys.exit(0)

from reward_task import scripts, reward, rewardtask
scripts.performance_test()
sys.exit()
"""

from goalenv import scripts as gevscripts
#gevscripts.graph_error_rate_per_step_example()
#gevscripts.bargraph_with_without_goalunits("error_bars.png")
#for net in range(10):
#gevscripts.rdm_distances_analysis()
#sys.exit()
#gevscripts.analyse_rdm_distances_output()
#gevscripts.rdm_distances_analysis()
#gevscripts.figure_steps_to_error("myfig.svg")

#from goalenv import scripts

#scripts.basic_analysis()


from atariACC import tests

tests.a2c_test()

sys.exit(0)

from modularACC import predictive
import rdm

for i in range(2):
    mod, _ = predictive.train_new_task(ultra=True, size_acc=50, iterations=20000, learning_rate_acc=0.1)
    utils.save_object("pred_acc_ultra2", mod)
rdm_acc, rdm_context = predictive.make_rdm_multiple_new_task("pred_acc_ultra2", num_networks=5, save_files=True, skips=[], rdm_type=rdm.EUCLIDIAN)
rdm_acc.save("rdm_acc_ultra2")
rdm_context.save("rdm_ctxt_ultra2")


for i in range(2):
    mod, _ = predictive.train_new_task(ultra=False, size_acc=50, iterations=20000, learning_rate_acc=0.1)
    utils.save_object("pred_acc2", mod)
rdm_acc, rdm_context = predictive.make_rdm_multiple_new_task("pred_acc2", num_networks=5, save_files=True, skips=[], rdm_type=rdm.EUCLIDIAN)
rdm_acc.save("rdm_acc2")
rdm_context.save("rdm_ctxt2")


"""
from modularACC import criticACC
import rdm
for i in range(2):
    mod, _ = criticACC.train(size_acc=20, iterations=2000, learning_rate_acc=0.1)
    utils.save_object("critic_acc", mod)
rdm_acc = criticACC.make_rdm_multiple("critic_acc", num_networks=3, save_files=True, skips=[], rdm_type=rdm.EUCLIDIAN)
rdm_acc.save("rdm_critic_acc")
exit()
#utils.save_object("rdm_acc", rdm_acc)
rdm = utils.load_object("rdm_acc")
rdm.save("rdm_critic_acc_log")
rdm.delete_entry(6)
rdm.delete_entry(0)
rdm.save("steps_1to5")
rdm.delete_entry(5)
rdm.delete_entry(0)
rdm.save("steps_1to4")
rdm.delete_entry(4)
rdm.delete_entry(0)
rdm.save("steps_1to3")
rdm.delete_entry(3)
rdm.delete_entry(0)
rdm.save("steps_1to2")
#rdm_context.save("rdm_ctxt")
"""
