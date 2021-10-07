# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import utils
import sys
import neuralnet as nn
from cognitiveload import cogloadtask

"""
for i in range(1
               ):
    #model = utils.load_object("bigmodel3")
    model = nn.ElmanGoalNet(size_hidden=50, size_observation=29, size_action=19,
                            size_goal1=0,#len(environment.GoalEnvData.goals1_list),
                            size_goal2=0,#len(environment.GoalEnvData.goals2_list),
                            algorithm=nn.RMSPROP, learning_rate=0.001, initialization="uniform",
                            last_action_inputs=True)
    model = goalenv2020.train(model=model, goals=False, num_iterations=200000, learning_rate=0.001, L2_reg=0.00001, noise=0.05, sequences=range(21))
    utils.save_object("bigmodel4", model)
#sys.exit()
model = utils.load_object("bigmodel4")
#sys.exit()
#test_data = utils.load_object("test_data_tsne")
#test_data = goalenv2020.generate_test_data(model, noise=0.5, one_run_per_step=True, goal1_noise=0., goal2_noise=0., goals=True, num_tests=3, sequence_ids=range(21))
test_data = goalenv2020.generate_test_data(model, noise=0.,  goal1_noise=0., goal2_noise=0., goals=False, num_tests=1, sequence_ids=range(21), noise_per_step=True, disruption_per_step=False, initialization="seminormal")
tsne_results, test_data = goalenv2020.analyse_test_data(test_data, do_error_analysis=True, do_rdm=False)
utils.save_object("test_data_tsne_disrupt", test_data)
utils.save_object("tsne_results_disrupt", tsne_results)
test_data = utils.load_object("test_data_tsne_disrupt")
tsne_results = utils.load_object("tsne_results_disrupt")

sys.exit()

"""



import cognitiveload.model3 as mod3
mod3.FAST_RDM = True
hrp=mod3.HierarchyGradientParams(regincrease="linear", regstrength=0.00001)
mod3.run_model3_multiple(from_file=None, num_networks=10, name="model3_test_gradient", hrp=hrp)
sys.exit(0)



import cognitiveload.model2 as mod2
mod2.FAST_RDM=True
mod2.run_model2_multiple(num_networks=10)
sys.exit(0)
""""""
# Try out the multilayer predictive network. On what task? --> PNAS.
"""
context_initialization = np.random.random(size=(1,15)).astype(np.float32)
print(context_initialization)
for i in range(90):
    print(i)
    model = nn.ElmanGoalNet(size_hidden=15, size_observation=9, size_action=8, size_goal1=0, size_goal2=0,
                            algorithm=nn.SGD, initialization="normal")
    model.context = context_initialization
    model, _ = pnas2018.train(model=model, iterations=10000, learning_rate=.1, loss_type=pnas2018.CROSS_ENTROPY,
                              context_initialization=None)
    #pnas2018.accuracy_test(model, context_initialization=None)
    utils.save_object("pnas_keepcontext_2seqs", model)
pnas2018.make_rdm_multiple_keepcontext("pnas_keepcontext_2seqs", num_networks=100)
#pnas2018.make_rdm_multiple_deepprednet("deepcontrolprednetpnas", 10, with_goals=False)
sys.exit()
"""
"""
for i in range(0):
    print(i)
    model = prederror_task.train(algorithm=nn.SGD, iterations=10000, learning_rate=0.5,
                                 hidden_units=15, l2reg=0.0001, mse=True)
    prederror_task.accuracy_test(model)
    utils.save_object('prederror_l2reg', model)

spearmanRDM = prederror_task.make_rdm_multiple('prederror_redo3', 100, type="spearman", skips=[11])#8, 15, 17, 25, 32, 49, 65, 68, 93])
euclidianRDM = prederror_task.make_rdm_multiple('prederror_redo3', 100, type="euclidian", skips=[11])#8, 15, 17, 25, 32, 49, 65, 68, 93])
sub = euclidianRDM-spearmanRDM
add = euclidianRDM+spearmanRDM
nonzero_idxs = np.where(add == 0)
add[nonzero_idxs] = 1.
normed_sub = np.divide(sub, add)
labels = []
for i, sequence in enumerate(prederror_task.seqs):
    for action in sequence[1:-1]:
        labels.append(str(i) + '_' + action)
import matplotlib.pyplot as plt
analysis.plot_rdm(normed_sub, labels, "normed_sub_matrix")
plt.savefig('normed_sub_rdm_redo3')
plt.clf()

np.savetxt('normed_sub_rdm_redo3' + ".txt", normed_sub, delimiter="\t", fmt='%.2e')

#prederror_task.make_rdm_multiple_predictive('prederror_prednet_wta15', 100, type='wta', skips=[41, 62, 75, 80, 81, 83, 85])
sys.exit()
"""
# using predictive net
for i in range(93):
    print(i)
    model = prederror_task.train_predictive_net(algorithm=nn.SGD, iterations=15000, learning_rate=0.02, hidden_units=15, type='wta')
    prederror_task.accuracy_test_predictive(model, 'wta')
    utils.save_object('prederror_prednet_wta15', model)

# using predictive net
for i in range(90):
    print(i)
    model = prederror_task.train_predictive_net(algorithm=nn.SGD, iterations=10000, learning_rate=0.02, hidden_units=15, type='sigmoid')
    prederror_task.accuracy_test_predictive(model, 'sigmoid')
    utils.save_object('prederror_prednet_sigmoid15', model)

sys.exit()

# Regular SGD with equiprobable sequences
for i in range(0):
    print(i)
    model = prederror_task.train(iterations=8000)
    prederror_task.accuracy_test(model)
    utils.save_object('prederror_basic_equiprobable', model)
#prederror_task.make_rdm_multiple('prederror_basic_equiprobable', 100, skips=[3, 19, 26, 28, 32, 35, 50, 55, 72, 76, 79, 80, 101, 110])


# Regular SGD with goals with equiprobable sequences
for i in range(0):
    print(i)
    model = prederror_task.train_with_goals(iterations=8000)
    prederror_task.accuracy_test_with_goals(model)
    utils.save_object('prederror_goals_equiprobable', model)
#prederror_task.make_rdm_multiple('prederror_goals_equiprobable', 100, with_goals=True)

# With 2 goals + gradient with equiprobable sequences
for i in range(0):
    print(i)
    model = prederror_task.train_with_goals(reg_strength=0.0001, iterations=10000)
    prederror_task.accuracy_test_with_goals(model)
    utils.save_object('prederror_gradient_equiprobable', model)
prederror_task.make_rdm_multiple_hierarchy('prederror_gradient', 100, skips=[17, 54, 91])
#prederror_task.make_rdm_multiple_hierarchy('prederror_gradient_equiprobable', 100)

sys.exit()
# Regular SGD
for i in range(0):
    model = prederror_task.train()
    prederror_task.accuracy_test(model)
    utils.save_object('prederror_basic', model)
#prederror_task.make_rdm_multiple('prederror_basic', 100)

# reload them and train them further
for i in range(0):
    print(i)
    model = utils.load_object('prederror_basic', latest=i)
    model = prederror_task.train(model, iterations=3000)
    utils.save_object('prederror_basic_v2', model)
#prederror_task.make_rdm_multiple('prederror_basic_v2', 100, skips=[3, 42, 79])

# With 2 goals
for i in range(0):
    print(i)
    model = prederror_task.train_with_goals()
    prederror_task.accuracy_test_with_goals(model)
    utils.save_object('prederror_goals', model)
#prederror_task.make_rdm_multiple('prederror_goals', 100, with_goals=True)

# reload them and train them further
for i in range(0):
    print(i)
    model = utils.load_object('prederror_goals', latest=i)
    model = prederror_task.train_with_goals(model, iterations=3000)
    utils.save_object('prederror_goals_v2', model)
#prederror_task.make_rdm_multiple('prederror_goals_v2', 100, with_goals=True, skips=[17, 45, 53])


# With 2 goals + gradient
for i in range(0):
    print(i)
    model = prederror_task.train_with_goals(reg_strength=0.0001, iterations=10000)
    prederror_task.accuracy_test_with_goals(model)
    utils.save_object('prederror_gradient', model)
prederror_task.make_rdm_multiple_hierarchy('prederror_gradient', 100, skips=[17, 54, 91])



#prederror_task.make_rdm_multiple_predictive('prederror_prednet', 100, skips=[16, 26, 35, 40, 50, 59, 61, 63, 79, 98, 100])

sys.exit()

"""
hyperparamstest.hyperparameter_analysis(file="deleteme.txt")
sys.exit()
names = ["sgdnormal150.00.0cross_entropy", "adamuniform500.0010.001mse", "rmspropnormal150.00.001cross_entropy"]
hyperparamstest.hyperparameter_individual_matrix_analysis(names)
sys.exit()
file = "hyperparams_test_rdm_spearman"
hyperparamstest.hyperparameter_analysis(file=file, type=analysis.SPEARMAN)
sys.exit()
hyperparamstest.reload(file)

file = "hyperparams_test_rdm_mahalanobis"
#hyperparamstest.hyperparameter_analysis(file=file, type=analysis.MAHALANOBIS)
hyperparamstest.reload(file)


file = "hyperparams_test_rdm_euclidian"
#hyperparamstest.hyperparameter_analysis(file=file, type=analysis.EUCLIDIAN)
hyperparamstest.reload(file)

sys.exit()
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

import analysis
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


#env = environment.GoalEnv()
#env.test_environment(task.sequences_list)
#sys.exit()
#model = goalenv2020.train(goals=False, num_iterations=100000, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

#model = utils.load_object("bigmodel1")
#goalenv2020.accuracy_test_botvinick(model, num_tests=100, sequences=[0, 1, 2, 3, 4, 5])
#sys.exit()

#model = utils.load_object("bigmodel1")

# bigmodel: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.002, L2_reg=0.00001, noise=0.1, sequences=range(21))
# bigmodel1: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.002, L2_reg=0.00001, noise=0., sequences=range(21))
# bigmodel2: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.002, L2_reg=0.00001, noise=0.05, sequences=range(21))
# bigmodel3: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.001, L2_reg=0.00001, noise=0.01, sequences=range(21)) --> 70% correct at 30000
# bigmodel4: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.003, L2_reg=0.00001, noise=0.01, sequences=range(21)) --> 60% correct
# bigmodel5:     model = goalenv2020.train(goals=True, num_iterations=50000, learning_rate=0.0001, L2_reg=0.00001, noise=0.01, sequences=range(21)) --> 23% but still improving
# bigmodel6: 6 model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.003, L2_reg=0.0, noise=0.01, sequences=range(21)) 46%, still improving
# bigmodel7: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.005, L2_reg=0.0, noise=0.01, sequences=range(21))

#goalenv2020.accuracy_test_botvinick(model, noise=0.1, goals=True, num_tests=10, sequence_ids=range(21))
#goalenv2020.accuracy_test_noise(model, noise=0.4, goals=True, num_tests=10, sequence_ids=range(21))
#goalenv2020.accuracy_test_botvinick(model, noise=0.5, noise_step=5, num_tests= 10, goals=True, sequence_ids=range(21))
for i in range(0):
    #model = utils.load_object("bigmodel3")
    model = nn.ElmanGoalNet(size_hidden=50, size_observation=29, size_action=19,
                            size_goal1=len(environment.GoalEnvData.goals1_list),
                            size_goal2=len(environment.GoalEnvData.goals2_list),
                            algorithm=nn.RMSPROP, learning_rate=0.001, initialization="uniform",
                            last_action_inputs=True)
    model = goalenv2020.train(model=model, goals=True, num_iterations=150000, learning_rate=0.001, L2_reg=0.00001, noise=0.05, sequences=range(21))
    utils.save_object("bigmodel4", model)
#sys.exit()
model = utils.load_object("bigmodel4")
#sys.exit()
#test_data = utils.load_object("test_data_tsne")
#test_data = goalenv2020.generate_test_data(model, noise=0.5, one_run_per_step=True, goal1_noise=0., goal2_noise=0., goals=True, num_tests=3, sequence_ids=range(21))
test_data = goalenv2020.generate_test_data(model, noise=0.,  goal1_noise=0., goal2_noise=0., goals=True, num_tests=1, sequence_ids=range(21), noise_per_step=True, disruption_per_step=False, initialization="seminormal")
tsne_results, test_data = goalenv2020.analyse_test_data(test_data, do_error_analysis=True, do_rdm=False)
utils.save_object("test_data_tsne_disrupt", test_data)
utils.save_object("tsne_results_disrupt", tsne_results)
test_data = utils.load_object("test_data_tsne_disrupt")
tsne_results = utils.load_object("tsne_results_disrupt")
#goalenv2020.plot_tsne(tsne_results, test_data, tsne_actions=True, filename="tsne_actions")
#goalenv2020.plot_tsne(tsne_results, test_data, tsne_subgoals=True, filename="tsne_subgoals")
#goalenv2020.plot_tsne(tsne_results, test_data, tsne_goals=True, filename="tsne_goals")
#goalenv2020.plot_tsne(tsne_results, test_data, tsne_sequences=True, filename="tsne_sequences")
#goalenv2020.plot_tsne(tsne_results, test_data, tsne_errors=True, filename="tsne_errors")
#goalenv2020.plot_tsne(tsne_results, test_data, tsne_actions=True, annotate=True, tsne_sequence=[1548, 1550, 1614], tsne_sequence_interval=[14, 34], filename="tsne_actions_plus_seqs")
sys.exit()
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
