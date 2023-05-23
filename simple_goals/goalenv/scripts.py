# Just collecting some random scripts that serve to produce data, networks, etc.
# These are horrible single-use code that is not going to be maintained so chances are they'll be broken as I change other bits of code.
# It's just nice to have them stored somewhere in the main code body rather than always having to revert to previous versions.

import re
import utils
import tensorflow as tf
from goalenv import goalenv2020
import goalenv.environment as env
from neural import neuralnet as nn, optimizers
import numpy as np

def analyse_rdm_distances_output():
    readfile = open("rdm_analysis_bonus.txt", "r")

    # writefile = open("distances2.txt", "a")
    lines = readfile.readlines()
    readfile.close()

    ee1 = []
    ee2 = []
    ed1 = []
    ed2 = []
    de1 = []
    de2 = []
    dd1 = []
    dd2 = []
    d_1 = []
    d_2 = []
    e_1 = []
    e_2 = []
    _d1 = []
    _d2 = []
    _e1 = []
    _e2 = []
    __1 = []
    __2 = []
    ets1v2 = []
    ddd1 = []
    ddd2 = []
    dd_1 = []
    dd_2 = []

    for i in range(len(lines)):
        if "1 - equal goals, equal sequence:" in lines[i]:
            ee1.append(float(lines[i + 1]))
        elif "2 - equal goals, equal sequence:" in lines[i]:
            ee2.append(float(lines[i + 1]))

        elif "1 - equal goals, different sequence:" in lines[i]:
            ed1.append(float(lines[i + 1]))
        elif "2 - equal goals, different sequence:" in lines[i]:
            ed2.append(float(lines[i + 1]))

        elif "1 - different goals and equal sequence:" in lines[i]:
            de1.append(float(lines[i + 1]))
        elif "2 - different goals and equal sequence:" in lines[i]:
            de2.append(float(lines[i + 1]))

        elif "1 - different goals, different sequence:" in lines[i]:
            dd1.append(float(lines[i + 1]))
        elif "2 - different goals, different sequence:" in lines[i]:
            dd2.append(float(lines[i + 1]))

        elif "1 - different goals:" in lines[i]:
            d_1.append(float(lines[i + 1]))
        elif "2 - different goals:" in lines[i]:
            d_2.append(float(lines[i + 1]))

        elif "1 - equal goals:" in lines[i]:
            e_1.append(float(lines[i + 1]))
        elif "2 - equal goals:" in lines[i]:
            e_2.append(float(lines[i + 1]))

        elif "1 - different sequence:" in lines[i]:
            _d1.append(float(lines[i + 1]))
        elif "2 - different sequence:" in lines[i]:
            _d2.append(float(lines[i + 1]))

        elif "1 - equal sequence:" in lines[i]:
            _e1.append(float(lines[i + 1]))
        elif "2 - equal sequence:" in lines[i]:
            _e2.append(float(lines[i + 1]))

        elif "1 - any:" in lines[i]:
            __1.append(float(lines[i + 1]))
        elif "2 - any:" in lines[i]:
            __2.append(float(lines[i + 1]))


        elif "1 - different goals, subgoals, sequence:" in lines[i]:
            ddd1.append(float(lines[i + 1]))
        elif "2 - different goals, subgoals, sequence:" in lines[i]:
            ddd2.append(float(lines[i + 1]))
        elif "1 - different goals, subgoals:" in lines[i]:
            dd_1.append(float(lines[i + 1]))
        elif "2 - different goals, subgoals:" in lines[i]:
            dd_2.append(float(lines[i + 1]))

        elif "1 vs 2, same sequences and steps" in lines[i]:
            ets1v2.append(float(lines[i + 1]))

    for seq in [ee1, ee2, ed1, ed2, de1, de2, dd1, dd2, d_1, d_2, e_1, e_2, _d1, _d2, _e1, _e2, __1, __2, ets1v2, ddd1,
                ddd2, dd_1, dd_2]:
        for val in seq:
            print(val)
        print("\n\n")

def analyse_ranks_output():
    readfile = open("ranks.txt", "r")

    #writefile = open("distances2.txt", "a")
    lines = readfile.readlines()
    readfile.close()

    regex = "([0-9]+);vs;([0-9]+);([0-9]+);vs;([0-9]+)"
    networks = []
    for i in range(len(lines)):
        if "rank comparisons" in lines[i]:
            network_ranks_x1 = []
            network_ranks_x2 = []
            network_ranks_x1_av = []
            network_ranks_x2_av = []

            for j in range(1, 14):
                netx1, netx2, netx1av, netx2av = re.findall(regex, lines[i + j])[0]
                """netx1 = res[0]
                netx2 = res[1]
                netx1av = res[2]
                netx2av = res[3]"""
                network_ranks_x1.append(netx1)
                network_ranks_x2.append(netx2)
                network_ranks_x1_av.append(netx1av)
                network_ranks_x2_av.append(netx2av)
            networks.append([network_ranks_x1, network_ranks_x2, network_ranks_x1_av, network_ranks_x2_av])
    for j, net in enumerate(networks):
        print("network "+str(j+1), end=";")
        for i in range(13):
            print(int(net[0][i])-int(net[1][i]), end=";")
        for i in range(13):
            print(int(net[2][i])-int(net[3][i]), end=";")
        print("")

def train_networks():
    for i in range(50):
        for g in [1]:
            model = nn.GoalNet(size_hidden=50, size_observation=29, size_action=19, # 50
                               size_goal1=len(env.GoalEnvData.goals1_list) * g,
                               size_goal2=len(env.GoalEnvData.goals2_list) * g,
                               algorithm=optimizers.ADAM, learning_rate=0.001,  # 0.001
                               L2_reg=0., #0.0001,
                               initialization=utils.HE,   # HE
                               nonlinearity=tf.nn.relu,     # relu
                               last_action_inputs=True)

            stopping = nn.ParamsStopping(max_iterations=20000, min_iterations=0, check_frequency=1000,
                                         stop_condition=goalenv2020.stop_condition, goals=bool(g), noise=0.0,
                                         context_initialization=utils.ZERO_INIT)
            goalenv2020.BONUS_ITERATIONS = 0
            model = goalenv2020.train(stop_params=stopping, model=model, goals=bool(g),
                                      noise=0.0, sequences=range(21), context_initialization=utils.ZERO_INIT) # normal context initialization to try and improve learning?
            utils.save_object("bigmodel1_"+str(g)+"goals_relu_adam_nonoise", model)

def basic_analysis():
    filename="ablations.csv"
    #filename="results_last_resort3.csv"
    results = []
    #for noise in [0., 0.01, 0.1, 0.2, 0.5, 1.]:
    for noise in [0]:#reversed([1., 2., 3., 4., 5.]):
        for goals in [True]:#, False]:
            if goals:
                #goal_multipliers = [0.5, 0.9, 1.0, 1.01, 1.05, 1.1, 1.2, 1.5, 2.]
                goal_multipliers = [1.0]#, 1.25, 1.5, 1.75]#, 2.]
            else:
                goal_multipliers = [1.0]
            for goal_multiplier in reversed(goal_multipliers):
                print("Hidden layer noise " + str(noise) + " - Goal: " + str(goals) + " (multiplier:"+str(goal_multiplier)+") \n")
                myfile = open(filename, "a")
                myfile.write("Hidden layer noise " + str(noise) + " - Goal: " + str(goals) + " (multiplier:"+str(goal_multiplier)+") \n")
                myfile.close()
                for network_id in range(50):#[14,18,20,22,26,27,33,38,42,43,46,49]:#range(50):
                    #print(network_id)
                    #print(goals)
                    #network_id=10
                    print("\nNETWORK:" + str(network_id+1))
                    goal_str = "1" if goals else "0"
                    model = utils.load_object("bigmodel1_"+ goal_str + "goals_relu_adam_nonoise", network_id)
                    #print(model.hidden_layer.layer.b.numpy()[0][0])

                    # Goals 1 and 2
                    #goal1 = utils.str_to_onehot("g_1_make_tea", goalenv.environment.GoalEnvData.goals1_list) * 3
                    #goal2 = utils.str_to_onehot("g_2_add_cream", goalenv.environment.GoalEnvData.goals2_list) * 3
                    #goal2 = utils.str_to_onehot("g_2_add_milk", goalenv.environment.GoalEnvData.goals2_list) * 1.5
                    #goal2 = utils.str_to_onehot("g_2_drink_tea", goalenv.environment.GoalEnvData.goals2_list) * 1.0
                    test_data = goalenv2020.generate_test_data(model, noise=0,
                                                               goal1_noise=0., goal2_noise=0.,
                                                               goals=goals, num_tests=100,
                                                               #goal_multiplier=goal_multiplier,
                                                               sequence_ids=[4],#, 16, 16],#range(21) ##[3, 16, 16],;  #0=coffee black, 2 =coffee 2 sugars 3 = coffee cream, 16 = tea milk
                                                               #switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                               #switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream, 25-28 3rd sugar in coffee
                                                               switch_sequence=0,
                                                               #lesion_observation_units=True,
                                                               #lesion_action_units=True,
                                                               lesion_goal2_units=False,  # goal 2 is lower level.
                                                               lesion_goal1_units=False,
                                                               noise_per_step=False,
                                                               constant_noise_to_input=0.,
                                                               noise_per_step_to_input=False,
                                                               #noise_per_step_to_hidden=2.0,
                                                               disruption_per_step=False,
                                                               constant_noise=0.25,
                                                               initialization=utils.ZERO_INIT,
                                                               clamped_goals=False,
                                                               verbose=False)
                    #utils.save_object("test_data_1", test_data)

                    result = goalenv2020.analyse_test_data(test_data, goals=goals,
                                                              do_special_representations=False,
                                                              do_tsne=False,
                                                              do_rdm=False,
                                                              mds_range=50,
                                                              mds_sequences=range(21),
                                                              verbose=True,
                                                              append_to_file=filename,
                                                              sequence_ids=[4])
                    results.append(result)
                    """
                    goalenv2020.plot_tsne(result[0], result[1], tsne_goals=False, tsne_subgoals=True,
                                          tsne_actions=False, tsne_sequences=False,
                                          tsne_errors=False, tsne_sequence=[0, 1, 2],  # tsne_sequence_interval=[1, 30],
                                          filename=str(network_id)+"mds", annotate=True, save_txt=True)
                    # utils.save_object("tsne_results_bigmodel1_yesgoals", tsne_results)
                    analysis.plot_tsne(str(network_id)+"mds_tsne.txt", str(network_id)+"tsne.svg")
                    """
    ranks = []
    for result in results:
        ranks.append(result[-3])
    # Turn this into numpy array
    ranks = np.asarray(ranks)
    print(ranks.mean(axis=0))

    ranks_xaverage = []
    for result in results:
        ranks_xaverage.append(result[-2])
    ranks_xaverage = np.asarray(ranks_xaverage)
    print(ranks_xaverage.mean(axis=0))

def more_basic_analysis():
    filename = "results4.csv"
    results = []
    # print("Hidden layer noise " + str(noise) + " - Goal: " + str(goals) + " (multiplier:"+str(goal_multiplier)+") \n")
    # myfile = open(filename, "a")
    # myfile.write("Hidden layer noise " + str(noise) + " - Goal: " + str(goals) + " (multiplier:"+str(goal_multiplier)+") \n")
    # myfile.close()
    goal_multiplier = 1.5
    goals = 1.
    for network_id in range(50):
        # print(network_id)
        # print(goals)
        # network_id=10
        print("\nNETWORK:" + str(network_id + 1))
        goal_str = "1" if goals else "0"
        model = utils.load_object("bigmodel1_" + goal_str + "goals_relu_adam_nonoise", network_id)
        goal1 = utils.str_to_onehot("g_1_make_tea", env.GoalEnvData.goals1_list)
        goal2 = utils.str_to_onehot("g_2_add_cream", env.GoalEnvData.goals2_list)
        # print(model.hidden_layer.layer.b.numpy()[0][0])
        test_data = goalenv2020.generate_test_data(model, noise=0.,
                                                   goal1_noise=0., goal2_noise=0.,
                                                   goals=goals, num_tests=5,
                                                   goal_multiplier=goal_multiplier,
                                                   sequence_ids=[3],
                                                   # 0=coffee black, 3 = coffee cream, 16 = tea milk
                                                   # switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                   switch_goal2=(range(18, 27), goal2),
                                                   # 18-27= coffee cream to milk , 14-23 = tea milk to cream
                                                   switch_sequence=2,
                                                   # lesion_observation_units=True,
                                                   # lesion_action_units=True,
                                                   lesion_goal2_units=False,
                                                   lesion_goal1_units=False,
                                                   noise_per_step=False,
                                                   constant_noise_to_input=0.,
                                                   noise_per_step_to_input=False,
                                                   noise_per_step_to_hidden=False,
                                                   disruption_per_step=False,
                                                   constant_noise=0.1,
                                                   initialization=utils.ZERO_INIT,
                                                   clamped_goals=False,
                                                   verbose=True)
        # utils.save_object("test_data_1", test_data)

        result = goalenv2020.analyse_test_data(test_data, goals=goals,
                                               do_special_representations=False,
                                               do_tsne=False,
                                               do_rdm=False,
                                               mds_range=50,
                                               mds_sequences=range(21),
                                               verbose=True,
                                               append_to_file=filename)
        results.append(result)

def rdm_distances_analysis():
    if True:  # New Cognitive Load model.
        # a) Load network. Just the last one.
        # b) Generate test data

        """
        test_data = goalenv2020.generate_test_data(model, noise=0.,
                       goal1_noise=0., goal2_noise=0.,
                       goals=True, num_tests=1,
                       goal_multiplier=1,
                       sequence_ids=range(21), ##[3, 16, 16],;  #0=coffee black, 3 = coffee cream, 16 = tea milk
                       #switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                       #switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                       switch_sequence=2,
                       noise_per_step=False,
                       noise_per_step_to_input=False,
                       disruption_per_step=False,
                       initialization=utils.SEMINORMAL,
                       clamped_goals=False,
                       hidden_goal_multiplier=1.)
                       #gain_multiplier = goal_multiplier,
                       #gain_multiplier_from=0,
                       #gain_multiplier_to=50)
        #c) Analyze test data:
        test_data = goalenv2020.analyse_test_data(test_data, goals=True, do_special_representations=True, do_tsne=True, do_rdm=True,
                                                  mds_range=50, mds_sequences=[0, 1, 2, 3])

        """
        # m1_egds_avg = 0
        # m2_egds_avg = 0
        # m1_dges_avg = 0
        # m2_digeqs_avg = 0
        # m1_eqgdiss_avg = 0
        # m2_eqgdiss_avg = 0
        # m1_m2_avg = 0
        # m1_eqgs_cnt = 0
        # m2_eqgs_cnt = 0
        # m1_digeqs_cnt = 0
        # m2_digeqs_cnt = 0
        # m1_eqgdiss_cnt = 0
        # m2_eqgdiss_cnt = 0
        # m1_m2_cnt = 0

        import rdm
        from os import path
        for network_id in range(50):
            try:
                my_rdm = utils.load_object("rdm_for_network" + str(network_id))
            except:  # generate the RDM
                model = utils.load_object("bigmodel1_1goals_relu_adam_nonoise", network_id)
                # b) Generate test data
                test_data = goalenv2020.generate_test_data(model, noise=0.0,
                                                           goal1_noise=0., goal2_noise=0.,
                                                           goals=True, num_tests=1,
                                                           goal_multiplier=1,
                                                           sequence_ids=range(21),
                                                           ##[3, 16, 16],;  #0=coffee black, 3 = coffee cream, 16 = tea milk
                                                           # switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                           # switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                                                           # switch_sequence=2,
                                                           noise_per_step=False,
                                                           noise_per_step_to_input=False,
                                                           disruption_per_step=False,
                                                           initialization=utils.SEMINORMAL,
                                                           clamped_goals=False)
                # hidden_goal_multiplier=1.)
                # gain_multiplier = goal_multiplier,
                # gain_multiplier_from=0,
                # gain_multiplier_to=50)
                # c) Analyze test data:
                test_data = goalenv2020.analyse_test_data(test_data, goals=True, do_special_representations=True,
                                                          do_tsne=False, do_rdm=True,
                                                          mds_range=50, mds_sequences=range(21))
                activations1 = test_data[-5]
                properties1 = test_data[-4]

                test_data = goalenv2020.generate_test_data(model, noise=0.,
                                                           goal1_noise=0., goal2_noise=0.,
                                                           goals=True, num_tests=1,
                                                           goal_multiplier=2,
                                                           sequence_ids=range(21),
                                                           ##[3, 16, 16],;  #0=coffee black, 3 = coffee cream, 16 = tea milk
                                                           # switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                           # switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                                                           # switch_sequence=2,
                                                           noise_per_step=False,
                                                           noise_per_step_to_input=False,
                                                           disruption_per_step=False,
                                                           initialization=utils.SEMINORMAL,
                                                           clamped_goals=False)
                # hidden_goal_multiplier=1.)
                # gain_multiplier = goal_multiplier,
                # gain_multiplier_from=0,
                # gain_multiplier_to=50)
                # c) Analyze test data:
                test_data = goalenv2020.analyse_test_data(test_data, goals=True, do_special_representations=True,
                                                          do_tsne=False, do_rdm=True,
                                                          mds_range=50, mds_sequences=range(21))
                activations2 = test_data[-5]
                properties2 = test_data[-4]

                for property in properties1:
                    property["multiplier"] = "1"
                for property in properties2:
                    property["multiplier"] = "2"

                # mark as erroneous the properties1 equivalents of properties2.
                for idx, property2 in enumerate(properties2):
                    if property2["error"] == "true":
                        # look for corresponding property1
                        for property1 in properties1:
                            if property1["seq"] == property2["seq"]:
                                property1["error"] = "true"

                # Now compute dimensionality based exclusively on the non-erroneous activations. Simple.
                # 1. remove all erroneous actions.
                errors1 = []
                for idx, prop in enumerate(properties1):
                    if prop["error"] == "true":
                        errors1.append(idx)
                for idx in reversed(errors1):
                    del activations1[idx]
                    del properties1[idx]
                errors2 = []
                for idx, prop in enumerate(properties2):
                    if prop["error"] == "true":
                        errors2.append(idx)
                for idx in reversed(errors2):
                    del activations2[idx]
                    del properties2[idx]

                matrices = []
                matrix1 = np.stack(activations1)
                matrix2 = np.stack(activations2)
                average1 = np.average(matrix1)
                average2 = np.average(matrix2)
                """ 
                ranks1 = []
                ranks1_xaverage = []
                print("rank per tolerance (combined):")
                for tolerance in [0.01, 0.1, 1.0, 1.5, 3, 5, 10., 15, 30, 50, 75, 100, 150]:
                    print("tol={0}, rank={1}".format(tolerance * average1,
                                                     np.linalg.matrix_rank(matrix1, tol=tolerance * average1)))
                    ranks1_xaverage.append(np.linalg.matrix_rank(matrix1, tol=tolerance * average1))
                    ranks1.append(np.linalg.matrix_rank(matrix1, tol=tolerance))
                ranks2 = []
                ranks2_xaverage = []
                print("rank per tolerance (combined):")
                for tolerance in [0.01, 0.1, 1.0, 1.5, 3, 5, 10., 15, 30, 50, 75, 100, 150]:
                    print("tol={0}, rank={1}".format(tolerance * average2,
                                                     np.linalg.matrix_rank(matrix2, tol=tolerance * average2)))
                    ranks2_xaverage.append(np.linalg.matrix_rank(matrix2, tol=tolerance * average2))
                    ranks2.append(np.linalg.matrix_rank(matrix2, tol=tolerance))
                # print the rank comparisons for 1 and 2
                print("rank comparisons")
                for i in range(len(ranks2)):
                    print("{0};vs;{1};{2};vs;{3};".format(ranks1[i], ranks2[i], ranks1_xaverage[i], ranks2_xaverage[i]))
                """

                # Now make an RDM with activations 1 and activations 2

                my_rdm = rdm.rdm(properties=properties1 + properties2, vectors=activations1 + activations2,
                                 type=rdm.EUCLIDIAN)
                # save the fucking rdms so I don't have to compute that again ugh
                utils.save_object("rdm_for_network" + str(network_id), my_rdm)

            print("\n\n----------------\nNETWORK " + str(network_id))

            # Equal goal equal sequence
            m1_eges = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                equals=["goal1", "goal2", "seq"])
            print("multiplier 1 - equal goals, equal sequence: \n{0}".format(m1_eges))
            # m1_eges_avg += m1_eges
            # m1_eges_cnt += 1

            m2_eges = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                equals=["goal1", "goal2", "seq"])
            print("multiplier 2 - equal goals, equal sequence: \n{0}".format(m2_eges))
            # m2_eges_avg += m2_eges
            # m2_eges_cnt += 1

            # Equal goals different sequence
            m1_egds = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                equals=["goal1", "goal2"], unequals=["seq"])
            print("multiplier 1 - equal goals, different sequence: \n{0}".format(m1_egds))
            # m1_egds_avg += m1_egds
            # m1_egds_cnt += 1

            m2_egds = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                equals=["goal1", "goal2"], unequals=["seq"])
            print("multiplier 2 - equal goals, different sequence: \n{0}".format(m2_egds))
            # m2_egds_avg += m2_egds
            # m2_egds_cnt += 1

            # Different goals equal sequence
            m1_dges = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                equals=["seq"], unequals_or=["goal1", "goal2"])
            print("multiplier 1 - different goals and equal sequence: \n{0}".format(m1_dges))
            # m1_dges_avg += m1_dges
            # m1_dges_cnt += 1

            m2_dges = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                equals=["seq"], unequals_or=["goal1", "goal2"])
            print("multiplier 2 - different goals and equal sequence: \n{0}".format(m2_dges))
            # m2_dges_avg += m2_dges
            # m2_dges_cnt += 1

            # Different goal different sequence
            m1_dgds = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                unequals=["seq"], unequals_or=["goal1", "goal2"])
            print("multiplier 1 - different goals, different sequence: \n{0}".format(m1_dgds))
            # m1_dgds_avg += m1_dgds
            # m1_dgds_cnt += 1

            m2_dgds = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                unequals=["seq"], unequals_or=["goal1", "goal2"])
            print("multiplier 2 - different goals, different sequence: \n{0}".format(m2_dgds))
            # m2_dgds_avg += m2_dgds
            # m2_dgds_cnt += 1

            # Different goal
            m1_dg = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                unequals_or=["goal1", "goal2"])
            print("multiplier 1 - different goals: \n{0}".format(m1_dg))
            # m1_dg_avg += m1_dg
            # m1_dg_cnt += 1

            m2_dg = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                unequals_or=["goal1", "goal2"])
            print("multiplier 2 - different goals: \n{0}".format(m2_dg))
            # m2_dg_avg += m2_dg
            # m2_dg_cnt += 1

            # Equal goal
            m1_eg = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                equals=["goal1", "goal2"], unequals=[])
            print("multiplier 1 - equal goals: \n{0}".format(m1_eg))
            # m1_eg_avg += m1_eg
            # m1_eg_cnt += 1

            m2_eg = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                equals=["goal1", "goal2"], unequals=[])
            print("multiplier 2 - equal goals: \n{0}".format(m2_eg))
            # m2_eg_avg += m2_eg
            # m2_eg_cnt += 1

            # Different sequence
            m1_ds = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                equals=[], unequals=["seq"])
            print("multiplier 1 - different sequence: \n{0}".format(m1_ds))
            # m1_ds_avg += m1_ds
            # m1_ds_cnt += 1

            m2_ds = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                equals=[], unequals=["seq"])
            print("multiplier 2 - different sequence: \n{0}".format(m2_ds))
            # m2_ds_avg += m2_ds
            # m2_ds_cnt += 1

            # Equal sequence
            m1_es = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                equals=["seq"], unequals=[])
            print("multiplier 1 - equal sequence: \n{0}".format(m1_es))
            # m1_eqgdiss_avg += m1_es
            # m1_eqgdiss_cnt += 1

            m2_es = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                equals=["seq"], unequals=[])
            print("multiplier 2 - equal sequence: \n{0}".format(m2_es))
            # m2_es_avg += m2_es
            # m2_es_cnt += 1

            # Any (average distance)
            m1_any = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"})
            print("multiplier 1 - any: \n{0}".format(m1_any))
            # m1_any_avg += m1_any
            # m1_any_cnt += 1

            m2_any = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"})
            print("multiplier 2 - any: \n{0}".format(m2_any))
            # m2_any_avg += m2_any
            # m2_any_cnt += 1

            # Same sequence and steps
            m1_m2 = my_rdm.get_average_key(
                keys_values={"error": "false"},
                equals=["goal1", "goal2", "seq", "step"], unequals=["multiplier"])
            print("multiplier 1 vs 2, same sequences and steps: \n{0}".format(m1_m2))
            # m1_m2_avg += m1_m2
            # m1_m2_cnt += 1

            # Different goal AND Different subgoal And Different sequence
            m1_ddd = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                unequals=["seq", "goal1", "goal2"])
            print("multiplier 1 - different goals, subgoals, sequence: \n{0}".format(m1_ddd))
            m2_ddd = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                unequals=["seq", "goal1", "goal2"])
            print("multiplier 2 - different goals, subgoals, sequence: \n{0}".format(m2_ddd))

            # Different goal AND Different subgoal
            m1_dd = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "error": "false"},
                unequals=["goal1", "goal2"])
            print("multiplier 1 - different goals, subgoals: \n{0}".format(m1_dd))
            m2_dd = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "error": "false"},
                unequals=["goal1", "goal2"])
            print("multiplier 2 - different goals, subgoals: \n{0}".format(m2_dd))
        exit()

        """    
        print("---------\nTOTALS !!")
        print(m1_eqgs_avg/m1_eqgs_cnt)
        print(m2_eqgs_avg/m2_eqgs_cnt)
        print(m1_digeqs_avg/m1_digeqs_cnt)
        print(m2_digeqs_avg/m2_digeqs_cnt)
        print(m1_eqgdiss_avg/m1_eqgdiss_cnt)
        print(m2_eqgdiss_avg/m2_eqgdiss_cnt)
        print(m1_m2_avg/m1_m2_cnt)
        exit()
        """
        # 2. With amplified goals. (a) RDM (b) MDS (c) t-SNE
        # THEN compare networks with and without goals?
        # For this I need 25 networks with goals and 25 without goals.
        # Make RDM average and MDS average; Euclidian and Spearman.


def lesion_analysis_or_something():  # probably obsolete??
    # Trained:
    # - "bigmodel1_yesgoals": 15 networks  (uniform initialization, sigmoid)
    # - "bigmodel1_yesgoals_2nd_batch": 1 network (same)
    # - "bigmodel1_yesgoals_relu": 4 networks, HE initialization, seminormal context initialization. UNSURE, RETRAIN
    # - "bigmodel1_yesgoals_relu": 4 networks, HE initialization, normal context initialization
    error_data_list = []
    print("ONLY GOALS")
    for model_type in ["goals"]:
        print(model_type)
        if model_type == "goals":
            goals = True
            model_files = "bigmodel1_yesgoals_relu_adam_nonoise"
        elif model_type == "no_goals":
            goals = False
            model_files = "bigmodel1_nogoals_relu_adam_nonoise"
        else:
            raise ValueError()
        for lesion in [7, 8, 9]:  # [1, 2, 3, 4, 5, 6]:
            for noise in [0.0]:
                print("\n\n\n" + str(noise))
                for i in range(10):
                    print(i)

                    model = utils.load_object(model_files, i)  # "bigmodel1_nogoals_relu_adam_nonoise", i)

                    lesion_goal1 = lesion_goal2 = lesion_actions = lesion_observation = False
                    if lesion == 1:
                        lesion_goal1 = True
                    elif lesion == 2:
                        lesion_goal2 = True
                    elif lesion == 3:
                        lesion_actions = True
                    elif lesion == 4:
                        lesion_goal1 = lesion_goal2 = True
                    elif lesion == 5:
                        lesion_goal1 = lesion_goal2 = lesion_actions = True
                    elif lesion == 6:
                        lesion_goal2 = lesion_actions = True
                    elif lesion == 7:
                        lesion_observation = True
                    elif lesion == 8:
                        lesion_goal1 = lesion_actions = True
                    elif lesion == 9:
                        lesion_observation = lesion_actions = True
                    test_data = goalenv2020.generate_test_data(model, noise=noise, goal1_noise=0., goal2_noise=0.,
                                                               goals=goals, num_tests=10, sequence_ids=range(21),
                                                               noise_per_step=False, noise_per_step_to_input=False,
                                                               disruption_per_step=False,
                                                               initialization=nn.SEMINORMAL,
                                                               lesion_goal1_units=lesion_goal1,
                                                               lesion_goal2_units=lesion_goal2,
                                                               lesion_action_units=lesion_actions,
                                                               lesion_observation_units=lesion_observation)

                    tsne_results, test_data, _, error_data, _ = goalenv2020.analyse_test_data(test_data,
                                                                                              do_rdm=False,
                                                                                              goals=False)
                    error_data_list.append(error_data)

    utils.write_lists_to_csv("error_results_lesion.csv", error_data_list, labels=goalenv2020.error_testing_labels)

    """
    error_data_list = []

    model = utils.load_object("bigmodel1_yesgoals_relu_adam_nonoise")

    test_data = goalenv2020.generate_test_data(model, noise=1.,
                                               goal1_noise=0., goal2_noise=0.,
                                               goals=True, num_tests=100,
                                               sequence_ids=range(21),
                                               noise_per_step=True,
                                               noise_per_step_to_input=False,
                                               disruption_per_step=False,
                                               initialization=nn.SEMINORMAL)
    print("generated")
    utils.save_object("test_data_error_test", test_data)
    test_data = utils.load_object("test_data_error_test")
    goalenv2020.VERBOSE=True
    tsne_results, test_data, _, error_data, _ = goalenv2020.analyse_test_data(test_data, do_rdm=False, goals=False)

    sys.exit()
    """
    error_data_list = []
    for model_type in ["goals", "no_goals"]:
        print(model_type)
        if model_type == "goals":
            goals = True
            model_files = "bigmodel1_yesgoals_relu_adam_nonoise"
        elif model_type == "no_goals":
            goals = False
            model_files = "bigmodel1_nogoals_relu_adam_nonoise"
        else:
            raise ValueError()
        for noise in [1.0, 2.0, 3.0]:
            print("\n\n\n" + str(noise))
            for i in range(10):
                print(i)
                model = nn.ElmanGoalNet(size_hidden=50, size_observation=29, size_action=19,
                                        size_goal1=len(env.GoalEnvData.goals1_list),
                                        size_goal2=len(env.GoalEnvData.goals2_list),
                                        algorithm=nn.ADAM, learning_rate=0.001,
                                        L2_reg=0.0001,
                                        initialization=nn.HE,
                                        nonlinearity=nn.RELU,
                                        last_action_inputs=True)

                stopping = nn.ParamsStopping(max_iterations=25000, min_iterations=3010, check_frequency=1000,
                                             stop_condition=goalenv2020.stop_condition, goals=True, noise=0.0)
                model = goalenv2020.train(stop_params=stopping, model=model, goals=True,
                                          noise=0.0, sequences=range(21), context_initialization=nn.SEMINORMAL)
                utils.save_object("bigmodel1_yesgoals_relu_adam_nonoise", model)

                model = utils.load_object(model_files, i)  # "bigmodel1_nogoals_relu_adam_nonoise", i)

                test_data = goalenv2020.generate_test_data(model, noise=noise,
                                                           goal1_noise=0., goal2_noise=0.,
                                                           goals=goals, num_tests=3,
                                                           sequence_ids=range(21),
                                                           noise_per_step=True,
                                                           noise_per_step_to_input=False,
                                                           disruption_per_step=False,
                                                           initialization=nn.SEMINORMAL)

                utils.save_object("test_data_error_test" + model_type + str(noise) + str(i), test_data)
                test_data = utils.load_object("test_data_error_test" + model_type + str(noise) + str(i))
                tsne_results, test_data, _, error_data, _ = goalenv2020.analyse_test_data(test_data, do_rdm=False,
                                                                                          goals=True)  # , mds_sequences=[2, 5, 11], mds_range=15)
                error_data_list.append(error_data)

                # utils.save_object("tsne_bigmodel1_yesgoals", tsne_results)
                utils.save_object("tsnetest_bigmodel1_yesgoals", test_data)
                goalenv2020.plot_tsne(tsne_results, test_data, tsne_goals=False, tsne_subgoals=False,
                                      tsne_actions=False, tsne_sequences=True,
                                      tsne_errors=True, tsne_sequence=[2, 5, 11], tsne_sequence_interval=[2, 14],
                                      filename="tsne", annotate=False)
                utils.save_object("tsne_results_bigmodel1_yesgoals", tsne_results)

    utils.write_lists_to_csv("loss_six.csv", error_data_list, labels=goalenv2020.error_testing_labels)

def basic_analysis2():
    from goalenv import goalenv2020
    from goalenv import environment as env
    import tensorflow as tf
    for i in range(0):
        print(i)
        model = nn.GoalNet(size_hidden=50, size_observation=29, size_action=19,
                           size_goal1=len(env.GoalEnvData.goals1_list),
                           size_goal2=len(env.GoalEnvData.goals2_list),
                           algorithm=nn.ADAM, learning_rate=0.001,
                           L2_reg=0.0001,
                           initialization=nn.HE,
                           nonlinearity=nn.RELU,
                           last_action_inputs=True)

        stopping = nn.ParamsStopping(max_iterations=25000, min_iterations=1, check_frequency=1000,
                                     stop_condition=goalenv2020.stop_condition, goals=True, noise=0.0)
        model = goalenv2020.train(stop_params=stopping, model=model, goals=True,
                                  noise=0.0, sequences=range(21), context_initialization=nn.SEMINORMAL,
                                  gradient=False)
        #utils.save_object("kitchenv_relu_adam_nonoise_", model) #gradient
        #utils.save_object("kitchenv_relu_adam_nonoise_nogoals", model) #no goals
        utils.save_object("kitchenv_relu_adam_goals", model) #goals
        #utils.save_object("bigmodel1_relu_adam_nonoise_gradient", model)
        #utils.save_object("bigmodel1_yesgoals_relu_adam_nonoise_goaltest", model)
    for i in range(0):
        print(i)
        model = nn.GoalNet(size_hidden=50, size_observation=29, size_action=19,
                           size_goal1=0,  #len(env.GoalEnvData.goals1_list),
                           size_goal2=0,  #len(env.GoalEnvData.goals2_list),
                           algorithm=nn.ADAM, learning_rate=0.001,
                           L2_reg=0.0001,
                           initialization=nn.HE,
                           nonlinearity=nn.RELU,
                           last_action_inputs=True)

        stopping = nn.ParamsStopping(max_iterations=25000, min_iterations=5001, check_frequency=1000,
                                     stop_condition=goalenv2020.stop_condition, goals=False, noise=0.0)
        model = goalenv2020.train(stop_params=stopping, model=model, goals=False,
                                  noise=0.0, sequences=range(21), context_initialization=nn.SEMINORMAL,
                                  gradient=False)
        # utils.save_object("kitchenv_relu_adam_nonoise_", model) #gradient
        # utils.save_object("kitchenv_relu_adam_nonoise_nogoals", model) #no goals
        utils.save_object("kitchenv_relu_adam_nogoals", model)  # goals
        # utils.save_object("bigmodel1_relu_adam_nonoise_gradient", model)
        # utils.save_object("bigmodel1_yesgoals_relu_adam_nonoise_goaltest", model)
    #sys.exit()
    for i in range(0):
        print(i)
        model = nn.GoalNet(size_hidden=50, size_observation=29, size_action=19,
                           size_goal1=len(env.GoalEnvData.goals1_list),
                           size_goal2=len(env.GoalEnvData.goals2_list),
                           algorithm=nn.ADAM, learning_rate=0.001,
                           L2_reg=0.0001,
                           initialization=nn.HE,
                           nonlinearity=nn.RELU,
                           last_action_inputs=True)

        stopping = nn.ParamsStopping(max_iterations=15000, min_iterations=10000, check_frequency=1000,
                                     stop_condition=goalenv2020.stop_condition, goals=True, noise=0.0)
        model = goalenv2020.train(stop_params=stopping, model=model, goals=True,
                                  noise=0.0, sequences=range(21), context_initialization=nn.SEMINORMAL,
                                  gradient=True)
        # utils.save_object("kitchenv_relu_adam_nonoise_", model) #gradient
        # utils.save_object("kitchenv_relu_adam_nonoise_nogoals", model) #no goals
        utils.save_object("kitchenv_relu_adam_gradient", model)  # goals
        # utils.save_object("bigmodel1_relu_adam_nonoise_gradient", model)
        # utils.save_object("bigmodel1_yesgoals_relu_adam_nonoise_goaltest", model)
    #analysis.plot_tsne("mds_tsne.txt", "tsne.svg")
    #sys.exit()

    import time
    start = time.time()

    error_data_list = []
    for model_type in ["goals"]:#, "nogoals"]:
        goals = model_type == "goals"
        for goal_multiplier in [1.]:
            print("goal multiplier:")
            print(goal_multiplier)
            for noise in [0]:
                print("noise:")
                print(noise)
                for clamped_goals in [False]: #[True, False]:
                    print("clamped goals")
                    print(clamped_goals)
                    for i in [11]:#range(1):
                        print("time elapsed: {0}s".format(time.time() - start))
                        print("Network:")
                        print(i)
                        model = utils.load_object("kitchenv_relu_adam_"+model_type, i)
                        #model = utils.load_object("bigmodel1_" + model_type +"_relu_adam_nonoise_goaltest", i)
                        #model = utils.load_object("bigmodel1_relu_adam_nonoise_gradient", i)
                        goal1 = utils.str_to_onehot("g_1_make_tea", env.GoalEnvData.goals1_list) * 4  #np.zeros((1, 2), np.float32)
                        #goal2 = utils.str_to_onehot("g_2_infuse_tea", env.GoalEnvData.goals2_list) #np.zeros((1, 9), np.float32)
                        #goal2 = utils.str_to_onehot("g_2_add_milk", env.GoalEnvData.goals2_list) * 2. # np.zeros((1, 9), np.float32)
                        goal2 = utils.str_to_onehot("g_2_add_cream", env.GoalEnvData.goals2_list) * 4

                        if False:
                            with tf.device('/cpu:0'):
                                #print("time elapsed: {0}s".format(time.time() - start))
                                test_data = goalenv2020.generate_test_data(model, noise=noise,
                                                                       goal1_noise=0., goal2_noise=0.,
                                                                       goals=goals, num_tests=1,
                                                                       goal_multiplier=goal_multiplier,
                                                                       sequence_ids=[3, 16, 16], ##[3, 16, 16],;  #0=coffee black, 3 = coffee cream, 16 = tea milk
                                                                       #switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                                       #switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                                                                       switch_sequence=2,
                                                                       noise_per_step=False,
                                                                       noise_per_step_to_input=False,
                                                                       disruption_per_step=False,
                                                                       initialization=nn.SEMINORMAL,
                                                                       clamped_goals = clamped_goals)
                                                                       #hidden_goal_multiplier=1.)
                                                                       #gain_multiplier = goal_multiplier,
                                                                       #gain_multiplier_from=0,
                                                                       #gain_multiplier_to=50)
                        #print("time elapsed: {0}s".format(time.time() - start))
                        print("generated data")

                        utils.save_object("control"+model_type+str(i), test_data)
                        #test_data = utils.load_object("control" + model_type + str(i))
                        goalenv2020.VERBOSE = True
                        tsne_results, test_data, _, error_data, _ = goalenv2020.analyse_test_data(test_data, do_rdm=True, rdm_sort=goalenv2020.GOAL + goalenv2020.SUBGOAL + goalenv2020.ACTION,
                                                                                                do_tsne=True, do_loss=True,
                                                                                                goals=goals, one_rdm=True)#, mds_sequences=[0, 1, 2])#, mds_range=15)
                        #utils.write_line_to_csv("goalmultiplierinputFINAL.csv", error_data)
                        #error_data_list.append(error_data)
                        utils.save_object("tsne_resultsmds"+model_type+str(i), tsne_results)
                        utils.save_object("mds"+model_type+str(i), test_data) # test dat ais updated by analysis
                        #tsne_results = utils.load_object("tsne_resultsmds"+model_type+str(i))
                        #test_data = utils.load_object("mds" + model_type + str(i))
                        goalenv2020.plot_tsne(tsne_results, test_data, tsne_goals=False, tsne_subgoals=True, tsne_actions=False, tsne_sequences=False,
                                  tsne_errors=False, tsne_sequence=[0, 1, 2], #tsne_sequence_interval=[1, 30],
                                  filename="mds", annotate=True, save_txt=True)
                        #utils.save_object("tsne_results_bigmodel1_yesgoals", tsne_results)
    analysis.plot_tsne("mds_tsne.txt", "tsne.svg")
    #utils.write_lists_to_csv("clamped_goals_noise0.csv", error_data_list, labels=goalenv2020.error_testing_labels)

def some_early_obsolete_scripts():
    # env = environment.GoalEnv()
    # env.test_environment(task.sequences_list)
    # sys.exit()
    # model = goalenv2020.train(goals=False, num_iterations=100000, learning_rate=0.0001, L2_reg=0.00001, noise=0., sequences=[0, 1, 2, 3, 4, 5])

    # model = utils.load_object("bigmodel1")
    # goalenv2020.accuracy_test_botvinick(model, num_tests=100, sequences=[0, 1, 2, 3, 4, 5])
    # sys.exit()

    # model = utils.load_object("bigmodel1")

    # bigmodel: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.002, L2_reg=0.00001, noise=0.1, sequences=range(21))
    # bigmodel1: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.002, L2_reg=0.00001, noise=0., sequences=range(21))
    # bigmodel2: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.002, L2_reg=0.00001, noise=0.05, sequences=range(21))
    # bigmodel3: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.001, L2_reg=0.00001, noise=0.01, sequences=range(21)) --> 70% correct at 30000
    # bigmodel4: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.003, L2_reg=0.00001, noise=0.01, sequences=range(21)) --> 60% correct
    # bigmodel5:     model = goalenv2020.train(goals=True, num_iterations=50000, learning_rate=0.0001, L2_reg=0.00001, noise=0.01, sequences=range(21)) --> 23% but still improving
    # bigmodel6: 6 model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.003, L2_reg=0.0, noise=0.01, sequences=range(21)) 46%, still improving
    # bigmodel7: model = goalenv2020.train(goals=True, num_iterations=30000, learning_rate=0.005, L2_reg=0.0, noise=0.01, sequences=range(21))

    # goalenv2020.accuracy_test_botvinick(model, noise=0.1, goals=True, num_tests=10, sequence_ids=range(21))
    # goalenv2020.accuracy_test_noise(model, noise=0.4, goals=True, num_tests=10, sequence_ids=range(21))
    # goalenv2020.accuracy_test_botvinick(model, noise=0.5, noise_step=5, num_tests= 10, goals=True, sequence_ids=range(21))
    for i in range(0):
        # model = utils.load_object("bigmodel3")
        model = nn.GoalNet(size_hidden=50, size_observation=29, size_action=19,
                           size_goal1=len(environment.GoalEnvData.goals1_list),
                           size_goal2=len(environment.GoalEnvData.goals2_list),
                           algorithm=nn.RMSPROP, learning_rate=0.001, initialization="uniform",
                           last_action_inputs=True)
        model = goalenv2020.train(model=model, goals=True, num_iterations=150000, learning_rate=0.001, L2_reg=0.00001,
                                  noise=0.05, sequences=range(21))
        utils.save_object("bigmodel4", model)
    # sys.exit()
    model = utils.load_object("bigmodel4")
    # sys.exit()
    # test_data = utils.load_object("test_data_tsne")
    # test_data = goalenv2020.generate_test_data(model, noise=0.5, one_run_per_step=True, goal1_noise=0., goal2_noise=0., goals=True, num_tests=3, sequence_ids=range(21))
    test_data = goalenv2020.generate_test_data(model, noise=0., goal1_noise=0., goal2_noise=0., goals=True, num_tests=1,
                                               sequence_ids=range(21), noise_per_step=True, disruption_per_step=False,
                                               initialization="seminormal")
    tsne_results, test_data = goalenv2020.analyse_test_data(test_data, do_error_analysis=True, do_rdm=False)
    utils.save_object("test_data_tsne_disrupt", test_data)
    utils.save_object("tsne_results_disrupt", tsne_results)
    test_data = utils.load_object("test_data_tsne_disrupt")
    tsne_results = utils.load_object("tsne_results_disrupt")
    # goalenv2020.plot_tsne(tsne_results, test_data, tsne_actions=True, filename="tsne_actions")
    # goalenv2020.plot_tsne(tsne_results, test_data, tsne_subgoals=True, filename="tsne_subgoals")
    # goalenv2020.plot_tsne(tsne_results, test_data, tsne_goals=True, filename="tsne_goals")
    # goalenv2020.plot_tsne(tsne_results, test_data, tsne_sequences=True, filename="tsne_sequences")
    # goalenv2020.plot_tsne(tsne_results, test_data, tsne_errors=True, filename="tsne_errors")
    # goalenv2020.plot_tsne(tsne_results, test_data, tsne_actions=True, annotate=True, tsne_sequence=[1548, 1550, 1614], tsne_sequence_interval=[14, 34], filename="tsne_actions_plus_seqs")
    sys.exit()
    # model = utils.load_object("bigmodel1")
    # goalenv2020.accuracy_test_botvinick(model, noise=0.15, goals=True, sequence_ids=range(21))
    # goalenv2020.accuracy_test_botvinick(model, noise=0.2, goals=True, sequence_ids=range(21))
    # goalenv2020.accuracy_test_botvinick(model, noise=0.25, goals=True, sequence_ids=range(21))
    # goalenv2020.accuracy_test_botvinick(model, noise=0.35, goals=True, sequence_ids=range(21))
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
    # for i in range(1):
    model = goalenv2020.train(goals=True, num_iterations=15000, learning_rate=0.0003, L2_reg=0.0001, noise=0.,
                              sequences=range(21))
    utils.save_object("bigmodel3", model)
    # model = utils.load_object("bigmodel")
    goalenv2020.accuracy_test_botvinick(model, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.05, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.1, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.2, goals=True, sequence_ids=range(21))
    goalenv2020.accuracy_test_botvinick(model, noise=0.3, noise_step=5, goals=True, sequence_ids=range(21))

    for i in range(0):
        print("Another 10,000 steps of training to reach " + str(i) + ",000 steps")
        goalenv2020.train(model, goals=True, num_iterations=10000, learning_rate=0.0003, L2_reg=0.0001, noise=0.,
                          sequences=range(21))
        utils.save_object("bigmodel3" + str(i), model)
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

    # goalenv.train(goals=False, num_iterations=100000, learning_rate=0.0003, L2_reg=0.0001, noise=0., sequences=[0, 1, 2, 3])
    # goalenv.train(goals=False, num_iterations=100000, learning_rate=0.0003, L2_reg=0.0001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
    # goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.0001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
    # goalenv.train(goals=False, num_iterations=50000, learning_rate=0.01, L2_reg=0.0001, noise=0., sequences=[0, 1])
    # goalenv.train(goals=False, num_iterations=50000, learning_rate=0.1, L2_reg=0.0001, noise=0., sequences=[0, 1])
    # goalenv.train(goals=False, num_iterations=200000, learning_rate=0.0001, L2_reg=0.000001, noise=0., sequences=[0, 1, 2, 3, 4, 5])
