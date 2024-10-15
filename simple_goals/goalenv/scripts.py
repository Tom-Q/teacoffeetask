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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import patches
import copy
import rdm

def analyse_rdm_distances_output():
    readfile = open("console_output_logs_0vs1.txt", "r")

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


def leave_one_out_analysis():
    # 1. load each network
    # 2. test it on every sequence
    # 3. ideally it should get all sequences right EXCEPT sometimes one.
    noise = 0.
    goal1_noise = 0.
    goal2_noise = 0.
    goals = True
    num_tests = 1
    sequence_ids = range(21)
    noise_per_step = False
    disruption_per_step = False
    context_initialization = utils.SEMINORMAL
    do_rdm = False
    for to_rem in [[0], [1], [2], [3], [4,6], [5,7], [8], [9,11], [10,12], [13], [14], [15], [16], [17,19], [18,20]]:
        for model_num in [0, 1]:
            print("excluded: {0}, model: {1}".format(to_rem, model_num))
            to_rem_str = ""
            for i in to_rem:
                to_rem_str += str(i)
            print("\n##################\n####################\n############# "+to_rem_str+" ###############\n###############\n##############\n")
            model = utils.load_object("bigmodel1_1goals_relu_adam_nonoise_excluding" + to_rem_str, model_num)
            test_data = goalenv2020.generate_test_data(model, noise=noise,
                                           goal1_noise=goal1_noise, goal2_noise=goal2_noise,
                                           goals=goals, num_tests=num_tests,
                                           sequence_ids=sequence_ids,
                                           noise_per_step=noise_per_step,
                                           disruption_per_step=disruption_per_step,
                                           initialization=context_initialization,
                                           verbose=False)
            tsne_results, test_data, total_errors, _, goal_errors, _, _, _, _, _ = goalenv2020.analyse_test_data(test_data,
                                                                                                     do_rdm=do_rdm,
                                                                                                     goals=goals,
                                                                                                     verbose=False,
                                                                                                     sequence_ids=sequence_ids)


def train_networks_leave1out():
    for g in [1]:
        sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        #for to_rem in [[0], [1], [2], [3], [4, 6], [5, 7], [8], [9, 11], [10, 12], [13], [14], [15], [16], [17, 19], [18, 20]]:
        for to_rem in [[15], [16], [17, 19], [18, 20]]:
            print("\n\nMODEL: MODEL: MODEL:"+str(to_rem)+"\nMODEL MODEL MODEL\n")
            seqs_minus_1 = copy.deepcopy(sequences)
            for i in to_rem:
                seqs_minus_1.remove(i)
            stopping = nn.ParamsStopping(max_iterations=20000, min_iterations=0, check_frequency=1000,
                                         stop_condition=goalenv2020.stop_condition, goals=bool(g), noise=0.0,
                                         context_initialization=utils.ZERO_INIT, sequence_ids=seqs_minus_1)
            goalenv2020.BONUS_ITERATIONS = 0
            model = nn.GoalNet(size_hidden=50, size_observation=29, size_action=19,  # 50
                               size_goal1=len(env.GoalEnvData.goals1_list) * g,
                               size_goal2=len(env.GoalEnvData.goals2_list) * g,
                               algorithm=optimizers.ADAM, learning_rate=0.001,  # 0.001
                               L2_reg=0.,  # 0.0001,
                               initialization=utils.HE,  # HE
                               nonlinearity=tf.nn.relu,  # relu
                               last_action_inputs=True)
            model = goalenv2020.train(stop_params=stopping, model=model, goals=bool(g),
                                      noise=0.0, sequences=seqs_minus_1, context_initialization=utils.ZERO_INIT) # normal context initialization to try and improve learning?
            to_rem_str = ""
            for i in to_rem:
                to_rem_str += str(i)
            utils.save_object("bigmodel1_"+str(g)+"goals_relu_adam_nonoise_excluding"+to_rem_str, model)


def train_networks_100_units(n):
    for g in [1, 0]:
        for i in range(n):
            sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            #for to_rem in [[0], [1], [2], [3], [4, 6], [5, 7], [8], [9, 11], [10, 12], [13], [14], [15], [16], [17, 19], [18, 20]]:
            #for to_rem in [[15], [16], [17, 19], [18, 20]]:

            print("\n\nMODEL: MODEL: MODEL:"+str(i)+"\nMODEL MODEL MODEL\n")
            seqs_minus_1 = copy.deepcopy(sequences)
            #    for i in to_rem:
            #        seqs_minus_1.remove(i)
            stopping = nn.ParamsStopping(max_iterations=20000, min_iterations=0, check_frequency=1000,
                                         stop_condition=goalenv2020.stop_condition, goals=bool(g), noise=0.0,
                                         context_initialization=utils.ZERO_INIT, sequence_ids=seqs_minus_1)
            goalenv2020.BONUS_ITERATIONS = 0
            model = nn.GoalNet(size_hidden=100, size_observation=29, size_action=19,  # 50
                               size_goal1=len(env.GoalEnvData.goals1_list) * g,
                               size_goal2=len(env.GoalEnvData.goals2_list) * g,
                               algorithm=optimizers.ADAM, learning_rate=0.001,  # 0.001
                               L2_reg=0.,  # 0.0001,
                               initialization=utils.HE,  # HE
                               nonlinearity=tf.nn.relu,  # relu
                               last_action_inputs=True)
            model = goalenv2020.train(stop_params=stopping, model=model, goals=bool(g),
                                      noise=0.0, sequences=seqs_minus_1, context_initialization=utils.ZERO_INIT) # normal context initialization to try and improve learning?
            utils.save_object("bigmodel1_"+str(g)+"goals_relu_adam_nonoise_100_units", model)

def control_analysis():
    to_rem = [[0], [1], [2], [3], [4, 6], [5, 7], [8], [9, 11], [10, 12], [13], [14], [15], [16], [17, 19], [18, 20]]
    #to_rem = [[13], [14], [15], [16], [17, 19], [18, 20]]
    filenames = []
    for i in to_rem:
        to_rem_str = ""
        for j in i:
            to_rem_str += str(j)
        filenames.append("bigmodel1_1goals_relu_adam_nonoise_excluding" + to_rem_str)
    for goal_multiplier in [1.0, 1.5, 2.0]:
        print("\n\n\n\t\t#############\n\t\t##############\n\t\t###############\n\t\t#############\n\t\t#############")
        print("\t\t#############\n\t\t##############\n\t\t###############\n\t\t#############\n\t\t#############")
        print(goal_multiplier)
        print("\n\n\n\t\t#############\n\t\t##############\n\t\t###############\n\t\t#############\n\t\t#############")
        print("\t\t#############\n\t\t##############\n\t\t###############\n\t\t#############\n\t\t#############")
        for idx, filename in enumerate(filenames):
            model = utils.load_object(filename)
            removed_seqs = to_rem[idx]
            print("\n\n\n\t\t#############\n\t\t##############\n\t\t###############\n\t\t#############\n\t\t#############")
            print(filename)

            for removed_seq in removed_seqs:
                print(removed_seq)
                control_analysis_1model(model, removed_seq, goal_multiplier)

def control_analysis_1model(model, sequence_id, goal_multiplier):
    filename="results_last_resort3.csv"
    results = []

    test_data = goalenv2020.generate_test_data(model, noise=0.0,
                                               goal1_noise=0., goal2_noise=0.,
                                               goals=True, num_tests=1,
                                               goal_multiplier=goal_multiplier,
                                               sequence_ids=[sequence_id],  #0=coffee black, 2 =coffee 2 sugars 3 = coffee cream, 16 = tea milk
                                               #lesion_observation_units=True,
                                               #lesion_action_units=True,
                                               lesion_goal2_units=False,  # goal 2 is lower level.
                                               lesion_goal1_units=False,
                                               noise_per_step=False,
                                               constant_noise_to_input=0.,
                                               noise_per_step_to_input=False,
                                               #noise_per_step_to_hidden=2.0,
                                               disruption_per_step=False,
                                               constant_noise=0.,
                                               initialization=utils.ZERO_INIT,
                                               clamped_goals=False,
                                               verbose=False)
    #utils.save_object("test_data_1", test_data)

    result = goalenv2020.analyse_test_data(test_data, goals=True,
                                              do_special_representations=False,
                                              do_tsne=False,
                                              do_rdm=False,
                                              do_dimensionality=False,
                                              #mds_range=50,
                                              mds_sequences=range(21),
                                              verbose=True,
                                              append_to_file=filename,
                                              sequence_ids=[sequence_id])
    results.append(result)
    #goalenv2020.plot_tsne(result[0], result[1], tsne_goals=False, tsne_subgoals=True,
    #                      tsne_actions=False, tsne_sequences=False,
    #                      tsne_errors=False, tsne_sequence=[0, 1, 2],  # tsne_sequence_interval=[1, 30],
    #                      filename=str(network_id)+"mds", annotate=True, save_txt=True)
    #utils.save_object("tsne_results_bigmodel1_yesgoals", tsne_results)
    #plot_tsne(str(network_id)+"mds_tsne.txt", str(network_id)+"tsne.svg")



def basic_analysis(nets, model_file_name, goals):  # Currently set up for the distance to error analysis. Can also do lots of other things
    #filename="distance_to_error_2.csv"
    filename="results_last_resort4.csv"
    results = []
    #for noise in [0., 0.01, 0.1, 0.2, 0.5, 1.]:
    for noise in [0.]:# reversed([1., 2., 3., 4., 5.]):
        #for goals in [False]:#, False]:
            if goals:
                #goal_multipliers = [0.5, 0.9, 1.0, 1.01, 1.05, 1.1, 1.2, 1.5, 2.]
                goal_multipliers = [1.]#, 2.0]#, 1.25, 1.5, 1.75]#, 2.]
            else:
                goal_multipliers = [1.0]
            for goal_multiplier in reversed(goal_multipliers):
                print("Hidden layer noise " + str(noise) + " - Goal: " + str(goals) + " (multiplier:"+str(goal_multiplier)+") \n")
                myfile = open(filename, "a")
                myfile.write("Hidden layer noise " + str(noise) + " - Goal: " + str(goals) + " (multiplier:"+str(goal_multiplier)+") \n")
                myfile.close()
                # 15 is the one that normally works
                for network_id in nets:#range(0, 50):#[6, 15, 16, 18, 21, 23, 26, 27, 33, 34, 36, 37, 38, 39, 41, 47]:#[14,18,20,22,26,27,33,38,42,43,46,49]:#range(50):
                    #print(network_id)
                    #print(goals)
                    #network_id=10
                    print("\nNETWORK:" + str(network_id+1))
                    goal_str = "1" if goals else "0"
                    #model = utils.load_object("bigmodel1_"+ goal_str + "goals_relu_adam_nonoise", network_id)
                    #model = utils.load_object("bigmodel1_" + goal_str + "goals_relu_adam_nonoise_100_units", network_id)
                    model = utils.load_object(model_file_name, network_id)
                    #print(model.hidden_layer.layer.b.numpy()[0][0])

                    # Goals 1 and 2
                    #goal1 = utils.str_to_onehot("g_1_make_tea", env.GoalEnvData.goals1_list)
                    #goal2 = utils.str_to_onehot("g_2_add_cream", goalenv.environment.GoalEnvData.goals2_list) * 3
                    #goal2 = utils.str_to_onehot("g_2_add_cream", env.GoalEnvData.goals2_list)
                    #goal2 = utils.str_to_onehot("g_2_drink_tea", goalenv.environment.GoalEnvData.goals2_list) * 1.0
                    test_data = goalenv2020.generate_test_data(model, noise=0.0,
                                                               goal1_noise=0., goal2_noise=0.,
                                                               goals=goals, num_tests=1,
                                                               goal_multiplier=goal_multiplier,
                                                               sequence_ids=range(21),
                                                               #sequence_ids=[0, 13],
                                                               #sequence_ids=[16, 16],  #0=coffee black, 2 =coffee 2 sugars 3 = coffee cream, 16 = tea milk
                                                               #switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                               #switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream, 25-28 3rd sugar in coffee
                                                               #switch_sequence=1,
                                                               #lesion_observation_units=True,
                                                               #lesion_action_units=True,
                                                               lesion_goal2_units=False,  # goal 2 is lower level.
                                                               lesion_goal1_units=False,
                                                               noise_per_step=False,
                                                               constant_noise_to_input=0.,
                                                               noise_per_step_to_input=False,
                                                               #noise_per_step_to_hidden=2.0,
                                                               disruption_per_step=False,
                                                               constant_noise=0.,
                                                               initialization=utils.ZERO_INIT,
                                                               clamped_goals=False,
                                                               verbose=False)
                    #utils.save_object("test_data_1", test_data)

                    result = goalenv2020.analyse_test_data(test_data, goals=goals,
                                                              do_special_representations=False,
                                                              do_tsne=False,
                                                              do_rdm=False,
                                                              do_dimensionality=True,
                                                              #mds_range=50,
                                                              #mds_sequences=range(21),
                                                              verbose=False,
                                                              append_to_file=filename,
                                                              sequence_ids=range(21))#[16, 16])
                    results.append(result)
                    #goalenv2020.plot_tsne(result[0], result[1], tsne_goals=False, tsne_subgoals=True,
                    #                      tsne_actions=False, tsne_sequences=False,
                    #                      tsne_errors=False, tsne_sequence=[0, 1, 2],  # tsne_sequence_interval=[1, 30],
                    #                      filename=str(network_id)+"mds", annotate=True, save_txt=True)
                    #utils.save_object("tsne_results_bigmodel1_yesgoals", tsne_results)
                    #plot_tsne(str(network_id)+"mds_tsne.txt", str(network_id)+"tsne.svg")
    #import sys
    #sys.exit()
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
    """
    # Get rid of all the steps past step 10.
    properties_flat = []
    activations_flat = []
    test_data = np.concatenate([result[1] for result in results])
    labels = []

    multipliers = ["2.0", "1.0", "0.0"] #, "0.5", "0.0"]
    for idx, result in enumerate(results):
        # set up the multiplier information
        for property in result[6]:
            property["goal_multiplier"] = multipliers[idx]

        # get rid of anything above step 10 for each of the 2 sequences each time
        i = 0
        while i < len(result[6]):
            prev_seq = result[6][i]["seq"]
            j = 0
            while i+j < len(result[6]) and prev_seq == result[6][i+j]["seq"]:
                if j < 10:
                    properties_flat.append(result[6][i+j])
                    activations_flat.append(result[5][i+j])
                    labels.append(result[6][i]["step"])
                j+=1
            i+=max(1, j)
    activations_flat = np.stack(activations_flat)
    """
    # Make an average RDM from the networks. This assumes all networks are correct, and it will average out different orders.
    rdm_sum = None
    for result in results:
        activations = result[5]
        properties = result[4]
        rdm_individual = rdm.rdm(properties, vectors=activations, type=rdm.EUCLIDIAN)
        if rdm_sum is None:
            rdm_sum = rdm_individual
        else:
            rdm_sum.matrix += rdm_individual.matrix

    # And that's my rdm
    rdm_sum.matrix /= len(results)

    # make an RDM combining all the results
    #activations_flat = np.concatenate([result[5] for result in results], axis=0)
    # add 0, 0.5, 1., 1.5, 2.0 to the properties
    #multipliers = ["2.0", "1.0", "0.0"] #, "0.5", "0.0"]
    #for idx, result in enumerate(results):
    #    for property in result[6]:
    #        property["goal_multiplier"] = multipliers[idx]
    ## stack up the properties
    #properties_flat = np.concatenate([result[6] for result in results])
    #test_data = np.concatenate([result[1] for result in results])

    # Make an rdm
    #import rdm
    #my_rdm = rdm.rdm(properties_flat, vectors=activations_flat, type=rdm.EUCLIDIAN)

    # make an MDS
    # Generate the MDS from the RDM.
    #import analysis
    #mdsy = analysis.mds(my_rdm.matrix)
    #print("mds done")

    #mdsy_idx = 0

    #colors = ['b', 'r', 'g', 'k', 'c', 'm'] * 10  #

    #i = 0
    #num_seqs = 0
    #while i < len(properties_flat):
    #    property = properties_flat[i]
    #    prev_seq = property["seq"]
    #    prev_mult = property["goal_multiplier"]
    #    labels = []
    #    j = 0
    #    while i+j < len(properties_flat) and prev_seq == properties_flat[i+j]["seq"] and prev_mult == properties_flat[i+j]["goal_multiplier"]:
    #        #if j == 0:
    #        #    labels.append("seq " + properties_flat[i+j]["seq"] + "/" + properties_flat[i+j]["goal_multiplier"])
    #        #else:
    #        labels.append(properties_flat[i+j]["step"])
    #        j+=1
    #    analysis.plot_mds_points(mdsy[i:i + j], range(j), labels=labels,  # labels,
    #                             style=colors[num_seqs] + '-', #'x-'
    #                             fontsize=32,
    #                             legend_label="seq " + properties_flat[i]["seq"] + "/" + properties_flat[i]["goal_multiplier"])
    #                            # style=list(colors.values())[(i)%len(colors)], fontsize=32)
    #    i+=j
    #    num_seqs+=1
    # specify order of items in legend
    #handles, labels = plt.gca().get_legend_handles_labels()
    #order = [0, 2, 4, 1, 3, 5]
    #labels = ['coffee 0x', 'tea 0x', 'coffee 1x', 'tea 1x', 'coffee 2x', 'tea 2x']
    #plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper left", fontsize=16)

    #plt.title("MDS")
    # plt.show()
    #plt.savefig("mds_test")
    #plt.clf()

    # also make a t-sne
    #from sklearn.manifold import TSNE
    #tsne = TSNE(perplexity=20.0, n_iter=1000)  # Defaults are perplexity=30 and n_iter=1000.
    # Reducing perplexity captures global structure better. increasing n_iter makes sure we've converged.
    #tsne_results = tsne.fit_transform(X=activations_flat)
    #results1 = np.concatenate([result[1] for result in results])
    #goalenv2020.plot_tsne(tsne_results, test_data, tsne_goals=False, tsne_subgoals=False,
    #                      tsne_actions=False, tsne_sequences=False,
    #                      tsne_errors=False, tsne_sequence=range(21), tsne_sequence_interval=[0, 20],
    #                      filename="tsne_test_5_"+str(nets[0]), annotate=False, save_txt=False, legend=nets[0]==0)#,
    #                      #custom_legend=[handles, labels, order])
    #plt.clf()
    # utils.save_object("tsne_results_bigmodel1_yesgoals", tsne_results)
    return rdm_sum

def basic_analysis_mds():
    filename="mds.csv"
    #filename="results_last_resort3.csv"
    results = []
    #for noise in [0., 0.01, 0.1, 0.2, 0.5, 1.]:
    for noise in [0]:#reversed([1., 2., 3., 4., 5.]):
        for goals in [True]:#, False]:
            if goals:
                #goal_multipliers = [0.5, 0.9, 1.0, 1.01, 1.05, 1.1, 1.2, 1.5, 2.]
                goal_multipliers = [1.0]#, 2.0]#, 1.25, 1.5, 1.75]#, 2.]
            else:
                goal_multipliers = [1.0]
            for goal_multiplier in reversed(goal_multipliers):
                print("Hidden layer noise " + str(noise) + " - Goal: " + str(goals) + " (multiplier:"+str(goal_multiplier)+") \n")
                myfile = open(filename, "a")
                myfile.write("Hidden layer noise " + str(noise) + " - Goal: " + str(goals) + " (multiplier:"+str(goal_multiplier)+") \n")
                myfile.close()
                for network_id in range(20, 25):#[6, 15, 16, 18, 21, 23, 26, 27, 33, 34, 36, 37, 38, 39, 41, 47]:#[14,18,20,22,26,27,33,38,42,43,46,49]:#range(50):
                    #print(network_id)
                    #print(goals)
                    #network_id=10
                    print("\nNETWORK:" + str(network_id+1))
                    goal_str = "1" if goals else "0"
                    model = utils.load_object("bigmodel1_"+ goal_str + "goals_relu_adam_nonoise", network_id)
                    #print(model.hidden_layer.layer.b.numpy()[0][0])

                    # Goals 1 and 2
                    #goal1 = utils.str_to_onehot("g_1_make_tea", env.GoalEnvData.goals1_list) * 3.
                    #goal2 = utils.str_to_onehot("g_2_add_cream", goalenv.environment.GoalEnvData.goals2_list) * 3
                    #goal2 = utils.str_to_onehot("g_2_add_cream", env.GoalEnvData.goals2_list) * 3.
                    #goal2 = utils.str_to_onehot("g_2_drink_tea", goalenv.environment.GoalEnvData.goals2_list) * 1.0
                    test_data = goalenv2020.generate_test_data(model, noise=noise,
                                                               goal1_noise=0., goal2_noise=0.,
                                                               goals=goals, num_tests=1,
                                                               goal_multiplier=[1., 3.],
                                                               sequence_ids=[1, 1],#[3, 16, 16],  #0=coffee black, 2 =coffee 2 sugars 3 = coffee cream, 16 = tea milk
                                                               #switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                               #switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream, 25-28 3rd sugar in coffee
                                                               #switch_sequence=2,
                                                               #lesion_observation_units=True,
                                                               #lesion_action_units=True,
                                                               lesion_goal2_units=False,  # goal 2 is lower level.
                                                               lesion_goal1_units=False,
                                                               noise_per_step=False,
                                                               constant_noise_to_input=0.,
                                                               noise_per_step_to_input=False,
                                                               #noise_per_step_to_hidden=2.0,
                                                               disruption_per_step=False,
                                                               constant_noise=0.0,
                                                               initialization=utils.ZERO_INIT,
                                                               clamped_goals=False,
                                                               verbose=False)
                    #utils.save_object("test_data_1", test_data)

                    result = goalenv2020.analyse_test_data(test_data, goals=goals,
                                                              do_special_representations=False,
                                                              do_tsne=False,
                                                              do_rdm=True,
                                                              mds_range=50,
                                                              mds_sequences=[0, 1], #range(0),#[3, 16, 16],#range(21),
                                                              verbose=False,
                                                              append_to_file=filename,
                                                              sequence_ids=[1, 1],
                                                              net_name=str(network_id))#[3, 16, 16])
                    results.append(result)
                    #goalenv2020.plot_tsne(result[0], result[1], tsne_goals=False, tsne_subgoals=True,
                    #                      tsne_actions=False, tsne_sequences=False,
                    #                      tsne_errors=False, tsne_sequence=[0, 1, 2],  # tsne_sequence_interval=[1, 30],
                    #                      filename=str(network_id)+"mds", annotate=True, save_txt=True)
                    # utils.save_object("tsne_results_bigmodel1_yesgoals", tsne_results)
                    #plot_tsne(str(network_id)+"mds_tsne.txt", str(network_id)+"tsne.svg")
    import sys
    sys.exit()
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
        goal1 = utils.str_to_onehot("g_1_make_tea", env.GoalEnvData.goals1_list) * 2
        goal2 = utils.str_to_onehot("g_2_add_cream", env.GoalEnvData.goals2_list) * 2
        import rdm
        from os import path
        for network_id in range(50):
            try:
                my_rdm = utils.load_object("rdm_for_network_a_switch_x3" + str(network_id))
            except:  # generate the RDM
                model = utils.load_object("bigmodel1_1goals_relu_adam_nonoise", network_id)
                # b) Generate test data
                test_data = goalenv2020.generate_test_data(model, noise=0.0,
                                                           goal1_noise=0., goal2_noise=0.,
                                                           goals=True, num_tests=1,
                                                           goal_multiplier=1,
                                                           sequence_ids=[16, 16],  #0=coffee black, 3 = coffee cream, 16 = tea milk
                                                           switch_goal1=(range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                           switch_goal2=(range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                                                           switch_sequence=1,
                                                           constant_noise=0.01,
                                                           noise_per_step=False,
                                                           noise_per_step_to_input=False,
                                                           disruption_per_step=False,
                                                           initialization=utils.ZERO_INIT,
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
                                                           sequence_ids=[16, 16],  #0=coffee black, 3 = coffee cream, 16 = tea milk
                                                           switch_goal1=(range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                           switch_goal2=(range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                                                           switch_sequence=1,
                                                           noise_per_step=False,
                                                           noise_per_step_to_input=False,
                                                           disruption_per_step=False,
                                                           initialization=utils.ZERO_INIT,
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

                # Add the multiplier and "switch" properties
                for property in properties1:
                    property["multiplier"] = "1"
                for property in properties2:
                    property["multiplier"] = "2"
                seq = 0
                for property in properties1:
                    if property["step"]=="1":
                        seq += 1
                    if seq == 1:
                        property["switch"] = "false"
                    else:
                        property["switch"] = "true"
                seq = 0
                for property in properties2:
                    if property["step"]=="1":
                        seq += 1
                    if seq == 1:
                        property["switch"] = "false"
                    else:
                        property["switch"] = "true"

                # mark as erroneous the properties1 equivalents of properties2.
                for idx, property2 in enumerate(properties2):
                    if property2["error"] == "true":
                        # look for corresponding property1
                        for property1 in properties1:
                            if property1["seq"] == property2["seq"]:
                                property1["error"] = "true"

                # Now compute dimensionality based exclusively on the non-erroneous activations. Simple.
                    #... or keep the erroneous activations!! if commented out
                # 1. remove all erroneous actions.
                #errors1 = []
                #for idx, prop in enumerate(properties1):
                #    if prop["error"] == "true":
                #        errors1.append(idx)
                #for idx in reversed(errors1):
                #    del activations1[idx]
                #    del properties1[idx]
                #errors2 = []
                #for idx, prop in enumerate(properties2):
                #    if prop["error"] == "true":
                #        errors2.append(idx)
                #for idx in reversed(errors2):
                #    del activations2[idx]
                #    del properties2[idx]

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
                utils.save_object("rdm_for_network_a_switch_x2" + str(network_id), my_rdm)

            print("\n\n----------------\nNETWORK " + str(network_id))


            # Equal goal, equal sequence
            m1m2_switch = my_rdm.get_average_key(
                keys_values={"switch": "true", "step":"17"},
                equals=["step"], unequals=["multiplier"])
            print("17 switch = true, dist. multiplier 1 vs multiplier 2: \n{0}".format(m1m2_switch))
            m1m2_switch = my_rdm.get_average_key(
                keys_values={"switch": "true"},
                equals=["step"], unequals=["multiplier"])
            print("!17 switch = true, dist. multiplier 1 vs multiplier 2: \n{0}".format(m1m2_switch))


            m1m2_noswitch = my_rdm.get_average_key(
                keys_values={"switch": "false", "step":"17"},
                equals=["step"], unequals=["multiplier"])
            print("17 switch = false, dist. multiplier 1 vs multiplier 2: \n{0}".format(m1m2_noswitch))
            m1m2_noswitch = my_rdm.get_average_key(
                keys_values={"switch": "false"},
                equals=["step"], unequals=["multiplier"])
            print("!17 switch = false, dist. multiplier 1 vs multiplier 2: \n{0}".format(m1m2_noswitch))


            m1_ss = my_rdm.get_average_key(
                keys_values={"multiplier": "1", "step":"17"},
                equals=["step"], unequals=["switch"])
            print("17 switch vs no switch multiplier 1: \n{0}".format(m1_ss))

            m1_ss = my_rdm.get_average_key(
                keys_values={"multiplier": "1"},
                equals=["step"], unequals=["switch"])
            print("!17 switch vs no switch multiplier 1: \n{0}".format(m1_ss))

            m2_ss = my_rdm.get_average_key(
                keys_values={"multiplier": "2", "step":"17"},
                equals=["step"], unequals=["switch"])
            print("17 switch vs no switch multiplier 2: \n{0}".format(m2_ss))
            m2_ss = my_rdm.get_average_key(
                keys_values={"multiplier": "2"},
                equals=["step"], unequals=["switch"])
            print("!17 switch vs no switch multiplier 2: \n{0}".format(m2_ss))

            """
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
    for i in range(10):
        print(i)
        model = nn.GoalNet(size_hidden=100, size_observation=29, size_action=19,
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
        utils.save_object("kitchenv_relu_adam_goals_100units", model) #goals
        #utils.save_object("bigmodel1_relu_adam_nonoise_gradient", model)
        #utils.save_object("bigmodel1_yesgoals_relu_adam_nonoise_goaltest", model)
    for i in range(10):
        print(i)
        model = nn.GoalNet(size_hidden=100, size_observation=29, size_action=19,
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
        utils.save_object("kitchenv_relu_adam_nogoals_100units", model)  # goals
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
                    for i in range(50):#range(1):
                        print("time elapsed: {0}s".format(time.time() - start))
                        print("Network:")
                        print(i)
                        #model = utils.load_object("kitchenv_relu_adam_"+model_type, i)
                        #model = utils.load_object("bigmodel1_" + model_type +"_relu_adam_nonoise_goaltest", i)
                        model = utils.load_object("bigmodel1_relu_adam_nonoise_gradient", i)
                        goal1 = utils.str_to_onehot("g_1_make_tea", env.GoalEnvData.goals1_list) * 4  #np.zeros((1, 2), np.float32)
                        #goal2 = utils.str_to_onehot("g_2_infuse_tea", env.GoalEnvData.goals2_list) #np.zeros((1, 9), np.float32)
                        #goal2 = utils.str_to_onehot("g_2_add_milk", env.GoalEnvData.goals2_list) * 2. # np.zeros((1, 9), np.float32)
                        goal2 = utils.str_to_onehot("g_2_add_cream", env.GoalEnvData.goals2_list) * 4

                        if True:
                            with tf.device('/cpu:0'):
                                #print("time elapsed: {0}s".format(time.time() - start))
                                test_data = goalenv2020.generate_test_data(model, noise=noise,
                                                                       goal1_noise=0., goal2_noise=0.,
                                                                       goals=goals, num_tests=1,
                                                                       goal_multiplier=goal_multiplier,
                                                                       sequence_ids=[3, 16, 16], ##[3, 16, 16],;  #0=coffee black, 3 = coffee cream, 16 = tea milk
                                                                       switch_goal1= (range(14, 23), goal1), #  does this work?? # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                                       switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
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
                        plot_tsne("mds_tsne.txt", "tsne" + str(i) + ".svg")
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


from matplotlib.ticker import FuncFormatter
def barplot_figure_errors(filename):
    # goals
    means_action_errors = (7.13, 37.42, 62.66, 75.52, 80.81)
    means_subseq_errors = (12.47, 22.73, 20.79, 17.64, 15.51)

    # no goals
    #means_action_errors = (6.25, 35.41, 59.96, 75.15, 81.25)
    #means_subseq_errors = (8.57, 18.64, 19.22, 16.22, 14.44)


    ind = [1, 2, 3, 4, 5]  # the x locations for the groups
    width = 0.35
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ind, means_subseq_errors, width, color='cornflowerblue')
    ax.bar(ind, means_action_errors, width, bottom=means_subseq_errors, color='orange')
    ax.bar(ind, [100]*len(means_subseq_errors), width, bottom=[a+b for (a, b) in zip(means_action_errors, means_subseq_errors)], color='bisque')
    ax.set_ylabel('Outcomes')
    ax.set_xlabel('Noise magnitude')
    #ax.set_title('Percentage of sequences displaying action or subsequence errors')
    plt.xticks(ind, ('1', '2', '3', '4', '5'))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim([0, 100])
    ax.legend(labels=['Subsequence error', 'Action error', 'Success'], loc="upper left")
    plt.tight_layout()
    #plt.show()
    #plt.title(title)
    plt.savefig(filename)
    plt.clf()

def plot_pca():
    pca_avgs_goal1 = 100*np.array([0.1876,0.1009,0.0825,0.0699,0.0606,0.0535,0.0467,0.0409,0.0364,0.0326,0.0291,0.0263,0.0235,0.0209,0.0182,0.0167,0.0147,0.0129,0.0118,0.0107])
    pca_conf_int_goal1 = np.array([0.0241,0.0233,0.0200,0.0177,0.0155,0.0140,0.0125,0.0117,0.0109,0.0101,0.0093,0.0089,0.0085,0.0079,0.0073,0.0068,0.0063,0.0059,0.0055,0.0051])

    pca_avgs_goal15 = 100 * np.array([0.2213,0.1211,0.0819,0.0681,0.0564,0.0487,0.0420,0.0374,0.0332,0.0295,0.0262,0.0238,0.0209,0.0188,0.0162,0.0149,0.0131,0.0119,0.0107,0.0095])
    pca_conf_int_goal15 = np.array([0.0257,0.0262,0.0224,0.0195,0.0167,0.0145,0.0130,0.0121,0.0112,0.0103,0.0093,0.0086,0.0077,0.0072,0.0066,0.0062,0.0056,0.0052,0.0048,0.0045])

    pca_avgs_nogoals = 100 * np.array([0.1256,0.1061,0.0936,0.0804,0.0706,0.0618,0.0535,0.0463,0.0407,0.0347,0.0305,0.0266,0.0233,0.0211,0.0189,0.0163,0.0143,0.0128,0.0114,0.0102])
    pca_conf_int_nogoals = np.array([0.0058,0.0092,0.0113,0.0121,0.0124,0.0127,0.0133,0.0133,0.0127,0.0118,0.0109,0.0101,0.0091,0.0082,0.0074,0.0066,0.0061,0.0057,0.0053,0.0049])

    sns.set()

    def cumulative_in_place(arr):
        prev = None
        for i in range(len(arr)):
            if i == 0:
                prev = arr[i]
            else:
                arr[i] += prev
                prev = arr[i]
    #for array in [pca_avgs_goal1, pca_avgs_goal15, pca_avgs_nogoals]:
    #    cumulative_in_place(array)

    N = 20
    x = np.arange(1, 21, 1)
    plt.plot(x, pca_avgs_nogoals, 'g-', label='Elman network')
    plt.fill_between(x, pca_avgs_nogoals - pca_conf_int_nogoals, pca_avgs_nogoals + pca_conf_int_nogoals, color='g', alpha=0.2)
    plt.plot(x, pca_avgs_goal1, 'b-', label='Goal network (goals × 1.0)')
    plt.fill_between(x, pca_avgs_goal1 - pca_conf_int_goal1, pca_avgs_goal1 + pca_conf_int_goal1, color='b', alpha=0.2)
    plt.plot(x, pca_avgs_goal15, 'r-', label='Goal network (goals × 1.5)')
    plt.fill_between(x, pca_avgs_goal15 - pca_conf_int_goal15, pca_avgs_goal15 + pca_conf_int_goal15, color='r', alpha=0.2)

    plt.xticks(np.arange(1, 20 + 1, 1.0))
    plt.ylim(ymin=0) #ymax=100)
    plt.xlim(xmin=0)
    plt.xlabel("Component")
    plt.ylabel("Variance explained (%)")
    plt.legend(title='Proportion of variance explained')
    plt.show()


def barplot_figure_ablations(filename):
    # Observation, Actions, Goal1, Goal2, Goal1 & 2,
    #means_action_errors = (81.48, 58.29, 18.10, 3.57, 5.10)
    #means_subseq_errors = (32.48, 22.57, 61.90, 57.10, 7.62)


    means_action_errors = (81.33, 71.05, 43.52, 7.14, 44.10)
    means_subseq_errors = (18.57, 18.86, 17.52, 55.71, 51.81)

    # no goals
    #means_action_errors = (6.25, 35.41, 59.96, 75.15, 81.25)
    #means_subseq_errors = (8.57, 18.64, 19.22, 16.22, 14.44)


    ind = [1, 2, 3, 4, 5]  # the x locations for the groups
    width = 0.35
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ind, means_subseq_errors, width, color='orange')
    ax.bar(ind, means_action_errors, width, bottom=means_subseq_errors, color='cornflowerblue')
    ax.bar(ind, [100]*len(means_subseq_errors), width, bottom=[a+b for (a, b) in zip(means_action_errors, means_subseq_errors)], color='bisque')
    ax.set_ylabel('Outcomes')
    ax.set_xlabel('Units lesioned')
    #ax.set_title('Percentage of sequences displaying action or subsequence errors')
    plt.xticks(ind, ['Observations', 'Actions', 'Subgoals', 'Goals', 'Subgoals and Goals'], rotation=20)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim([0, 100])
    ax.legend(labels=['Subsequence error', 'Action error', 'Success'], loc="upper right")
    plt.tight_layout()
    #plt.show()
    #plt.title(title)
    plt.savefig(filename)
    plt.clf()

def bargraph_with_without_goalunits(filename):
    # Updated info based on 20 networks of each, 1 run each.
    # goals
    means_action_errors_goals = (0.02, 0.76, 5.36, 59.88, 93.51)
    means_subseq_errors_goals = (0.05, 2.33, 18.04, 33.38, 4.63)

    # no goals
    means_action_errors_nogoals = (0.07, 0.28, 5.72, 50.24, 90.12)
    means_subseq_errors_nogoals = (0.17, 0.93, 25.86, 43.62, 8.09)

    ind = [1., 2., 3., 4., 5.]  # the x locations for the groups
    width = 0.2
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    idxs_goals = [id+0.11 for id in ind]
    plt.rcParams['hatch.color'] = 'white'  # using "edgecolor" causes a white edge to appear in between the different bars.
    ax.bar(idxs_goals, means_subseq_errors_goals, width, color='orangered', hatch='////')
    ax.bar(idxs_goals, means_action_errors_goals, width, bottom=means_subseq_errors_goals, color='orange', hatch='////')
    ax.bar(idxs_goals, [100]*len(means_subseq_errors_goals), width, bottom=[a+b for (a, b) in zip(means_action_errors_goals, means_subseq_errors_goals)], color='bisque', hatch='////')

    idxs_nogoals = [id-0.11 for id in ind]
    ax.bar(idxs_nogoals, means_subseq_errors_nogoals, width, color='orangered')
    ax.bar(idxs_nogoals, means_action_errors_nogoals, width, bottom=means_subseq_errors_nogoals, color='orange')
    ax.bar(idxs_nogoals, [100] * len(means_subseq_errors_goals), width, bottom=[a + b for (a, b) in zip(means_action_errors_nogoals, means_subseq_errors_nogoals)], color='bisque')
    ax.set_ylabel('Outcomes')
    ax.set_xlabel('Noise magnitude')
    #ax.set_title('Percentage of sequences displaying action or subsequence errors')
    plt.xticks(ind, ('0.01', '0.1', '0.2', '0.5', '1.0'))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    formatter = FuncFormatter(lambda y, pos: "%d%%" % (y))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylim([0, 100])
    legend_elements = [patches.Patch(facecolor='orangered', label='Subsequence error'),
                       patches.Patch(facecolor='orange', label='Action error'),
                       patches.Patch(facecolor='bisque', label='Correct'),
                       #Patch(facecolor='white', edgecolor='grey', label='Goals'),
                       patches.Patch(facecolor='grey', edgecolor='white', label='Flat network'),
                       patches.Patch(facecolor='grey', edgecolor='white', hatch='////', label='Goal network')]

    ax.legend(handles=legend_elements, loc='upper left')

    #ax.legend(labels=['Subsequence error', 'Action error', 'Success'], loc="upper left")
    plt.tight_layout()
    #plt.show()
    #plt.title(title)
    plt.savefig(filename)
    plt.clf()

def loss_big_plot(filename="loss_plots"):
    fig = plt.figure()
    plt.clf()
    #plt.title('Loss per timestep between noise injection and error')
    plotcount = 0
    for length in [3, 6, 10]:
        for goals in [True, False]:
            plotcount += 1
            ax1 = fig.add_subplot(320 + plotcount)
            _loss_plot(ax1, length, 2, goals)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def _loss_plot(ax, length, noise_level, goals):
    if length == 3 and noise_level == 2:
        if goals:
            noise_actions = [3.28,	3.26,	3.26,	5.02]
            noise_goals = [2.69,	3.35,	3.89,	4.31]
        else:
            noise = [3.26,	3.30,	3.16,	5.17]
    elif length == 6 and noise_level == 2:
        if goals:
            noise_actions = [3.62, 3.67, 3.01, 3.75, 3.83, 4.74, 6.46]
            noise_goals = [3.22,	3.42,	4.93,	6.83,	7.73,	8.81,	10.47]
        else:
            noise = [3.76,	3.53,	3.11,	3.13,	3.23,	3.36,	5.25]
    elif length == 10 and noise_level == 2:
        if goals:
            noise_actions = [3.71,	3.41,	3.24,	2.85,	3.29,	3.03,	2.61,	3.43,	3.60,	4.12,	6.75]
            noise_goals = [3.84,	2.57,	3.32,	4.21,	3.41,	4.25,	6.00,	8.51,	8.82,	9.62,	11.64]
        else:
            noise = [4.91,	4.85,	5.10,	4.48,	4.87,	4.22,	3.64,	3.88,	4.25,	4.30,	6.11]

    ax.title.set_text('Error occurs on step ' + str(length))
    if goals:
        timestep = range(len(noise_actions))
        ax.plot(timestep, noise_goals, color='orangered', marker='+', label="Loss: goals")
        ax.plot(timestep, noise_actions, color='orange', marker='+', label="Loss: actions")
    else:
        timestep = range(len(noise))
        ax.plot(timestep, noise, color='orange', marker='+', label="Loss: actions")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_ticks(timestep)

    #ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Timesteps since noise injection")

import matplotlib.lines as plt_lines
def plot_tsne(filefrom, fileto):
    points = np.loadtxt(filefrom)
    points = points.T#reshape((2, -1))
    print(points)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    points_seq1 = points[:, 0:39]
    points_seq2 = points[:, 39:74]
    points_seq3 = points[:, 74:109]

    # plot seq 1
    ax.plot(points_seq1[0, :], points_seq1[1, :], color='orange', linestyle='solid', linewidth=0.6)#, marker='+')

    # Plot seq 2
    ax.plot(points_seq2[0, :], points_seq2[1, :], color='brown', linestyle='solid', linewidth=0.6)#, marker='+')

    # Plot seq 3
    ax.plot(points_seq3[0, :], points_seq3[1, :], color='#79afdaff', linestyle='solid', linewidth=0.6)#, marker='+')

    # start points
    ax.plot(points_seq1[0, 0], points_seq1[1, 0], color='orange', marker='>')
    ax.plot(points_seq2[0, 0], points_seq2[1, 0], color='brown', marker='>')
    ax.plot(points_seq3[0, 0], points_seq3[1, 0], color='#79afdaff', marker='>')

    # End points
    ax.plot(points_seq1[0, -1], points_seq1[1, -1], color='orange', marker='o')
    ax.plot(points_seq2[0, -1], points_seq2[1, -1], color='brown', marker='o')
    ax.plot(points_seq3[0, -1], points_seq3[1, -1], color='#79afdaff', marker='o')

    ax.plot(points_seq3[0, 27:36], points_seq3[1, 27:36], linestyle='solid', color='#79afdaff', linewidth=1.4)#, marker='v', markerfacecolor='black')
    ax.plot(points_seq3[0, 13:23], points_seq3[1, 13:23], linestyle='solid', color='#79afdaff', linewidth=1.4)#marker='v', markerfacecolor='white')
    ax.plot(points_seq3[0, 28:36], points_seq3[1, 28:36], linestyle='', color='#79afdaff', linewidth=1.4, marker='.')  # , marker='v', markerfacecolor='black')
    ax.plot(points_seq3[0, 14:23], points_seq3[1, 14:23], linestyle='', color='#79afdaff', linewidth=1.4, marker='.')  # marker='v', markerfacecolor='white')

    legend_elements = [plt_lines.Line2D([0], [0], color='orange', label='Coffee cream', linestyle='solid', linewidth=0.6),
                       plt_lines.Line2D([0], [0], color='brown', label='Tea milk', linestyle='solid', linewidth=0.6), #linestyle='dashed'),
                       plt_lines.Line2D([0], [0], color='#79afdaff', label='Tea cream', linestyle='solid', linewidth=0.6), #linestyle='dotted'),
                       #plt_lines.Line2D([0], [0], marker='>', color='w', label='Sequence start', markerfacecolor='k'),
                       #plt_lines.Line2D([0], [0], marker='o', color='w', label='Sequence end', linestyle='', markerfacecolor='k'),
                       plt_lines.Line2D([0], [0], marker='.', linestyle='solid', linewidth=1.4, color='#79afdaff', label='Control intervention')#, markerfacecolor='black'),
                       #plt_lines.Line2D([0], [0], marker='v', linestyle='dotted', color='#79afdaff', label='Clamped: Cream', markerfacecolor='white'),
                       ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig(fileto, format='svg')
    plt.clf()


from numpy import genfromtxt
import seaborn as sns
import pandas as pd
def graph_error_rate_per_step_example(fileto="testfig.svg"):
    errors = genfromtxt("results/example_data.csv", delimiter=',')

    trials_per_step = np.zeros_like(errors)
    for net in range(50):
        remainders = 100
        for step in range(50):
            trials_per_step[net,step] = remainders
            remainders -= errors[net,step]

    errors = np.divide(errors, trials_per_step)
    errors *= 100
    errors = pd.DataFrame(errors, columns=["1: fixate cupboard", "2: open", "3: fixate coffee jar", "4: take", "5: open",
                          "6: fixate mug", "7: add to mug", "8: fixate coffee jar", "9: close",
                          "10: fixate cupboard", "11: put down", "12: fixate spoon", "13: take", "14: fixate mug",
                          "15: stir ", "16: fixate table ", "17: put down ", "18: fixate sugar box ", "19: take",
                          "20: fixate mug ", "21: add to mug", "22: fixate cupboard", "23: close", "24: fixate spoon",
                          "25: take", "26: fixate mug", "27: stir", "28: fixate table", "29: put down",
                          "30: fixate fridge", "31: open", "32: fixate cream", "33: take", "34: fixate mug",
                          "35: add to mug", "36: fixate fridge", "37: put down", "38: close", "39: fixate spoon",
                          "40: take", "41: fixate mug", "42: stir", "43: fixate table", "44: put down",
                          "45: fixatemug", "46: take", "47: sip", "48: fixate table", "49: put down", "50: good coffee!"])

    fig = plt.figure(figsize=(8, 12), dpi=600)

    plot = sns.barplot(data=errors, orient="h", estimator=np.mean, ci="sd", color='lightblue', errwidth=0.4) #x="Sequence step", y="Error %",

    plot.set_xlabel('Error rate (%)', fontsize=16)
    plot.set_ylabel('Sequence steps', fontsize=16)
    plot.tick_params(labelsize=12)
    plt.tight_layout()
    #plt.subplots(figsize=(100, 50))
    #figure(figsize=(30, 15), dpi=300)


    fig = plot.get_figure()
    fig.savefig(fileto)
    #g = sns.catplot(data=errors, x="day", y="total_bill", kind="violin", inner=None)
    #sns.swarmplot(data=errors, x="day", y="total_bill", color="k", size=3, ax=g.ax)

    #violin_parts = plt.violinplot(errors, widths=.8, showmeans=True, vert=False)
    #plt.tight_layout()
    #plt.savefig(fileto, format='svg')
    #plt.clf()

import matplotlib
def control_robustness_plot(fileto="robustness.svg"):
    avgs = [[171.14, 171.08, 170.96, 166.54, 160.62],
    [77.68, 78.92, 79.7, 81.72, 75.4],
    [26.34, 29.12, 30.82, 30.88, 30.52],
    [9.8, 10.82, 10.6, 12.22, 12.3],
    [3.98, 4.52, 4.98, 5.56, 5.82]]


    ci95high = [[173.6845329, 173.6133859, 174.0716106, 169.5290811, 163.9535618],
    [80.26465809, 81.75490706, 82.7905033, 84.74625468, 78.49463401],
    [27.93593937, 30.87908066, 32.74981585, 32.7560683, 32.63292482],
    [10.74519382, 11.81396788, 11.59944187, 13.43023441, 13.53712101],
    [4.610248366, 5.235382057, 5.658121889, 6.178820266, 6.614942937]]


    ci95low = [[168.5954671, 168.5466141, 167.8483894, 163.5509189, 157.2864382],
    [75.09534191, 76.08509294, 76.6094967, 78.69374532, 72.30536599],
    [24.74406063, 27.36091934, 28.89018415, 29.0039317, 28.40707518],
    [8.854806184, 9.826032116, 9.600558128, 11.00976559, 11.06287899],
    [3.349751634, 3.804617943, 4.301878111, 4.941179734, 5.025057063]]

    for seq in [avgs, ci95high, ci95low]:
        for i, lvl1 in enumerate(seq):
            for j, lvl2 in enumerate(lvl1):
                seq[i][j]= 100*seq[i][j]/210

    fig, ax = plt.subplots()
    for i in range(0, 5):
        ax.plot([1.0, 1.25, 1.50, 1.75, 2.0], avgs[i], marker='+', label=str(i+1))
        ax.fill_between([1.0, 1.25, 1.50, 1.75, 2.0], ci95low[i], ci95high[i], color='b', alpha=.1)
    plt.xticks(ticks=[1.0, 1.25, 1.50, 1.75, 2.0])
    plt.yscale("log")

    legend = ax.legend(loc='center left', bbox_to_anchor=(1,0.5), shadow=False)#, fontsize='x-large')
    legend.set_title("Noise magnitude")
    ax.yaxis.grid(which='major', color='gray', linestyle='dashed', alpha=0.5)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    ax.yaxis.grid(which='minor', color='gray', linestyle='dashed', alpha=0.2)
    ax.set_yticks(ticks=[1, 5,  10, 50, 100])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('Goal multiplier')
    plt.ylabel('Correct sequences (%)')
    plt.tight_layout()
    plt.savefig(fileto, format='svg')
    plt.clf()

def figure_steps_to_error(fileto):
    # Data order:
    # goals subseqs
    # goals actions
    # nogoals subseqs
    # nogoals_actions

    avg =[
        [68.92, 37.44, 22.5, 10.84, 5.42, 3.48, 2.4, 1.46, 1.12, 0.94, 0.48, 0.62, 0.58, 0.4, 0.42, 0.32, 0.36, 0.26, 0.38, 0.18, 0.18, 0.16, 0.1, 0.1, 0.02, 0.04, 0.04, 0, 0, 0, 0, 0.02, 0.02, 0, 0.02, 0, 0.02, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0.02, 0, 0],
        [16.22, 12.7, 11.96, 8.02, 7.48, 7.96, 8.44, 7.06, 7.24, 6.74, 6.82, 6.98, 7.24, 5.76, 5.2, 4.96, 5.28, 4.58, 4.54, 3.9, 4.14, 3.68, 2.7, 2.28, 2.28, 1.92, 1.62, 1.6, 1.56, 1.22, 1.04, 0.86, 0.72, 0.6, 0.42, 0.2, 0.2, 0.14, 0.08, 0.06, 0.08, 0, 0.04, 0.02, 0, 0, 0, 0, 0, 0],
        [75.96, 40.38, 23.82, 10.24, 7.08, 4.12, 2.86, 1.74, 1.22, 1.02, 1.04, 0.78, 0.6, 0.74, 0.72, 0.6, 0.56, 0.6, 0.58, 0.6, 0.32, 0.38, 0.36, 0.24, 0.18, 0.16, 0.16, 0.12, 0.04, 0.06, 0.04, 0.1, 0, 0.02, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [11.96, 8.86, 8.22, 6.16, 6.04, 5.92, 6.58, 5.2, 5.5, 5.48, 5.46, 5, 4.96, 4.08, 4.56, 4.58, 4.34, 3.72, 3.62, 3.32, 3.24, 2.76, 1.94, 1.82, 1.86, 1.4, 1.42, 1.4, 1.32, 0.72, 1, 0.78, 0.84, 0.56, 0.48, 0.2, 0.2, 0.06, 0, 0.04, 0, 0, 0, 0.02, 0, 0, 0.02, 0, 0, 0.02]
        ]

    ci = [
        [4.909639065, 2.1105207, 1.920877979, 1.340842807, 0.7173378825, 0.6374000174, 0.5566729942, 0.4266238197, 0.2963489115, 0.3368050251, 0.2884568798, 0.2686328421, 0.2758977715, 0.2223728979, 0.2875984604, 0.1855181802, 0.2050980652, 0.1498337959, 0.2139869797, 0.148507807, 0.1369595623, 0.1557668844, 0.1035089508, 0.08612465302, 0.04019150474, 0.05625638778, 0.05625638778, 0, 0, 0, 0, 0.04019150474, 0.04019150474, 0, 0.04019150474, 0, 0.04019150474, 0, 0, 0.04019150474, 0, 0, 0, 0, 0, 0, 0, 0.04019150474, 0, 0],
        [1.533448849, 1.230441042, 0.9167950323, 0.9355355818, 1.017810576, 0.9692329662, 0.8596373667, 0.8328550326, 0.8232003202, 0.7525934688, 0.7545183753, 0.7228316475, 0.9268086937, 0.5871093725, 0.6671186937, 0.6792633663, 0.6890929116, 0.7150363562, 0.6822899161, 0.511135284, 0.5911662656, 0.5181495301, 0.511135284, 0.4061578736, 0.4522432784, 0.4212973737, 0.3931663517, 0.4699737782, 0.3984551692, 0.341084556, 0.3629518095, 0.2436649612, 0.2815162429, 0.2435973047, 0.1996408345, 0.1283870525, 0.1406409694, 0.1149763222, 0.0778834422, 0.06817823947, 0.0778834422, 0, 0.05625638778, 0.04019150474, 0, 0, 0, 0, 0, 0],
        [5.143487211, 2.091479859, 1.718363697, 1.065782673, 1.015540773, 0.6200975017, 0.579905997, 0.3712584182, 0.3262648206, 0.3322720416, 0.2433264905, 0.2887424524, 0.2367340276, 0.2859891838, 0.2370123746, 0.2693069532, 0.2641156026, 0.2502724394, 0.2573511982, 0.2435973047, 0.1942000057, 0.1711934352, 0.16993713, 0.1226079546, 0.1102933706, 0.1052460644, 0.1329286582, 0.109543577, 0.05625638778, 0.06817823947, 0.05625638778, 0.1183670138, 0, 0.04019150474, 0.05625638778, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.095375579, 0.9023883794, 0.8955675632, 0.7710922523, 0.6840994189, 0.686216485, 0.7640701707, 0.5625638778, 0.7644152597, 0.6269706484, 0.6399035317, 0.6621586346, 0.5711708955, 0.5962469435, 0.5971309276, 0.5657485854, 0.635768743, 0.6572614618, 0.6742215225, 0.5117476346, 0.424105014, 0.4890855853, 0.4196116286, 0.4047349649, 0.4180373907, 0.3347924725, 0.314954139, 0.4334845807, 0.3219418941, 0.2370123746, 0.309196967, 0.2999424911, 0.2590747651, 0.2160567569, 0.208920097, 0.1283870525, 0.1148328707, 0.06817823947, 0, 0.05625638778, 0, 0, 0, 0.04019150474, 0, 0, 0.04019150474, 0, 0, 0.04019150474]
        ]


    fig, ax = plt.subplots()
    names=["Elman network: action errors", "Elman network: subsequence errors", "Goal network: action errors", "Goal network: subsequence errors"]
    linestyles=["-", "-", "--", "--"]
    for i in range(4):
        ax.plot(range(50), avg[i], marker='', linestyle= linestyles[i], linewidth = 0.8, label = names[i])
        ax.fill_between(range(50), np.asarray(avg[i]) - np.asarray(ci[i]), np.asarray(avg[i]) + np.asarray(ci[i]), color='b', alpha=.05)
    plt.xticks(ticks=[0, 10, 20, 30, 40, 50])
    #plt.yscale("log")

    legend = ax.legend(loc='upper right', shadow=False)#, fontsize='x-large')
    #legend.set_title("Noise magnitude")
    ax.yaxis.grid(which='major', color='gray', linestyle='dashed', alpha=0.5)
    ax.set_ylim(ymin=0)
    ax.set_ylim(ymax=75)
    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=50)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    ax.yaxis.grid(which='minor', color='gray', linestyle='dashed', alpha=0.2)
    #ax.set_yticks(ticks=[1, 5,  10, 50, 100])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('Delay from noise to error')
    plt.ylabel('Average number of errors')
    plt.tight_layout()
    plt.savefig(fileto, format='svg')
    plt.clf()