# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import utils
import sys
from neural import neuralnet as nn, optimizers, layers
import analysis
import goalenv
from goalenv import environment as env, goalenv2020
import tensorflow as tf

utils.initialize_random_seeds(1)

# Free energy tutorial

import matplotlib.pyplot as plt
import math


if True:  # TRAINING NETWORKS
    for i in range(50):
        for g in [0, 1]:
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

    sys.exit()

# 25 networks with goals
# 25 networks without goals
# Todo: take a sequence and a network.
# For this sequence, compare:
# 1. With goals. (a) RDM (b) MDS (c) t-SNE

if True: # Regenerate all data.
    for constant_noise in [1.]:
        myfile = open("results.csv", "a")
        myfile.write("Noise "+ str(constant_noise) +"\n")
        myfile.close()
        for goals in [True]:
            for network_id in range(20):
                #print(network_id)
                #print(goals)
                #network_id=10
                print("\nNETWORK:" + str(network_id+1))
                goal_str = "1" if goals else "0"
                model = utils.load_object("bigmodel1_"+ goal_str + "goals_relu_adam_nonoise", network_id)
                test_data = goalenv2020.generate_test_data(model, noise=0.,
                                                           goal1_noise=0., goal2_noise=0.,
                                                           goals=goals, num_tests=1,
                                                           goal_multiplier=2.0,
                                                           sequence_ids=range(21),
                                                           ##[3, 16, 16],;  #0=coffee black, 3 = coffee cream, 16 = tea milk
                                                           # switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                           # switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                                                           # switch_sequence=2,
                                                           #lesion_observation_units=True,
                                                           #lesion_action_units=True,
                                                           lesion_goal2_units=False,
                                                           lesion_goal1_units=False,
                                                           noise_per_step=False,
                                                           noise_per_step_to_input=False,
                                                           disruption_per_step=False,
                                                           constant_noise=constant_noise,
                                                           initialization=utils.ZERO_INIT,
                                                           clamped_goals=False,
                                                           verbose=False)
                #utils.save_object("test_data_1", test_data)

                test_data = goalenv2020.analyse_test_data(test_data, goals=goals,
                                                          do_special_representations=False,
                                                          do_tsne=False,
                                                          do_rdm=False,
                                                          mds_range=50,
                                                          mds_sequences=range(21),
                                                          verbose=True,
                                                          append_to_file="results.csv")
    sys.exit(0)
if True:  # New Cognitive Load model.
    #a) Load network. Just the last one.
    #b) Generate test data

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
    m1_eqgs_avg = 0
    m2_eqgs_avg = 0
    m1_digeqs_avg = 0
    m2_digeqs_avg = 0
    m1_eqgdiss_avg = 0
    m2_eqgdiss_avg = 0
    m1_m2_avg = 0
    m1_eqgs_cnt = 0
    m2_eqgs_cnt = 0
    m1_digeqs_cnt = 0
    m2_digeqs_cnt = 0
    m1_eqgdiss_cnt = 0
    m2_eqgdiss_cnt = 0
    m1_m2_cnt = 0
    for network_id in range(21):
        model = utils.load_object("bigmodel1_1goals_relu_adam_nonoise", network_id)
        #b) Generate test data
        test_data = goalenv2020.generate_test_data(model, noise=0.0,
                       goal1_noise=0., goal2_noise=0.,
                       goals=True, num_tests=1,
                       goal_multiplier=1,
                       sequence_ids=range(21), ##[3, 16, 16],;  #0=coffee black, 3 = coffee cream, 16 = tea milk
                       #switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                       #switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                       #switch_sequence=2,
                       noise_per_step=False,
                       noise_per_step_to_input=False,
                       disruption_per_step=False,
                       initialization=utils.SEMINORMAL,
                       clamped_goals=False)
                       #hidden_goal_multiplier=1.)
                       #gain_multiplier = goal_multiplier,
                       #gain_multiplier_from=0,
                       #gain_multiplier_to=50)
        #c) Analyze test data:
        test_data = goalenv2020.analyse_test_data(test_data, goals=True, do_special_representations=True, do_tsne=True, do_rdm=True,
                                                  mds_range=50, mds_sequences=range(21))
        activations1 = test_data[-2]
        properties1 = test_data[-1]

        test_data = goalenv2020.generate_test_data(model, noise=0.,
                       goal1_noise=0., goal2_noise=0.,
                       goals=True, num_tests=1,
                       goal_multiplier=3,
                       sequence_ids=range(21), ##[3, 16, 16],;  #0=coffee black, 3 = coffee cream, 16 = tea milk
                       #switch_goal1= (range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                       #switch_goal2= (range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                       #switch_sequence=2,
                       noise_per_step=False,
                       noise_per_step_to_input=False,
                       disruption_per_step=False,
                       initialization=utils.SEMINORMAL,
                       clamped_goals=False)
                       #hidden_goal_multiplier=1.)
                       #gain_multiplier = goal_multiplier,
                       #gain_multiplier_from=0,
                       #gain_multiplier_to=50)
        #c) Analyze test data:
        test_data = goalenv2020.analyse_test_data(test_data, goals=True, do_special_representations=True, do_tsne=True, do_rdm=True,
                                                  mds_range=50, mds_sequences=range(21))
        activations2 = test_data[-2]
        properties2 = test_data[-1]

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


        # Now make an RDM with activations 1 and activations 2
        import rdm

        my_rdm = rdm.rdm(properties=properties1 + properties2, vectors=activations1 + activations2,
                         type=rdm.EUCLIDIAN)

        # my_rdm.plot_rdm("goal comparison")
        #my_rdm.save("goal_comparison", figsize=50, fontsize=0.5)

        print("\n\n----------------\nNETWORK "+str(network_id))
        m1_eqgs = my_rdm.get_average_key(
            keys_values={"multiplier": "1", "error": "false"},
            equals=["goal1", "goal2", "seq"])
        print("multiplier 1 - equal goals and sequence: \n{0}".format(m1_eqgs))
        m1_eqgs_avg += m1_eqgs
        m1_eqgs_cnt += 1

        m2_eqgs = my_rdm.get_average_key(
            keys_values={"multiplier": "2", "error": "false"},
            equals=["goal1", "goal2", "seq"])
        print("multiplier 2 - equal goals and sequence: \n{0}".format(m2_eqgs))
        m2_eqgs_avg += m2_eqgs
        m2_eqgs_cnt += 1

        m1_digeqs = my_rdm.get_average_key(
                                    keys_values={"multiplier": "1", "error": "false"},
                                    equals=["seq"], unequals=["goal2"])
        print("multiplier 1 - different goals and same sequence: \n{0}".format(m1_digeqs))
        m1_digeqs_avg += m1_digeqs
        m1_digeqs_cnt += 1

        m2_digeqs = my_rdm.get_average_key(
                                    keys_values={"multiplier": "2", "error": "false"},
                                    equals=["seq"], unequals=["goal2"])
        print("multiplier 2 - different goals and same sequence: \n{0}".format(m2_digeqs))
        m2_digeqs_avg += m2_digeqs
        m2_digeqs_cnt += 1

        m1_eqgdiss = my_rdm.get_average_key(
                                    keys_values={"multiplier": "1", "error": "false"},
                                    equals=["goal1", "goal2"], unequals=["seq"])
        print("multiplier 1 - equal goals, different sequence: \n{0}".format(m1_eqgdiss))
        m1_eqgdiss_avg += m1_eqgdiss
        m1_eqgdiss_cnt += 1

        m2_eqgdiss = my_rdm.get_average_key(
                                    keys_values={"multiplier": "2", "error": "false"},
                                    equals=["goal1", "goal2"], unequals=["seq"])
        print("multiplier 2 - equal goals, different sequence: \n{0}".format(m2_eqgdiss))
        m2_eqgdiss_avg += m2_eqgdiss
        m2_eqgdiss_cnt += 1

        m1_m2 = my_rdm.get_average_key(
                                    keys_values={"error": "false"},
                                    equals=["goal1", "goal2", "seq", "step"], unequals=["multiplier"])
        print("multiplier 1 vs 2, same sequences and steps: \n{0}".format(m1_m2))
        m1_m2_avg += m1_m2
        m1_m2_cnt += 1

    print("---------\nTOTALS !!")
    print(m1_eqgs_avg/m1_eqgs_cnt)
    print(m2_eqgs_avg/m2_eqgs_cnt)
    print(m1_digeqs_avg/m1_digeqs_cnt)
    print(m2_digeqs_avg/m2_digeqs_cnt)
    print(m1_eqgdiss_avg/m1_eqgdiss_cnt)
    print(m2_eqgdiss_avg/m2_eqgdiss_cnt)
    print(m1_m2_avg/m1_m2_cnt)
    exit()
        # 2. With amplified goals. (a) RDM (b) MDS (c) t-SNE
        # THEN compare networks with and without goals?
        # For this I need 25 networks with goals and 25 without goals.
        # Make RDM average and MDS average; Euclidian and Spearman.


if False: # rdm test
    import numpy as np
    import rdm
    rdm_vals = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
    properties = []
    properties.append({"label": 'seq 1 action 1', "sequence": '1', "action": '1', "order": 'arithmetic first'})
    properties.append({"label": 'seq 1 action 2', "sequence": '1', "action": '2', "order": 'arithmetic first'})
    properties.append({"label": 'seq 2 action 1', "sequence": '2', "action": '1', "order": 'beverage first'})
    properties.append({"label": 'seq 2 action 2', "sequence": '2', "action": '2', "order": 'beverage first'})
    my_rdm = rdm.rdm(properties, rdm_vals)
    my_rdm.save('original', 'Original')
    my_rdm.sort_by('action')
    my_rdm.save('action', 'Action sorted')
    my_rdm.sort_by('order')
    my_rdm.save('order', 'Order sorted')
    my_rdm.sort_by('sequence')
    my_rdm.save('sequence', 'sequence sorted') #should be same as order.
    sys.exit()

if False:
    from reward_task import reward
    import rdm
    for i in range(5):
        print(i)
        model = reward.train_with_goals(nonlinearity=tf.nn.relu, learning_rate = 0.02)
        #utils.save_object("reward_task_with_goals", model)
        utils.save_object("reward_task_with_goals_sigmoid", model)
    cond="none"
    nets = "reward_task_with_goals_sigmoid"
    #nets = "reward_task_with_4goals"
    num_nets =5
    for type in [rdm.SPEARMAN, rdm.EUCLIDIAN]:
        cond = type
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1111"+cond, gain=[1,1,1,1])#, skips=[0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0022"+cond, gain=[0,0,2,2])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_2200"+cond, gain=[2,2,0,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0044"+cond, gain=[0,0,4,4])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_4400"+cond, gain=[4,4,0,0])
        """
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0000"+cond, gain=[0,0,0,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0011"+cond, gain=[0,0,1,1])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1100"+cond, gain=[1,1,0,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1122"+cond, gain=[1,1,2,2])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1110"+cond, gain=[1,1,1,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0001"+cond, gain=[0,0,0,1])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1112"+cond, gain=[1,1,1,2])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_5_5_52"+cond, gain=[.5,.5,.5,2])
        """
    sys.exit()

if False:
    from pnas import pnas2018, pnashierarchy
    for i in range(10):
        print(i)
        #model, _ = pnas2018.train(iterations=5000, learning_rate=0.1, size_hidden=100)
        model = nn.TripleACCNet(size_observation=7, size_action=8,
                                layer_sizes=[layers.RCPLayerSizes(input_bottomup=7, output_bottomup=96,
                                                                  input_topdown=96, output_topdown=96),
                                             layers.RCPLayerSizes(input_bottomup=96, output_bottomup=96,
                                                                  input_topdown=8, output_topdown=96)],
                                output_layer=True)
        model.learning_rate = 0.001
        pnas2018.trainACC(model=model, iterations=10000)
        pnas2018.accuracy_testACC(model)
        model.delete_tapes()  #can't save tapes
        utils.save_object("rdm_acc", model)

    #gain = [1., 1., .0, 0.0]
    pnas2018.make_rdm_multipleACC("rdm_acc", num_networks=5, rdm_type=analysis.SPEARMAN, save_name="spearman_acc")
    #pnas2018.make_rdm_multiple_gain("rdm_gains", num_networks=25, rdm_type=analysis.EUCLIDIAN, save_name="euclidian_zeroed_goalstea", gain=gain)
    #pnas2018.make_rdm_multiple("rdm_gains", num_networks=25, rdm_type=analysis.EUCLIDIAN, save_name="euclidian_nogain", with_goals=True)
    #pnas2018.make_rdm_multiple("rdm_gains", num_networks=25, rdm_type=analysis.SPEARMAN, save_name="spearman_nogain", with_goals=True)

    sys.exit()

if False:
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

    sys.exit()

# Trained:
# - "bigmodel1_yesgoals": 15 networks  (uniform initialization, sigmoid)
# - "bigmodel1_yesgoals_2nd_batch": 1 network (same)
# - "bigmodel1_yesgoals_relu": 4 networks, HE initialization, seminormal context initialization. UNSURE, RETRAIN
# - "bigmodel1_yesgoals_relu": 4 networks, HE initialization, normal context initialization
if False:
    from goalenv import goalenv2020



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
        for lesion in [7, 8, 9]:#[1, 2, 3, 4, 5, 6]:
            for noise in [0.0]:
                print("\n\n\n" + str(noise))
                for i in range(10):
                    print(i)

                    model = utils.load_object(model_files, i)  # "bigmodel1_nogoals_relu_adam_nonoise", i)

                    lesion_goal1 = lesion_goal2 = lesion_actions = lesion_observation = False
                    if lesion == 1: lesion_goal1 = True
                    elif lesion == 2: lesion_goal2 = True
                    elif lesion == 3: lesion_actions = True
                    elif lesion == 4: lesion_goal1 = lesion_goal2 = True
                    elif lesion == 5: lesion_goal1 = lesion_goal2 = lesion_actions = True
                    elif lesion == 6: lesion_goal2 = lesion_actions = True
                    elif lesion == 7: lesion_observation = True
                    elif lesion == 8: lesion_goal1 = lesion_actions = True
                    elif lesion == 9: lesion_observation = lesion_actions = True
                    test_data = goalenv2020.generate_test_data(model, noise=noise,  goal1_noise=0., goal2_noise=0.,
                                                               goals=goals, num_tests=10, sequence_ids=range(21),
                                                               noise_per_step=False, noise_per_step_to_input=False,
                                                               disruption_per_step=False, initialization=nn.SEMINORMAL,
                                                               lesion_goal1_units=lesion_goal1,
                                                               lesion_goal2_units=lesion_goal2,
                                                               lesion_action_units=lesion_actions,
                                                               lesion_observation_units=lesion_observation)

                    tsne_results, test_data, _, error_data, _ = goalenv2020.analyse_test_data(test_data, do_rdm=False, goals=False)
                    error_data_list.append(error_data)

    utils.write_lists_to_csv("error_results_lesion.csv", error_data_list, labels=goalenv2020.error_testing_labels)
    sys.exit()
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
                                        size_goal1=len(environment.GoalEnvData.goals1_list),
                                        size_goal2=len(environment.GoalEnvData.goals2_list),
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

                model = utils.load_object(model_files, i)  #"bigmodel1_nogoals_relu_adam_nonoise", i)

                test_data = goalenv2020.generate_test_data(model, noise=noise,
                                                           goal1_noise=0., goal2_noise=0.,
                                                           goals=goals, num_tests=3,
                                                           sequence_ids = range(21),
                                                           noise_per_step = True,
                                                           noise_per_step_to_input = False,
                                                           disruption_per_step = False,
                                                           initialization = nn.SEMINORMAL)

                utils.save_object("test_data_error_test"+model_type+str(noise)+str(i), test_data)
                test_data = utils.load_object("test_data_error_test"+model_type+str(noise)+str(i))
                tsne_results, test_data, _, error_data, _ = goalenv2020.analyse_test_data(test_data, do_rdm=False, goals=True)  #, mds_sequences=[2, 5, 11], mds_range=15)
                error_data_list.append(error_data)

                #utils.save_object("tsne_bigmodel1_yesgoals", tsne_results)
                utils.save_object("tsnetest_bigmodel1_yesgoals", test_data)
                goalenv2020.plot_tsne(tsne_results, test_data, tsne_goals=False, tsne_subgoals=False, tsne_actions=False, tsne_sequences=True,
                          tsne_errors=True, tsne_sequence=[2, 5, 11], tsne_sequence_interval=[2, 14], filename="tsne", annotate=False)
                utils.save_object("tsne_results_bigmodel1_yesgoals", tsne_results)

    utils.write_lists_to_csv("loss_six.csv", error_data_list, labels=goalenv2020.error_testing_labels)

    sys.exit()

if False:
    import cognitiveload.model4 as mod4
    import cognitiveload.model3 as mod3
    import cognitiveload.model2 as mod2
    mod3.FAST_RDM = True
    import cognitiveload.cogloadtask as task
    import rdm
    import tensorflow as tf

    # 0. Mod 4 stuff
    hrp = mod4.HierarchyGradientParams(regincrease="linear", regstrength=0.00005)#regstrength=0.00005)
    nnparams = nn.ParamsGoalNet(algorithm=optimizers.SGD,
                                nonlinearity=tf.nn.tanh,
                                initialization=utils.NORMAL,
                                learning_rate=0.01,
                                size_action=None,  # these will get filled automatically
                                size_observation=None,  #
                                size_hidden=50,
                                L1_reg=0, L2_reg=0)

    stopping = nn.ParamsStopping(max_iterations=20000, min_iterations=1000, check_frequency=200,
                                 stop_condition=mod4.stop_condition, blanks=True, min_accuracy=1.)

    nnparams.size_hidden = 100
    nnparams.learning_rate = 0.01
    nnparams.initialization = utils.NORMAL
    for i in [1.0, 2.0, 4.0, 10.0]:
        for j in [100]:
            print(i)
            print(j)
            mod4.BEV_GOAL_MULTIPLIER = i
            mod4.RDM_PERCENT = j
            mod4.run_model4_multiple(stopping_params=stopping,
                                     num_networks=50, from_file="model4_fixed_lr001_reg000005",
                                     name="model4_fixed_lr001_reg000005_"+str(int(i))+'_' +str(j),
                                     nnparams=nnparams,
                                     blanks=True,
                                     type=rdm.EUCLIDIAN,
                                     hrp=hrp,
                                     skips=[0, 6, 28, 46])
    """
    mod4.BEV_GOAL_MULTIPLIER = 1.5
    mod4.run_model4_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="mod4_normal_l20.0_sig100_sgd_48",
                             name="50_50_euclidian_bev15",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN,
                             hrp=hrp)#,

    mod4.BEV_GOAL_MULTIPLIER = 1.0
    mod4.run_model4_multiple(stopping_params=stopping,
                             num_networks=5, from_file="mod4_normal_l20.0_sig100_sgd_48",
                             name="50_50_euclidian_bev10",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN,
                             hrp=hrp)#,
    """
    sys.exit(0)

    # 1. one network with tanh. Make RDMs at 1, 10, 100, 1000
    hrp = mod3.HierarchyGradientParams(regincrease="linear", regstrength=0.0)
    nnparams = nn.ParamsGoalNet(algorithm=optimizers.SGD,
                                nonlinearity=tf.nn.tanh,
                                initialization=utils.NORMAL,
                                learning_rate=0.01,
                                size_action=None,  # these will get filled automatically
                                size_observation=None,  #
                                size_hidden=25,
                                L1_reg=0, L2_reg=0)

    stopping = nn.ParamsStopping(max_iterations=20000, min_iterations=1000, check_frequency=200,
                                 stop_condition=mod3.stop_condition, blanks=True, min_accuracy=1.)
    #
    #nnparams.algorithm = optimizers.SGD
    #nnparams.initialization = utils.NORMAL
    #nnparams.nonlinearity = tf.nn.sigmoid
    #nnparams.learning_rate = 0.03
    #nnparams.size_hidden = 100
    #mod2.run_model2_multiple(stopping_params=stopping,
    #                         num_networks=5,  #from_file="normal_l20.0_sig_sgd",
    #                         name="normal_sig_100_sgd_005_48",
    #                         nnparams=nnparams,
    #                         blanks=True,
    #                         type=rdm.EUCLIDIAN)

    # works but doesn't look very good
    #nnparams.algorithm = optimizers.SGD
    #nnparams.initialization = utils.NORMAL
    #nnparams.nonlinearity = tf.nn.tanh
    #nnparams.learning_rate = 0.01
    #nnparams.size_hidden = 50
    #mod2.run_model2_multiple(stopping_params=stopping,
    #                             num_networks=5,  #from_file="normal_l20.0_sig_sgd",
    #                             name="normal_tanh_50_sgd_001_48",
    #                             nnparams=nnparams,
    #                             blanks=True,
    #                             type=rdm.EUCLIDIAN)

    # doesnt work.
    #nnparams.algorithm = optimizers.ADAM
    #nnparams.initialization = utils.NORMAL
    #nnparams.nonlinearity = tf.nn.sigmoid
    #nnparams.learning_rate = 0.001
    #nnparams.size_hidden = 25
    #mod2.run_model2_multiple(stopping_params=stopping,
    #                         num_networks=5,  # from_file="normal_l20.0_sig_sgd",
    #                         name="normal_tanh_25_adam_0001_48",
    #                         nnparams=nnparams,
    #                         blanks=True,
    #                         type=rdm.EUCLIDIAN)

    # works
    #nnparams.algorithm = optimizers.SGD
    #nnparams.initialization = utils.NORMAL
    #nnparams.nonlinearity = tf.nn.sigmoid
    #nnparams.learning_rate = 0.03
    #nnparams.size_hidden = 100
    #mod2.run_model2_multiple(stopping_params=stopping,
    #                         num_networks=5, #from_file="normal_sig_25_sgd_001_48",
    #                         name="normal_sig_100_sgd_003_96",
    #                         nnparams=nnparams,
    #                         blanks=True,
    #                         type=rdm.EUCLIDIAN)

    # doesn't work : sgd norm sig 0.01 100
    # works : sgd norm sig 0.03 100  ~bit longer than 0.05
    # works : sgd norm sig 0.05 100 - 6100 7100 5300 6300 5900
    # works : sgd norm sig 0.1 100 - 5500 11300 7900 7900 7300
    # works : sgd norm sig 0.05 75 - 8100 6300 8900 10100 7900
    # works : sgd norm sig 0.05 50 - 8100 10500 9300 10900 9900
    # works : sgd norm sig 0.05 40 - 13300 13500 10500 12100 14700
    # works : sgd norm sig 0.1 50 - 13300 9700 10700 7100 12700
    # doesn't work : sgd norm sig 0.03 50 - 18500 20000 - discontinued.

    nnparams.algorithm = optimizers.ADAM
    nnparams.initialization = utils.HE
    nnparams.nonlinearity = tf.nn.relu
    nnparams.learning_rate = 0.001
    nnparams.size_hidden = 25
    mod2.run_model2_multiple(stopping_params=stopping,
                                 num_networks=50,  from_file="nogoals_he_relu_25_adam_0001",
                                 name="nogoals_he_relu_25_adam_0001_96",
                                 nnparams=nnparams,
                                 blanks=True,
                                 type=rdm.EUCLIDIAN)

    nnparams.algorithm = optimizers.SGD
    nnparams.initialization = utils.XAVIER
    nnparams.nonlinearity = tf.nn.sigmoid
    nnparams.learning_rate = 0.005
    nnparams.size_hidden = 100
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=50,  from_file="nogoals_xavier_sigmoid_100_sgd_0005_48",
                             name="nogoals_xavier_sigmoid_100_sgd_0005_96",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN,
                             skips=None)

    sys.exit()
    nnparams.algorithm = optimizers.SGD
    nnparams.initialization = utils.XAVIER
    nnparams.nonlinearity = tf.nn.sigmoid
    nnparams.learning_rate = 0.005
    nnparams.size_hidden = 100
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=50,  from_file="goals_uniform_sigmoid_100_sgd_0005_48",
                             name="goals_uniform_sigmoid_100_sgd_0005_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN,
                             skips=None)

    sys.exit()

    # works well and fast
    # tanh
    nnparams.algorithm = optimizers.ADAM
    nnparams.initialization = utils.XAVIER
    nnparams.learning_rate = 0.001
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  #from_file="normal_l20.0_sig_sgd",
                             name="xavier_l20.0_tanh25_adam_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    # works but learning is slow (~10,000 or so)
    nnparams.algorithm = optimizers.SGD
    nnparams.initialization = utils.NORMAL
    nnparams.learning_rate = 0.01
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  #from_file="normal_l20.0_sig_sgd",
                             name="normal_l20.0_tanh25_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    # works but learning is slow (~15,000 or so)
    nnparams.learning_rate = 0.005
    nnparams.initialization = utils.XAVIER
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  #from_file="normal_l20.0_sig_sgd",
                             name="xavier_l20.0_tanh25_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    # works well and fast
    # tanh
    nnparams.size_hidden = 40
    nnparams.learning_rate = 0.02
    nnparams.initialization = utils.NORMAL
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=5,  #from_file="normal_l20.0_sig_sgd",
                             name="normal_l20.0_sig40_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    # works well and fast
    # tanh
    nnparams.size_hidden = 50
    nnparams.learning_rate = 0.01
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=5,  #from_file="normal_l20.0_sig_sgd",
                             name="normal_l20.0_sig50_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    nnparams.size_hidden = 60
    nnparams.learning_rate = 0.005
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=5,  #from_file="normal_l20.0_sig_sgd",
                             name="normal_l20.0_sig60_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    # Doesn't work (slow convergence and/or diverges)
    # tanh
    nnparams.size_hidden = 25
    nnparams.learning_rate = 0.02
    nnparams.initialization = utils.XAVIER
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=5,  #from_file="normal_l20.0_sig_sgd",
                             name="xavier002_l20.0_sig25_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    # Doesnt work (diverges)
    # tanh
    nnparams.size_hidden = 25
    nnparams.learning_rate = 0.1
    nnparams.initialization = utils.XAVIER
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=5,  #from_file="normal_l20.0_sig_sgd",
                             name="xavier01_l20.0_sig25_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)
    sys.exit()
    # Use the easy arithmetic sequences :-)
    for optimizer in [optimizers.SGD]:#, optimizers.ADAM]:
        for initialization in [utils.NORMAL]:#, utils.HE]:
            for nonlinearity in [tf.nn.relu]:#[tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu]:
                for L2_reg in [0.001]:#, 0.0]:
                    if optimizer == optimizers.SGD:
                        if nonlinearity == tf.nn.sigmoid:
                            lr = 0.1 # sigmoid is super slow to learn
                        else:
                            lr = 0.005
                    else: # ADAM
                        if nonlinearity == tf.nn.sigmoid:
                            lr = 0.01
                        else:
                            lr = 0.001
                    hrp=mod3.HierarchyGradientParams(regincrease="linear", regstrength=0.0)
                    nnparams = nn.ParamsGoalNet(algorithm=optimizer,
                                                nonlinearity=nonlinearity,
                                                initialization=initialization,
                                                learning_rate=lr,
                                                size_action=None,  # these will get filled automatically
                                                size_observation=None,  #
                                                size_hidden=25,
                                                L1_reg=0, L2_reg=L2_reg)
                    stopping = nn.ParamsStopping(max_iterations=15000, min_iterations=200, check_frequency=200,
                                                 stop_condition=mod3.stop_condition, blanks=True, min_accuracy=1.)
                    if nonlinearity == tf.nn.sigmoid:
                        str_nl = "_sig_"
                    elif nonlinearity == tf.nn.tanh:
                        str_nl = "_tanh_"
                    else:
                        str_nl = "_relu_"
                    mod2.run_model2_multiple(stopping_params=stopping,
                                             num_networks=2, #from_file=initialization + "_l2"+str(L2_reg) + str_nl + optimizer[0],
                                             name=initialization + "_l2"+str(L2_reg) + str_nl + optimizer[0],
                                             nnparams=nnparams,
                                             blanks=True,
                                             type=rdm.EUCLIDIAN)
    sys.exit()
    #i=-1
    #print(i)
    #hrp.reg_strength = 0.001
    #nnparams.L2_reg = 0.0

    hrp.reg_strength = 0.0
    # matrix processing test
    import rdm
    import numpy as np

    #test_rdm = rdm.rdm(props, matrix_values=mat)
    #averaged_rdm = test_rdm.average_values(["v", "n"], ["s"])
    #averaged_rdm.save(filename="test rdm")

    #mod3.run_model3_multiple(stopping_params=stopping,
    #                         num_networks=20, from_file="model3_euclidian_activations", #"model3_distances_spearman",
    #                         name="model3_spearman_avdistances",
    #                         hrp=None,
    #                         nnparams=nnparams,
    #                         blanks=True,
    #                         type=rdm.EUCLIDIAN)
    #sys.exit()
    if False:
        rdms = mod2._generate_rdm_from_files("activations_model2_old", "properties_file", 5)
        final_rdm = None
        for my_rdm in rdms:
            if final_rdm is None:
                final_rdm = mod2.process_rdmatrix(rdm.rdm(properties=my_rdm.properties, matrix_values=my_rdm.matrix.copy()), True)
            else:
                final_rdm.matrix += mod2.process_rdmatrix(my_rdm, True).matrix
        final_rdm.matrix /= 5

        mod2.set_rdm_labels(final_rdm)
        # Save it
        final_rdm.save("test_rdm", title="RDM training combined")#, dpi=200, figsize=60, fontsize=0.5)
        sys.exit()

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model2_rdm_rich",
                             name="model2_rdm_rich",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)
    sys.exit()

    mod3.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, #from_file="model3_euclidian_distances",
                             name="model3_rdm",
                             hrp=None,
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)
    sys.exit()
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model3_plusminus",
                             name="model3_spearman_activations",
                             hrp=None,
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_ACTIVATIONS,
                             type=rdm.SPEARMAN)

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  from_file="model2_plusminus",
                             name="model2_spearman_distances",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_DISTANCES,
                             type=rdm.SPEARMAN)


    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  from_file="model2_plusminus",
                             name="model2_euclidian_activations",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_ACTIVATIONS,
                             type=rdm.EUCLIDIAN)

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  from_file="model2_plusminus",
                             name="model2_euclidian_distances",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_DISTANCES,
                             type=rdm.EUCLIDIAN)

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  from_file="model2_plusminus",
                             name="model2_spearman_activations",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_ACTIVATIONS,
                             type=rdm.SPEARMAN)

    sys.exit()

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  from_file="model2_",
                             name="model2_distances_euclidian",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_DISTANCES,
                             type=analysis.EUCLIDIAN)


    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  from_file="model2_",
                             name="model2_activations_euclidian",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_ACTIVATIONS,
                             type=analysis.EUCLIDIAN)


    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  from_file="model2_",
                             name="model2_distances_euclidian",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_DISTANCES,
                             type=analysis.EUCLIDIAN)


    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  from_file="model2_",
                             name="model2_activations_euclidian",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_ACTIVATIONS,
                             type=analysis.EUCLIDIAN)
    sys.exit()
    #hrp=mod3.HierarchyGradientParams(regincrease="linear", regstrength=0.00002)
    #mod3.run_model3_multiple(from_file="model3_test_gradient_2goals00002", num_networks=2, name="model3_test_gradient_2goals00002_test2", hrp=hrp)

    # model3_ADAMRELU are best normal ones.
    # "model3_ADAMRELU_gradient" is a first try for gradient ones. Didn't work.
    # "model3_ADAMRELU_gradient0_0001" is a second try for gradient ones.
    # Best perf so far: ADAM, RELU, HE, 0.0003, 300,000  (1, 1, .99, .83, .97, .99, .99, .97, .76, 1.0.
    # With no reg: (1., .93, .99, .99, .95, .)
    hrp=mod3.HierarchyGradientParams(regincrease="linear", regstrength=0.00001)
    nnparams = nn.ParamsGoalNet(algorithm=nn.ADAM,
                                nonlinearity=nn.RELU,
                                initialization=nn.HE,
                                learning_rate=0.002,
                                size_action=None,  # these will get filled automatically
                                size_observation=None,  #
                                size_hidden=100,
                                L1_reg=0, L2_reg=0.00001)

    stopping = nn.ParamsStopping(max_iterations=150001, min_iterations=10000, check_frequency=1000,
                                 stop_condition=mod3.stop_condition, blanks=True, min_accuracy=0.95)

    i=-1
    print(i)
    hrp.reg_strength = 0.00003
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=25, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 3. about 100,000 iterations. Massive left/right difference.=

    sys.exit()
    i=0
    print(i)
    hrp.reg_strength=0.0
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 0. about 50,000 iterations. No left/right difference

    i+=1
    print(i)
    hrp.reg_strength = 0.000001
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 1. about 45,000 iterations, high variance in training time. No left/right difference

    i+=1
    print(i)
    hrp.reg_strength = 0.00001
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 2. 50,000 iterations agian. No left/right difference

    i+=1
    print(i)
    hrp.reg_strength = 0.0001
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 3. about 100,000 iterations. Massive left/right difference.=

    i+=1
    print(i)
    hrp.reg_strength = 0.001
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 4. Accuracy about 15 to 20%, always hits max iteration (151001). Even more massive difference.

    i+=1
    print(i)
    hrp.reg_strength=0.0
    nnparams.L2_reg = 0.000001
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 5. about 50000, no effect.  Combined number quite self-similar, maybe an idiosyncracy?


    i+=1
    print(i)
    hrp.reg_strength=0.0
    nnparams.L2_reg = 0.00001
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 6. 50000 again. Seems to have more contrast compared to 5. Combined numbers less self-simialr.


    i+=1
    print(i)
    hrp.reg_strength=0.0
    nnparams.L2_reg = 0.0001
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 7.  50000 again. Feels like a mashup of 5 and 6. Impression = this doesn4t matter at this point

    i+=1
    print(i)
    hrp.reg_strength=0.0
    nnparams.L2_reg = 0.001
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 8. 50000 again. Nothing obvious


    i+=1
    print(i)
    hrp.reg_strength=0.0
    nnparams.L2_reg = 0.01
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 9. max iterations, accuracy about 75-90%. Looks high contrast again, ish.


    i+=1
    print(i)
    hrp.reg_strength=0.000001
    hrp.reg_increase="square"
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 10. about 90000 iterations. Clear effect, big differences.


    i+=1
    print(i)
    hrp.reg_strength=0.00001
    hrp.reg_increase="square"
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 11. about 150,000 iterations, some finish some dont

    i+=1
    print(i)
    hrp.reg_strength=0.0001
    hrp.reg_increase="square"
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 12. max iterations, accuracy about 35%

    i+=1
    print(i)
    hrp.reg_strength=0.001
    hrp.reg_increase="square"
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="test"+str(i),#"model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 13. max iterations, accuracy 5 to 30%


    sys.exit()
    print('with blanks now')

    stopping = nn.ParamsStopping(max_iterations=150001, min_iterations=10000, check_frequency=1000,
                                 stop_condition=mod3.stop_condition, blanks=True)
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5, #from_file="model3_nodummy"
                             name="model3_withdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    sys.exit()

    #3 and 6 need extra training.
    updated_models = []
    for i in range(20):
        print(i)
        model = utils.load_object("model3_ADAMRELU_gradient", i)
        hidden_activation, accuracy_totals, accuracy_fullseqs = mod3.test_network_ari(model)
        print(accuracy_fullseqs)


    print("\n\n\nLEARNING RATE = 0.002, 100 HIDDEN UNITS, HRP STR = 0.000001")
    hrp.reg_strength = 0.00001
    mod3.run_model3_multiple(num_networks=20, from_file="model3_ADAMRELU_gradient",
                             name="model3_ADAMRELU_gradient",
                             hrp=hrp,
                             nnparams=nnparams,
                             iterations=150000,
                             early_stopping_from=100000)


    sys.exit()

    print("\n\n\nLEARNING RATE = 0.01 AGAIN, 100 HIDDEN UNITS")
    nnparams.learning_rate = 0.01
    for i in range(4):
        mod3.run_model3_multiple(num_networks=1, name="test10",
                                 hrp=hrp,
                                 nnparams=nnparams,
                                 iterations=150000)

    print("\n\n\nLEARNING RATE = 0.003 again, 100 HIDDEN UNITS")
    nnparams.learning_rate = 0.003
    for i in range(4):
        mod3.run_model3_multiple(num_networks=1, name="test10",
                                 hrp=hrp,
                                 nnparams=nnparams,
                                 iterations=150000)

    print("\n\n\nLEARNING RATE = 0.0001, 50 HIDDEN UNITS")
    nnparams.learning_rate = 0.0001
    for i in range(10):
        mod3.run_model3_multiple(num_networks=1, name="test3",
                                 hrp=hrp,
                                 nnparams=nnparams,
                                 iterations=500000)

    print("\n\n\nLEARNING RATE = 0.001, 100 HIDDEN UNITS")
    nnparams = nn.ParamsGoalNet(algorithm=nn.ADAM,
                                nonlinearity=nn.RELU,
                                initialization=nn.HE,
                                learning_rate=0.001,
                                size_action=None,  # these will get filled automatically
                                size_observation=None,
                                size_hidden=100,
                                L1_reg=0, L2_reg=0.000001)
    for i in range(10):
        mod3.run_model3_multiple(num_networks=1, name="test4",
                                 hrp=hrp,
                                 nnparams=nnparams,
                                 iterations=500000)

    print("\n\n\nLEARNING RATE = 0.0003, 100 HIDDEN UNITS")
    nnparams.learning_rate = 0.0003
    for i in range(10):
        mod3.run_model3_multiple(num_networks=1, name="test5",
                                 hrp=hrp,
                                 nnparams=nnparams,
                                 iterations=500000)

    print("\n\n\nLEARNING RATE = 0.0001, 100 HIDDEN UNITS")
    nnparams.learning_rate = 0.0001
    for i in range(10):
        mod3.run_model3_multiple(num_networks=1, name="test6",
                                 hrp=hrp,
                                 nnparams=nnparams,
                                 iterations=500000)

    print("\n\n\nLEARNING RATE = 0.001, 25 HIDDEN UNITS")
    nnparams = nn.ParamsGoalNet(algorithm=nn.ADAM,
                                nonlinearity=nn.RELU,
                                initialization=nn.HE,
                                learning_rate=0.001,
                                size_action=None,  # these will get filled automatically
                                size_observation=None,
                                size_hidden=25,
                                L1_reg=0, L2_reg=0.000001)
    for i in range(10):
        mod3.run_model3_multiple(num_networks=1, name="test7",
                                 hrp=hrp,
                                 nnparams=nnparams,
                                 iterations=500000)

    print("\n\n\nLEARNING RATE = 0.0001, 25 HIDDEN UNITS")
    nnparams.learning_rate = 0.0001
    for i in range(10):
        mod3.run_model3_multiple(num_networks=1, name="test8",
                                 hrp=hrp,
                                 nnparams=nnparams,
                                 iterations=500000)

    sys.exit(0)
"""
print("\n\nNOW DOING: 0.003")
for i in range(3):
    mod3.run_model3_multiple(num_networks=1, name="model3_test_gradient_2goals00005_relutest",
                         hrp=hrp,
                         algorithm=nn.ADAM,
                         nonlinearity=nn.RELU,
                         initialization=nn.HE,
                         learning_rate=0.003,
                         iterations=300000)
print("\n\nNOW DOING: 0.0003")
for i in range(3):
    mod3.run_model3_multiple(num_networks=1, name="model3_test_gradient_2goals00005_relutest",
                             hrp=hrp,
                             algorithm=nn.RMSPROP,
                             nonlinearity=nn.TANH,
                             initialization=nn.XAVIER,
                             learning_rate=0.0003,
                             iterations=300000)
print("\n\nNOW DOING: 0.0001")
for i in range(3):
    mod3.run_model3_multiple(num_networks=1, name="model3_test_gradient_2goals00005_relutest",
                             hrp=hrp,
                             algorithm=nn.RMSPROP,
                             nonlinearity=nn.TANH,
                             initialization=nn.XAVIER,
                             learning_rate=0.001,
                             iterations=300000)
print("\n\nNOW DOING: 0.003")
for i in range(3):
    mod3.run_model3_multiple(num_networks=1, name="model3_test_gradient_2goals00005_relutest",
                             hrp=hrp,
                             algorithm=nn.RMSPROP,
                             nonlinearity=nn.TANH,
                             initialization=nn.XAVIER,
                             learning_rate=0.003,
                             iterations=300000)
"""




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
    model = nn.GoalNet(size_hidden=50, size_observation=29, size_action=19,
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
