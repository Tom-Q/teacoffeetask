# Entry point of the program. Mostly loads scripts from scripts, which themselves rely on more serious code.
import utils
import sys
import neuralnet as nn
from cognitiveload import cogloadtask
import numpy as np

import analysis
from pnas import pnas2018
"""
#for i in range(25):
#    print(i)
#    model, _ = pnas2018.train(iterations=5000, learning_rate=0.1, size_hidden=100)
#    utils.save_object("rdm_measure_test100", model)

rdms=[]
#num networks should be 10 and 5
rdms.append(pnas2018.make_rdm_multiple("rdm_measure_test", num_networks=10, rdm_type=analysis.SPEARMAN, save_name="spearman_1")[0])
rdms.append(pnas2018.make_rdm_multiple("rdm_measure_test", num_networks=10, rdm_type=analysis.EUCLIDIAN, save_name="euclidian_1")[0])
rdms.append(pnas2018.make_rdm_multiple("rdm_measure_test", num_networks=10, rdm_type=analysis.MAHALANOBIS, save_name="mahalanobis_1")[0])
rdms.append(pnas2018.make_rdm_multiple_ldt("rdm_measure_test", num_networks=10, noise_after=.1, num_samples=20, save_name="LDt_1_01"))
rdms.append(pnas2018.make_rdm_multiple_ldt("rdm_measure_test", num_networks=10, noise_after=.5, num_samples=20, save_name="LDt_1_05"))
rdms.append(pnas2018.make_rdm_multiple_ldt("rdm_measure_test", num_networks=10, noise_after=1., num_samples=20, save_name="LDt_1_1"))

rdms.append(pnas2018.make_rdm_multiple("rdm_measure_test100", num_networks=5, rdm_type=analysis.SPEARMAN, save_name="spearman_2")[0])
rdms.append(pnas2018.make_rdm_multiple("rdm_measure_test100", num_networks=5, rdm_type=analysis.EUCLIDIAN, save_name="euclidian_2")[0])
rdms.append(pnas2018.make_rdm_multiple("rdm_measure_test100", num_networks=5, rdm_type=analysis.MAHALANOBIS, save_name="mahalanobis_2")[0])
rdms.append(pnas2018.make_rdm_multiple_ldt("rdm_measure_test100", num_networks=5, noise_after=.1, num_samples=20, save_name="LDt_2_01"))
rdms.append(pnas2018.make_rdm_multiple_ldt("rdm_measure_test100", num_networks=5, noise_after=.5, num_samples=20, save_name="LDt_2_05"))
rdms.append(pnas2018.make_rdm_multiple_ldt("rdm_measure_test100", num_networks=5, noise_after=1., num_samples=20, save_name="LDt_2_1"))


rdm = analysis.rdm_of_rdms(rdms, type=analysis.PEARSON)
analysis.save_rdm(rdm, filename="rdm_of_rdms", labels=["spearman_1", "euclidian_1", "mahalanobis_1",
                                                       "LDt_1_01", "LDt_1_05", "LDt_1_1",
                                                       "spearman_2", "euclidian_2", "mahalanobis_2",
                                                       "LDt_2_01", "LDt_2_05", "LDt_2_1"])
sys.exit()
"""
#analysis.loss_big_plot()
#analysis.bargraph_with_without_goalunits("myfile4.png")
#analysis.barplot_figure_errors("myfile2.png")
#analysis.barplot_figure_ablations("myfile3.png")
#analysis.plot_tsne("mds_tsne.txt", "tsne.png")
#sys.exit()
# import time
# from pnas import pnas2018
# for i in range(20):
#     model = nn.ElmanGoalNet(size_hidden=100, size_observation=9, size_action=8, size_goal1=0, size_goal2=0,
#                             #algorithm=nn.SGD, initialization="normal", learning_rate=0.1)
#                             algorithm=nn.ADAM, nonlinearity=nn.RELU, initialization=nn.HE, learning_rate=0.01, L2_reg=0.00)
#     model, _ = pnas2018.train(model, noise=0.0, initial_context=pnas2018.ZEROS, iterations=1000)
#     utils.save_object("pnasoverfittest100_1000", model)
#pnas2018.make_rdm_multiple_ldt("pnasldttest_0_0_noiseafter_sgd50", noise_during=0.0, noise_after=0.10, num_networks=10,
#                               num_samples=50, initial_context=pnas2018.ZEROS, log_scale=False)
#pnas2018.make_rdm_multiple("pnasoverfittest100_1000", num_networks=20, with_goals=False, title="-", save_files=True)

#sys.exit()



if True:
    from goalenv import goalenv2020
    from goalenv import environment as env

    for i in range(20):
        print(i)
        model = nn.ElmanGoalNet(size_hidden=50, size_observation=29, size_action=19,
                                size_goal1=len(env.GoalEnvData.goals1_list),
                                size_goal2=len(env.GoalEnvData.goals2_list),
                                algorithm=nn.ADAM, learning_rate=0.001,
                                L2_reg=0.0001,
                                initialization=nn.HE,
                                nonlinearity=nn.RELU,
                                last_action_inputs=True)

        stopping = nn.ParamsStopping(max_iterations=40000, min_iterations=3001, check_frequency=1000,
                                     stop_condition=goalenv2020.stop_condition, goals=True, noise=0.0)
        model = goalenv2020.train(stop_params=stopping, model=model, goals=True,
                                  noise=0.0, sequences=range(21), context_initialization=nn.SEMINORMAL)

        utils.save_object("bigmodel1_yesgoals_relu_adam_nonoise_goaltest", model)
    sys.exit()


    error_data_list = []
    for model_type in ["yesgoals"]:#, "nogoals"]:
        goals = model_type == "yesgoals"
        for goal_multiplier in [2]:#, 3]:#, 2]:
            print("goal multiplier:")
            print(goal_multiplier)
            for noise in [0, 1, 2, 3]:
                print("noise:")
                print(noise)
                for clamped_goals in [True]:#, False]:
                    for i in [0, 1, 2, 3, 4, 5]: #range(5, 7):
                        print("Network:")
                        print(i)
                        model = utils.load_object("bigmodel1_" + model_type +"_relu_adam_nonoise_goaltest", i)

                        #goal1 = utils.str_to_onehot("g_1_make_tea", env.GoalEnvData.goals1_list) * 4  #np.zeros((1, 2), np.float32)
                        #goal2 = utils.str_to_onehot("g_2_infuse_tea", env.GoalEnvData.goals2_list) #np.zeros((1, 9), np.float32)
                        #goal2 = utils.str_to_onehot("g_2_add_milk", env.GoalEnvData.goals2_list) * 2. # np.zeros((1, 9), np.float32)
                        #goal2 = utils.str_to_onehot("g_2_add_cream", env.GoalEnvData.goals2_list) * 4

                        if True:
                            test_data = goalenv2020.generate_test_data(model, noise=noise,
                                                                   goal1_noise=0., goal2_noise=0.,
                                                                   goals=goals, num_tests=1,
                                                                   goal_multiplier=goal_multiplier,
                                                                   sequence_ids=range(21), ##[3, 16, 16],;  #0=coffee black, 3 = coffee cream, 16 = tea milk
                                                                   switch_goal1= None, #(range(28, 36), goal1),  # 28, 36 for tea cream. 18, 23 for coffee as tea.
                                                                   switch_goal2= None, #(range(14, 23), goal2), #18-27= coffee cream to milk , 14-23 = tea milk to cream
                                                                   #switch_sequence=2,
                                                                   noise_per_step=True,
                                                                   noise_per_step_to_input=False,
                                                                   disruption_per_step=False,
                                                                   initialization=nn.SEMINORMAL,
                                                                   clamped_goals = clamped_goals)
                        print("generated data")

                        #utils.save_object("control"+model_type+str(i), test_data)
                        #test_data = utils.load_object("control" + model_type + str(i))
                        goalenv2020.VERBOSE = False
                        tsne_results, test_data, _, error_data, _ = goalenv2020.analyse_test_data(test_data, do_rdm=False, do_tsne=False, do_loss=True,
                                                                                               goals=True, mds_sequences=[0, 1, 2])#, mds_range=15)
                        error_data_list.append(error_data)
                        #utils.save_object("tsne_resultsmds"+model_type+str(i), tsne_results)
                        #utils.save_object("mds"+model_type+str(i), test_data) # test dat ais updated by analysis
                        #tsne_results = utils.load_object("tsne_resultsmds"+model_type+str(i))
                        #test_data = utils.load_object("mds" + model_type + str(i))
                        #goalenv2020.plot_tsne(tsne_results, test_data, tsne_goals=False, tsne_subgoals=True, tsne_actions=False, tsne_sequences=False,
                        #          tsne_errors=False, tsne_sequence=[0, 1, 2], #tsne_sequence_interval=[1, 30],
                        #          filename="mds", annotate=True, save_txt=True)
                        #utils.save_object("tsne_results_bigmodel1_yesgoals", tsne_results)

    utils.write_lists_to_csv("clamped_goals_4.csv", error_data_list, labels=goalenv2020.error_testing_labels)

    sys.exit()

# Trained:
# - "bigmodel1_yesgoals": 15 networks  (uniform initialization, sigmoid)
# - "bigmodel1_yesgoals_2nd_batch": 1 network (same)
# - "bigmodel1_yesgoals_relu": 4 networks, HE initialization, seminormal context initialization. UNSURE, RETRAIN
# - "bigmodel1_yesgoals_relu": 4 networks, HE initialization, normal context initialization
if True:
    from goalenv import goalenv2020
    from goalenv import environment
    """
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
                #model = nn.ElmanGoalNet(size_hidden=50, size_observation=29, size_action=19,
                #                        size_goal1=len(environment.GoalEnvData.goals1_list),
                #                        size_goal2=len(environment.GoalEnvData.goals2_list),
                #                        algorithm=nn.ADAM, learning_rate=0.001,
                #                        L2_reg=0.0001,
                #                        initialization=nn.HE,
                #                        nonlinearity=nn.RELU,
                #                        last_action_inputs=True)
        
                #stopping = nn.ParamsStopping(max_iterations=25000, min_iterations=3010, check_frequency=1000,
                #                             stop_condition=goalenv2020.stop_condition, goals=True, noise=0.0)
                #model = goalenv2020.train(stop_params=stopping, model=model, goals=True,
                #                          noise=0.0, sequences=range(21), context_initialization=nn.SEMINORMAL)
                #utils.save_object("bigmodel1_yesgoals_relu_adam_nonoise", model)

                #model = utils.load_object(model_files, i)  #"bigmodel1_nogoals_relu_adam_nonoise", i)

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

if True:
    import cognitiveload.model3 as mod3
    mod3.FAST_RDM = True
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
