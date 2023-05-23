import cognitiveload.model4 as mod4
import cognitiveload.model3 as mod3
import cognitiveload.model2 as mod2
import cognitiveload.cogloadtask as task
import rdm
import tensorflow as tf
from neural import neuralnet as nn, optimizers
import utils
import numpy as np

def training_and_params():
    mod3.FAST_RDM = True

    """
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
    # sys.exit(0)

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
    # nnparams.algorithm = optimizers.SGD
    # nnparams.initialization = utils.NORMAL
    # nnparams.nonlinearity = tf.nn.sigmoid
    # nnparams.learning_rate = 0.03
    # nnparams.size_hidden = 100
    # mod2.run_model2_multiple(stopping_params=stopping,
    #                         num_networks=5,  #from_file="normal_l20.0_sig_sgd",
    #                         name="normal_sig_100_sgd_005_48",
    #                         nnparams=nnparams,
    #                         blanks=True,
    #                         type=rdm.EUCLIDIAN)

    # works but doesn't look very good
    # nnparams.algorithm = optimizers.SGD
    # nnparams.initialization = utils.NORMAL
    # nnparams.nonlinearity = tf.nn.tanh
    # nnparams.learning_rate = 0.01
    # nnparams.size_hidden = 50
    # mod2.run_model2_multiple(stopping_params=stopping,
    #                             num_networks=5,  #from_file="normal_l20.0_sig_sgd",
    #                             name="normal_tanh_50_sgd_001_48",
    #                             nnparams=nnparams,
    #                             blanks=True,
    #                             type=rdm.EUCLIDIAN)

    # doesnt work.
    # nnparams.algorithm = optimizers.ADAM
    # nnparams.initialization = utils.NORMAL
    # nnparams.nonlinearity = tf.nn.sigmoid
    # nnparams.learning_rate = 0.001
    # nnparams.size_hidden = 25
    # mod2.run_model2_multiple(stopping_params=stopping,
    #                         num_networks=5,  # from_file="normal_l20.0_sig_sgd",
    #                         name="normal_tanh_25_adam_0001_48",
    #                         nnparams=nnparams,
    #                         blanks=True,
    #                         type=rdm.EUCLIDIAN)

    # works
    # nnparams.algorithm = optimizers.SGD
    # nnparams.initialization = utils.NORMAL
    # nnparams.nonlinearity = tf.nn.sigmoid
    # nnparams.learning_rate = 0.03
    # nnparams.size_hidden = 100
    # mod2.run_model2_multiple(stopping_params=stopping,
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

    # --> TO USE:  works : sgd norm sig 0.05 100 - 6100 7100 5300 6300 5900

    # tanh 30 96 normal 001
    nnparams.algorithm = optimizers.SGD
    nnparams.initialization = utils.NORMAL
    nnparams.nonlinearity = tf.nn.tanh
    nnparams.learning_rate = 0.01
    nnparams.size_hidden = 30
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=50, from_file="nogoals_normal_tanh_30_sgd_001_96",
                             name="nogoals_normal_tanh_30_sgd_001_96",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN,
                             skips=None)
    #sys.exit()

    nnparams.algorithm = optimizers.ADAM
    nnparams.initialization = utils.HE
    nnparams.nonlinearity = tf.nn.relu
    nnparams.learning_rate = 0.001
    nnparams.size_hidden = 25
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=50, from_file="nogoals_he_relu_25_adam_0001",
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
                             num_networks=50, from_file="nogoals_xavier_sigmoid_100_sgd_0005_48",
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
                             num_networks=50, from_file="goals_uniform_sigmoid_100_sgd_0005_48",
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
                             num_networks=25,  # from_file="normal_l20.0_sig_sgd",
                             name="xavier_l20.0_tanh25_adam_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    # works but learning is slow (~10,000 or so)
    nnparams.algorithm = optimizers.SGD
    nnparams.initialization = utils.NORMAL
    nnparams.learning_rate = 0.01
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  # from_file="normal_l20.0_sig_sgd",
                             name="normal_l20.0_tanh25_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    # works but learning is slow (~15,000 or so)
    nnparams.learning_rate = 0.005
    nnparams.initialization = utils.XAVIER
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  # from_file="normal_l20.0_sig_sgd",
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
                             num_networks=5,  # from_file="normal_l20.0_sig_sgd",
                             name="normal_l20.0_sig40_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    # works well and fast
    # tanh
    nnparams.size_hidden = 50
    nnparams.learning_rate = 0.01
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="normal_l20.0_sig_sgd",
                             name="normal_l20.0_sig50_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)

    nnparams.size_hidden = 60
    nnparams.learning_rate = 0.005
    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="normal_l20.0_sig_sgd",
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
                             num_networks=5,  # from_file="normal_l20.0_sig_sgd",
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
                             num_networks=5,  # from_file="normal_l20.0_sig_sgd",
                             name="xavier01_l20.0_sig25_sgd_48",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)
    sys.exit()
    # Use the easy arithmetic sequences :-)
    for optimizer in [optimizers.SGD]:  # , optimizers.ADAM]:
        for initialization in [utils.NORMAL]:  # , utils.HE]:
            for nonlinearity in [tf.nn.relu]:  # [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu]:
                for L2_reg in [0.001]:  # , 0.0]:
                    if optimizer == optimizers.SGD:
                        if nonlinearity == tf.nn.sigmoid:
                            lr = 0.1  # sigmoid is super slow to learn
                        else:
                            lr = 0.005
                    else:  # ADAM
                        if nonlinearity == tf.nn.sigmoid:
                            lr = 0.01
                        else:
                            lr = 0.001
                    hrp = mod3.HierarchyGradientParams(regincrease="linear", regstrength=0.0)
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
                                             num_networks=2,
                                             # from_file=initialization + "_l2"+str(L2_reg) + str_nl + optimizer[0],
                                             name=initialization + "_l2" + str(L2_reg) + str_nl + optimizer[0],
                                             nnparams=nnparams,
                                             blanks=True,
                                             type=rdm.EUCLIDIAN)
    sys.exit()
    # i=-1
    # print(i)
    # hrp.reg_strength = 0.001
    # nnparams.L2_reg = 0.0

    hrp.reg_strength = 0.0
    # matrix processing test
    import rdm
    import numpy as np

    # test_rdm = rdm.rdm(props, matrix_values=mat)
    # averaged_rdm = test_rdm.average_values(["v", "n"], ["s"])
    # averaged_rdm.save(filename="test rdm")

    # mod3.run_model3_multiple(stopping_params=stopping,
    #                         num_networks=20, from_file="model3_euclidian_activations", #"model3_distances_spearman",
    #                         name="model3_spearman_avdistances",
    #                         hrp=None,
    #                         nnparams=nnparams,
    #                         blanks=True,
    #                         type=rdm.EUCLIDIAN)
    # sys.exit()
    if False:
        rdms = mod2._generate_rdm_from_files("activations_model2_old", "properties_file", 5)
        final_rdm = None
        for my_rdm in rdms:
            if final_rdm is None:
                final_rdm = mod2.process_rdmatrix(rdm.rdm(properties=my_rdm.properties, matrix_values=my_rdm.matrix.copy()),
                                                  True)
            else:
                final_rdm.matrix += mod2.process_rdmatrix(my_rdm, True).matrix
        final_rdm.matrix /= 5

        mod2.set_rdm_labels(final_rdm)
        # Save it
        final_rdm.save("test_rdm", title="RDM training combined")  # , dpi=200, figsize=60, fontsize=0.5)
        sys.exit()

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model2_rdm_rich",
                             name="model2_rdm_rich",
                             nnparams=nnparams,
                             blanks=True,
                             type=rdm.EUCLIDIAN)
    sys.exit()

    mod3.run_model2_multiple(stopping_params=stopping,
                             num_networks=25,  # from_file="model3_euclidian_distances",
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
                             num_networks=25, from_file="model2_plusminus",
                             name="model2_spearman_distances",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_DISTANCES,
                             type=rdm.SPEARMAN)

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model2_plusminus",
                             name="model2_euclidian_activations",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_ACTIVATIONS,
                             type=rdm.EUCLIDIAN)

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model2_plusminus",
                             name="model2_euclidian_distances",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_DISTANCES,
                             type=rdm.EUCLIDIAN)

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model2_plusminus",
                             name="model2_spearman_activations",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_ACTIVATIONS,
                             type=rdm.SPEARMAN)

    sys.exit()

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model2_",
                             name="model2_distances_euclidian",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_DISTANCES,
                             type=analysis.EUCLIDIAN)

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model2_",
                             name="model2_activations_euclidian",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_ACTIVATIONS,
                             type=analysis.EUCLIDIAN)

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model2_",
                             name="model2_distances_euclidian",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_DISTANCES,
                             type=analysis.EUCLIDIAN)

    mod2.run_model2_multiple(stopping_params=stopping,
                             num_networks=25, from_file="model2_",
                             name="model2_activations_euclidian",
                             nnparams=nnparams,
                             blanks=True,
                             mode=task.RDM_MODE_AVERAGE_ACTIVATIONS,
                             type=analysis.EUCLIDIAN)
    sys.exit()
    # hrp=mod3.HierarchyGradientParams(regincrease="linear", regstrength=0.00002)
    # mod3.run_model3_multiple(from_file="model3_test_gradient_2goals00002", num_networks=2, name="model3_test_gradient_2goals00002_test2", hrp=hrp)

    # model3_ADAMRELU are best normal ones.
    # "model3_ADAMRELU_gradient" is a first try for gradient ones. Didn't work.
    # "model3_ADAMRELU_gradient0_0001" is a second try for gradient ones.
    # Best perf so far: ADAM, RELU, HE, 0.0003, 300,000  (1, 1, .99, .83, .97, .99, .99, .97, .76, 1.0.
    # With no reg: (1., .93, .99, .99, .95, .)
    hrp = mod3.HierarchyGradientParams(regincrease="linear", regstrength=0.00001)
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

    i = -1
    print(i)
    hrp.reg_strength = 0.00003
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=25,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 3. about 100,000 iterations. Massive left/right difference.=

    sys.exit()
    i = 0
    print(i)
    hrp.reg_strength = 0.0
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 0. about 50,000 iterations. No left/right difference

    i += 1
    print(i)
    hrp.reg_strength = 0.000001
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 1. about 45,000 iterations, high variance in training time. No left/right difference

    i += 1
    print(i)
    hrp.reg_strength = 0.00001
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 2. 50,000 iterations agian. No left/right difference

    i += 1
    print(i)
    hrp.reg_strength = 0.0001
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 3. about 100,000 iterations. Massive left/right difference.=

    i += 1
    print(i)
    hrp.reg_strength = 0.001
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 4. Accuracy about 15 to 20%, always hits max iteration (151001). Even more massive difference.

    i += 1
    print(i)
    hrp.reg_strength = 0.0
    nnparams.L2_reg = 0.000001
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 5. about 50000, no effect.  Combined number quite self-similar, maybe an idiosyncracy?


    i += 1
    print(i)
    hrp.reg_strength = 0.0
    nnparams.L2_reg = 0.00001
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 6. 50000 again. Seems to have more contrast compared to 5. Combined numbers less self-simialr.


    i += 1
    print(i)
    hrp.reg_strength = 0.0
    nnparams.L2_reg = 0.0001
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 7.  50000 again. Feels like a mashup of 5 and 6. Impression = this doesn4t matter at this point

    i += 1
    print(i)
    hrp.reg_strength = 0.0
    nnparams.L2_reg = 0.001
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 8. 50000 again. Nothing obvious


    i += 1
    print(i)
    hrp.reg_strength = 0.0
    nnparams.L2_reg = 0.01
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 9. max iterations, accuracy about 75-90%. Looks high contrast again, ish.


    i += 1
    print(i)
    hrp.reg_strength = 0.000001
    hrp.reg_increase = "square"
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 10. about 90000 iterations. Clear effect, big differences.


    i += 1
    print(i)
    hrp.reg_strength = 0.00001
    hrp.reg_increase = "square"
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 11. about 150,000 iterations, some finish some dont

    i += 1
    print(i)
    hrp.reg_strength = 0.0001
    hrp.reg_increase = "square"
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 12. max iterations, accuracy about 35%

    i += 1
    print(i)
    hrp.reg_strength = 0.001
    hrp.reg_increase = "square"
    nnparams.L2_reg = 0.0
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="test" + str(i),  # "model3_withoutdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    # 13. max iterations, accuracy 5 to 30%


    sys.exit()
    print('with blanks now')

    stopping = nn.ParamsStopping(max_iterations=150001, min_iterations=10000, check_frequency=1000,
                                 stop_condition=mod3.stop_condition, blanks=True)
    mod3.run_model3_multiple(stopping_params=stopping,
                             num_networks=5,  # from_file="model3_nodummy"
                             name="model3_withdummy_150000_goals",
                             hrp=hrp,
                             nnparams=nnparams,
                             blanks=True)
    sys.exit()

    # 3 and 6 need extra training.
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