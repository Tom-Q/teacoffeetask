from pnas import pnas2018, pnashierarchy
import utils

def obsolete_pnas_stuff():
    for i in range(10):
        print(i)
        # model, _ = pnas2018.train(iterations=5000, learning_rate=0.1, size_hidden=100)
        model = nn.TripleACCNet(size_observation=7, size_action=8,
                                layer_sizes=[layers.RCPLayerSizes(input_bottomup=7, output_bottomup=96,
                                                                  input_topdown=96, output_topdown=96),
                                             layers.RCPLayerSizes(input_bottomup=96, output_bottomup=96,
                                                                  input_topdown=8, output_topdown=96)],
                                output_layer=True)
        model.learning_rate = 0.001
        pnas2018.trainACC(model=model, iterations=10000)
        pnas2018.accuracy_testACC(model)
        model.delete_tapes()  # can't save tapes
        utils.save_object("rdm_acc", model)

    # gain = [1., 1., .0, 0.0]
    pnas2018.make_rdm_multipleACC("rdm_acc", num_networks=5, rdm_type=analysis.SPEARMAN, save_name="spearman_acc")
    # pnas2018.make_rdm_multiple_gain("rdm_gains", num_networks=25, rdm_type=analysis.EUCLIDIAN, save_name="euclidian_zeroed_goalstea", gain=gain)
    # pnas2018.make_rdm_multiple("rdm_gains", num_networks=25, rdm_type=analysis.EUCLIDIAN, save_name="euclidian_nogain", with_goals=True)
    # pnas2018.make_rdm_multiple("rdm_gains", num_networks=25, rdm_type=analysis.SPEARMAN, save_name="spearman_nogain", with_goals=True)

def predictive_error_task():
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
        model = prederror_task.train_predictive_net(algorithm=nn.SGD, iterations=15000, learning_rate=0.02,
                                                    hidden_units=15, type='wta')
        prederror_task.accuracy_test_predictive(model, 'wta')
        utils.save_object('prederror_prednet_wta15', model)

    # using predictive net
    for i in range(90):
        print(i)
        model = prederror_task.train_predictive_net(algorithm=nn.SGD, iterations=10000, learning_rate=0.02,
                                                    hidden_units=15, type='sigmoid')
        prederror_task.accuracy_test_predictive(model, 'sigmoid')
        utils.save_object('prederror_prednet_sigmoid15', model)


    # Regular SGD with equiprobable sequences
    for i in range(0):
        print(i)
        model = prederror_task.train(iterations=8000)
        prederror_task.accuracy_test(model)
        utils.save_object('prederror_basic_equiprobable', model)
    # prederror_task.make_rdm_multiple('prederror_basic_equiprobable', 100, skips=[3, 19, 26, 28, 32, 35, 50, 55, 72, 76, 79, 80, 101, 110])

    # Regular SGD with goals with equiprobable sequences
    for i in range(0):
        print(i)
        model = prederror_task.train_with_goals(iterations=8000)
        prederror_task.accuracy_test_with_goals(model)
        utils.save_object('prederror_goals_equiprobable', model)
    # prederror_task.make_rdm_multiple('prederror_goals_equiprobable', 100, with_goals=True)

    # With 2 goals + gradient with equiprobable sequences
    for i in range(0):
        print(i)
        model = prederror_task.train_with_goals(reg_strength=0.0001, iterations=10000)
        prederror_task.accuracy_test_with_goals(model)
        utils.save_object('prederror_gradient_equiprobable', model)
    prederror_task.make_rdm_multiple_hierarchy('prederror_gradient', 100, skips=[17, 54, 91])
    # prederror_task.make_rdm_multiple_hierarchy('prederror_gradient_equiprobable', 100)

    sys.exit()
    # Regular SGD
    for i in range(0):
        model = prederror_task.train()
        prederror_task.accuracy_test(model)
        utils.save_object('prederror_basic', model)
    # prederror_task.make_rdm_multiple('prederror_basic', 100)

    # reload them and train them further
    for i in range(0):
        print(i)
        model = utils.load_object('prederror_basic', latest=i)
        model = prederror_task.train(model, iterations=3000)
        utils.save_object('prederror_basic_v2', model)
    # prederror_task.make_rdm_multiple('prederror_basic_v2', 100, skips=[3, 42, 79])

    # With 2 goals
    for i in range(0):
        print(i)
        model = prederror_task.train_with_goals()
        prederror_task.accuracy_test_with_goals(model)
        utils.save_object('prederror_goals', model)
    # prederror_task.make_rdm_multiple('prederror_goals', 100, with_goals=True)

    # reload them and train them further
    for i in range(0):
        print(i)
        model = utils.load_object('prederror_goals', latest=i)
        model = prederror_task.train_with_goals(model, iterations=3000)
        utils.save_object('prederror_goals_v2', model)
    # prederror_task.make_rdm_multiple('prederror_goals_v2', 100, with_goals=True, skips=[17, 45, 53])

    # With 2 goals + gradient
    for i in range(0):
        print(i)
        model = prederror_task.train_with_goals(reg_strength=0.0001, iterations=10000)
        prederror_task.accuracy_test_with_goals(model)
        utils.save_object('prederror_gradient', model)
    prederror_task.make_rdm_multiple_hierarchy('prederror_gradient', 100, skips=[17, 54, 91])

    # prederror_task.make_rdm_multiple_predictive('prederror_prednet', 100, skips=[16, 26, 35, 40, 50, 59, 61, 63, 79, 98, 100])
    with tf.device('/cpu:0'):
        for i in range(100):
            print(i + 1)
            model = prederror_task.train_predictive_net(iterations=15000, algorithm=nn.RMSPROP, learning_rate=0.005,
                                                        hidden_units=25)
            prederror_task.accuracy_test_predictive(model)
            utils.save_object("prederorr_predictivenet_rmsprop005_h25", model)
        for i in range(50):
            print(i + 1)
            model = pnas2018.train_predictive_net(iterations=15000, algorithm=nn.RMSPROP, learning_rate=0.005,
                                                  hidden_units=25)
            pnas2018.accuracy_test_predictive(model)
            utils.save_object("pnas_predictivenet_rmsprop005_h25", model)

    sys.exit()

def hyperparameters_analysis():
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
    # hyperparamstest.hyperparameter_analysis(file=file, type=analysis.MAHALANOBIS)
    hyperparamstest.reload(file)

    file = "hyperparams_test_rdm_euclidian"
    # hyperparamstest.hyperparameter_analysis(file=file, type=analysis.EUCLIDIAN)
    hyperparamstest.reload(file)

def pnas_timing_tests():
    start = timeit.default_timer()

    with tf.device('/gpu:0'):

        goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.00001, sequences=[0])
        goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.00001, sequences=[0, 1])
        goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.00001, sequences=[0, 1, 2, 3])
        goalenv.train(goals=False, num_iterations=100000, learning_rate=0.001, L2_reg=0.00001, sequences=[0, 1, 2, 3, 4, 5])

        stop = timeit.default_timer()
        #print('Time: ', stop - start)

        sys.exit("DONE THANK U")

def pnas_rdm_tests():
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