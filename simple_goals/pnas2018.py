import neuralnet as nn
import utils
import numpy as np
import tensorflow as tf
import scripts
import analysis
import matplotlib.pyplot as plt

all_inputs = ["start", "coffee", "milk", "water", "stir", "tea", "sugar", "serve_coffee", "serve_tea"]
all_outputs = ["coffee", "milk", "water", "stir", "sugar", "tea", "serve_coffee", "serve_tea"]
seq1 = ["start", "coffee", "milk", "stir", "water", "stir", "serve_coffee"]
seq2 = ["start", "coffee", "water", "stir", "milk", "stir", "serve_coffee"]
seq3 = ["start", "tea", "sugar", "stir", "water", "stir", "serve_tea"]
seq4 = ["start", "tea", "water", "stir", "sugar", "stir", "serve_tea"]
seqs = [seq1, seq2, seq3, seq4]
goals = [[[0., 1.]]*6, [[0., 1.]]*6, [[1., 0]]*6, [[1, 0]]*6]
goals = [np.asarray(goal, dtype=np.float32).reshape((-1, 1, 2)) for goal in goals]
sequence_probabilities=[0.25, 0.25, 0.25, 0.25]

def train(mse=False, noise= 0., iterations=5000, reg= 0.0, lopsided = False, special_seq=False):
    model = nn.NeuralNet(size_hidden=15, size_observation=9, size_action=8, size_goal1=0, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = 0.5 if mse else 0.1
    model.L2_regularization = reg

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        seqid = utils.idx_from_probabilities(sequence_probabilities)

        sequence = seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            #model.context = np.zeros((1, model.size_hidden), dtype=np.float32) #np.float32(np.random.uniform(0.01, 0.99, (1, model.size_hidden)))
            model.context = np.float32(np.random.randint(0, 1, (1, model.size_hidden)))
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            if mse:
                loss, _ = model.train_MSE(targets, None, None, tape)
            else:
                loss, _ = model.train_obsolete(targets, None, None, tape)

        # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_goals = utils.rolling_avg(rng_avg_goals, ratios[0] == 1,
                                          speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print(
                "{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_goals))
    return model

def train_with_goals(noise=0, iterations=10000, learning_rate=0.1):
    model = nn.NeuralNet(size_hidden=15, size_observation=9, size_action=8, size_goal1=2, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = learning_rate
    model.L2_regularization = 0.

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        seqid = utils.idx_from_probabilities(sequence_probabilities)

        goal = goals[seqid]
        sequence = seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.goal1 = goal[0]
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                # Add noise
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            loss, _ = model.train_obsolete(targets, goal, None, tape)
        # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_goals = utils.rolling_avg(rng_avg_goals, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print(
                "{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_goals))
    return model


def train_hierarchical_nogoals(noise=0, iterations=10000, learning_rate=0.1, reg_strength=0.001, reg_increase="linear"):
    model = nn.NeuralNet(size_hidden=15, size_observation=9, size_action=8, size_goal1=0, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = learning_rate
    model.L2_regularization = 0.

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        seqid = utils.idx_from_probabilities(sequence_probabilities)

        #goal = goals[seqid]
        sequence = seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            #model.goal1 = np.zeros_like(goal[0])
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                # Add noise
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            cols = model.size_hidden
            # Regularization in the hidden layer weights
            # Recurrent hidden to hidden connections
            extra_loss = utils.weight_regularization_calculator(model.hidden_layer.w,
                                                                [0, model.size_hidden], [0, cols],
                                                                reg_strength, reg_type="recurrent", reg_increase=reg_increase)
            # Prev action to hidden
            #extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
            #                                                     [model.size_hidden+9, model.size_hidden+9+model.size_action], [0, cols],
            #                                                     reg_strength, reg_type="input_right", reg_increase=reg_increase)
            # Prev goal to hidden
            #extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
            #                                                     [model.size_hidden+9+model.size_action, model.size_hidden+9+model.size_action+2], [0, cols],
            #                                                     reg_strength, reg_type="input_left", reg_increase=reg_increase)

            #Regularization in the output layers (goals and actions) weights
            # hidden to next action
            #extra_loss += utils.weight_regularization_calculator(model.action_layer.w,
            #                                                     [0, model.size_hidden], [0, model.size_action],
            #                                                     reg_strength, reg_type="output_right", reg_increase=reg_increase)

            # Hidden to next goal
            #extra_loss += utils.weight_regularization_calculator(model.goal1_layer.w,
            #                                                     [0, model.size_hidden], [0, model.size_action],
            #                                                     reg_strength, reg_type="output_left", reg_increase=reg_increase)

            # Regularization of the observation (only goes to the action side)
            extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                                 [model.size_hidden, model.size_hidden+model.size_observation],
                                                                 [0, cols],
                                                                 reg_strength, reg_type="input_right", reg_increase=reg_increase)

            loss, _ = model.train_obsolete(targets, None, None, tape, extra_loss)
            #if(episode%100 == 0):
            #    print(loss.numpy()-extra_loss.numpy(), extra_loss.numpy())
        # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_goals = utils.rolling_avg(rng_avg_goals, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print(
                "{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_goals))
    return model

def train_hierarchical(noise=0, iterations=10000, learning_rate=0.1, reg_strength=0.001, reg_increase="linear"):
    model = nn.NeuralNet(size_hidden=15, size_observation=9, size_action=8, size_goal1=2, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = learning_rate
    model.L2_regularization = 0.

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        seqid = utils.idx_from_probabilities(sequence_probabilities)

        goal = goals[seqid]
        sequence = seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.random.normal(0., .1, )#np.zeros((1, model.size_hidden), dtype=np.float32)
            model.goal1 = np.zeros_like(goal[0])
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                # Add noise
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            cols = model.size_hidden
            # Regularization in the hidden layer weights
            # Recurrent hidden to hidden connections
            extra_loss = utils.weight_regularization_calculator(model.hidden_layer.w,
                                                                [0, model.size_hidden], [0, cols],
                                                                reg_strength, reg_type="recurrent", reg_increase=reg_increase)
            # Prev action to hidden
            extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                                 [model.size_hidden+9, model.size_hidden+9+model.size_action], [0, cols],
                                                                 reg_strength, reg_type="input_right", reg_increase=reg_increase)
            # Prev goal to hidden
            extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                                 [model.size_hidden+9+model.size_action, model.size_hidden+9+model.size_action+2], [0, cols],
                                                                 reg_strength, reg_type="input_left", reg_increase=reg_increase)

            #Regularization in the output layers (goals and actions) weights
            # hidden to next action
            extra_loss += utils.weight_regularization_calculator(model.action_layer.w,
                                                                 [0, model.size_hidden], [0, model.size_action],
                                                                 reg_strength, reg_type="output_right", reg_increase=reg_increase)
            # Hidden to next goal
            extra_loss += utils.weight_regularization_calculator(model.goal1_layer.w,
                                                                 [0, model.size_hidden], [0, model.size_action],
                                                                 reg_strength, reg_type="output_left", reg_increase=reg_increase)

            # Regularization of the observation (only goes to the action side)
            extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                                 [model.size_hidden, model.size_hidden+model.size_observation],
                                                                 [0, cols],
                                                                 reg_strength, reg_type="input_right", reg_increase=reg_increase)

            loss, _ = model.train_obsolete(targets, goal, None, tape, extra_loss)

        # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_goals = utils.rolling_avg(rng_avg_goals, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print(
                "{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_goals))
    return model


def make_models(num_models):
    models = []
    for i in range(num_models):
        models.append(train())
        accuracy_test(models[-1])

def accuracy_test(model):
    hidden_activation = []
    all_choices = []
    for sequence in seqs:
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            model.h_action_wta.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    accuracy_totals = np.zeros((len(seq1)-1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1
    accuracy_totals /= 4
    print(accuracy_totals)
    return hidden_activation

def accuracy_test_with_goals(model):
    hidden_activation = []
    all_choices = []
    for j, sequence in enumerate(seqs):
        goal = goals[j]
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.goal1 = goal[0]
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            model.h_action_wta.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    accuracy_totals = np.zeros((len(seq1)-1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1
    accuracy_totals /= 4
    print(accuracy_totals)
    return hidden_activation

def accuracy_test_reg_hierarchy(model, model_num=None):
    hidden_activation = []
    all_choices = []
    for j, sequence in enumerate(seqs):
        goal = goals[j]
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.goal1 = np.zeros_like(goal[0])
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            model.h_action_wta.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    accuracy_totals = np.zeros((len(seq1)-1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1
    accuracy_totals /= 4
    if model_num is not None:
        print(model_num, accuracy_totals)
    else:
        print(accuracy_totals)
    return hidden_activation


def accuracy_test_reg_hierarchy_nogoals(model, model_num=None):
    hidden_activation = []
    all_choices = []
    for j, sequence in enumerate(seqs):
        #goal = goals[j]
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            #model.goal1 = np.zeros_like(goal[0])
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            model.h_action_wta.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    accuracy_totals = np.zeros((len(seq1)-1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1
    accuracy_totals /= 4
    if model_num is not None:
        print(model_num, accuracy_totals)
    else:
        print(accuracy_totals)
    return hidden_activation

def make_rdm_and_mds(name, with_goals=False):
    model = utils.load_object(name, 1)  # eg 'noise_test2'
    if with_goals:
        hidden = accuracy_test_with_goals(model)
    else:
        hidden = accuracy_test(model)
    # Turn into a list of simple vectors
    for i, tensor in enumerate(hidden):
        hidden[i] = tensor.numpy().reshape(-1)
    rdmatrix = analysis.rdm_spearman(hidden)
    labels = []
    for i, sequence in enumerate(seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(rdmatrix, labels, "Spearman rho matrix")

    for i in range(4):
        mdsy = analysis.mds(hidden[6*i:6*i+6])
        analysis.plot_mds_points(mdsy, range(len(mdsy)), labels=labels[6*i:6*i+6])

    mdsy = analysis.mds(hidden)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6*i:6*i+6], range(6), labels=labels[6*i:6*i+6], style=style, show=(i==3))


def make_rdm_and_mds_reg_hierarchy(name):
    model = utils.load_object(name, 1)  # eg 'noise_test2'
    hidden = accuracy_test_reg_hierarchy(model)
    # Turn a list of tensors into a list of np vectors
    for i, tensor in enumerate(hidden):
        hidden[i] = tensor.numpy().reshape(-1)

    # Now cut that in two
    left_units = [vector[:len(hidden)/2] for vector in hidden]
    right_units = [vector[len(hidden)/2:] for vector in hidden]

    # Make the labels for the rdms
    labels = []
    for i, sequence in enumerate(seqs):
        for action in sequence[1:]:
            labels.append(str(i) + '_' + action)
    for side in [[left_units, "left units"], [right_units, "right_units"]]:
        rdm = analysis.rdm_spearman(side[0])
        analysis.plot_rdm(rdm, labels, "Spearman rho matrix"+side[1])

        #for i in range(4):
        #    mdsy = analysis.mds(side[0][6*i:6*i+6])
        #    analysis.plot_mds_points(mdsy, range(len(mdsy)), labels=labels[6*i:6*i+6])

        mdsy = analysis.mds(side[0])
        for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
            analysis.plot_mds_points(mdsy[6*i:6*i+6], range(6), labels=labels[6*i:6*i+6], style=style)




def make_rdm_multiple_hierarchy(name, num_networks, title="-", save_files=True, file_save_name=None):
    if file_save_name == None:
        file_save_name = name
    # Make one rdm for each network
    rdmatrices_left = []
    rdmatrices_right = []
    for i in range(num_networks):
        # Skip number 52
        #if i == 53 or i == 1:
        #    continue
        model = utils.load_object(name, i)
        hidden = accuracy_test_reg_hierarchy(model, i)

        # Turn a list of tensors into a list of np vectors
        for i, tensor in enumerate(hidden):
            hidden[i] = tensor.numpy().reshape(-1)

        # Now cut that in two and make an RDM for each
        cutoff = int(len(hidden[0])//2)
        left_units = [vector[:cutoff] for vector in hidden]
        rdm_left= analysis.rdm_spearman(left_units)
        rdmatrices_left.append(rdm_left)

        right_units = [vector[cutoff:] for vector in hidden]
        rdm_right = analysis.rdm_spearman(right_units)
        rdmatrices_right.append(rdm_right)

    # Do the same processing for each side (low level/left and high_level/right)
    for side in [[rdmatrices_left, "_goals"], [rdmatrices_right, "_actions"]]:
        # Now average over all matrices
        avg_matrix = None
        for matrix in side[0]:
            if avg_matrix is None:
                avg_matrix = matrix
            else:
                avg_matrix += matrix
        avg_matrix = avg_matrix / num_networks
        side_name = file_save_name+side[1]
        np.savetxt(side_name+"_rdm_mat"+utils.datestr()+".csv", avg_matrix, delimiter=",")
        labels = []
        for i, sequence in enumerate(seqs):
            for action in sequence[1:]:
                labels.append(str(i)+'_'+action)
        analysis.plot_rdm(avg_matrix, labels, title+side_name + " spearman rho matrix")
        if save_files:
            plt.savefig(side_name+'_rdm'+utils.datestr())
        plt.clf()

        mdsy = analysis.mds(avg_matrix)
        for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
            analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
        plt.title(title+side_name)
        if save_files:
            plt.savefig(side_name + '_mds'+utils.datestr())
        plt.clf()


def make_rdm_multiple_hierarchy_nogoals(name, num_networks, title="-", save_files=True, file_save_name=None, cutoff=None):
    if file_save_name == None:
        file_save_name = name
    # Make one rdm for each network
    rdmatrices_left = []
    rdmatrices_right = []
    for i in range(num_networks):
        if i == 19:
            continue
        model = utils.load_object(name, i)
        hidden = accuracy_test_reg_hierarchy_nogoals(model, i)

        # Turn a list of tensors into a list of np vectors
        for i, tensor in enumerate(hidden):
            hidden[i] = tensor.numpy().reshape(-1)

        # Now cut that in two and make an RDM for each
        if cutoff is None:
            cutoff = int(len(hidden[0])//2)
        left_units = [vector[:cutoff] for vector in hidden]
        rdm_left = analysis.rdm_spearman(left_units)
        rdmatrices_left.append(rdm_left)

        right_units = [vector[cutoff:] for vector in hidden]
        rdm_right = analysis.rdm_spearman(right_units)
        rdmatrices_right.append(rdm_right)

    # Do the same processing for each side (low level/left and high_level/right)
    for side in [[rdmatrices_left, "_goals"], [rdmatrices_right, "_actions"]]:
        # Now average over all matrices
        avg_matrix = None
        for matrix in side[0]:
            if avg_matrix is None:
                avg_matrix = matrix
            else:
                avg_matrix += matrix
        avg_matrix = avg_matrix / num_networks
        side_name = file_save_name+side[1]
        np.savetxt(side_name+"_rdm_mat"+utils.datestr()+".csv", avg_matrix, delimiter=",")
        labels = []
        for i, sequence in enumerate(seqs):
            for action in sequence[1:]:
                labels.append(str(i)+'_'+action)
        analysis.plot_rdm(avg_matrix, labels, title+side_name + " spearman rho matrix")
        if save_files:
            plt.savefig(side_name+'_rdm'+utils.datestr())
        plt.clf()

        mdsy = analysis.mds(avg_matrix)
        for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
            analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
        plt.title(title+side_name)
        if save_files:
            plt.savefig(side_name + '_mds'+utils.datestr())
        plt.clf()

def make_rdm_multiple(name, num_networks, with_goals=False, title="-", save_files=True):
    # Make one rdm for each network
    rdmatrices = []
    for i in range(num_networks):
        model = utils.load_object(name, i)
        if with_goals:
            hidden = accuracy_test_with_goals(model)
        else:
            hidden = accuracy_test(model)
        # Turn into a list of simple vectors
        for i, tensor in enumerate(hidden):
            hidden[i] = tensor.numpy().reshape(-1)
        rdmatrix = analysis.rdm_spearman(hidden)
        rdmatrices.append(rdmatrix)

    # Now average over all matrices
    avg_matrix = None
    for matrix in rdmatrices:
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix = avg_matrix / num_networks
    np.savetxt(name+"_rdm_mat.csv", avg_matrix, delimiter=",")
    labels = []
    for i, sequence in enumerate(seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(avg_matrix, labels, title + " spearman rho matrix")
    if save_files:
        plt.savefig(name+'_rdm')
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
    plt.title(title)
    if save_files:
        plt.savefig(name + '_mds')
    plt.clf()

import predictivenet

def train_predictive_net(model=None, iterations=5000, learning_rate=0.1, algorithm=nn.RMSPROP, hidden_units=15):
    if model is None:
        model = predictivenet.PredictiveNet(size_hidden=hidden_units, algorithm=algorithm, size_observation=len(all_inputs), size_action=len(all_outputs))
    num_episodes = iterations
    model.learning_rate = learning_rate

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_full_seq = 0.
    rng_avg_preds = 0.

    for episode in range(num_episodes):
        seqid = utils.idx_from_probabilities(sequence_probabilities)
        sequence = seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        action_targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        prediction_targets = utils.liststr_to_onehot(sequence[1:], all_inputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        model.prediction_linear = np.zeros((1, model.size_observation), dtype=np.float32)  #initial prediction = 0
        # run the network
        # Initialize context with random/uniform values.
        with tf.GradientTape() as tape:
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            for i in range(len(action_targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                #model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_wta).reshape((-1, len(action_targets[0])))    # reshape to (x, 8)
            ratios = scripts.evaluate([tchoices], [action_targets])
            tpreds = np.array(model.h_prediction_wta).reshape((-1, len(prediction_targets[0])))
            ratios_predictions = scripts.evaluate([tpreds], [prediction_targets])

            # Train model, record loss. NOTE: targets and predictions are identical for this task!!!
            loss, gradients = model.train_MSE(action_targets, prediction_targets, tape)

        # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_preds = utils.rolling_avg(rng_avg_preds, ratios_predictions[0], speed)
        rng_avg_full_seq = utils.rolling_avg(rng_avg_full_seq, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            grad_avg = sum([np.sum(tf.reduce_sum(tf.abs(gradient)).numpy()) for gradient in gradients])/sum([tf.size(gradient).numpy() for gradient in gradients])
            grad_max = max([np.max(tf.reduce_max(tf.abs(gradient)).numpy()) for gradient in gradients])
            print("{0}: avg loss={1}, \tactions={2}, \tfull_seq={3}, \tpredictions={4}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_full_seq, rng_avg_preds))

    return model


def accuracy_test_predictive(model, test_number=None):
    hidden_activation = []
    all_choices = []
    all_predictions = []
    for sequence in seqs:
        seq_choices = []
        seq_predictions = []
        all_predictions.append(seq_predictions)
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        action_targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        prediction_targets = utils.liststr_to_onehot(sequence[1:], all_inputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.prediction_linear = np.zeros((1, model.size_observation), dtype=np.float32)  #initial prediction = 0,
            # Reset the previous action
            for i in range(len(action_targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)

            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_wta).reshape((-1, len(action_targets[0])))
            prediction = np.array(model.h_prediction_wta).reshape((-1, len(prediction_targets[0])))
            model.h_action_wta.clear()
            model.h_prediction_wta.clear()
            seq_choices.append(choice)
            seq_predictions.append(prediction)

    # Now evaluate accuracy:
    optimal_accuracy = np.asarray([.5, .5, 1., 1., 1., 1.])
    accuracy = np.zeros((len(seq1) - 1))
    accuracy_weighted = np.zeros((len(seq1) - 1))
    for i in range(len(all_choices)):
        action_targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(action_targets)):
            if (all_choices[i][0][j] == action_targets[j]).all():
                accuracy_weighted[j] += 1 * sequence_probabilities[i]
                accuracy[j] += 1/len(all_choices)
    optimal_actions = np.array_equal(accuracy_weighted, optimal_accuracy)

    optimal_accuracy_preds = [.5, .5, 1, 1, 1, 1]
    accuracy_preds = np.zeros((len(seq1) - 1))
    accuracy_preds_weighted = np.zeros((len(seq1) - 1))
    for i in range(len(all_predictions)):
        prediction_targets = utils.liststr_to_onehot(seqs[i][1:], all_inputs)
        for j in range(len(prediction_targets)):
            if (all_predictions[i][0][j] == prediction_targets[j]).all():
                accuracy_preds_weighted[j] += 1 * sequence_probabilities[i]
                accuracy_preds[j] += 1/len(all_predictions)
    optimal_predictions = np.array_equal(accuracy_preds_weighted, optimal_accuracy_preds)

    if test_number is None:
        print(accuracy, accuracy_weighted, optimal_actions, accuracy_preds, accuracy_preds_weighted, optimal_predictions)
    else:
        print("Actions: {0} ({1}) - network {2} -- {3}".format(accuracy, accuracy_weighted, test_number, optimal_actions))
    if not optimal_actions or not optimal_predictions:
        print("actions:")
        for i in range(len(seqs)):
            print([utils.onehot_to_str(all_choices[i][0][j], all_outputs) for j in range(len(action_targets))])
        print("predictions:")
        for i in range(len(seqs)):
            print([utils.onehot_to_str(all_predictions[i][0][j], all_inputs) for j in range(len(prediction_targets))])
    return hidden_activation, optimal_actions and optimal_predictions


def make_rdm_multiple_predictive(name, num_networks,title="-", save_files=True):
    # Make one rdm for each network
    optimal_list = []
    rdmatrices = []
    for i in range(num_networks):
        model = utils.load_object(name, i)
        hidden, optimal = accuracy_test_predictive(model, i)
        optimal_list.append(optimal)
        if optimal:
            # Turn into a list of simple vectors
            for i, tensor in enumerate(hidden):
                hidden[i] = tensor.numpy().reshape(-1)
            rdmatrix = analysis.rdm_spearman(hidden)
            rdmatrices.append(rdmatrix)
    print("{0} networks, of which {1} achieve optimal accuracy".format(num_networks, optimal_list.count(True)))
    # Now average over all matrices
    avg_matrix = None
    for matrix in rdmatrices:
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix = avg_matrix / len(rdmatrices)

    # delete the unwanted rows and columns:
    #avg_matrix = np.delete(avg_matrix, [0, 6, 12], 0)
    #avg_matrix = np.delete(avg_matrix, [0, 6, 12], 1)
    nps = 6  # number of elements per sequence

    if save_files:
        np.savetxt(name+".csv", avg_matrix, delimiter=",")
    labels = []
    for i, sequence in enumerate(seqs):
        for action in sequence[0:-1]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(avg_matrix, labels, title + " spearman rho matrix")
    if save_files:
        plt.savefig(name+'_rdm')
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.']):
        analysis.plot_mds_points(mdsy[nps * i:nps * i + nps], range(nps), labels=labels[nps * i:nps * i + nps], style=style)
    plt.title(title)
    if save_files:
        plt.savefig(name + '_mds')
    plt.clf()
    return avg_matrix
