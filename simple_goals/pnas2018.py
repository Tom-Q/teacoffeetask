import neuralnet as nn
import utils
import numpy as np
import tensorflow as tf
import scripts
import analysis

all_inputs = ["start", "coffee", "milk", "water", "stir", "tea", "sugar"]
all_outputs = ["coffee", "milk", "water", "stir", "sugar", "tea", "serve_coffee", "serve_tea"]
seq1 = ["start", "coffee", "milk", "stir", "water", "stir", "serve_coffee"]
seq2 = ["start", "coffee", "water", "stir", "milk", "stir", "serve_coffee"]
seq3 = ["start", "tea", "sugar", "stir", "water", "stir", "serve_tea"]
seq4 = ["start", "tea", "water", "stir", "sugar", "stir", "serve_tea"]
seqs = [seq1, seq2, seq3, seq4]
goals = [[[0., 1.]]*6, [[0., 1.]]*6, [[1., 0]]*6, [[1, 0]]*6]
goals = [np.asarray(goal, dtype=np.float32).reshape((-1, 1, 2)) for goal in goals]

def train_pnas():
    model = nn.NeuralNet(size_hidden=15, size_observation=7, size_action=8, size_goal1=0, size_goal2=0)
    num_episodes = 10000
    model.learning_rate = .5
    model.L2_regularization = 0.

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        sequence = seqs[np.random.randint(len(seqs))]
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32) #np.float32(np.random.uniform(0.01, 0.99, (1, model.size_hidden)))
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            loss = model.train(targets, None, None, tape)

        # Monitor progress using rolling averages.
        speed = 2. / (
                    episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
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

def train_pnas_with_goals():
    model = nn.NeuralNet(size_hidden=15, size_observation=7, size_action=8, size_goal1=2, size_goal2=0)
    num_episodes = 10000
    model.learning_rate = .5
    model.L2_regularization = 0.

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        seqid = np.random.randint(len(seqs))
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
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            loss = model.train(targets, goal, None, tape)

        # Monitor progress using rolling averages.
        speed = 2. / (
                    episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
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


def make_pnas_models(num_models):
    models = []
    for i in range(num_models):
        models.append(train_pnas())
        accuracy_test_pnas(models[-1])

def accuracy_test_pnas(model):
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

def accuracy_test_pnas_with_goals(model):
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

def make_rdm_and_mds_pnas(name, with_goals=False):
    model = utils.load_object(name, 1)  # eg 'noise_test2'
    if with_goals:
        hidden = accuracy_test_pnas_with_goals(model)
    else:
        hidden = accuracy_test_pnas(model)
    # Turn into a list of simple vectors
    for i, tensor in enumerate(hidden):
        hidden[i] = tensor.numpy().reshape(-1)
    rdmatrix = analysis.rdm_spearman(hidden)
    labels = []
    for i, sequence in enumerate(seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.show_rdm(rdmatrix, labels, "Spearman rho matrix")

    for i in range(4):
        mdsy = analysis.mds(hidden[6*i:6*i+6])
        analysis.plot_mds_points(mdsy, range(len(mdsy)), labels=labels[6*i:6*i+6])

    mdsy = analysis.mds(hidden)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6*i:6*i+6], range(6), labels=labels[6*i:6*i+6], style=style, show=(i==3))


def make_rdm_multiple(name, num_networks, with_goals=False):
    # Make one rdm for each network
    rdmatrices = []
    for i in range(num_networks):
        model = utils.load_object(name, i)
        if with_goals:
            hidden = accuracy_test_pnas_with_goals(model)
        else:
            hidden = accuracy_test_pnas(model)
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
    labels = []
    for i, sequence in enumerate(seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.show_rdm(avg_matrix, labels, "Spearman rho matrix")