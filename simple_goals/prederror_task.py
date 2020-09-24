import neuralnet as nn
import utils
import numpy as np
import tensorflow as tf
import scripts
import analysis

all_inputs = ["start", "coffee", "milk", "cream", "water", "stir", "tea", "stea", "scofcream", "scofmilk", "sugar"]
all_outputs = ["start", "coffee", "milk", "cream", "water", "stir", "tea", "stea", "scofcream", "scofmilk", "sugar"]
seq1 = ['start', 'coffee', 'water', 'stir', 'cream', 'scofcream', 'start']  #60%
seq2 = ['start', 'coffee', 'water', 'stir', 'milk', 'scofmilk', 'start']  # 20%
seq3 = ['start', 'tea', 'water', 'stir', 'sugar', 'stea', 'start']  # 20%
goals = [[[0., 1.]] * 7, [[0., 1.]] * 7, [[1., 0]] * 7]
goals = [np.asarray(goal, dtype=np.float32).reshape((-1, 1, 2)) for goal in goals]
seqs = [seq1, seq2, seq3]
probas = [0.6, 0.2, 0.2]

def accuracy_test(model, test_number=None):
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
    accuracy_totals = np.zeros((len(seq1) - 1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1 * probas[i]
    #accuracy_totals /= len(peng_seqs)
    if test_number is None:
        print(accuracy_totals)
    else:
        print("{0} - network {1}".format(accuracy_totals, test_number))
    return hidden_activation

def train(model = None, mse=False, noise= 0., iterations=5000, reg= 0.0):
    if model is None:
        model = nn.NeuralNet(size_hidden=15, size_observation=len(all_inputs), size_action=len(all_inputs), size_goal1=0, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = 0.5 if mse else 0.1
    model.L2_regularization = reg

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        decider = np.random.uniform()
        if decider < 0.6:
            seqid = 0
        elif decider < 0.8:
            seqid = 1
        else:
            seqid = 2

        sequence = seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32) #np.float32(np.random.uniform(0.01, 0.99, (1, model.size_hidden)))
            #model.context = np.float32(np.random.uniform(0.01, 0.99, (1, model.size_hidden)))
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
                loss = model.train_MSE(targets, None, None, tape)
            else:
                loss = model.train_obsolete(targets, None, None, tape)

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


def accuracy_test_with_goals(model, test_number=None):
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
    accuracy_totals = np.zeros((len(seq1) - 1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1
    accuracy_totals /= 3
    if test_number is None:
        print(accuracy_totals)
    else:
        print("{0} - network {1}".format(accuracy_totals, test_number))
    return hidden_activation


def train_with_goals(model = None, mse=False, noise=0., iterations=5000, reg=0.0, lopsided = True):
    if model is None:
        model = nn.NeuralNet(size_hidden=15, size_observation=len(all_inputs), size_action=len(all_inputs), size_goal1=2, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = 0.5 if mse else 0.1
    model.L2_regularization = reg

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        decider = np.random.uniform()
        if decider < 0.6:
            seqid = 0
        elif decider < 0.8:
            seqid = 1
        else:
            seqid = 2

        sequence = seqs[seqid]
        goal = goals[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32) #np.float32(np.random.uniform(0.01, 0.99, (1, model.size_hidden)))
            #model.context = np.float32(np.random.uniform(0.01, 0.99, (1, model.size_hidden)))
            model.goal1 = goal[0]
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
                loss = model.train_MSE(targets, None, None, tape)
            else:
                loss = model.train_obsolete(targets, None, None, tape)

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


def make_rdm_multiple(name, num_networks, with_goals=False, title="-"):
    # Make one rdm for each network
    rdmatrices = []
    for i in range(num_networks):
        model = utils.load_object(name, i)
        if with_goals:
            hidden = accuracy_test_with_goals(model, i)
        else:
            hidden = accuracy_test(model, i)
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
    # Remove every 6th column and row ("b" state)
    for i in range(5, 6*3, 6):
        print(i)
    avg_matrix = np.delete(avg_matrix, range(5, 6*3, 6), axis = 0)
    avg_matrix = np.delete(avg_matrix, range(5, 6*3, 6), axis = 1)
    np.savetxt(name+".csv", avg_matrix, delimiter=",")
    labels = []
    for i, sequence in enumerate(seqs):
        for action in sequence[1:-1]:
            labels.append(str(i)+'_'+action)
    analysis.show_rdm(avg_matrix, labels, title+" spearman rho matrix")

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.']):
        analysis.plot_mds_points(mdsy[5 * i:5 * i + 5], range(5), labels=labels[5 * i:5 * i + 5], style=style,
                                 show=(i == 2), title=title)
