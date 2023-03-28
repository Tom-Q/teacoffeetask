from neural import neuralnet as nn
import utils
import numpy as np
import tensorflow as tf
import scripts
import analysis
import matplotlib.pyplot as plt
from pnas import pnashierarchy

np.set_printoptions(precision=3)
all_inputs = ["start", "coffee", "milk", "cream", "water", "stir", "tea", "servetea", "servecoffee", "sugar", "end"]
all_outputs = ["start", "coffee", "milk", "cream", "water", "stir", "tea", "servetea", "servecoffee", "sugar", "end"]
seq1 = ['start', 'coffee', 'water', 'stir', 'cream', 'servecoffee', 'end']  #60%
seq2 = ['start', 'coffee', 'water', 'stir', 'milk', 'servecoffee', 'end']  # 20%
seq3 = ['start', 'tea', 'water', 'stir', 'sugar', 'servetea', 'end']  # 20%
goals = [[[0., 1.]] * 7, [[0., 1.]] * 7, [[1., 0]] * 7]
goals = [np.asarray(goal, dtype=np.float32).reshape((-1, 1, 2)) for goal in goals]
seqs = [seq1, seq2, seq3]
sequence_probabilities = [0.6, 0.2, 0.2]
#sequence_probabilities = [1./3, 1./3, 1./3]
optimal_accuracy = np.asarray([0.8, 1., 1., 0.8, 1., 1.])
optimal_accuracy_goals = np.asarray([1., 1., 1., 0.8, 1., 1.])
#optimal_accuracy = np.asarray([2./3, 1., 1., 2./3, 1., 1.])
#optimal_accuracy_goals = np.asarray([1., 1., 1., 2./3, 1., 1.])

def accuracy_test(model, test_number=None):
    hidden_activation = []
    all_choices = []
    all_choices_probas = []
    for sequence in seqs:

        seq_choices = []
        all_choices.append(seq_choices)

        seq_choices_probas = []
        all_choices_probas.append(seq_choices_probas)

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
            choice = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            seq_choices_probas.append(np.array(model.h_action_activation))
            model.h_action_collapsed.clear()
            model.h_action_activation.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    full_accuracy = np.zeros((3,6,11), dtype=np.float32)
    for i in range(len(all_choices_probas)):
        targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(targets)):
            full_accuracy[i, j, :] = targets[j] - 1./(1+np.exp(-all_choices_probas[i][0][j]))

    accuracy = np.zeros((len(seq1) - 1))
    accuracy_weighted = np.zeros((len(seq1) - 1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_weighted[j] += 1 * sequence_probabilities[i]
                accuracy[j] += 1/len(all_choices)
    optimal = np.array_equal(accuracy_weighted, optimal_accuracy)
    if test_number is None:
        print(accuracy, accuracy_weighted, optimal)
    else:
        print("{0} ({1}) - network {2} -- {3}".format(accuracy, accuracy_weighted, test_number, optimal))
    if not optimal:
        for i in range(len(seqs)):
            print([utils.onehot_to_str(all_choices[i][0][j], all_outputs) for j in range(len(targets))])
    return hidden_activation, optimal, full_accuracy


def train(model = None, mse=False, noise= 0., iterations=5000, l2reg=0.0, learning_rate=0.1, algorithm=nn.SGD, hidden_units=15):
    if model is None:
        model = nn.GoalNet(size_hidden=hidden_units, algorithm=algorithm, size_observation=len(all_inputs), size_action=len(all_inputs), size_goal1=0, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = learning_rate
    model.L2_regularization = l2reg

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_full_seq = 0.

    for episode in range(num_episodes):
        seqid = utils.idx_from_probabilities(sequence_probabilities)
        sequence = seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)

        # run the network
        # Initialize context with random/uniform values.
        with tf.GradientTape() as tape:
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])

            # Train model, record loss.
            if mse:
                loss, gradients = model.train_MSE(targets, None, None, tape)
            else:
                loss, gradients = model.train_obsolete(targets, None, None, tape)

        # Monitor progress using averages
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_full_seq = utils.rolling_avg(rng_avg_full_seq, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            grad_avg = sum([np.sum(tf.reduce_sum(tf.abs(gradient)).numpy()) for gradient in gradients])/sum([tf.size(gradient).numpy() for gradient in gradients])
            grad_max = max([np.max(tf.reduce_max(tf.abs(gradient)).numpy()) for gradient in gradients])
            print("{0}: avg loss={1}, \tactions={2}, \tfull_seq={3}, \tgrad_avg={4}, \tgrad_max={5}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_full_seq, grad_avg, grad_max))

    return model


def train_with_goals(model=None, mse=False, learning_rate=0.1, noise=0., iterations=5000, l2reg=0.0,
                     algorithm=nn.SGD, hidden_units=15,
                     reg_strength=0., reg_increase="square"):
    num_goals = 2
    if model is None:
        model = nn.GoalNet(size_hidden=hidden_units, algorithm=algorithm, size_observation=len(all_inputs), size_action=len(all_inputs), size_goal1=num_goals, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = 0.5 if mse else learning_rate
    model.L2_regularization = l2reg

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
        targets_goal1 = goal
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.goal1 = goal[0]
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])

            cols = model.size_hidden
            # Regularization in the hidden layer weights
            # Recurrent hidden to hidden connections
            extra_loss = pnashierarchy.weight_regularization_calculator(model.hidden_layer.w,
                                                          [0, model.size_hidden], [0, cols],
                                                          reg_strength, reg_type="recurrent", reg_increase=reg_increase)
            # Prev action to hidden
            # extra_loss += weight_regularization_calculator(model.hidden_layer.w,
            #                                               [model.size_hidden+9, model.size_hidden+9+model.size_action],
            #                                               [0, cols],
            #                                               reg_strength, reg_type="input_right", reg_increase=reg_increase)
            # Prev goal to hidden
            extra_loss += pnashierarchy.weight_regularization_calculator(model.hidden_layer.w,
                                                          [model.size_hidden+9+model.size_action, model.size_hidden+9+model.size_action+num_goals],
                                                          [0, cols],
                                                          reg_strength, reg_type="input_left", reg_increase=reg_increase)

            # SWITCHED OUTPUT LEFT AND OUTPUT RIGHT.
            #Regularization in the output layers (goals and actions) weights
            # hidden to next action
            extra_loss += pnashierarchy.weight_regularization_calculator(model.action_layer.w,
                                                           [0, model.size_hidden], [0, model.size_action],
                                                           reg_strength, reg_type="output_right", reg_increase=reg_increase)
            # Hidden to next goal
            extra_loss += pnashierarchy.weight_regularization_calculator(model.goal1_layer.w,
                                                           [0, model.size_hidden], [0, model.size_action],
                                                           reg_strength, reg_type="output_left", reg_increase=reg_increase)

            # Regularization of the observation (only goes to the action side)
            #extra_loss += weight_regularization_calculator(model.hidden_layer.w,
            #                                                     [model.size_hidden, model.size_hidden+model.size_observation],
            #                                                     [0, cols],
            #                                                     reg_strength, reg_type="input_right", reg_increase=reg_increase)

            loss, _ = model.train_obsolete(targets, goal, None, tape, extra_loss)

            # Train model, record loss.
            #if mse:
            #    loss = model.train_MSE(targets, None, None, tape)
            #else:
            #    loss, gradients = model.train_obsolete(targets, targets_goal1, None, tape)
        # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_goals = utils.rolling_avg(rng_avg_goals, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print("{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
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
            choice = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            model.h_action_collapsed.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    accuracy = np.zeros((len(seq1) - 1))
    accuracy_weighted = np.zeros((len(seq1) - 1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(seqs[i][1:], all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_weighted[j] += 1 * sequence_probabilities[i]
                accuracy[j] += 1/len(all_choices)
    optimal = np.array_equal(accuracy_weighted, optimal_accuracy_goals)
    if test_number is None:
        print(accuracy, accuracy_weighted, optimal)
    else:
        print("{0} ({1}) - network {2} -- {3}".format(accuracy, accuracy_weighted, test_number, optimal))
    if not optimal:
        for i in range(len(seqs)):
            print([utils.onehot_to_str(all_choices[i][0][j], all_outputs) for j in range(len(targets))])
    return hidden_activation, optimal


def make_rdm_multiple(name, num_networks, type="spearman", with_goals=False, title="-", save_files=True, skips=[]):
    # Make one rdm for each network
    optimal_list = []
    rdmatrices = []
    error_per_step = np.zeros((3, 6, 11), dtype=np.float32)  # number of steps
    hidden_avg = []
    hiddens =  np.zeros((3, 6), dtype=np.float32)
    for i in range(num_networks+len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        if with_goals:
            hidden, optimal = accuracy_test_with_goals(model, i)
        else:
            hidden, optimal, error_per_step_model = accuracy_test(model, i)
            error_per_step += error_per_step_model
        optimal_list.append(optimal)
        # Turn into a list of simple vectors
        for j, tensor in enumerate(hidden):
            hidden[j] = tensor.numpy().reshape(-1)
            hidden_avg.append(np.average(hidden[j]))  # Get the average activation for that time-step
        if type == "spearman":
            rdmatrix = analysis.rdm_spearman(hidden)
        elif type == "euclidian":
            rdmatrix = analysis.rdm_euclidian(hidden)
        else:
            raise Exception("RDM type " + type + " not implemented")
        rdmatrices.append(rdmatrix)

    # Now
    i = j = 0
    for act_avg in hidden_avg:
        hiddens[i, j] += act_avg
        j += 1
        if j>5:
            j = 0
            i += 1
            if i>2:
                i = 0
                j = 0
    print(hiddens/num_networks)
    print("{0} networks, of which {1} achieve optimal accuracy".format(num_networks, optimal_list.count(True)))

    # Hidden activation per step averages
    #for i, hidden_act in enumerate(hidden):
    #    hidden[i] = np.average(hidden_act)
    #hidden_avg = hidden.reshape(18, 11)
    #print(enumerate(hidden_avg))

    # Error per step averages
    error_avg = error_per_step/100.
    error_avg = error_avg.reshape(18, 11)
    print(error_avg)

    # Now average over all matrices
    avg_matrix = None
    for matrix in rdmatrices:
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix = avg_matrix / num_networks

    # delete the unwanted rows and columns:
    avg_matrix = np.delete(avg_matrix, [0, 6, 12], 0)
    avg_matrix = np.delete(avg_matrix, [0, 6, 12], 1)
    nps = 5 # number of elements per sequence

    if save_files:
        np.savetxt(name+"_rdm_mat_"+type+utils.datestr()+".txt", avg_matrix, delimiter="\t", fmt='%.2e')
        np.savetxt(name+"_errors"+utils.datestr()+".txt", error_avg, delimiter="\t", fmt='%.2e')
    labels = []
    for i, sequence in enumerate(seqs):
        for action in sequence[1:-1]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(avg_matrix, labels, title + " " + type + " matrix")
    if save_files:
        plt.savefig(name+'_rdm_'+type)
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.']):
        analysis.plot_mds_points(mdsy[nps * i:nps * i + nps], range(nps), labels=labels[nps * i:nps * i + nps], style=style)
    plt.title(title)
    if save_files:
        plt.savefig(name + '_mds')
    plt.clf()
    return avg_matrix


# This should achieve 100% prediction success.
def train_predictive_net(model=None, iterations=5000, learning_rate=0.1, algorithm=nn.RMSPROP, l2reg = 0, hidden_units=15, type='sigmoid'):
    inputs_str = ["start", "coffee", "milk", "cream", "water", "stir", "tea", "serve", "sugar", "end"]
    outputs_str = ["start", "coffee", "milk", "cream", "water", "stir", "tea", "servetea", "servecoffee", "sugar", "end"]
    seq1in = ['start', 'coffee',  'water', 'stir', 'cream', 'serve', 'end']  # 60%
    seq1t = ['start', 'coffee', 'water', 'stir', 'cream', 'servecoffee', 'end']  # 60%
    seq2in = ['start', 'coffee', 'water', 'stir', 'milk', 'serve', 'end']  # 20%
    seq2t = ['start', 'coffee',  'water', 'stir', 'milk', 'servecoffee', 'end']  # 20%
    seq3in = ['start', 'tea',  'water', 'stir', 'sugar', 'serve', 'end']  # 20%
    seq3t = ['start', 'tea', 'water', 'stir', 'sugar', 'servetea', 'end']  # 20%

    inputs_seqs = [seq1in, seq2in, seq3in]
    target_seqs = [seq1t, seq2t, seq3t]

    if model is None:
        model = nn.PredictiveNet(size_hidden=hidden_units,
                                 algorithm=algorithm,
                                 size_observation=len(inputs_str),
                                 size_action=len(outputs_str))
    num_episodes = iterations
    model.learning_rate = learning_rate

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_full_seq = 0.
    rng_avg_preds = 0.

    for episode in range(num_episodes):
        seqid = utils.idx_from_probabilities(sequence_probabilities)
        sequence_i = inputs_seqs[seqid]
        sequence_t = target_seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence_i[:-1], inputs_str)
        action_targets = utils.liststr_to_onehot(sequence_t[:-1], outputs_str)
        prediction_targets = utils.liststr_to_onehot(sequence_i[1:], inputs_str)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        model.prediction_linear = np.zeros((1, model.size_observation), dtype=np.float32)  #initial prediction = 0
        # run the network
        # Initialize context with random/uniform values.
        with tf.GradientTape() as tape:
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.prediction_wta = np.zeros((1, model.size_observation), dtype=np.float32)
            model.prediction_probability = np.zeros((1, model.size_observation), dtype=np.float32)
            for i in range(len(action_targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                #model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_wta).reshape((-1, len(action_targets[0])))
            ratios = scripts.evaluate([tchoices], [action_targets])
            tpreds = np.array(model.h_prediction_wta).reshape((-1, len(prediction_targets[0])))
            ratios_predictions = scripts.evaluate([tpreds], [prediction_targets])

            # Train model, record loss. NOTE: targets and predictions are identical for this task!!!
            loss, gradients = model.train(tape, [action_targets, prediction_targets])

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


def accuracy_test_predictive(model, test_number=None, type='sigmoid'):
    inputs_str = ["start", "coffee", "milk", "cream", "water", "stir", "tea", "serve", "sugar", "end"]
    outputs_str = ["start", "coffee", "milk", "cream", "water", "stir", "tea", "servetea", "servecoffee", "sugar", "end"]
    seq1in = ['start', 'coffee',  'water', 'stir', 'cream', 'serve', 'end']  # 60%
    seq1t = ['start', 'coffee', 'water', 'stir', 'cream', 'servecoffee', 'end']  # 60%
    seq2in = ['start', 'coffee', 'water', 'stir', 'milk', 'serve', 'end']  # 20%
    seq2t = ['start', 'coffee',  'water', 'stir', 'milk', 'servecoffee', 'end']  # 20%
    seq3in = ['start', 'tea',  'water', 'stir', 'sugar', 'serve', 'end']  # 20%
    seq3t = ['start', 'tea', 'water', 'stir', 'sugar', 'servetea', 'end']  # 20%
    inputs_seqs = [seq1in, seq2in, seq3in]
    target_seqs = [seq1t, seq2t, seq3t]

    hidden_activation = []
    all_choices = []
    all_predictions = []
    for i in range(len(inputs_seqs)):
        sequence_i = inputs_seqs[i]
        sequence_t = target_seqs[i]
        seq_choices = []
        seq_predictions = []
        all_predictions.append(seq_predictions)
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence_i[:-1], inputs_str)
        action_targets = utils.liststr_to_onehot(sequence_t[:-1], outputs_str)
        prediction_targets = utils.liststr_to_onehot(sequence_i[1:], inputs_str)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.prediction_linear = np.zeros((1, model.size_observation), dtype=np.float32)  #initial prediction = 0,
            # Reset the previous action
            for i in range(len(action_targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation, type)
                hidden_activation.append(model.context)

            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_collapsed).reshape((-1, len(action_targets[0])))
            prediction = np.array(model.h_prediction_wta).reshape((-1, len(prediction_targets[0])))
            model.h_action_collapsed.clear()
            model.h_prediction_wta.clear()
            seq_choices.append(choice)
            seq_predictions.append(prediction)

    # Now evaluate accuracy:
    optimal_accuracy = np.asarray([1., 1., 1., 1., 1., 1.])
    accuracy = np.zeros((len(seq1) - 1))
    accuracy_weighted = np.zeros((len(seq1) - 1))
    for i in range(len(all_choices)):
        action_targets = utils.liststr_to_onehot(target_seqs[i][:-1], outputs_str)
        for j in range(len(action_targets)):
            if (all_choices[i][0][j] == action_targets[j]).all():
                accuracy_weighted[j] += 1 * sequence_probabilities[i]
                accuracy[j] += 1/len(all_choices)
    optimal_actions = np.array_equal(accuracy_weighted, optimal_accuracy)

    optimal_accuracy_preds = [.8, 1, 1, .8, 1, 1]
    accuracy_preds = np.zeros((len(seq1) - 1))
    accuracy_preds_weighted = np.zeros((len(seq1) - 1))
    for i in range(len(all_predictions)):
        prediction_targets = utils.liststr_to_onehot(inputs_seqs[i][1:], inputs_str)
        for j in range(len(prediction_targets)):
            if (all_predictions[i][0][j] == prediction_targets[j]).all():
                accuracy_preds_weighted[j] += 1 * sequence_probabilities[i]
                accuracy_preds[j] += 1/len(all_predictions)
    optimal_predictions = np.array_equal(accuracy_preds_weighted, optimal_accuracy_preds)

    if test_number is None:
        print(accuracy, accuracy_weighted, optimal_actions, accuracy_preds, accuracy_preds_weighted, optimal_predictions)
    else:
        print("Actions: {0} ({1}) - network {2} -- {3}".format(accuracy, accuracy_weighted, test_number, optimal_actions and optimal_predictions))
    if not optimal_actions or not optimal_predictions:
        print("actions:")
        for i in range(len(seqs)):
            print([utils.onehot_to_str(all_choices[i][0][j], outputs_str) for j in range(len(action_targets))])
        print("predictions:")
        for i in range(len(seqs)):
            print([utils.onehot_to_str(all_predictions[i][0][j], inputs_str) for j in range(len(prediction_targets))])
    return hidden_activation, optimal_actions and optimal_predictions


def make_rdm_multiple_predictive(name, num_networks, type='sigmoid', title="-", save_files=True, skips=[]):
    # Make one rdm for each network
    optimal_list = []
    rdmatrices = []
    for i in range(num_networks+len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        hidden, optimal = accuracy_test_predictive(model, i, type=type)
        optimal_list.append(optimal)
        # Turn into a list of simple vectors
        for i, tensor in enumerate(hidden):
            hidden[i] = tensor.numpy().reshape(-1)
        if optimal:
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
    avg_matrix = np.delete(avg_matrix, [0, 6, 12], 0)
    avg_matrix = np.delete(avg_matrix, [0, 6, 12], 1)
    nps = 5  # number of elements per sequence

    if save_files:
        np.savetxt(name+"_rdm_mat"+utils.datestr()+".txt", avg_matrix, delimiter="\t", fmt='%.2e')
    labels = []
    for i, sequence in enumerate(seqs):
        for action in sequence[1:-1]:
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

"""
# TODO: (good to have when I find more time) refactor this to match the pnas2018hierarchy stuff...
# ...this is basically identical to that
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
            model.goal1 = goal[0]#np.zeros_like(goal[0]) #
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
    accuracy_totals /= 4
    if model_num is not None:
        print(model_num, accuracy_totals)
    else:
        print(accuracy_totals)
    return hidden_activation
"""


def make_rdm_multiple_hierarchy(name, num_networks, title="-", save_files=True, file_save_name=None, skips=[]):
    if file_save_name == None:
        file_save_name = name
    # Make one rdm for each network
    rdmatrices_left = []
    rdmatrices_right = []
    for i in range(num_networks + len(skips)):
        # Skip number
        if skips is not None and i in skips:
           continue
        model = utils.load_object(name, i)
        hidden, _ = accuracy_test_with_goals(model, i)

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

    matrices = []
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

        # delete the unwanted rows and columns:
        avg_matrix = np.delete(avg_matrix, [0, 6, 12], 0)
        avg_matrix = np.delete(avg_matrix, [0, 6, 12], 1)
        nps = 5  # number of elements per sequence

        side_name = file_save_name+side[1]
        np.savetxt(side_name+"_rdm_mat"+utils.datestr()+".txt", avg_matrix, delimiter="\t", fmt='%.2e')
        labels = []
        for i, sequence in enumerate(seqs):
            for action in sequence[1:-1]:
                labels.append(str(i)+'_'+action)

        analysis.plot_rdm(avg_matrix, labels, title+side_name + " spearman rho matrix")
        if save_files:
            plt.savefig(side_name+'_rdm'+utils.datestr())
        plt.clf()

#        nps = 5  # number of elements per sequence
        mdsy = analysis.mds(avg_matrix)
        for i, style in enumerate(['ro-', 'b|--', 'gx-.']):
            analysis.plot_mds_points(mdsy[nps * i:nps * i + nps], range(nps), labels=labels[nps * i:nps * i + nps],
                                     style=style)
        plt.title(title+side_name)
        if save_files:
            plt.savefig(side_name + '_mds'+utils.datestr())
        plt.clf()
        matrices.append(avg_matrix)
    return matrices