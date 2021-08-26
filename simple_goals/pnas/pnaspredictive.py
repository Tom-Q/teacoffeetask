from pnas import pnas2018task
import utils
import numpy as np
import tensorflow as tf
import analysis
import matplotlib.pyplot as plt
import neuralnet as nn
import scripts

def train_predictive_net(model=None, iterations=5000, learning_rate=0.1, algorithm=nn.RMSPROP, hidden_units=15):
    if model is None:
        model = nn.PredictiveNet(size_hidden=hidden_units, algorithm=algorithm,
                                 size_observation=len(pnas2018task.all_inputs),
                                 size_action=len(pnas2018task.all_outputs))
    num_episodes = iterations
    model.learning_rate = learning_rate

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_full_seq = 0.
    rng_avg_preds = 0.

    for episode in range(num_episodes):
        seqid = utils.idx_from_probabilities(pnas2018task.sequence_probabilities)
        sequence = pnas2018task.seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        action_targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        prediction_targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_inputs)
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

def accuracy_test_predictive(model, test_number=None):
    hidden_activation = []
    all_choices = []
    all_predictions = []
    for sequence in pnas2018task.seqs:
        seq_choices = []
        seq_predictions = []
        all_predictions.append(seq_predictions)
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        action_targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        prediction_targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_inputs)
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
    accuracy = np.zeros((len(pnas2018task.seq1) - 1))
    accuracy_weighted = np.zeros((len(pnas2018task.seq1) - 1))
    for i in range(len(all_choices)):
        action_targets = utils.liststr_to_onehot(pnas2018task.seqs[i][1:], pnas2018task.all_outputs)
        for j in range(len(action_targets)):
            if (all_choices[i][0][j] == action_targets[j]).all():
                accuracy_weighted[j] += 1 * pnas2018task.sequence_probabilities[i]
                accuracy[j] += 1/len(all_choices)
    optimal_actions = np.array_equal(accuracy_weighted, optimal_accuracy)

    optimal_accuracy_preds = [.5, .5, 1, 1, 1, 1]
    accuracy_preds = np.zeros((len(pnas2018task.seq1) - 1))
    accuracy_preds_weighted = np.zeros((len(pnas2018task.seq1) - 1))
    for i in range(len(all_predictions)):
        prediction_targets = utils.liststr_to_onehot(pnas2018task.seqs[i][1:], pnas2018task.all_inputs)
        for j in range(len(prediction_targets)):
            if (all_predictions[i][0][j] == prediction_targets[j]).all():
                accuracy_preds_weighted[j] += 1 * pnas2018task.sequence_probabilities[i]
                accuracy_preds[j] += 1/len(all_predictions)
    optimal_predictions = np.array_equal(accuracy_preds_weighted, optimal_accuracy_preds)

    if test_number is None:
        print(accuracy, accuracy_weighted, optimal_actions, accuracy_preds, accuracy_preds_weighted, optimal_predictions)
    else:
        print("Actions: {0} ({1}) - network {2} -- {3}".format(accuracy, accuracy_weighted, test_number, optimal_actions))
    if not optimal_actions or not optimal_predictions:
        print("actions:")
        for i in range(len(pnas2018task.seqs)):
            print([utils.onehot_to_str(all_choices[i][0][j], pnas2018task.all_outputs) for j in range(len(action_targets))])
        print("predictions:")
        for i in range(len(pnas2018task.seqs)):
            print([utils.onehot_to_str(all_predictions[i][0][j], pnas2018task.all_inputs) for j in range(len(prediction_targets))])
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
    for i, sequence in enumerate(pnas2018task.seqs):
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
