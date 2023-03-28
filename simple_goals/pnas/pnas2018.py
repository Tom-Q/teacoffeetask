from neural import neuralnet as nn, optimizers
import utils
import numpy as np
import tensorflow as tf
import scripts
import analysis
import matplotlib.pyplot as plt
from pnas import pnashierarchy, pnas2018task
import random
import rdm

MSE='mse'
CROSS_ENTROPY='cross_entropy'
ZEROS = 'zeros'
UNIFORM = 'uniform'

def train(model=None, noise=0., iterations=5000, l1reg=0.0, l2reg= 0.0, algorithm=optimizers.SGD,
          size_hidden=15, learning_rate=None, loss_type='cross_entropy',
          initial_context=ZEROS):
    if model is None:
        model = nn.GoalNet(size_hidden=size_hidden, size_observation=9, size_action=8, size_goal1=0, size_goal2=0,
                           algorithm=algorithm, initialization="normal")
    num_episodes = iterations
    if learning_rate is not None:  # Else keep the model's learning rate
        model.learning_rate = learning_rate
    model.L1_regularization = l1reg
    model.L2_regularization = l2reg

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_sequence = 0.

    for episode in range(num_episodes):
        model.new_episode(initial_context=initial_context)
        seqid = utils.idx_from_probabilities(pnas2018task.sequence_probabilities)

        sequence = pnas2018task.seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        # run the network
        #TODO: uncomment
        with tf.GradientTape() as ptape, tf.GradientTape() as rtape, tf.GradientTape() as ctape:
            model.PredictionTape = ptape
            model.RewardTape = rtape
            model.ControlTape = ctape
            #with tf.GradientTape(persistent=True) as tape:
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                if noise != 0.:
                    model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            #if episode % 2 == 0:
                # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            if loss_type==MSE:
                #loss, _ = model.train_MSE(targets, None, None, tape)
                loss, _ = model.train_MSE(targets, None, None, None)
            elif loss_type==CROSS_ENTROPY:
                # TODO: return this to its original statate
                #loss, _ = model.train_obsolete(targets, None, None, tape)
                #else:
                #loss, _ = model.train(tape, targets)
                loss = model.train(None, targets)
        #del tape

        #if episode % 2 == 0:
            # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_sequence = utils.rolling_avg(rng_avg_sequence, ratios[0] == 1,
                                          speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print(
                "{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_sequence))
    return model, rng_avg_sequence


def test_one_sequence(model, sequence_num, turn_goal_step=None, goal_to_turn=None):
    hidden_activation = []
    all_choices = []
    results = []

    for trials in range(100):
        sequence = pnas2018task.seqs[sequence_num]
        goal = pnas2018task.goals[sequence_num]
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.float32(np.abs(np.random.randint(0, 2, (1, model.size_hidden))-0.1))  # np.zeros((1, model.size_hidden), dtype=np.float32)
            model.goal1 = goal[0]
            # Reset the previous action
            for i in range(len(targets)):
                if i == turn_goal_step:
                    model.goal1 = goal_to_turn
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            model.clear_history()
            results.append(choice)

    # Now, count the number of unique result sequences, and the number of occurences for each unique sequence
    unique_results = []
    unique_results_counts = []
    for result in results:
        unique = True
        for i, unique_result in enumerate(unique_results):
            if np.array_equal(result, unique_result):
                unique_results_counts[i] += 1
                unique = False
                break
        if unique:
            unique_results.append(result)
            unique_results_counts.append(1)

    # Sort in order of frequency
    unique_results = [unique_result for (_, unique_result) in sorted(zip(unique_results_counts, unique_results), key=lambda pair: pair[0], reverse=True)]
    unique_results_counts = sorted(unique_results_counts, reverse=True)

    # Print the target sequence
    full_sequence_str = ""
    for row in targets:
        full_sequence_str += utils.onehot_to_str(row, pnas2018task.all_outputs) + "; "
    print("target: "+ full_sequence_str)

    # Now print the results
    for i, unique_result in enumerate(unique_results):
        full_sequence_str = ""
        for row in unique_result:
            full_sequence_str += utils.onehot_to_str(row, pnas2018task.all_outputs) + "; "
        print(str(unique_results_counts[i])+"%: "+ full_sequence_str)

    return hidden_activation


def make_models(num_models):
    models = []
    for i in range(num_models):
        models.append(train())
        accuracy_test(models[-1])


def accuracy_test(model, name=None, noise=0., initial_context=ZEROS):
    hidden_activation = []
    all_choices = []
    for sequence in pnas2018task.seqs:
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            model.new_episode(initial_context=initial_context)
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                if noise != 0.:
                    model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            model.h_action_collapsed.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    accuracy_totals = np.zeros((len(pnas2018task.seq1) - 1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(pnas2018task.seqs[i][1:], pnas2018task.all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1
    accuracy_totals /= 4
    if name is not None:
        print(name, accuracy_totals)
    else:
        print(accuracy_totals)
    return hidden_activation, accuracy_totals


def make_rdm_and_mds(name, with_goals=False):
    model = utils.load_object(name, 1)  # eg 'noise_test2'
    if with_goals:
        hidden = pnashierarchy.accuracy_test_with_goals(model)
    else:
        hidden = accuracy_test(model)
    # Turn into a list of simple vectors
    for i, tensor in enumerate(hidden):
        hidden[i] = tensor.numpy().reshape(-1)
    rdmatrix = analysis.rdm_spearman(hidden)
    labels = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(rdmatrix, labels, "Spearman rho matrix")

    for i in range(4):
        mdsy = analysis.mds(hidden[6*i:6*i+6])
        analysis.plot_mds_points(mdsy, range(len(mdsy)), labels=labels[6*i:6*i+6])

    mdsy = analysis.mds(hidden)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6*i:6*i+6], range(6), labels=labels[6*i:6*i+6], style=style, show=(i==3))


def make_rdm_noisy(name, num_networks, noise, num_runs_per_network=10, title="-", save_files=True, skips=[],
                      rdm_type=rdm.EUCLIDIAN):
    # Make one rdm for each network
    rdmatrices = []
    for i in range(num_networks + len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        hiddens = []
        for j in range(num_runs_per_network):
            hidden, _ = accuracy_test(model, name=str(i), noise=noise)
            for k, tensor in enumerate(hidden):
                hidden[k] = tensor.numpy().reshape(-1)
            hiddens.append(hidden)
        rdmatrix = rdm.rdm_noisy_mahalanobis(hiddens)
        rdmatrices.append(rdmatrix)

    # Now average over all matrices
    avg_matrix = None
    for matrix in rdmatrices:
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix = avg_matrix / num_networks
    name = name + '_' + rdm_type
    np.savetxt(name + "_rdm_mat.txt", avg_matrix, delimiter="\t", fmt='%.2e')
    labels = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for action in sequence[1:]:
            labels.append(str(i) + '_' + action)
    analysis.plot_rdm(avg_matrix, labels, title + " spearman rho matrix")
    if save_files:
        plt.savefig(name + '_rdm')
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
    plt.title(title)
    if save_files:
        plt.savefig(name + '_mds')
    plt.clf()
    return avg_matrix


def make_rdm_multiple(name, num_networks, with_goals=False, title="-", save_files=True, skips=[],
                      rdm_type=rdm.EUCLIDIAN, noise=0., save_name=None):
    # Make one rdm for each network
    hidden_activations = []
    rdmatrices = []
    for i in range(num_networks+len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        if with_goals:
            hidden = pnashierarchy.accuracy_test_with_goals(model)
        else:
            hidden, _ = accuracy_test(model, name=str(i), noise=noise)
        hidden_activations.append(hidden)
        # Turn into a list of simple vectors
        for k, tensor in enumerate(hidden):
            hidden[k] = tensor.numpy().reshape(-1)

        if rdm_type == rdm.SPEARMAN:
            rdmatrix = rdm.rdm_spearman(hidden)
        elif rdm_type == rdm.MAHALANOBIS:
            rdmatrix = rdm.rdm_mahalanobis(hidden)
            #rdmatrix = analysis.rdm_noisy2_mahalanobis(hidden)
        elif rdm_type == rdm.EUCLIDIAN:
            rdmatrix = rdm.rdm_euclidian(hidden)
        elif rdm_type == rdm.CRAPPYNOBIS:
            rdmatrix = rdm.rdm_crappynobis(hidden)
        else:
            raise ValueError("Only implemented rdm types are mahalanobis, spearman, euclidian")
        rdmatrices.append(rdmatrix)

    # Now average over all matrices
    avg_matrix = None
    for matrix in rdmatrices:
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix = avg_matrix / num_networks
    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        if save_name is None:
            save_name = name
        np.savetxt(save_name+"_rdm.txt", avg_matrix, delimiter="\t", fmt='%.2e')
    labels = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(avg_matrix, labels, title + " spearman rho matrix")
    if save_files:
        plt.savefig(save_name+'_rdm')
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
    plt.title(title)
    if save_files:
        plt.savefig(save_name + '_mds')
    plt.clf()
    return avg_matrix, hidden_activations


def make_rdm_multiple_ldt(name, num_networks, with_goals=False, title="-", save_files=True, skips=[],
                          rdm_type=rdm.SPEARMAN, noise_during=0., noise_after=0., num_samples=2,
                          initial_context=ZEROS, log_scale=False, save_name=None):
    # Make one rdm for each network
    #hidden_activations = []
    rdmatrices = []
    for i in range(num_networks+len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        hidden = []
        for j in range(num_samples):
            if with_goals:
                hidden += pnashierarchy.accuracy_test_with_goals(model)
            else:
                sample_hidden_activations, _ = accuracy_test(model, name=str(i), noise=noise_during, initial_context=initial_context)
                hidden += sample_hidden_activations
            #hidden_activations.append(hidden)
            # Turn into a list of simple vectors
        for k, tensor in enumerate(hidden):
            hidden[k] = tensor.numpy().reshape(-1)

        # Reshape hidden into a samples x states x activations ndarray.
        # At this point hidden is as follow: sample1 state1 activation ndarray, sample1 state2 activation nd array, etc.
        num_states = 24 # = 6
        num_activations = model.size_hidden
        hidden = np.concatenate(hidden, axis=0).reshape((num_samples, num_states, num_activations))
        rdmatrix = analysis.rdm_ldt(hidden, noise=noise_after)
        rdmatrices.append(rdmatrix)

    # Now average over all matrices
    avg_matrix = None
    for matrix in rdmatrices:
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix = avg_matrix / num_networks
    if log_scale:
        avg_matrix = np.where(avg_matrix != 0,  np.log10(avg_matrix), 0)
    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        if save_name is None:
            save_name = name
        np.savetxt(save_name+"_rdm_mat.txt", avg_matrix, delimiter="\t", fmt='%.2e')
    labels = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(avg_matrix, labels, title + " LD-t matrix" + " (log scale)" if log_scale else "")
    if save_files:
        plt.savefig(save_name+'_rdm.jpeg')
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
    plt.title(title)
    if save_files:
        plt.savefig(name + '_mds')
    plt.clf()
    return avg_matrix #, hidden_activations


def make_rdm_multiple_gain(name, num_networks, title="-", save_files=True, skips=[],
                      rdm_type=rdm.SPEARMAN, noise=0., save_name=None, gain=[1, 1, 2, 2.]):
    with_goals = True
    # Make one rdm for each network
    hidden_activations = []
    rdmatrices = []
    for i in range(num_networks+len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        if with_goals:
            hidden = pnashierarchy.accuracy_test_with_goals(model, gain=gain)
        else:
            hidden, _ = accuracy_test(model, name=str(i), noise=noise)
        hidden_activations.append(hidden)
        # Turn into a list of simple vectors
        for k, tensor in enumerate(hidden):
            hidden[k] = tensor.numpy().reshape(-1)

        if rdm_type == rdm.SPEARMAN:
            rdmatrix = rdm.rdm_spearman(hidden)
        elif rdm_type == rdm.MAHALANOBIS:
            rdmatrix = rdm.rdm_mahalanobis(hidden)
            #rdmatrix = analysis.rdm_noisy2_mahalanobis(hidden)
        elif rdm_type == rdm.EUCLIDIAN:
            rdmatrix = rdm.rdm_euclidian(hidden)
        elif rdm_type == rdm.CRAPPYNOBIS:
            rdmatrix = rdm.rdm_crappynobis(hidden)
        else:
            raise ValueError("Only implemented rdm types are mahalanobis, spearman, euclidian")
        rdmatrices.append(rdmatrix)

    # Now average over all matrices
    avg_matrix = None
    for matrix in rdmatrices:
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix = avg_matrix / num_networks
    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        if save_name is None:
            save_name = name
        np.savetxt(save_name+"_rdm.txt", avg_matrix, delimiter="\t", fmt='%.2e')
    labels = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(avg_matrix, labels, title + " spearman rho matrix")
    if save_files:
        plt.savefig(save_name+'_rdm')
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
    plt.title(title)
    if save_files:
        plt.savefig(save_name + '_mds')
    plt.clf()
    return avg_matrix, hidden_activations


def accuracy_test_keepcontext(model, name=None, num_samples=100):
    num_actions = len(pnas2018task.seq1)-1
    # List of 4 lists of 6 empty lists
    hidden_activation = [[[] for _ in range(num_actions)] for _ in range(len(pnas2018task.seqs))]

    all_choices = []

    # Make a list of 400 sequences
    sequences = [0, 1, 2, 3]*num_samples
    random.shuffle(sequences)

    for seq_id in sequences:
        sequence = pnas2018task.seqs[seq_id]
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            model.new_episode()
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                #model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation[seq_id][i].append(model.context.numpy())
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            model.h_action_collapsed.clear()
            seq_choices.append(choice)

    hidden_activations_averaged = []
    # Average the hidden activations per action
    for seq in hidden_activation:
        for action in seq:
            average = np.zeros_like(action[0])
            for instance in action:
                average += instance
            average /= len(action)
            hidden_activations_averaged.append(average)

    # Now evaluate accuracy:
    accuracy_totals = np.zeros(num_actions)
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(pnas2018task.seqs[sequences[i]][1:], pnas2018task.all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1
    accuracy_totals /= 4 * num_samples
    if name is not None:
        print(name, accuracy_totals)
    else:
        print(accuracy_totals)
    return hidden_activations_averaged, accuracy_totals


def make_rdm_multiple_keepcontext(name, num_networks, with_goals=False, title="-", save_files=True, skips=[],
                      rdm_type=rdm.SPEARMAN):
    # Make one rdm for each network
    hidden_activations = []
    rdmatrices = []
    for i in range(num_networks+len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        if with_goals:
            hidden = pnashierarchy.accuracy_test_with_goals(model)
        else:
            hidden, _ = accuracy_test_keepcontext(model, name=str(i))
        hidden_activations.append(hidden)
        # Turn into a list of simple vectors - already done in accuracy test in this version.
        for i, vector in enumerate(hidden):
            hidden[i] = vector.reshape(-1)

        if rdm_type == rdm.SPEARMAN:
            rdmatrix = rdm.rdm_spearman(hidden)
        elif rdm_type == rdm.MAHALANOBIS:
            rdmatrix = rdm.rdm_mahalanobis(hidden)
        elif rdm_type == rdm.EUCLIDIAN:
            rdmatrix = rdm.rdm_euclidian(hidden)
        elif rdm_type == rdm.CRAPPYNOBIS:
            rdmatrix = rdm.rdm_crappynobis(hidden)
        else:
            raise ValueError("Only implemented rdm types are mahalanobis, spearman, euclidian")
        rdmatrices.append(rdmatrix)

    # Now average over all matrices
    avg_matrix = None
    for matrix in rdmatrices:
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix = avg_matrix / num_networks
    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        np.savetxt(name+"_rdm_mat.txt", avg_matrix, delimiter="\t", fmt='%.2e')
    labels = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(avg_matrix, labels, title + " spearman rho matrix")
    if save_files:
        plt.savefig(name+'_rdm.jpeg')
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
    plt.title(title)
    if save_files:
        plt.savefig(name + '_mds')
    plt.clf()
    return avg_matrix, hidden_activations


def trainACC(model, noise=0., iterations=5000, l1reg=0.0, l2reg= 0.0, algorithm=optimizers.SGD,
          size_hidden=15, learning_rate=None, initial_context=ZEROS):
    num_episodes = iterations
    if learning_rate is not None:  # Else keep the model's learning rate
        model.learning_rate = learning_rate
    model.L1_regularization = l1reg
    model.L2_regularization = l2reg

    rng_avg_oloss, rng_avg_ploss, rng_avg_closs, rng_avg_l2loss = 0., 0., 0., 0.
    rng_avg_actions, rng_avg_sequence = 0., 0.

    for episode in range(num_episodes):
        model.new_episode(initial_context=initial_context)
        seqid = utils.idx_from_probabilities(pnas2018task.sequence_probabilities)

        sequence = pnas2018task.seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        # run the network

        with tf.GradientTape() as ptape, tf.GradientTape() as rtape, tf.GradientTape() as ctape:
            model.PredictionTape = ptape
            model.RewardTape = rtape
            model.ControlTape = ctape
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                if noise != 0.:
                    model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            ol, pl, cl, l2l = model.train(None, targets)

        #if episode % 2 == 0:
            # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_oloss = utils.rolling_avg(rng_avg_oloss, ol, speed)
        rng_avg_ploss = utils.rolling_avg(rng_avg_ploss, pl, speed)
        rng_avg_closs = utils.rolling_avg(rng_avg_closs, cl, speed)
        rng_avg_l2loss = utils.rolling_avg(rng_avg_l2loss, l2l, speed)

        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_sequence = utils.rolling_avg(rng_avg_sequence, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print(
                "{0}: avg loss outcome={1}, \tpred={2}, \tctrl={3}, \tl2={4}, \tactions={5}, \tfull_sequence={6}".format(
                    episode, rng_avg_oloss, rng_avg_ploss, rng_avg_closs, rng_avg_l2loss, rng_avg_actions, rng_avg_sequence))
    return model, rng_avg_sequence

def make_rdm_multipleACC(name, num_networks, title="-", save_files=True, skips=[],
                      rdm_type=rdm.SPEARMAN, noise=0., save_name=None):
    # Make one rdm for each network
    hidden_activations = []
    rdmatrices = []
    for i in range(num_networks+len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        hidden, _ = accuracy_testACC(model, name=str(i), noise=noise)
        hidden_activations.append(hidden)
        # Turn into a list of simple vectors
        for k, tensor in enumerate(hidden):
            hidden[k] = tensor.numpy().reshape(-1)

        if rdm_type == rdm.SPEARMAN:
            rdmatrix = rdm.rdm_spearman(hidden)
        elif rdm_type == rdm.MAHALANOBIS:
            rdmatrix = rdm.rdm_mahalanobis(hidden)
            #rdmatrix = analysis.rdm_noisy2_mahalanobis(hidden)
        elif rdm_type == rdm.EUCLIDIAN:
            rdmatrix = rdm.rdm_euclidian(hidden)
        elif rdm_type == rdm.CRAPPYNOBIS:
            rdmatrix = rdm.rdm_crappynobis(hidden)
        else:
            raise ValueError("Only implemented rdm types are mahalanobis, spearman, euclidian")
        rdmatrices.append(rdmatrix)

    # Now average over all matrices
    avg_matrix = None
    for matrix in rdmatrices:
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix = avg_matrix / num_networks
    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        if save_name is None:
            save_name = name
        np.savetxt(save_name+"_rdm.txt", avg_matrix, delimiter="\t", fmt='%.2e')
    labels = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(avg_matrix, labels, title + " spearman rho matrix")
    if save_files:
        plt.savefig(save_name+'_rdm')
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
    plt.title(title)
    if save_files:
        plt.savefig(save_name + '_mds')
    plt.clf()
    return avg_matrix, hidden_activations


def accuracy_testACC(model, name=None, noise=0., initial_context=ZEROS):
    hidden_activation = []
    all_choices = []
    for sequence in pnas2018task.seqs:
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        model.action_logits = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as ptape, tf.GradientTape() as rtape, tf.GradientTape() as ctape:
            model.PredictionTape = ptape
            model.RewardTape = rtape
            model.ControlTape = ctape
            model.new_episode(initial_context=initial_context)
            # Reset the previous action
            for i in range(len(targets)):
                # model.action = np.zeros((1, model.size_action), dtype=np.float32)
                if noise != 0.:
                    model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            model.h_action_collapsed.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    accuracy_totals = np.zeros((len(pnas2018task.seq1) - 1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(pnas2018task.seqs[i][1:], pnas2018task.all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1
    accuracy_totals /= 4
    if name is not None:
        print(name, accuracy_totals)
    else:
        print(accuracy_totals)
    return hidden_activation, accuracy_totals
