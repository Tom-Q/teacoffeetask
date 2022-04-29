# This is extremely similar to PNAS. This is mostly copy-pasted code with minor changes; not worth the effort to
# factorize that code!
from pnas import pnas2018, pnashierarchy
import neuralnet as nn
import utils
from reward_task import rewardtask
import numpy as np
import tensorflow as tf
import scripts
import analysis
import matplotlib.pyplot as plt


def train_multiple(number, filename, from_file=False):
    # train the models
    if not from_file:
        for i in range(number):
            print(i)
            model, _ = train()
            accuracy_test(model)
            utils.save_object(filename, model)

    # make the rdms
    make_rdm_multiple(filename, number)

def train(model=None, noise=0., iterations=5000, l1reg=0.0, l2reg= 0.0, algorithm=nn.SGD,
          size_hidden=15, learning_rate=None, loss_type='cross_entropy',
          initial_context=pnas2018.ZEROS):
    if model is None:
        model = nn.ElmanGoalNet(size_hidden=size_hidden, size_observation=len(rewardtask.all_inputs),
                                size_action=len(rewardtask.all_outputs), size_goal1=0, size_goal2=0,
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
        seqid = utils.idx_from_probabilities(rewardtask.sequence_probabilities)

        sequence = rewardtask.seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], rewardtask.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], rewardtask.all_outputs)
        # run the network
        with tf.GradientTape(persistent=True) as tape:
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            #if episode % 2 == 0:
                # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            if loss_type==pnas2018.MSE:
                loss, _ = model.train_MSE(targets, None, None, tape)
            elif loss_type==pnas2018.CROSS_ENTROPY:
                loss, _ = model.train_obsolete(targets, None, None, tape)
            else:
                loss, _ = model.train(tape, targets)
        del tape

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


def make_rdm_multiple(name, num_networks, with_goals=False, title="-", save_files=True, skips=[],
                      rdm_type=analysis.SPEARMAN, noise=0., save_name=None):
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

        if rdm_type == analysis.SPEARMAN:
            rdmatrix = analysis.rdm_spearman(hidden)
        elif rdm_type == analysis.MAHALANOBIS:
            rdmatrix = analysis.rdm_mahalanobis(hidden)
            #rdmatrix = analysis.rdm_noisy2_mahalanobis(hidden)
        elif rdm_type == analysis.EUCLIDIAN:
            rdmatrix = analysis.rdm_euclidian(hidden)
        elif rdm_type ==analysis.CRAPPYNOBIS:
            rdmatrix = analysis.rdm_crappynobis(hidden)
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
    for i, sequence in enumerate(rewardtask.seqs):
        for action in sequence[1:]:
            labels.append(str(i)+'_'+action)
    analysis.plot_rdm(avg_matrix, labels, title + " spearman rho matrix")
    if save_files:
        plt.savefig(save_name+'_rdm')
    plt.clf()

    mdsy = analysis.mds(avg_matrix)
    for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
        analysis.plot_mds_points(mdsy[rewardtask.length * i:rewardtask.length * i + rewardtask.length],
                                 range(rewardtask.length),
                                 labels=labels[rewardtask.length * i:rewardtask.length * i + rewardtask.length],
                                 style=style)
    plt.title(title)
    if save_files:
        plt.savefig(save_name + '_mds')
    plt.clf()
    return avg_matrix, hidden_activations



def accuracy_test(model, name=None, noise=0., initial_context=pnas2018.ZEROS):
    hidden_activation = []
    all_choices = []
    for sequence in rewardtask.seqs:
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], rewardtask.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], rewardtask.all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            model.new_episode(initial_context=initial_context)
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_wta).reshape((-1, len(targets[0])))
            model.h_action_wta.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    accuracy_totals = np.zeros((len(rewardtask.seq1) - 1))
    for i in range(len(all_choices)):
        targets = utils.liststr_to_onehot(rewardtask.seqs[i][1:], rewardtask.all_outputs)
        for j in range(len(targets)):
            if (all_choices[i][0][j] == targets[j]).all():
                accuracy_totals[j] += 1
    accuracy_totals /= 4
    if name is not None:
        print(name, accuracy_totals)
    else:
        print(accuracy_totals)
    return hidden_activation, accuracy_totals