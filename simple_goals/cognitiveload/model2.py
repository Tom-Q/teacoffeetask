import utils
import cognitiveload.cogloadtask as task
import analysis
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neuralnet as nn

# Model 2: we're training only one network. This network is trained to perform all 3 tasks.
# To achieve this, when training for bev or ari we train on the same time-steps (=with a blank timestep in between)

def ff_ari(nnet, seq_ari):
    zeros = np.zeros_like(utils.str_to_onehot(seq_ari[0], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[0], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[1], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[2], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[3], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[4], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[5], task.symbols))

def make_targets_ari(seq_ari):
    targets = []
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_ari[6], task.symbols)))
    return targets

def ff_bev(nnet, seq_bev):
    zeros = np.zeros_like(utils.str_to_onehot(seq_bev[0], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[0], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[1], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[2], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[3], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[4], task.symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[5], task.symbols))

def make_targets_bev(seq_bev):
    targets = []
    targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.symbols)))
    return targets

def ff_all(nnet, seq_bev, seq_ari):
    nnet.feedforward(utils.str_to_onehot(seq_bev[0], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[0], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[1], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[1], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[2], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[2], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[3], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[3], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[4], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[4], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[5], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[5], task.symbols))

def make_targets_all(seq_bev, seq_ari):
    targets = []
    targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_ari[6], task.symbols)))
    return targets

ONLY_ARI = 0
ONLY_BEV = 1
BOTH = 2
def train_all(nnet, num_training_steps = 1000000):
    i=0
    avg_loss = 0.
    while i < num_training_steps:
        # Pick a random arithmetic seq:
        # and a random beverage seq
        seq_ari = random.choice(task.arithmetic_seqs)
        seq_bev = random.choice(task.beverage_seqs)
        # 1/3rd = only ari, 1/3rd = only bev, 1/3rd = combo.
        mode = np.random.randint(0, 3)

        with tf.GradientTape() as tape:
            nnet.new_episode()
            if mode == ONLY_ARI:
                targets = make_targets_ari(seq_ari)
                ff_ari(nnet, seq_ari)
            elif mode == ONLY_BEV:
                targets = make_targets_bev(seq_bev)
                ff_ari(nnet, seq_bev)
            elif mode == BOTH:
                targets = make_targets_all(seq_bev,seq_ari)
                ff_all(nnet, seq_bev, seq_ari)

            loss = nnet.train(tape, targets)
            loss = loss.numpy()[0]
            avg_loss = 0.999 * avg_loss + 0.001 * loss
            if i % 100 == 0 or i > (num_training_steps - 20):
                print('{0}, avgloss={1}'.format(i, avg_loss))
            if mode == BOTH:
                if i % 1019 == 0:
                    print(seq_ari)
                    print(np.argmax(nnet.action) - 9)
            i += 1
    nnet.new_episode() # just clear up the network history to avoid any bad surprises

def test_network_all(model):
    sequence_length = len(make_targets_all(task.beverage_seqs[0], task.arithmetic_seqs[0])) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for seq_bev in task.beverage_seqs:
        for seq_ari in task.arithmetic_seqs:
            model.new_episode()
            ff_all(model, seq_bev, seq_ari)
            context = [c.numpy().flatten() for c in model.h_context]
            hidden_activation.append(context)

            # Now also test whether the model was correct or not: compare targets with actual outputs
            targets = make_targets_all(seq_bev, seq_ari)
            outputs = [a.flatten() for a in model.h_action_wta]
            accuracy_sequence = np.zeros_like(accuracy_totals)
            sequence_fail = False
            for i in range(sequence_length):
                ta = targets[i].action_one_hot
                op = outputs[i]
                if ta is None or (op == ta[0]).all():
                    accuracy_sequence[i] += 1
                else:
                    sequence_fail = True
            if not sequence_fail:
                accuracy_fullseqs += 1
            accuracy_totals += accuracy_sequence
    accuracy_totals /= 4*4*17
    accuracy_fullseqs /= 4*4*17

    # So I've got all the networks and for each I've computed accuracy and extracted hidden layer activation.
    # Now to do some averaging: average all the ++, +-, -+, and -- sequences together. Also deal with accuracy.
    # The sequences are ordered like so: 4 bev. sequences, each has 4*17 arithmetic sequences.
    averaged_up_hidden_activation = []
    for i in range(4):
        for j in range(4):
            # add up those 17 sequences
            averaged_up_hidden_activation.append(np.zeros_like(hidden_activation[0]))
            for k in range(17):
                idx = i*4*17+j*17+k
                averaged_up_hidden_activation[-1] += hidden_activation[idx]
            averaged_up_hidden_activation[-1] /= 17

    return averaged_up_hidden_activation, accuracy_totals, accuracy_fullseqs

def test_network_ari(model):
    sequence_length = len(make_targets_ari(task.arithmetic_seqs[0])) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for seq_ari in task.arithmetic_seqs:
        model.new_episode()
        ff_ari(model, seq_ari)
        context = [c.numpy().flatten() for c in model.h_context]
        hidden_activation.append(context)

        # Now also test whether the model was correct or not: compare targets with actual outputs
        targets = make_targets_ari(seq_ari)
        outputs = [a.flatten() for a in model.h_action_wta]
        accuracy_sequence = np.zeros_like(accuracy_totals)
        sequence_fail = False
        for i in range(sequence_length):
            ta = targets[i].action_one_hot
            op = outputs[i]
            if ta is None or (op == ta[0]).all():
                accuracy_sequence[i] += 1
            else:
                sequence_fail = True
        if not sequence_fail:
            accuracy_fullseqs += 1
        accuracy_totals += accuracy_sequence
    accuracy_totals /= 4*17
    accuracy_fullseqs /= 4*17

    # So I've got all the networks and for each I've computed accuracy and extracted hidden layer activation.
    # Now to do some averaging: average all the ++, +-, -+, and -- sequences together. Also deal with accuracy.
    # The sequences are ordered like so: 4 bev. sequences, each has 4*17 arithmetic sequences.
    averaged_up_hidden_activation = []
    for i in range(4):
        # add up those 17 sequences
        averaged_up_hidden_activation.append(np.zeros_like(hidden_activation[0]))
        for j in range(17):
            idx = i*17+j
            averaged_up_hidden_activation[-1] += hidden_activation[idx]
        averaged_up_hidden_activation[-1] /= 17

    return averaged_up_hidden_activation, accuracy_totals, accuracy_fullseqs

def test_network_bev(model):
    sequence_length = len(make_targets_bev(task.beverage_seqs[0])) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for seq_bev in task.beverage_seqs:
        model.new_episode()
        ff_bev(model, seq_bev)
        context = [c.numpy().flatten() for c in model.h_context]
        hidden_activation.append(context)

        # Now also test whether the model was correct or not: compare targets with actual outputs
        targets = make_targets_bev(seq_bev)
        outputs = [a.flatten() for a in model.h_action_wta]
        accuracy_sequence = np.zeros_like(accuracy_totals)
        sequence_fail = False
        for i in range(sequence_length):
            ta = targets[i].action_one_hot
            op = outputs[i]
            if ta is None or (op == ta[0]).all():
                accuracy_sequence[i] += 1
            else:
                sequence_fail = True
        if not sequence_fail:
            accuracy_fullseqs += 1
        accuracy_totals += accuracy_sequence
    accuracy_totals /= 4
    accuracy_fullseqs /= 4

    return hidden_activation, accuracy_totals, accuracy_fullseqs

def generate_rdm_all(nnet, name, rdm_type=analysis.SPEARMAN, save_files=True, title="RDM training combined"):
    if rdm_type != analysis.SPEARMAN:
        raise Exception("not implemented")
    hidden_both, accuracy_totals_both, accuracy_fullseqs_both = test_network_all(nnet)
    hidden_ari, accuracy_totals_ari, accuracy_fullseqs_ari = test_network_ari(nnet)
    hidden_bev, accuracy_totals_bev, accuracy_fullseqs_bev = test_network_bev(nnet)
    print("Both: {0}, {1}".format(accuracy_totals_both, accuracy_fullseqs_both))
    print("Ari: {0}, {1}".format(accuracy_totals_ari, accuracy_fullseqs_ari))
    print("Bev: {0}, {1}".format(accuracy_totals_bev, accuracy_fullseqs_bev))

    hidden = utils.flatten_onelevel(hidden_bev) +\
             utils.flatten_onelevel(hidden_ari) +\
             utils.flatten_onelevel(hidden_both)
    rdmatrix = analysis.rdm_spearman(hidden)
    labels = utils.flatten_onelevel(task.label_seqs_bev) +\
             utils.flatten_onelevel(task.label_seqs_ari) +\
             utils.flatten_onelevel(task.label_seqs_all)
    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        np.savetxt(name+"_rdm_mat.txt", rdmatrix, delimiter="\t", fmt='%.2e')
    analysis.plot_rdm(rdmatrix, labels, title + " spearman rho matrix", figsize=30, fontsize=0.6)
    if save_files:
        plt.savefig(name+'_rdm.png', dpi=300, bbox_inches='tight')
    plt.clf()


def run_model2():
    # COMBINED #
    num_training_steps = 100000
    nnet = nn.ElmanGoalNet(size_hidden=15, initialization=nn.UNIFORM, size_goal1=0, size_goal2=0,
                           size_observation=len(task.symbols), size_action=len(task.symbols), learning_rate=0.01, algorithm=nn.ADAM)
    nnet.L2_regularization = 0.00001
    train_all(nnet, num_training_steps)
    utils.save_object("cogloadtasknet", nnet)
    nnet = utils.load_object("cogloadtasknet")
    generate_rdm_all(nnet, name="cogloadtasknet")
