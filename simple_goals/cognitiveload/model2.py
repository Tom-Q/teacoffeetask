import utils
import cognitiveload.cogloadtask as task
import analysis
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neuralnet as nn

# No blansk approach

def ff_ari_noblanks(nnet, seq_ari):
    nnet.feedforward(utils.str_to_onehot(seq_ari[0], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[1], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[2], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[3], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[4], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[5], task.symbols))

def make_targets_ari_noblanks(seq_ari):
    targets = []
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_ari[6], task.symbols)))
    return targets

def ff_bev_noblanks(nnet, seq_bev):
    nnet.feedforward(utils.str_to_onehot(seq_bev[0], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[1], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[2], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[3], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[4], task.symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[5], task.symbols))

def make_targets_bev_noblanks(seq_bev):
    targets = []
    targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.symbols)))
    return targets


def test_network_ari_noblanks(model):
    sequence_length = len(make_targets_ari_noblanks(task.arithmetic_seqs[0])) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for seq_ari in task.arithmetic_seqs:
        model.new_episode()
        ff_ari_noblanks(model, seq_ari)
        context = [c.numpy().flatten() for c in model.h_context]
        hidden_activation.append(context)

        # Now also test whether the model was correct or not: compare targets with actual outputs
        targets = make_targets_ari_noblanks(seq_ari)
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

def test_network_bev_noblanks(model):
    sequence_length = len(make_targets_bev_noblanks(task.beverage_seqs[0])) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for seq_bev in task.beverage_seqs:
        model.new_episode()
        ff_bev_noblanks(model, seq_bev)
        context = [c.numpy().flatten() for c in model.h_context]
        hidden_activation.append(context)

        # Now also test whether the model was correct or not: compare targets with actual outputs
        targets = make_targets_bev_noblanks(seq_bev)
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


def train_all_noblanks(nnet, num_training_steps = 1000000):
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
            if mode == task.ONLY_ARI:
                targets = make_targets_ari_noblanks(seq_ari)
                ff_ari_noblanks(nnet, seq_ari)
            elif mode == task.ONLY_BEV:
                targets = make_targets_bev_noblanks(seq_bev)
                ff_ari_noblanks(nnet, seq_bev)
            elif mode == task.BOTH:
                targets = make_targets_all(seq_bev,seq_ari)
                ff_all(nnet, seq_bev, seq_ari)

            loss = nnet.train(tape, targets)
            loss = loss.numpy()[0]
            avg_loss = 0.999 * avg_loss + 0.001 * loss
            if i % 100 == 0 or i > (num_training_steps - 20):
                print('{0}, avgloss={1}'.format(i, avg_loss))
            if mode == task.BOTH:
                if i % 1019 == 0:
                    print(seq_ari)
                    print(np.argmax(nnet.action) - 9)
            i += 1
    nnet.new_episode() # just clear up the network history to avoid any bad surprises

def generate_rdm_all_noblanks(nnet, name, rdm_type=analysis.SPEARMAN, save_files=True, title="RDM training combined"):
    if rdm_type != analysis.SPEARMAN:
        raise Exception("not implemented")
    hidden_both, accuracy_totals_both, accuracy_fullseqs_both = test_network_all(nnet)
    hidden_ari, accuracy_totals_ari, accuracy_fullseqs_ari = test_network_ari_noblanks(nnet)
    hidden_bev, accuracy_totals_bev, accuracy_fullseqs_bev = test_network_bev_noblanks(nnet)
    print("Both: {0}, {1}".format(accuracy_totals_both, accuracy_fullseqs_both))
    print("Ari: {0}, {1}".format(accuracy_totals_ari, accuracy_fullseqs_ari))
    print("Bev: {0}, {1}".format(accuracy_totals_bev, accuracy_fullseqs_bev))

    hidden = utils.flatten_onelevel(hidden_bev) +\
             utils.flatten_onelevel(hidden_ari) +\
             utils.flatten_onelevel(hidden_both)
    rdmatrix = analysis.rdm_spearman(hidden)
    labels = utils.flatten_onelevel(task.label_seqs_bev_noblanks) +\
             utils.flatten_onelevel(task.label_seqs_ari_noblanks) +\
             utils.flatten_onelevel(task.label_seqs_all)
    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        np.savetxt(name+"_rdm_mat.txt", rdmatrix, delimiter="\t", fmt='%.2e')
    analysis.plot_rdm(rdmatrix, labels, title + " spearman rho matrix", figsize=30, fontsize=0.6)
    if save_files:
        plt.savefig(name+'_rdm.png', dpi=300, bbox_inches='tight')
    plt.clf()


def run_model2_noblanks(from_file=False):
    # COMBINED #
    if not from_file:
        num_training_steps = 100000
        nnet = nn.ElmanGoalNet(size_hidden=15, initialization=nn.UNIFORM, size_goal1=0, size_goal2=0,
                           size_observation=len(task.symbols), size_action=len(task.symbols),
                           learning_rate=0.005, algorithm=nn.ADAM)
        nnet.L2_regularization = 0.00001
        train_all_noblanks(nnet, num_training_steps)
        utils.save_object("cogloadtasknet_noblanks", nnet)
    nnet = utils.load_object("cogloadtasknet_noblanks")
    generate_rdm_all_noblanks(nnet, name="cogloadtasknet_noblanks")


# Blanks
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
            if mode == task.ONLY_ARI:
                targets = make_targets_ari(seq_ari)
                ff_ari(nnet, seq_ari)
            elif mode == task.ONLY_BEV:
                targets = make_targets_bev(seq_bev)
                ff_ari(nnet, seq_bev)
            elif mode == task.BOTH:
                targets = make_targets_all(seq_bev,seq_ari)
                ff_all(nnet, seq_bev, seq_ari)

            loss = nnet.train(tape, targets)
            loss = loss.numpy()[0]
            avg_loss = 0.999 * avg_loss + 0.001 * loss
            if i % 100 == 0 or i > (num_training_steps - 20):
                print('{0}, avgloss={1}'.format(i, avg_loss))
            if mode == task.BOTH:
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

    if task.FAST_RDM:
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
        hidden_activation = averaged_up_hidden_activation
    return hidden_activation, accuracy_totals, accuracy_fullseqs

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

    if task.FAST_RDM:
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
        hidden_activation = averaged_up_hidden_activation
    return hidden_activation, accuracy_totals, accuracy_fullseqs

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

def generate_rdm_all(nnet, name, rdm_type=analysis.SPEARMAN, save_files=True, title="RDM training combined",
                     from_file=False, delete_blank_states=True):
    if not from_file:
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
        # save the massive rdm for debug purposes (so that I don't have to generate it all over again everytime).
        utils.save_object("rdmatrix_test", rdmatrix)
    else:
        rdmatrix = utils.load_object("rdmatrix_test")

    if not task.FAST_RDM: # the rdm is massive and we need to average out of the fluff
        rdmatrix = average_arithmetic_sequences(rdmatrix)

    if delete_blank_states:
        rdmatrix = delete_blanks(rdmatrix)
        labels = utils.flatten_onelevel(task.label_seqs_bev_noblanks) + \
                 utils.flatten_onelevel(task.label_seqs_ari_noblanks) + \
                 utils.flatten_onelevel(task.label_seqs_all)
    else:
        labels = utils.flatten_onelevel(task.label_seqs_bev) +\
                 utils.flatten_onelevel(task.label_seqs_ari) +\
                 utils.flatten_onelevel(task.label_seqs_all)
    """
    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        np.savetxt(name+"_rdm_mat.txt", rdmatrix, delimiter="\t", fmt='%.2e')
    analysis.plot_rdm(rdmatrix, labels, title + " spearman rho matrix", figsize=30, fontsize=0.6)
    if save_files:
        plt.savefig(name+'_rdm.png', dpi=300, bbox_inches='tight')
    plt.clf()
    """
    return rdmatrix, labels


def run_model2_multiple(from_file=None, num_networks=1):
    if from_file is not None:
        networks = utils.load_objects(from_file, num_networks)
    else:
        networks = []
        for i in range(num_networks):
            nnet = nn.ElmanGoalNet(size_hidden=25, initialization=nn.UNIFORM, size_goal1=0, size_goal2=0,
                                   size_observation=len(task.symbols), size_action=len(task.symbols),
                                   learning_rate=0.001, algorithm=nn.ADAM)
            nnet.L2_regularization = 0.00001
            train_all(nnet, num_training_steps=200000)
            utils.save_object("cogloadtasknet", nnet)
            networks.append(nnet)
            # Print some stuff
            hidden_activation, accuracy_totals, accuracy_fullseqs = test_network_all(nnet)
            print("network {0}: ")
            print(accuracy_totals)
            print(accuracy_fullseqs)
    sum_rdm = None
    labels = None
    for net in networks:
        rdm, labels = generate_rdm_all(net, name="cogloadtasknet", from_file = False, )
        if sum_rdm is None:
            sum_rdm = rdm
        else:
            sum_rdm += rdm
    average_rdm = sum_rdm/num_networks
    name = "cogloadtasknet"
    utils.save_rdm(average_rdm, name, labels,  title="RDM training combined")

def run_model2(from_file=False):
    if not from_file:
        # COMBINED #
        num_training_steps = 100000
        nnet = nn.ElmanGoalNet(size_hidden=15, initialization=nn.UNIFORM, size_goal1=0, size_goal2=0,
                               size_observation=len(task.symbols), size_action=len(task.symbols),
                               learning_rate=0.01, algorithm=nn.ADAM)
        nnet.L2_regularization = 0.00001
        train_all(nnet, num_training_steps)
        utils.save_object("cogloadtasknet", nnet)
    nnet = utils.load_object("cogloadtasknet")
    generate_rdm_all(nnet, name="cogloadtasknet", from_file=True)


def run_model2_deleteblanks(from_file=False):
    if not from_file:
        # COMBINED #
        num_training_steps = 100000
        nnet = nn.ElmanGoalNet(size_hidden=15, initialization=nn.UNIFORM, size_goal1=0, size_goal2=0,
                               size_observation=len(task.symbols), size_action=len(task.symbols),
                               learning_rate=0.01, algorithm=nn.ADAM)
        nnet.L2_regularization = 0.00001
        train_all(nnet, num_training_steps)
        utils.save_object("cogloadtasknet_deleteblanks", nnet)
    nnet = utils.load_object("cogloadtasknet")
    generate_rdm_all(nnet, name="cogloadtasknet_deleteblanks", delete_blank_states=True)


def delete_blanks(rdm):
    # List of indexes to delete. basically the pattern is : data - blank - data - blank - data - data - blank - etc.
    # Initially coded that, but I think writing the whole sequence is actually less bug prone.
    to_delete = [1,  3,  5,  7,  9,  # beverage task
                 12, 14, 16, 18, 20,
                 23, 25, 27, 29, 31,
                 34, 36, 38, 40, 42,
                 45, 47, 49, 51, 53,  # math task
                 56, 58, 60, 62, 64,
                 67, 69, 71, 73, 75,
                 78, 80, 82, 84, 86]
    rdm = np.delete(rdm, to_delete, 0)
    rdm = np.delete(rdm, to_delete, 1)
    return rdm

def average_arithmetic_sequences(rdm, distinguish_operators=True):
    # There's 4 beverage sequences in between, with total length 4 * 6
    ari_seqs_per_cat = 17  # number of sequences e.g. in the ++ category, i.e. 1 + 2 + 3 = 6.
    ari_cats = 4  # number of arithmetic categories: ++,+-, -+, --
    bev_seqs = 4  # number of beverage sequences: tea water first, etc.
    len_nomix_seq = 11  # number of steps in a non-mixed sequence (including blanks!)
    len_mixed_seq = 12  #

    # These are the lengths BEFORE all these averages and other corrections are applied; on the RAW rdm
    len_ari = len_nomix_seq * ari_cats * ari_seqs_per_cat  #
    len_bev = len_nomix_seq * bev_seqs  # 44 (=24+ 20 blank)

    # must start from the end of the RDM, otherwise the indexes will change all the time.
    if distinguish_operators:
        # Mixed sequences.
        for i in range(3, -1, -1):   # 3,2,1, 0: corresponding to the beverage sequences starting from the end
            for j in range(3, -1, -1): # corresponding to the arithmetic sequences starting from the end
                rdm = average_onetype_ari_sequence(rdm,
                                                   len_bev + len_ari +  # the non-mixed seqs
                                                   len_mixed_seq * ari_cats * ari_seqs_per_cat * i +  # total length of fully processed mixed sequences for previous beverages
                                                   len_mixed_seq * ari_seqs_per_cat * j,  # total length of fully processed mixed sequences for the current beverage
                                                   len_mixed_seq,
                                                   ari_seqs_per_cat)

        # Num only sequences: indexes 0 to 12x17x4
        # Clean up all sequences one by one, by averaging arithmetic sequence distances
        rdm = average_onetype_ari_sequence(rdm, len_bev + 11 * 17 * 3, 11, 17) # --
        rdm = average_onetype_ari_sequence(rdm, len_bev + 11 * 17 * 2, 11, 17) # -+
        rdm = average_onetype_ari_sequence(rdm, len_bev + 11 * 17, 11, 17) # +-
        rdm = average_onetype_ari_sequence(rdm, len_bev, 11, 17) # ++

    else: # average all sequences in the same way.
        for i in range(3, -1, -1):
            rdm = average_onetype_ari_sequence(rdm,
                                               len_bev + len_ari +  # the non-mixed seqs
                                               len_mixed_seq * ari_cats * ari_seqs_per_cat * i,  # total length of fully processed mixed sequences for previous beverages
                                               len_mixed_seq,
                                               ari_seqs_per_cat * ari_cats)

        # Num only sequences: indexes 0 to 12x17x4
        # Clean up all sequences one by one, by averaging arithmetic sequence distances
        rdm = average_onetype_ari_sequence(rdm, len_bev + 11 * 17 * 3, 11, 17) # --
        rdm = average_onetype_ari_sequence(rdm, len_bev + 11 * 17 * 2, 11, 17) # -+
        rdm = average_onetype_ari_sequence(rdm, len_bev + 11 * 17, 11, 17) # +-
        rdm = average_onetype_ari_sequence(rdm, len_bev, 11, 17) # ++
    return rdm


# This cleans up the massive RDM by averaging the rows and columns corresponding to a single category
# (e.g. all the "++" sequences).
# This must be called once for each sequence that needs to be cleaned up, meaning: ++, +-, -+, --
def average_onetype_ari_sequence(rdm, idx0, seq_size, num_seqs):
    """
    @param rdm: The RDM
    @param idx0: index of the first element of the first sequence to clean up
    @param seq_size: length of the sequence
    @param num_seqs: number of consecutive sequences
    @return: an RDM that has been cleaned up
    """
    # Add up columns
    for i in range(1, num_seqs):
        rdm[idx0:idx0+seq_size, :] += rdm[idx0+i*seq_size: idx0+(i+1)*seq_size, :]
    # Add up rows
    for i in range(1, num_seqs):
        rdm[:, idx0:idx0+seq_size] += rdm[:, idx0+i*seq_size: idx0+(i+1)*seq_size]
    # Delete the excess rows and columns. All but 1 sequence.
    rdm = np.delete(rdm, range(idx0+seq_size, idx0+num_seqs*seq_size), 0)
    rdm = np.delete(rdm, range(idx0+seq_size, idx0+num_seqs*seq_size), 1)
    # Do the averaging on rows and columns
    rdm[idx0:idx0+seq_size, :] /= num_seqs
    rdm[:, idx0:idx0+seq_size] /= num_seqs
    return rdm
