import utils
import cognitiveload.cogloadtask as task
import analysis
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neuralnet as nn
import copy

# No blansk approach

def ff_ari_noblanks(nnet, seq_ari):
    nnet.feedforward(utils.str_to_onehot(seq_ari[0], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[1], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[2], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[3], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[4], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[5], task.input_symbols))

def make_targets_ari_noblanks(seq_ari):
    targets = []
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_ari[6], task.output_symbols)))
    return targets

def ff_bev_noblanks(nnet, seq_bev):
    nnet.feedforward(utils.str_to_onehot(seq_bev[0], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[1], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[2], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[3], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[4], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[5], task.input_symbols))

def make_targets_bev_noblanks(seq_bev):
    targets = []
    targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.output_symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.output_symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.output_symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.output_symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.output_symbols)))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.output_symbols)))
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
    zeros = np.zeros_like(utils.str_to_onehot(seq_ari[0], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_ari[0], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[1], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[2], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[3], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[4], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_ari[5], task.input_symbols))

def make_targets_ari(seq_ari):
    target0 = utils.str_to_onehot(seq_ari[1], task.output_symbols)
    target1 = utils.str_to_onehot(seq_ari[2], task.output_symbols)
    target2 = utils.str_to_onehot(seq_ari[3], task.output_symbols)
    target3 = utils.str_to_onehot(seq_ari[4], task.output_symbols)
    target4 = utils.str_to_onehot(seq_ari[5], task.output_symbols)
    target5 = utils.str_to_onehot(seq_ari[6], task.output_symbols)

    targets = []
    targets.append(task.Target(target0))
    targets.append(task.Target(None))
    targets.append(task.Target(target1))
    targets.append(task.Target(None))
    targets.append(task.Target(target2))
    targets.append(task.Target(None))
    targets.append(task.Target(target3))
    targets.append(task.Target(None))
    targets.append(task.Target(target4))
    targets.append(task.Target(None))
    targets.append(task.Target(target5))
    return targets


def ff_bev(nnet, seq_bev):
    zeros = np.zeros_like(utils.str_to_onehot(seq_bev[0], task.input_symbols))
    nnet.feedforward(utils.str_to_onehot(seq_bev[0], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[1], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[2], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[3], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[4], task.input_symbols))
    nnet.feedforward(zeros)
    nnet.feedforward(utils.str_to_onehot(seq_bev[5], task.input_symbols))


def make_targets_bev(seq_bev):
    targets = []
    targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.output_symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.output_symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.output_symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.output_symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.output_symbols)))
    targets.append(task.Target(None))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.output_symbols)))
    return targets


def ff_all(nnet, seq_bev, seq_ari, start):
    if start == task.START_BEV:
        nnet.feedforward(utils.str_to_onehot(seq_bev[0], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[0], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[1], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[1], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[2], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[2], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[3], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[3], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[4], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[4], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[5], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[5], task.input_symbols))
    elif start == task.START_ARI:
        nnet.feedforward(utils.str_to_onehot(seq_ari[0], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[0], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[1], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[1], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[2], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[2], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[3], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[3], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[4], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[4], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[5], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[5], task.input_symbols))

def make_targets_all(seq_bev, seq_ari, start):
    target0 = utils.str_to_onehot(seq_ari[1], task.output_symbols)
    target1 = utils.str_to_onehot(seq_ari[2], task.output_symbols)
    target2 = utils.str_to_onehot(seq_ari[3], task.output_symbols)
    target3 = utils.str_to_onehot(seq_ari[4], task.output_symbols)
    target4 = utils.str_to_onehot(seq_ari[5], task.output_symbols)
    target5 = utils.str_to_onehot(seq_ari[6], task.output_symbols)

    targets = []
    if start == task.START_BEV:
        targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.output_symbols)))
        targets.append(task.Target(target0))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.output_symbols)))
        targets.append(task.Target(target1))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.output_symbols)))
        targets.append(task.Target(target2))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.output_symbols)))
        targets.append(task.Target(target3))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.output_symbols)))
        targets.append(task.Target(target4))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.output_symbols)))
        targets.append(task.Target(target5))
    elif start == task.START_ARI:
        targets.append(task.Target(target0))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.output_symbols)))
        targets.append(task.Target(target1))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.output_symbols)))
        targets.append(task.Target(target2))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.output_symbols)))
        targets.append(task.Target(target3))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.output_symbols)))
        targets.append(task.Target(target4))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.output_symbols)))
        targets.append(task.Target(target5))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.output_symbols)))
    return targets


def train_all(stopping_params, nnet, blanks=True): #nnet, num_training_steps = 1000000):
    i=0
    avg_loss = 0.
    while not stopping_params.is_time_to_stop(nnet, i): #i < num_training_steps:
        # Pick a random arithmetic seq
        # and a random beverage seq
        seq_ari = random.choice(task.arithmetic_seqs_easy)
        seq_bev = random.choice(task.beverage_seqs)

        mode = np.random.choice([task.ONLY_ARI, task.ONLY_BEV, task.BOTH + task.START_BEV, task.BOTH + task.START_ARI])

        with tf.GradientTape() as tape:
            nnet.new_episode()
            if mode == task.ONLY_ARI:
                targets = make_targets_ari(seq_ari)
                ff_ari(nnet, seq_ari)
            elif mode == task.ONLY_BEV:
                targets = make_targets_bev(seq_bev)
                ff_ari(nnet, seq_bev)
            elif mode == task.BOTH+task.START_BEV:
                targets = make_targets_all(seq_bev,seq_ari, task.START_BEV)
                ff_all(nnet, seq_bev, seq_ari, task.START_BEV)
            elif mode == task.BOTH+task.START_ARI:
                targets = make_targets_all(seq_bev,seq_ari, task.START_ARI)
                ff_all(nnet, seq_bev, seq_ari, task.START_ARI)

            loss = nnet.train(tape, targets)
            loss = loss.numpy()[0]
            avg_loss = 0.999 * avg_loss + 0.001 * loss
            if i % 1000 == 0: # or i > (num_training_steps - 20):
                print('{0}, avgloss={1}'.format(i, avg_loss))
            i += 1
    nnet.new_episode() # just clear up the network history to avoid any bad surprises


def test_network_all(model):
    sequence_length = len(make_targets_all(task.beverage_seqs[0], task.arithmetic_seqs[0], task.START_BEV)) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for start in task.STARTS:
        for seq_bev in task.beverage_seqs:
            for seq_ari in task.arithmetic_seqs:
                model.new_episode()
                ff_all(model, seq_bev, seq_ari, start)
                context = [c.numpy().flatten() for c in model.h_context]
                hidden_activation.append(context)

                # Now also test whether the model was correct or not: compare targets with actual outputs
                targets = make_targets_all(seq_bev, seq_ari, start)
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

def generate_rdm_all(nnet, name, rdm_type=analysis.EUCLIDIAN, save_files=True, title="RDM training combined",
                     from_file=False, delete_blank_states=True):
    # There's 3 ways to go about this.
    # 1. AVERAGE THE ACTIVATIONS.
    # --> First we average activations for a given thing, then we compute distance across those averaged activations.
    # For instance, activations for tea water first math1 beverage first step 1 is averaged with
    # activations for tea water first math2 math first step 1.
    # 2. AVERAGE THE DISTANCES.
    # The RDM is generated for all steps separately, then distances are averaged. For instance, tea water first math1
    # beverage first step 1 is averaged with tea water first math2 math first step 1 to produce part of the input for
    # tea water first step 1. As a result distances on the main diagonal are not equal to zero!
    # 3. AVERAGE THE DISTANCES BUT WITHOUT THE CROSS-TALK.
    # Same as above but cross-talk is prevented, such that the diagonal is equal to zero. This is sort of like
    # controlling for some of the variations caused by ordering or different other task.

    if not from_file:

        hidden_both, accuracy_totals_both, accuracy_fullseqs_both = test_network_all(nnet)
        hidden_ari, accuracy_totals_ari, accuracy_fullseqs_ari = test_network_ari(nnet)
        hidden_bev, accuracy_totals_bev, accuracy_fullseqs_bev = test_network_bev(nnet)
        print("Both: {0}, {1}".format(accuracy_totals_both, accuracy_fullseqs_both))
        print("Ari: {0}, {1}".format(accuracy_totals_ari, accuracy_fullseqs_ari))
        print("Bev: {0}, {1}".format(accuracy_totals_bev, accuracy_fullseqs_bev))

        hidden = utils.flatten_onelevel(hidden_bev) +\
                 utils.flatten_onelevel(hidden_ari) +\
                 utils.flatten_onelevel(hidden_both)
        process_activations(hidden, task.RDM_MODE_AVERAGE_DISTANCES_SMART)

        if rdm_type == analysis.SPEARMAN:
            rdmatrix = analysis.rdm_spearman(hidden)
        elif rdm_type == analysis.EUCLIDIAN:
            rdmatrix = analysis.rdm_euclidian(hidden)
        else:
            raise Exception("not implemented")
        # save the massive rdm for debug purposes (so that I don't have to generate it all over again everytime).
        utils.save_object("rdmatrix_test", rdmatrix)
    else:
        rdmatrix = utils.load_object("rdmatrix_test")

    return process_matrix(rdmatrix, delete_blank_states, task.RDM_MODE_AVERAGE_DISTANCES_SMART)

def process_activations(activations, mode):
    if mode == task.RDM_MODE_AVERAGE_ACTIVATIONS:
        start_idx= 11 * 4 * 2  # where the combined sequences start. At this point we still
                               # have activations for blank states
        #1. Average activations for both arithmetic starts vs. beverage start
        for idx in range(start_idx, start_idx+12*4*4):
            if idx % 2 == 0:  # Beverage sequences
                activations[idx] += activations[idx + 12 * 16 + 1, :]
            else:  # arithmetic sequences
                activations[idx] += activations[idx + 12 * 16 - 1, :]

            activations[idx] /= 2.
        # Remove the averaged-out entries
        activations = activations[:start_idx+12*4*4]

        # Average activations across sequence types
        # For the next step, work on a copy
        activations_copy = copy.deepcopy(activations)
        # Average sequence tea1, tea2, coffee1, coffee2
        seq_length = 12  # bev1 ari1 bev2 ... = 12 steps
        for seq_type in range(0, 4):  # four sequences = tea water first, tea water second, etc.
            for seq_combi in range(0, 4):  # for each sequence four combination: tea1 ari1, tea1 ari2, etc.
                start_idx_seq_in = start_idx + seq_type * 48 + seq_length * seq_combi
                start_idx_seq_out = start_idx + seq_type * 6
                for step in range(6):
                    activations_copy[start_idx_seq_out + step] += activations[start_idx_seq_in + step * 2]

        # Now the arithmetic sequences
        for seq_type in range(4):
            for seq_combi in range(4):
                start_idx_seq_in = start_idx + 1 + seq_combi * 48 + seq_type * 12
                start_idx_seq_out = start_idx + 24 + seq_type * 6
                for step in range(6):
                    activations_copy[start_idx_seq_out + step] += activations[start_idx_seq_in + step * 2]

        # dont forget to divide after all these additions
        activations_copy[start_idx:] /= 4
        # Cut off the leftovers
        activations = activations_copy[:start_idx + 6*4*2]
    return activations

CROSSTALK = "crosstalk"
NOCROSSTALK = "nocrosstalk"
def process_matrix(rdmatrix, delete_blank_states, mode=task.RDM_MODE_AVERAGE_ACTIVATIONS):
    # start by deleting blank states. Note, currently this will break the rest of the processing.
    if delete_blank_states:
        rdmatrix = delete_blanks(rdmatrix)

    if mode == task.RDM_MODE_AVERAGE_ACTIVATIONS:
        pass   # nothing to do. Everything was done from the activations
    elif mode == task.RDM_MODE_AVERAGE_DISTANCES:
        # average the 16 combo bev first and 16 combo ari first. Must be done first row-by-row then column-by-column
        start_idx_bev_first = 6*8 if delete_blank_states else 11 * 8
        rdmatrix = average_orderings(rdmatrix, start_idx_bev_first)
        rdmatrix = average_combined(rdmatrix, start_idx_bev_first)
    elif mode == task.RDM_MODE_AVERAGE_DISTANCES_SMART:
        # The fancy one! The best way to do it, I THINK, is to set all unwanted values to 0, then do the average distances.
        # So we zero out all the cross terms - terms which have nothing in common.
        # etc.
        # Merge the two different orders (bev first and ari first).
        start_idx_bev_first = 6*8

        # Here too I should first delete unwanted crosstalk. How could I miss that.
        # In the bottom right it's again just a diagonal.
        mask = np.zeros_like(rdmatrix)
        mask[0:48, :] = 1.
        mask[:, 0:48] = 1.
        for i in range(32):
            start_row=48+i*12
            end_row=48+(i+1)*12
            for j in range(16):
                mask[start_row:end_row, j*48+((i%4)*12):j*48+((i%4)*12)+12] = 1.
        for i in range(9):
            mask[i*48:(i+1)*48, i*48:(i+1)*48] = 1.
        mask[48:48*5, 48*5:48*9] = 0.
        mask[48 * 5:48 * 9, 48:48 * 5] = 0.

        # Merge:
        rdmatrix = np.multiply(rdmatrix, mask)
        # Average out the two orders (bev 1st and ari 1st)
        rdmatrix = average_orderings(rdmatrix, start_idx_bev_first, crosstalk=NOCROSSTALK)
        # Re-arrange the matrix
        rdmatrix = average_combined_nocrosstalk(rdmatrix, start_idx_bev_first)
    else:
        raise NotImplementedError("only 3 modes available, check 'em")

    # RDM organization is always the same at the end of processing: beverage, math, bev. combined, math combined.
    labels = utils.flatten_onelevel(task.label_seqs_bev_noblanks) + \
             utils.flatten_onelevel(task.label_seqs_ari)
    labels *= 2
    return rdmatrix, labels

def run_model2_multiple(stopping_params, nnparams, blanks, from_file=None,
                        num_networks=1, name="model2"):
    #    from_file=None, num_networks=1):
    if from_file is not None:
        networks = utils.load_objects(from_file, num_networks)
    else:
        networks = []
        for i in range(num_networks):
            nnparams.size_goal1=0
            nnparams.size_action=len(task.output_symbols)
            nnparams.size_observation=len(task.input_symbols)
            nnet = nn.ElmanGoalNet(params=nnparams) #size_hidden=25, initialization=nn.UNIFORM, size_goal1=0, size_goal2=0,
                                   #size_observation=len(task.symbols), size_action=len(task.symbols),
                                   #learning_rate=0.001, algorithm=nn.ADAM)
            #nnet.L2_regularization = 0.00001
            train_all(stopping_params, nnet, blanks=blanks) #nnet, num_training_steps=200000)
            utils.save_object(name, nnet)
            networks.append(nnet)
            # Print some stuff
            hidden_activation, accuracy_totals, accuracy_fullseqs = test_network_all(nnet)
            print("network {0}: ")
            print(accuracy_totals)
            print(accuracy_fullseqs)
    sum_rdm = None
    labels = None
    for net in networks:
        rdm, labels = generate_rdm_all(net, name=name, from_file = False)
        if sum_rdm is None:
            sum_rdm = rdm
        else:
            sum_rdm += rdm
    average_rdm = sum_rdm/num_networks
    utils.save_rdm(average_rdm, name, labels,  title="RDM training combined")

def run_model2(from_file=False):
    if not from_file:
        # COMBINED #
        num_training_steps = 100000
        nnet = nn.ElmanGoalNet(size_hidden=15, initialization=nn.UNIFORM, size_goal1=0, size_goal2=0,
                               size_observation=len(task.input_symbols), size_action=len(task.output_symbols),
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
                               size_observation=len(task.input_symbols), size_action=len(task.output_symbols),
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

# The rdmatrix has two orderings, bev first then arithmetic first. This averages distances across the two orders.
def average_orderings(rdmatrix, start_idx, crosstalk=CROSSTALK):
    # each order has 12*16 rdm rows.
    # rows
    for idx in range(start_idx, start_idx+12*16):
        if idx % 2 == 0:  # Beverage sequences
            rdmatrix[idx, :] += rdmatrix[idx + 12 * 16 + 1, :]
        else:  # arithmetic sequences
            rdmatrix[idx, :] += rdmatrix[idx + 12 * 16 - 1, :]

        rdmatrix[idx, :] /= 2.
    # columns
    for idx in range(start_idx, start_idx+12*16):
        if idx % 2 == 0:
            rdmatrix[:, idx] += rdmatrix[:, idx + 12 * 16 + 1]
        else:
            rdmatrix[:, idx] += rdmatrix[:, idx + 12 * 16 - 1]

        rdmatrix[:, idx] /= 2.

    # in the case where the matrix was pruned to avoid crosstalk, we end up doing less averaging
    # and so we've divided out too much
    if crosstalk == NOCROSSTALK:
        rdmatrix[start_idx:, start_idx:] *= 2

    # Cut off the part that we've averaged out.
    return rdmatrix[:start_idx+12*16,:start_idx+12*16]

# The rdmatrix has 4x4 = 16 different combined sequences (after merging the two orderings).
# This averages distances for all instances of each sequence, e.g. all 4 instances of coffee water first are averaged.
def average_combined(rdmatrix, start_idx):
    rdmatrix_in = rdmatrix

    rdmatrix_in = _average_combined_rows(start_idx, rdmatrix_in)
    rdmatrix_in = rdmatrix_in.T
    rdmatrix_in = _average_combined_rows(start_idx, rdmatrix_in)
    # No need to undo the transpose, since the matrix is symmetric

    # Delete unneeded rdmatrix_out steps. We should have: 6*4 + 6*4 + 6*4 + 6*4 = 96
    rdmatrix_in = rdmatrix_in[0:96, 0:96]

    return rdmatrix_in

def _average_combined_rows(start_idx, rdmatrix_in):
    # Make a copy to avoid overtwriting useful info
    rdmatrix_out = copy.deepcopy(rdmatrix_in)

    # Average sequence tea1, tea2, coffee1, coffee2
    seq_length = 12  # bev1 ari1 bev2 ... = 12 steps
    for seq_type in range(0, 4): # four sequences = tea water first, tea water second, etc.
        for seq_combi in range(0, 4): # for each sequence four combination: tea1 ari1, tea1 ari2, etc.
            start_idx_seq_in = start_idx + seq_type * 48 + seq_length * seq_combi
            start_idx_seq_out = start_idx + seq_type * 6
            for step in range(6):
                rdmatrix_out[start_idx_seq_out + step, :] += rdmatrix_in[start_idx_seq_in + step*2, :]

    # Now the arithmetic sequences
    for seq_type in range(4):
        for seq_combi in range(4):
            start_idx_seq_in = start_idx + 1 + seq_combi * 48 + seq_type * 12
            start_idx_seq_out = start_idx + 24 + seq_type * 6
            for step in range(6):
                rdmatrix_out[start_idx_seq_out + step, :] += rdmatrix_in[start_idx_seq_in + step * 2, :]

    # dont forget to divide after all these additions
    rdmatrix_out[48:96, :] /= 4

    return rdmatrix_out


# This does an averaging of distances while avoiding taking into account any "cross talk" distances. This insures that
# the diagonal remains 0 and removes any additional distances caused by differences in secondary task
# (e.g. to obtain the distance between bev1 step1 and ari2 step 2, we don't take into account
# the difference between bev1 ari1 step 1 and bev2 ari2 step2.)
def average_combined_nocrosstalk(rdmatrix, start_idx):

    # This takes care of the simple and simplexcomplex parts
    rdmatrix_simple = copy.deepcopy(rdmatrix)
    rdmatrix_simple = _average_combined_rows(start_idx, rdmatrix_simple)
    rdmatrix_simple = rdmatrix_simple.T
    rdmatrix_simple = _average_combined_rows(start_idx, rdmatrix_simple)

    # The more complicated complex x complex part
    rdmatrix_complex = rdmatrix[48:, 48:]
    # 1. beverage x beverage
    # a) average per 48x48 square diagonal
    bevxbev = np.zeros((48,48))
    for i in range(4):
        for j in range(4):
            # Average the square diagonal 12x12
            bevxbev_square = np.zeros((12,12))
            offset_row = i*48
            offset_col = j*48
            for k in range(4):
                bevxbev_square[:,:] += rdmatrix_complex[offset_row+k*12:offset_row+(k+1)*12,
                                                        offset_col+(k*12):offset_col+(k+1)*12]
            bevxbev_square /= 4
            # Now fill in the bevxbev matrix at the corresponding place
            bevxbev[i*12:(i+1)*12, j*12:(j+1)*12] = bevxbev_square[:,:]
    # b) remove math steps. That means just removing every odd row/column.
    bevxbev = np.delete(bevxbev, range(1, 48, 2), 0) # rows
    bevxbev = np.delete(bevxbev, range(1, 48, 2), 1) # columns

    # 2. math x math
    # a) average 48x48 squares on the main diagonal
    mathxmath = np.zeros((48,48))
    for i in range(4):
        mathxmath[:, :] += rdmatrix_complex[i*48:(i+1)*48, i*48:(i+1)*48]
    mathxmath /= 4

    # b) remove beverage steps
    mathxmath = np.delete(mathxmath, range(0, 48, 2), 0)  # rows
    mathxmath = np.delete(mathxmath, range(0, 48, 2), 1)  # columns

    # 3. beverage x math
    # a. rearrange main diagonal
    bevxmath = np.zeros((48, 48))
    for i in range(4):
        for j in range(4):
            bevxmath[i*12:(i+1)*12, j*12:(j+1)*12] = rdmatrix_complex[(i*4+j)*12:(i*4+j+1)*12, (i*4+j)*12:(i*4+j+1)*12]

    # b. remove bxb, mxm, and mxb steps
    bevxmath = np.delete(bevxmath, range(1, 48, 2), 0) # rows
    bevxmath = np.delete(bevxmath, range(0, 48, 2), 1)  # columns

    # c. Transpose
    mathxbev = np.transpose(bevxmath)

    # Put all this together in a 96x96 matrix
    bevxx = np.concatenate((bevxbev, bevxmath), axis=1)
    mathxx = np.concatenate((mathxbev, mathxmath), axis=1)
    xxx = np.concatenate((bevxx, mathxx), axis=0)

    rdmatrix_simple = rdmatrix_simple[0:96, 0:96]
    rdmatrix_simple[48:, 48:] = xxx[:,:]

    return rdmatrix_simple


def stop_condition(nnet, blanks, min_accuracy=1.):
    _, accuracy_both, _ = test_network_all(nnet)
    if blanks:
        _, accuracy_ari, _ = test_network_ari(nnet)
        _, accuracy_bev, _ = test_network_bev(nnet)
    else:
        _, accuracy_ari, _ = test_network_ari_noblanks(nnet)
        _, accuracy_bev, _ = test_network_bev_noblanks(nnet)

    if not blanks:
        return np.all(accuracy_both >= [.75, .75, .5, .5, 1., 1., 1., 1., 1., 1., 1., min_accuracy]) and \
           np.all(accuracy_ari >= [1., .5, 1., 1., 1., min_accuracy]) and \
           np.all(accuracy_bev >= [.5, .5, 1., 1., 1., 1.])
    elif blanks:
        return np.all(accuracy_both >= [.75, .75, .5, .5, 1., 1., 1., 1., 1., 1., 1., min_accuracy]) and \
        np.all(accuracy_ari >= [1., .5, 1., 1., 1., 1., 1., 1., 1., 1., min_accuracy]) and \
        np.all(accuracy_bev >= [.5, 1., .5, 1., 1., 1., 1., 1., 1., 1., 1.])
