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
        rdmatrix = analysis.rdm_euclidian(hidden)
        # save the massive rdm for debug purposes (so that I don't have to generate it all over again everytime).
        utils.save_object("rdmatrix_test", rdmatrix)
    else:
        rdmatrix = utils.load_object("rdmatrix_test")

    return process_matrix(rdmatrix, delete_blank_states)


def process_matrix(rdmatrix, delete_blank_states):
    if delete_blank_states:
        rdmatrix = delete_blanks(rdmatrix)
        #labels = utils.flatten_onelevel(task.label_seqs_bev_noblanks) + \
        #         utils.flatten_onelevel(task.label_seqs_ari) + \
        #         utils.flatten_onelevel(task.label_seqs_all_collapsed)
        labels = utils.flatten_onelevel(task.label_seqs_bev_noblanks) + \
                 utils.flatten_onelevel(task.label_seqs_ari)
        labels *= 2

    else:
        labels = utils.flatten_onelevel(task.label_seqs_bev) +\
                 utils.flatten_onelevel(task.label_seqs_ari) +\
                 utils.flatten_onelevel(task.label_seqs_all_collapsed)

    # Now do the averaging and re-arranging.
    #if collapse_rdm:
    # average the 16 combo bev first and 16 combo ari first. Must be done first row-by-row then column-by-column
    start_idx_bev_first = 6*8 if delete_blank_states else 11 * 8
    #rdmatrix = average_combined_seqs(rdmatrix, start_idx_bev_first)
    rdmatrix = merge_both_orders(rdmatrix, start_idx_bev_first)
    rdmatrix = rearrange_rdm2(rdmatrix, start_idx_bev_first)
    return rdmatrix, labels

def average_combined_seqs(rdmatrix, start_idx):
    # average the 16 combo bev first and 16 combo ari first. Must be done first row-by-row then column-by-column
    start_idx_bev_first = start_idx
    start_idx_ari_first = start_idx + 16 * 12
    rdmatrix = average_onetype_ari_sequence(rdmatrix, idx0=start_idx_ari_first, seq_size=12, num_seqs=16)
    rdmatrix = average_onetype_ari_sequence(rdmatrix, idx0=start_idx_bev_first, seq_size=12, num_seqs=16)
    return rdmatrix

def rearrange_rdm(rdmatrix, labels, start_idx):
    # re-arrange to have all the beverage together and all the arithmetic together
    # Make a list
    new_order = list(range(len(rdmatrix[0, :])))
    # Easiest to do this by hand
    swapped = [0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23, 1, 3, 5, 7, 9, 11, 12, 14, 16, 18, 20, 22]
    swapped = [val + start_idx for val in swapped]
    new_order[start_idx:] = swapped[:]
    rdmatrix = utils.reorder_matrix(rdmatrix, new_order)
    labels = utils.reorder_list(labels, new_order)
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

def average_arithmetic_sequences(rdm, distinguish_operators=True):
    # There's 4 beverage sequences in between, with total length 4 * 6
    ari_seqs_per_cat = 17  # number of sequences e.g. in the ++ category, i.e. 1 + 2 + 3 = 6.
    ari_cats = 4  # number of arithmetic categories: ++,+-, -+, --
    bev_seqs = 4  # number of beverage sequences: tea water first, etc.
    len_nomix_seq = 11  # number of steps in a non-mixed sequence (including blanks!)
    len_mixed_seq = 12  # number of steps in the total sequence

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

# The rdmatrix has two orderings, bev first then arithmetic first. This averages distances across the two orders.
def merge_both_orders(rdmatrix, start_idx):
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

    # Cut off the part that we've averaged.
    return rdmatrix[:start_idx+12*16,:start_idx+12*16]

import copy
# The rdmatrix has 4x4 = 16 different combined sequences (after merging the two orderings).
# This averages distances for all instances of each sequence, e.g. all 4 instances of coffee water first are averaged.
def rearrange_rdm2(rdmatrix, start_idx):
    rdmatrix_in = rdmatrix

    rdmatrix_in = _rearrange2_rows(start_idx, rdmatrix_in)
    rdmatrix_in = rdmatrix_in.T
    rdmatrix_in = _rearrange2_rows(start_idx, rdmatrix_in)
    # No need to undo the transpose, since the matrix is symmetric

    # Delete unneeded rdmatrix_out steps. We should have: 6*4 + 6*4 + 6*4 + 6*4 = 96
    rdmatrix_in = rdmatrix_in[0:96, 0:96]

    return rdmatrix_in

def _rearrange2_rows(start_idx, rdmatrix_in):
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

def _rearrange2_columns(start_idx, rdmatrix_in, rdmatrix_out):
    # Average sequence tea1, tea2, coffee1, coffee2
    seq_length = 12  # bev1 ari1 bev2 ... = 12 steps
    for seq_type in range(4): # four sequences = tea water first, tea water second, etc.
        for seq_combi in range(4): # for each sequence four combination: tea1 ari1, tea1 ari2, etc.
            start_idx_seq_in = start_idx + seq_type * 48 + seq_length * seq_combi
            start_idx_seq_out = start_idx + seq_type * 6
            for step in range(6):
                rdmatrix_in[:, start_idx_seq_out + step] += rdmatrix_in[:, start_idx_seq_in + step*2]
    rdmatrix_in[:, 48:72]/=4

    # Now the arithmetic parts
    #for seq_type in range(4):
    #    for seq_combi in range(4):
    #        start_idx_seq_in = start_idx + 1 + seq_combi * 48 + seq_type * 12
    #        start_idx_seq_out = start_idx + 24 + seq_type * 6
    #        for step in range(6):
    #            rdmatrix_out[start_idx_seq_out+step, :] += rdmatrix_in[start_idx_seq_in + step*2, :]
    #rdmatrix_out[72:96, :] /= 4

def new_order_combined(start_combined, len_combined):
    new_order = list(range(len_combined + start_combined))
    for i in range(start_combined, start_combined + len_combined):
        if i < start_combined + len_combined/2:
            new_order[i] = start_combined + 2*(i-start_combined)
        else:
            new_order[i] = start_combined + 1 + 2*(i - start_combined - len_combined//2)
    return new_order


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
