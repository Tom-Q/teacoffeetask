# This is like model 2 but with goal units.

import utils
import cognitiveload.cogloadtask as task
import cognitiveload.model2 as model2
import analysis
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neuralnet as nn

class HierarchyGradientParams(object):
    def __init__(self, regstrength=0.001, regincrease="linear"):
        self.reg_strength = regstrength
        self.reg_increase = regincrease


# Model 3: with goals units. TODO: only 2 goal units, beverage and math.

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
        utils.save_object(name+"rdmat", rdmatrix)
    else:
        rdmatrix = utils.load_object(name+"rdmat")

    if not task.FAST_RDM: # the rdm is massive and we need to average out of the fluff
        rdmatrix = model2.average_arithmetic_sequences(rdmatrix)

    if delete_blank_states:
        rdmatrix = model2.delete_blanks(rdmatrix)
        labels = utils.flatten_onelevel(task.label_seqs_bev_noblanks) + \
                 utils.flatten_onelevel(task.label_seqs_ari_noblanks) + \
                 utils.flatten_onelevel(task.label_seqs_all)
    else:
        labels = utils.flatten_onelevel(task.label_seqs_bev) +\
                 utils.flatten_onelevel(task.label_seqs_ari) +\
                 utils.flatten_onelevel(task.label_seqs_all)

    return rdmatrix, labels

def generate_rdm_all_gradient(nnet, name, rdm_type=analysis.SPEARMAN, save_files=True, title="RDM training combined",
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

        hidden_left = []
        hidden_right = []
        for vector in hidden:
            hidden_left.append(vector[:len(vector)//2])
            hidden_right.append(vector[len(vector)//2:])

        # Now cut the hidden layer in two.
        rdmatrix_left = analysis.rdm_spearman(hidden_left)
        rdmatrix_right = analysis.rdm_spearman(hidden_right)
        # save the massive rdm for debug purposes (so that I don't have to generate it all over again everytime).
        utils.save_object(name+"rdmatright", rdmatrix_right)
        utils.save_object(name+"rdmatleft", rdmatrix_left)
    else:
        rdmatrix_left = utils.load_object(name+"rdmatleft")
        rdmatrix_right = utils.load_object(name+"rdmatright")

    if not task.FAST_RDM: # the rdm is massive and we need to average out of the fluff
        rdmatrix_right = model2.average_arithmetic_sequences(rdmatrix_right)
        rdmatrix_left = model2.average_arithmetic_sequences(rdmatrix_left)

    if delete_blank_states:
        rdmatrix_left = model2.delete_blanks(rdmatrix_left)
        rdmatrix_right = model2.delete_blanks(rdmatrix_right)
        labels = utils.flatten_onelevel(task.label_seqs_bev_noblanks) + \
                 utils.flatten_onelevel(task.label_seqs_ari_noblanks) + \
                 utils.flatten_onelevel(task.label_seqs_all)
    else:
        labels = utils.flatten_onelevel(task.label_seqs_bev) +\
                 utils.flatten_onelevel(task.label_seqs_ari) +\
                 utils.flatten_onelevel(task.label_seqs_all)

    return rdmatrix_left, rdmatrix_right, labels

def apply_extra_loss(network, hrp):
    if hrp is None:
        return 0.
    cols = network.size_hidden
    num_goals = network.size_goal1
    # Regularization in the hidden layer weights
    # Recurrent hidden to hidden connections
    extra_loss = utils.weight_regularization_calculator(network.hidden_layer.w,
                                                  [0, network.size_hidden], [0, cols],
                                                  hrp.reg_strength, reg_type="recurrent", reg_increase=hrp.reg_increase)
    # Prev action to hidden
    # extra_loss += weight_regularization_calculator(model.hidden_layer.w,
    #                                               [model.size_hidden+9, model.size_hidden+9+model.size_action],
    #                                               [0, cols],
    #                                               reg_strength, reg_type="input_right", reg_increase=reg_increase)
    # Prev goal to hidden
    extra_loss += utils.weight_regularization_calculator(network.hidden_layer.w,
                                                   [network.size_hidden + 9 + network.size_action,
                                                    network.size_hidden + 9 + network.size_action + num_goals],
                                                   [0, cols],
                                                   hrp.reg_strength, reg_type="input_left", reg_increase=hrp.reg_increase)

    # SWITCHED OUTPUT LEFT AND OUTPUT RIGHT.
    # Regularization in the output layers (goals and actions) weights
    # hidden to next action
    extra_loss += utils.weight_regularization_calculator(network.action_layer.w,
                                                   [0, network.size_hidden], [0, network.size_action],
                                                   hrp.reg_strength, reg_type="output_right", reg_increase=hrp.reg_increase)
    # Hidden to next goal
    extra_loss += utils.weight_regularization_calculator(network.goal1_layer.w,
                                                   [0, network.size_hidden], [0, network.size_action],
                                                   hrp.reg_strength, reg_type="output_left", reg_increase=hrp.reg_increase)
    return extra_loss

def train_all(nnet, num_training_steps = 1000000, hrp=None):
    i=0
    avg_loss = 0.
    while i < num_training_steps:
        # Pick a random arithmetic seq:
        # and a random beverage seq
        seq_ari_id = random.randint(0, len(task.arithmetic_seqs)-1)
        seq_ari = task.arithmetic_seqs[seq_ari_id]
        seq_bev_id = random.randint(0, len(task.beverage_seqs)-1)
        seq_bev = task.beverage_seqs[seq_bev_id]
        goal_targets_bev = task.goal_target_bev[seq_bev_id]

        # 1/3rd = only ari, 1/3rd = only bev, 1/3rd = combo.
        mode = np.random.randint(0, 3)

        with tf.GradientTape() as tape:
            nnet.new_episode()
            if mode == task.ONLY_ARI:
                targets = make_targets_ari(seq_ari)
                ff_ari(nnet, seq_ari)
            elif mode == task.ONLY_BEV:
                targets = make_targets_bev(seq_bev, goal_targets_bev)
                ff_ari(nnet, seq_bev)
            elif mode == task.BOTH:
                targets = make_targets_all(seq_bev, seq_ari, goal_targets_bev)
                ff_all(nnet, seq_bev, seq_ari)

            loss = nnet.train(tape, targets, apply_extra_loss(nnet, hrp))
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

def run_model3_multiple(from_file=None, num_networks=1, name="model3", hrp=None):
    if from_file is not None:
        networks = utils.load_objects(from_file, num_networks)
    else:
        networks = []
        for i in range(num_networks):
            nnet = nn.ElmanGoalNet(size_hidden=25, initialization=nn.UNIFORM, size_goal1=3, size_goal2=0,
                                   size_observation=len(task.symbols), size_action=len(task.symbols),
                                   learning_rate=0.001, algorithm=nn.ADAM)
            nnet.L2_regularization = 0.00001
            train_all(nnet, num_training_steps=200000, hrp=hrp)
            utils.save_object(name, nnet)
            networks.append(nnet)
            # Print some stuff
            hidden_activation, accuracy_totals, accuracy_fullseqs = test_network_all(nnet)
            print("network {0}: ")
            print(accuracy_totals)
            print(accuracy_fullseqs)
    if hrp is None:
        sum_rdm = None
        labels = None
        for net in networks:
            rdm, labels = generate_rdm_all(net, name=name, from_file=False)
            if sum_rdm is None:
                sum_rdm = rdm
            else:
                sum_rdm += rdm
        average_rdm = sum_rdm/num_networks
        utils.save_rdm(average_rdm, name, labels,  title="RDM training combined")
    else:
        sum_rdm_left = sum_rdm_right = None
        labels = None
        for net in networks:
            rdmleft, rdmright, labels = generate_rdm_all_gradient(net, name=name, from_file=False)
            if sum_rdm_left is None:
                sum_rdm_left = rdmleft
                sum_rdm_right = rdmright
            else:
                sum_rdm_left += rdmleft
                sum_rdm_right += rdmright
        average_rdm_left = sum_rdm_left/num_networks
        average_rdm_right = sum_rdm_right/num_networks
        utils.save_rdm(average_rdm_left, name+"left", labels,  title="RDM training combined: left (goals)")
        utils.save_rdm(average_rdm_right, name+"right", labels,  title="RDM training combined: right (actions)")

def make_targets_ari(seq_ari):
    targets = []
    targets.append(task.Target(None, None))
    targets.append(task.Target(None, task.goal_target_ari[0]))
    targets.append(task.Target(None, None))
    targets.append(task.Target(None, task.goal_target_ari[1]))
    targets.append(task.Target(None, None))
    targets.append(task.Target(None, task.goal_target_ari[2]))
    targets.append(task.Target(None, None))
    targets.append(task.Target(None, task.goal_target_ari[3]))
    targets.append(task.Target(None, None))
    targets.append(task.Target(None, task.goal_target_ari[4]))
    targets.append(task.Target(utils.str_to_onehot(seq_ari[6], task.symbols), task.goal_target_ari[-1]))
    return targets

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

def make_targets_bev(seq_bev, goal_targets_bev):
    targets = []
    targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.symbols), None))#, goal_targets_bev[0]))
    targets.append(task.Target(None, goal_targets_bev[0]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.symbols), None))#, goal_targets_bev[1]))
    targets.append(task.Target(None, goal_targets_bev[1]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.symbols), None))#, goal_targets_bev[2]))
    targets.append(task.Target(None, goal_targets_bev[2]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.symbols), None))#, goal_targets_bev[3]))
    targets.append(task.Target(None, goal_targets_bev[3]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.symbols), None))#, goal_targets_bev[4]))
    targets.append(task.Target(None, goal_targets_bev[4]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.symbols), None))#, goal_targets_bev[5]))
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

def make_targets_all(seq_bev, seq_ari, goal_targets_bev):
    targets = []
    targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.symbols), task.goal_target_ari[0]))
    targets.append(task.Target(None, goal_targets_bev[0]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.symbols), task.goal_target_ari[1]))
    targets.append(task.Target(None, goal_targets_bev[1]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.symbols), task.goal_target_ari[2]))
    targets.append(task.Target(None, goal_targets_bev[2]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.symbols), task.goal_target_ari[3]))
    targets.append(task.Target(None, goal_targets_bev[3]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.symbols), task.goal_target_ari[4]))
    targets.append(task.Target(None, goal_targets_bev[4]))
    targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.symbols), task.goal_target_ari[5]))
    targets.append(task.Target(utils.str_to_onehot(seq_ari[6], task.symbols), goal_targets_bev[5]))
    return targets


def test_network_all(model):
    sequence_length = len(make_targets_all(task.beverage_seqs[0], task.arithmetic_seqs[0], task.goal_target_bev[0])) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for idx, seq_bev in enumerate(task.beverage_seqs):
        bev_targets = task.goal_target_bev[idx]
        for seq_ari in task.arithmetic_seqs:
            model.new_episode()
            ff_all(model, seq_bev, seq_ari)
            context = [c.numpy().flatten() for c in model.h_context]
            hidden_activation.append(context)

            # Now also test whether the model was correct or not: compare targets with actual outputs
            targets = make_targets_all(seq_bev, seq_ari, bev_targets)
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
    sequence_length = len(make_targets_bev(task.beverage_seqs[0], task.goal_target_bev[0])) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for idx, seq_bev in enumerate(task.beverage_seqs):
        bev_targets = task.goal_target_bev[idx]
        model.new_episode()
        ff_bev(model, seq_bev)
        context = [c.numpy().flatten() for c in model.h_context]
        hidden_activation.append(context)

        # Now also test whether the model was correct or not: compare targets with actual outputs
        targets = make_targets_bev(seq_bev, bev_targets)
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