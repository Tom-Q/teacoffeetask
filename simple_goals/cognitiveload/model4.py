# This is like model 3 but with different goal units.

import utils
import cognitiveload.cogloadtask as task
import cognitiveload.model2 as model2
import analysis
import random
import tensorflow as tf
import numpy as np
from neural import neuralnet as nn
import rdm

RDM_PERCENT = 50
BEV_GOAL_MULTIPLIER = 1.0

class HierarchyGradientParams(object):
    def __init__(self, regstrength=0.001, regincrease="linear"):
        self.reg_strength = regstrength
        self.reg_increase = regincrease


# Model 4: with goals units.

def generate_rdm_all(nnet, name, rdm_type=rdm.EUCLIDIAN, save_files=True, title="RDM training combined",
                     from_file=False, delete_blank_states=True, collapse_rdm=True):
    if not from_file:
        hidden_both, accuracy_totals_both, accuracy_fullseqs_both, properties_both = test_network_all(nnet)
        hidden_ari, accuracy_totals_ari, accuracy_fullseqs_ari, properties_ari = test_network_ari(nnet, blanks=True)
        hidden_bev, accuracy_totals_bev, accuracy_fullseqs_bev, properties_bev = test_network_bev(nnet, blanks=True)
        print("Both: {0}, {1}".format(accuracy_totals_both, accuracy_fullseqs_both))
        print("Ari: {0}, {1}".format(accuracy_totals_ari, accuracy_fullseqs_ari))
        print("Bev: {0}, {1}".format(accuracy_totals_bev, accuracy_fullseqs_bev))

        hidden = utils.flatten_onelevel(hidden_bev) +\
                 utils.flatten_onelevel(hidden_ari) +\
                 utils.flatten_onelevel(hidden_both)

        properties = properties_bev + properties_ari + properties_both

        rdmatrix = rdm.rdm(properties, type=rdm_type, vectors=hidden)
        utils.save_object(name+"rdmat", rdmatrix)
    else:
        rdmatrix = utils.load_object(name+"rdmat")
    rdmatrix = model2.process_rdmatrix(rdmatrix, delete_blank_states)
    return rdmatrix

import copy
def generate_rdm_all_gradient(nnet, name, blanks, rdm_type=rdm.SPEARMAN, save_files=True, title="RDM training combined",
                     from_file=False, delete_blank_states=True):
    if not from_file:
        hidden_both, accuracy_totals_both, accuracy_fullseqs_both, properties_both = test_network_all(nnet)
        hidden_ari, accuracy_totals_ari, accuracy_fullseqs_ari, properties_ari = test_network_ari(nnet, blanks)
        hidden_bev, accuracy_totals_bev, accuracy_fullseqs_bev, properties_bev = test_network_bev(nnet, blanks)
        print("Both: {0}, {1}".format(accuracy_totals_both, accuracy_fullseqs_both))
        print("Ari: {0}, {1}".format(accuracy_totals_ari, accuracy_fullseqs_ari))
        print("Bev: {0}, {1}".format(accuracy_totals_bev, accuracy_fullseqs_bev))

        properties = properties_bev + properties_ari + properties_both
        hidden = utils.flatten_onelevel(hidden_bev) +\
                 utils.flatten_onelevel(hidden_ari) +\
                 utils.flatten_onelevel(hidden_both)

        hidden_left = []
        hidden_right = []
        for vector in hidden:
            hidden_left.append(vector[:RDM_PERCENT*len(vector)//100])
            hidden_right.append(vector[(100-RDM_PERCENT)*len(vector)//100:])
        properties_right = copy.deepcopy(properties)
        # save the massive rdm for debug purposes (so that I don't have to generate it all over again everytime).
        rdmatrix_left = rdm.rdm(properties, type=rdm_type, vectors=hidden_left)
        rdmatrix_right = rdm.rdm(properties_right, type=rdm_type, vectors=hidden_right)

        utils.save_object(name+"rdmatright", rdmatrix_right)
        utils.save_object(name+"rdmatleft", rdmatrix_left)
    else:
        rdmatrix_left = utils.load_object(name+"rdmatleft")
        rdmatrix_right = utils.load_object(name+"rdmatright")

    #rdmatrix_left.save(filename="unprocessed_left", dpi=100, fontsize=2, figsize=40)
    #rdmatrix_right.save(filename="unprocessed_right", dpi=100, fontsize=2, figsize=40)
    rdmatrix_left = model2.process_rdmatrix(rdmatrix_left, delete_blank_states)
    rdmatrix_right = model2.process_rdmatrix(rdmatrix_right, delete_blank_states)
    return rdmatrix_left, rdmatrix_right


def apply_extra_loss(network, hrp):
    if hrp is None:
        return 0.

    # New code.
    # Regularization in the hidden layer weights
    # Recurrent hidden to hidden connections
    # print("recurrent")
    model = network
    extra_loss = 0.

    # Recurrent
    extra_loss += utils.weight_regularization_calculator(model.hidden_layer.layer.w,
                                                         [0, model.size_hidden], [0, model.size_hidden],
                                                         hrp.reg_strength, reg_type="recurrent", reg_increase=hrp.reg_increase)

    # Prev action to hidden
    # print("action-->hidden")
    extra_loss += utils.weight_regularization_calculator(model.hidden_layer.layer.w,
                                                         [model.size_hidden + model.size_observation,
                                                          model.size_hidden + model.size_observation + model.size_action],
                                                         [0, model.size_hidden],
                                                         hrp.reg_strength, reg_type="input_left",
                                                         reg_increase=hrp.reg_increase)

    # middle=0.25)
    # Prev goal to hidden
    # print("goal-->hidden")
    extra_loss += utils.weight_regularization_calculator(model.hidden_layer.layer.w,
                                                         [model.size_hidden + model.size_observation + model.size_action,
                                                          model.size_hidden + model.size_observation + model.size_action + model.size_goal1],
                                                         [0, model.size_hidden],
                                                         hrp.reg_strength, reg_type="input_right", reg_increase=hrp.reg_increase)

    # Regularization in the output layers (goals and actions) weights
    # hidden to next action
    extra_loss += utils.weight_regularization_calculator(model.action_layer.w,
                                                         [0, model.size_hidden], [0, model.size_action],
                                                         hrp.reg_strength, reg_type="output_left",
                                                         reg_increase=hrp.reg_increase)

    # Hidden to next goal
    # print("hidden->goal")
    extra_loss += utils.weight_regularization_calculator(model.goal1_layer.w,
                                                         [0, model.size_hidden], [0, model.size_goal1],
                                                         hrp.reg_strength, reg_type="output_right",
                                                         reg_increase=hrp.reg_increase)

    return extra_loss


def train_all(stopping_params, nnet, hrp=None, blanks=True):
    i=0
    avg_loss = 0.
    while not stopping_params.is_time_to_stop(nnet, i):
        # Pick a random arithmetic seq:
        # and a random beverage seq
        seq_ari_id = random.randint(0, len(task.arithmetic_seqs)-1)
        seq_ari = task.arithmetic_seqs[seq_ari_id]
        seq_bev_id = random.randint(0, len(task.beverage_seqs)-1)
        seq_bev = task.beverage_seqs[seq_bev_id]
        mode = np.random.choice([task.ONLY_ARI, task.ONLY_BEV, task.BOTH + task.START_BEV, task.BOTH + task.START_ARI])

        with tf.GradientTape() as tape:
            nnet.new_episode()
            if mode == task.ONLY_ARI:
                targets = make_targets_ari(seq_ari, blanks)
                ff_ari(nnet, seq_ari, blanks)
            elif mode == task.ONLY_BEV:
                targets = make_targets_bev(seq_bev, blanks)
                ff_ari(nnet, seq_bev, blanks)
            elif mode == task.BOTH+task.START_BEV:
                targets = make_targets_all(seq_bev, seq_ari,task.START_BEV)
                ff_all(nnet, seq_bev, seq_ari, task.START_BEV)
            elif mode == task.BOTH+task.START_ARI:
                targets = make_targets_all(seq_bev, seq_ari, task.START_ARI)
                ff_all(nnet, seq_bev, seq_ari, task.START_ARI)
            extra_loss = apply_extra_loss(nnet, hrp)
            loss = nnet.train_MSE(tape, targets, extra_loss)
            loss = loss.numpy() #[0]
            avg_loss = 0.999 * avg_loss + 0.001 * loss
            if i % 100 == 0:
                _, accuracy_both, _, _ = test_network_all(nnet)
                _, accuracy_ari, _, _ = test_network_ari(nnet, blanks)
                _, accuracy_bev, _, _ = test_network_bev(nnet, blanks)
                print('{0}, avgloss={1}, accuracies=both{2}, ari{3}, bev{4}'.format(i, avg_loss, accuracy_both, accuracy_ari, accuracy_bev))
            i += 1
    print("Training complete after " + str(i) + " iterations")
    nnet.new_episode()  # just clear up the network history to avoid any bad surprises


def stop_condition(nnet, blanks, min_accuracy=1.):
    _, accuracy_both, _, _ = test_network_all(nnet)
    _, accuracy_ari, _, _ = test_network_ari(nnet, blanks)
    _, accuracy_bev, _, _ = test_network_bev(nnet, blanks)
    if not blanks:
        return np.all(accuracy_both >= [.75, .75, .5, .5, 1., 1., 1., 1., 1., 1., 1., min_accuracy]) and \
           np.all(accuracy_ari >= [1., .5, 1., 1., 1., min_accuracy]) and \
           np.all(accuracy_bev >= [.5, .5, 1., 1., 1., 1.])
    elif blanks:
        return np.all(accuracy_both >= [.5, .5, .75, .75, .75, .75, 1., 1., 1., 1., 1., min_accuracy]) and \
        np.all(accuracy_ari >= [.5, 1., 1., 1., .5, 1., 1., 1., 1., 1., min_accuracy]) and \
        np.all(accuracy_bev >= [.5, 1., .5, 1., 1., 1., 1., 1., 1., 1., 1.])


def run_model4_multiple(stopping_params, nnparams, blanks, from_file=False,
                        num_networks=1, name="model4", hrp=None,
                        type=rdm.EUCLIDIAN, skips=None):
    if from_file:
        networks = []
        if skips is None:
            skips = []
        for i in range(num_networks + len(skips)):
            if i not in skips:
                networks.append(utils.load_object(from_file, i))
    else:
        networks = []
        for i in range(num_networks):
            nnparams.size_goal1=4
            nnparams.size_action=len(task.output_symbols)
            nnparams.size_observation=len(task.input_symbols)
            nnet = nn.GoalNet(params=nnparams)
            nnet.action_layer.nonlinearity = tf.nn.sigmoid  # Sigmoid layers for goal outputs make the best sense.
            nnet.goal1_layer.nonlinearity = tf.nn.sigmoid  # Sigmoid layers for goal outputs make the best sense.
            train_all(stopping_params, nnet, hrp=hrp, blanks=blanks)
            utils.save_object(name, nnet)
            networks.append(nnet)
            # Print some stuff
            hidden_activation, accuracy_totals, accuracy_fullseqs, properties = test_network_all(nnet)
            print("network {0}: ".format(i))
            print(accuracy_totals)
            print(accuracy_fullseqs)

    # pattern of sequences, for the mds
    if hrp is None:
        final_rdm = None
        for i, net in enumerate(networks):
            print(i)
            net_rdm = generate_rdm_all(net, name=name, from_file=from_file, rdm_type=type)
            if final_rdm is None:
                final_rdm = rdm.rdm(properties=net_rdm.properties, matrix_values=net_rdm.matrix.copy())
            else:
                final_rdm.matrix += net_rdm.matrix
        final_rdm.matrix /= num_networks
        model2.set_rdm_labels(final_rdm)
        # Save it
        #(self, filename, title="", labels=None, image=True, csv=True, figsize=None, fontsize=None,
        #color=RDM_COLOR_SEQUENTIAL, dpi=300):
        final_rdm.save(name, title="RDM training combined")#, dpi=200, figsize=60, fontsize=0.5)
        #analysis.make_mds(final_rdm.matrix, name, labels=final_rdm.get_labels(), title="MDS training combined", pattern=pattern)
    else:
        #raise NotImplementedError("obsolete")
        sum_rdm_left = sum_rdm_right = None
        labels = None
        for idx, net in enumerate(networks):
            print(idx)
            rdmleft, rdmright = generate_rdm_all_gradient(net, name=name, blanks=blanks, from_file=False, delete_blank_states=True,
                                                          rdm_type=type)
            if sum_rdm_left is None:
                sum_rdm_left = rdmleft
                sum_rdm_right = rdmright
            else:
                sum_rdm_left.matrix += rdmleft.matrix
                sum_rdm_right.matrix += rdmright.matrix
        sum_rdm_left.matrix /= num_networks
        sum_rdm_right.matrix /= num_networks

        model2.set_rdm_labels(sum_rdm_left)
        model2.set_rdm_labels(sum_rdm_right)

        sum_rdm_left.save(name+"left", title="RDM training combined - left")
        sum_rdm_right.save(name+"right", title="RDM training combined - right")

        #analysis.make_mds(average_rdm_left, name+"left", labels=labels, title="MDS training combined: left (goals)", pattern=pattern)
        #analysis.make_mds(average_rdm_right, name+"right", labels=labels, title="MDS training combined: right (actions)", pattern=pattern)




def make_targets_ari(seq_ari, blanks):
    # Goal target is always: single, math.
    goal_target = ["single", "math"]
    target0 = utils.str_to_onehot(seq_ari[1], task.output_symbols)
    target1 = utils.str_to_onehot(seq_ari[2], task.output_symbols)
    target2 = utils.str_to_onehot(seq_ari[3], task.output_symbols)
    target3 = utils.str_to_onehot(seq_ari[4], task.output_symbols)
    target4 = utils.str_to_onehot(seq_ari[5], task.output_symbols)
    target5 = utils.str_to_onehot(seq_ari[6], task.output_symbols)

    targets = []
    if blanks:
        targets.append(task.TargetMod4(target0, goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(target1, goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(target2, goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(target3, goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(target4, goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(target5, goal_target))
    else:
        targets.append(task.TargetMod4(target0, goal_target))
        targets.append(task.TargetMod4(target1, goal_target))
        targets.append(task.TargetMod4(target2, goal_target))
        targets.append(task.TargetMod4(target3, goal_target))
        targets.append(task.TargetMod4(target4, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_ari[6], task.output_symbols), goal_target))
    return targets


# Blanks
def ff_ari(nnet, seq_ari, blanks):
    zeros = np.zeros_like(utils.str_to_onehot(seq_ari[0], task.input_symbols))
    for i in range(6):
        nnet.feedforward(utils.str_to_onehot(seq_ari[i], task.input_symbols))
        if i < 5 and blanks:
            nnet.feedforward(zeros)

import warnings
def ff_bev(nnet, seq_bev, blanks):
    warnings.warn("Bev goal multiplication is on")
    zeros = np.zeros_like(utils.str_to_onehot(seq_bev[0], task.input_symbols))
    for i in range(6):
        nnet.goal1[0, 1] *= BEV_GOAL_MULTIPLIER
        nnet.feedforward(utils.str_to_onehot(seq_bev[i], task.input_symbols))
        if i < 5 and blanks:
            nnet.feedforward(zeros)

def make_targets_bev(seq_bev, blanks):
    goal_target = ["single", "beverage"]
    targets = []
    if blanks:
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[1], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[2], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[3], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[4], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[5], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(None, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[6], task.output_symbols), goal_target))
    else:
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[1], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[2], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[3], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[4], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[5], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[6], task.output_symbols), goal_target))
    return targets


def ff_all(nnet, seq_bev, seq_ari, start):
    warnings.warn("Bev goal multiplication is on")
    for i in range(12//2):
        if start == task.START_BEV:
            nnet.goal1[0, 1] *= BEV_GOAL_MULTIPLIER
            nnet.feedforward(utils.str_to_onehot(seq_bev[i], task.input_symbols))
            nnet.feedforward(utils.str_to_onehot(seq_ari[i], task.input_symbols))
        elif start == task.START_ARI:
            nnet.feedforward(utils.str_to_onehot(seq_ari[i], task.input_symbols))
            nnet.goal1[0, 1] *= BEV_GOAL_MULTIPLIER
            nnet.feedforward(utils.str_to_onehot(seq_bev[i], task.input_symbols))
        else:
            raise NotImplementedError("only starts are ari and bev")

def make_targets_all(seq_bev, seq_ari, start):
    goal_target = ["combined", "beverage", "math"]
    target0 = utils.str_to_onehot(seq_ari[1], task.output_symbols)
    target1 = utils.str_to_onehot(seq_ari[2], task.output_symbols)
    target2 = utils.str_to_onehot(seq_ari[3], task.output_symbols)
    target3 = utils.str_to_onehot(seq_ari[4], task.output_symbols)
    target4 = utils.str_to_onehot(seq_ari[5], task.output_symbols)
    target5 = utils.str_to_onehot(seq_ari[6], task.output_symbols)

    targets = []
    if start == task.START_BEV:
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[1], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target0, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[2], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target1, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[3], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target2, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[4], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target3, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[5], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target4, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[6], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target5, goal_target))
    elif start == task.START_ARI:
        targets.append(task.TargetMod4(target0, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[1], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target1, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[2], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target2, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[3], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target3, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[4], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target4, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[5], task.output_symbols), goal_target))
        targets.append(task.TargetMod4(target5, goal_target))
        targets.append(task.TargetMod4(utils.str_to_onehot(seq_bev[6], task.output_symbols), goal_target))
    return targets


def test_network_all(model):
    sequence_length = len(make_targets_all(task.beverage_seqs[0], task.arithmetic_seqs[0], start=task.START_BEV)) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    properties = []

    for start in task.STARTS:
        for seq_bev_id, seq_bev in enumerate(task.beverage_seqs):
            for seq_ari_id, seq_ari in enumerate(task.arithmetic_seqs):
                model.new_episode()
                ff_all(model, seq_bev, seq_ari, start)
                context = [c.numpy().flatten() for c in model.h_context]
                hidden_activation.append(context)

                # Now also test whether the model was correct or not: compare targets with actual outputs
                targets = make_targets_all(seq_bev, seq_ari, start)
                outputs = [a.flatten() for a in model.h_action_collapsed]
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
                # Record properties
                for timestep in range(sequence_length):
                    if start == task.START_ARI:
                        seq1_type = "ari" if timestep % 2 == 0 else "bev"
                    else:
                        seq1_type = "bev" if timestep % 2 == 0 else "ari"
                    p = {}
                    p['start_seq'] = "ari" if start == task.START_ARI else "bev"
                    p['interleaved'] = "yes"
                    p['timestep'] = str(timestep)
                    p['timestep_seq1'] = str(timestep // 2)
                    p['seq1_type'] = seq1_type
                    op1 = '+' if seq_ari_id < 2 else '-'
                    op2 = '+' if seq_ari_id % 2 == 0 else '-'
                    tc = 'c' if seq_bev_id < 2 else 't'
                    wf = '1' if seq_bev_id % 2 else '2'
                    if seq1_type == "ari":
                        p['seq2_type'] = "bev"
                        p['seq1_ari_op1'] = op1
                        p['seq1_ari_op2'] = op2
                        p['seq1_bev_tc'] = None
                        p['seq1_bev_wf'] = None
                        p['seq2_ari_op1'] = None
                        p['seq2_ari_op2'] = None
                        p['seq2_bev_tc'] = tc
                        p['seq2_bev_wf'] = wf
                        p['target'] = task.arithmetic_seqs[seq_ari_id][timestep // 2 + 1]
                        p['input'] = task.arithmetic_seqs[seq_ari_id][timestep // 2]
                    else: # seq1_type =="bev"
                        p['seq2_type'] = "ari"
                        p['seq1_ari_op1'] = None
                        p['seq1_ari_op2'] = None
                        p['seq1_bev_tc'] = tc
                        p['seq1_bev_wf'] = wf
                        p['seq2_ari_op1'] = op1
                        p['seq2_ari_op2'] = op2
                        p['seq2_bev_tc'] = None
                        p['seq2_bev_wf'] = None
                        p['target'] = task.beverage_seqs[seq_bev_id][timestep // 2 + 1]
                        p['input'] = task.beverage_seqs[seq_bev_id][timestep // 2 + 1]
                    p['blank'] = "no"  # no blanks
                    properties.append(p)

    accuracy_totals /= len(task.arithmetic_seqs) * len(task.beverage_seqs) * len(task.STARTS)
    accuracy_fullseqs /= len(task.arithmetic_seqs) * len(task.beverage_seqs) * len(task.STARTS)
    return hidden_activation, accuracy_totals, accuracy_fullseqs, properties


def test_network_ari(model, blanks):
    sequence_length = len(make_targets_ari(task.arithmetic_seqs[0], blanks))  # wasteful but works
    hidden_activation = []
    properties = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for seq_ari_id, seq_ari in enumerate(task.arithmetic_seqs):
        model.new_episode()
        ff_ari(model, seq_ari, blanks)
        context = [c.numpy().flatten() for c in model.h_context]
        hidden_activation.append(context)

        # Now also test whether the model was correct or not: compare targets with actual outputs
        targets = make_targets_ari(seq_ari, blanks)
        outputs = [a.flatten() for a in model.h_action_collapsed]
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
        for timestep in range(sequence_length):
            p = {}
            p['start_seq'] = "ari"
            p['interleaved'] = "no"
            p['timestep'] = str(timestep)
            p['timestep_seq1'] = str(timestep // 2)
            p['seq1_type'] = "ari"
            p['seq2_type'] = None
            p['seq1_ari_op1'] = '+' if seq_ari_id < 2 else '-'
            p['seq1_ari_op2'] = '+' if seq_ari_id % 2 == 0 else '-'
            p['seq1_bev_tc'] = None
            p['seq1_bev_wf'] = None
            p['seq2_ari_op1'] = None
            p['seq2_ari_op2'] = None
            p['seq2_bev_tc'] = None
            p['seq2_bev_wf'] = None
            p['target'] = task.arithmetic_seqs[seq_ari_id][timestep // 2 + 1]
            p['input'] = task.arithmetic_seqs[seq_ari_id][timestep // 2]
            p['blank'] = "no" if timestep in [0, 2, 4, 6, 8, 10] else "yes"
            properties.append(p)

    accuracy_totals /= 4
    accuracy_fullseqs /= 4
    return hidden_activation, accuracy_totals, accuracy_fullseqs, properties


def test_network_bev(model, blanks):
    sequence_length = len(make_targets_bev(task.beverage_seqs[0], blanks)) # wasteful but works
    hidden_activation = []
    properties = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for seq_bev_id, seq_bev in enumerate(task.beverage_seqs):
        model.new_episode()
        ff_bev(model, seq_bev, blanks)
        context = [c.numpy().flatten() for c in model.h_context]
        hidden_activation.append(context)

        # Now also test whether the model was correct or not: compare targets with actual outputs
        targets = make_targets_bev(seq_bev, blanks)
        outputs = [a.flatten() for a in model.h_action_collapsed]
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
        for timestep in range(sequence_length):
            p = {}
            p['start_seq'] = "bev"
            p['interleaved'] = "no"
            p['timestep'] = str(timestep)
            p['timestep_seq1'] = str(timestep // 2)
            p['seq1_type'] = "bev"
            p['seq2_type'] = None
            p['seq1_ari_op1'] = None
            p['seq1_ari_op2'] = None
            p['seq1_bev_tc'] = 'c' if seq_bev_id < 2 else 't'
            p['seq1_bev_wf'] = '1' if seq_bev_id % 2 == 0 else '2'
            p['seq2_ari_op1'] = None
            p['seq2_ari_op2'] = None
            p['seq2_bev_tc'] = None
            p['seq2_bev_wf'] = None
            p['target'] = task.beverage_seqs[seq_bev_id][timestep // 2 + 1]
            p['input'] = task.beverage_seqs[seq_bev_id][timestep // 2]
            p['blank'] = "no" if timestep in [0, 2, 4, 6, 8, 10] else "yes"
            properties.append(p)

    accuracy_totals /= 4
    accuracy_fullseqs /= 4

    return hidden_activation, accuracy_totals, accuracy_fullseqs, properties