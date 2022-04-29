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


# Model 3: with goals units.

def generate_rdm_all(nnet, name, rdm_type=analysis.SPEARMAN, save_files=True, title="RDM training combined",
                     from_file=False, delete_blank_states=True, collapse_rdm=True):
    if not from_file:
        if rdm_type != analysis.SPEARMAN:
            raise Exception("not implemented")
        hidden_both, accuracy_totals_both, accuracy_fullseqs_both = test_network_all(nnet)
        hidden_ari, accuracy_totals_ari, accuracy_fullseqs_ari = test_network_ari(nnet, blanks=True)
        hidden_bev, accuracy_totals_bev, accuracy_fullseqs_bev = test_network_bev(nnet, blanks=True)
        print("Both: {0}, {1}".format(accuracy_totals_both, accuracy_fullseqs_both))
        print("Ari: {0}, {1}".format(accuracy_totals_ari, accuracy_fullseqs_ari))
        print("Bev: {0}, {1}".format(accuracy_totals_bev, accuracy_fullseqs_bev))

        hidden = utils.flatten_onelevel(hidden_bev) +\
                 utils.flatten_onelevel(hidden_ari) +\
                 utils.flatten_onelevel(hidden_both)
        rdmatrix = analysis.rdm_euclidian(hidden)

        utils.save_object(name+"rdmat", rdmatrix)
    else:
        rdmatrix = utils.load_object(name+"rdmat")

    return model2.process_matrix(rdmatrix, delete_blank_states)


def generate_rdm_all_gradient(nnet, name, blanks, rdm_type=analysis.SPEARMAN, save_files=True, title="RDM training combined",
                     from_file=False, delete_blank_states=True):
    if not from_file:
        if rdm_type != analysis.SPEARMAN:
            raise Exception("not implemented")
        hidden_both, accuracy_totals_both, accuracy_fullseqs_both = test_network_all(nnet)
        hidden_ari, accuracy_totals_ari, accuracy_fullseqs_ari = test_network_ari(nnet, blanks)
        hidden_bev, accuracy_totals_bev, accuracy_fullseqs_bev = test_network_bev(nnet, blanks)
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

    rdmatrix_left, labels = model2.process_matrix(rdmatrix_left, delete_blank_states)
    rdmatrix_right, _ = model2.process_matrix(rdmatrix_right, delete_blank_states)

    return rdmatrix_left, rdmatrix_right, labels


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
    extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                         [0, model.size_hidden], [0, model.size_hidden],
                                                         hrp.reg_strength, reg_type="recurrent", reg_increase=hrp.reg_increase)

    # Prev action to hidden
    # print("action-->hidden")
    extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                         [model.size_hidden + model.size_observation,
                                                          model.size_hidden + model.size_observation + model.size_action],
                                                         [0, model.size_hidden],
                                                         hrp.reg_strength, reg_type="input_right",
                                                         reg_increase=hrp.reg_increase)

    # middle=0.25)
    # Prev goal to hidden
    # print("goal-->hidden")
    extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                         [model.size_hidden + model.size_observation + model.size_action,
                                                          model.size_hidden + model.size_observation + model.size_action + model.size_goal1],
                                                         [0, model.size_hidden],
                                                         hrp.reg_strength, reg_type="input_left", reg_increase=hrp.reg_increase)

    # Regularization in the output layers (goals and actions) weights
    # hidden to next action
    extra_loss += utils.weight_regularization_calculator(model.action_layer.w,
                                                         [0, model.size_hidden], [0, model.size_action],
                                                         hrp.reg_strength, reg_type="output_right",
                                                         reg_increase=hrp.reg_increase)

    # Hidden to next goal
    # print("hidden->goal")
    extra_loss += utils.weight_regularization_calculator(model.goal1_layer.w,
                                                         [0, model.size_hidden], [0, model.size_goal1],
                                                         hrp.reg_strength, reg_type="output_left",
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


            loss = nnet.train(tape, targets, apply_extra_loss(nnet, hrp))
            loss = loss.numpy()[0]
            avg_loss = 0.999 * avg_loss + 0.001 * loss
            if i % 1000 == 0:
                _, accuracy_both, _ = test_network_all(nnet)
                _, accuracy_ari, _ = test_network_ari(nnet, blanks)
                _, accuracy_bev, _ = test_network_bev(nnet, blanks)
                print('{0}, avgloss={1}, accuracies=both{2}, ari{3}, bev{4}'.format(i, avg_loss, accuracy_both, accuracy_ari, accuracy_bev))
            i += 1
    print("Training complete after " + str(i) + " iterations")
    nnet.new_episode()  # just clear up the network history to avoid any bad surprises


def stop_condition(nnet, blanks, min_accuracy=1.):
    _, accuracy_both, _ = test_network_all(nnet)
    _, accuracy_ari, _ = test_network_ari(nnet, blanks)
    _, accuracy_bev, _ = test_network_bev(nnet, blanks)
    if not blanks:
        return np.all(accuracy_both >= [.75, .75, .5, .5, 1., 1., 1., 1., 1., 1., 1., min_accuracy]) and \
           np.all(accuracy_ari >= [1., .5, 1., 1., 1., min_accuracy]) and \
           np.all(accuracy_bev >= [.5, .5, 1., 1., 1., 1.])
    elif blanks:
        return np.all(accuracy_both >= [.75, .75, .5, .5, 1., 1., 1., 1., 1., 1., 1., min_accuracy]) and \
        np.all(accuracy_ari >= [1., 1., .5, 1., 1., 1., 1., 1., 1., 1., min_accuracy]) and \
        np.all(accuracy_bev >= [.5, 1., .5, 1., 1., 1., 1., 1., 1., 1., 1.])


def run_model3_multiple(stopping_params, nnparams, blanks, from_file=None,
                        num_networks=1, name="model3", hrp=None):
    if from_file is not None:
        networks = utils.load_objects(from_file, num_networks)
    else:
        networks = []
        for i in range(num_networks):
            nnparams.size_goal1=2
            nnparams.size_action=len(task.output_symbols)
            nnparams.size_observation=len(task.input_symbols)
            nnet = nn.ElmanGoalNet(params=nnparams)
            train_all(stopping_params, nnet, hrp=hrp, blanks=blanks)
            utils.save_object(name, nnet)
            networks.append(nnet)
            # Print some stuff
            hidden_activation, accuracy_totals, accuracy_fullseqs = test_network_all(nnet)
            print("network {0}: ")
            print(accuracy_totals)
            print(accuracy_fullseqs)

    # pattern of sequences, for the mds
    pattern = [6]*4 + [6]*4 + [12]*4
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

        # Save it
        utils.save_rdm(average_rdm, name, labels,  title="RDM training combined")
        analysis.make_mds(average_rdm, name, labels=labels, title="MDS training combined", pattern=pattern)
    else:
        sum_rdm_left = sum_rdm_right = None
        labels = None
        for net in networks:
            rdmleft, rdmright, labels = generate_rdm_all_gradient(net, name=name, blanks=blanks, from_file=False, delete_blank_states=True)
            if sum_rdm_left is None:
                sum_rdm_left = rdmleft
                sum_rdm_right = rdmright
            else:
                sum_rdm_left += rdmleft
                sum_rdm_right += rdmright
        average_rdm_left = sum_rdm_left/num_networks
        average_rdm_right = sum_rdm_right/num_networks

        utils.save_rdm(average_rdm_left, name+"left", labels,  title="RDM training combined: left (goals)", fontsize=1.)
        utils.save_rdm(average_rdm_right, name+"right", labels,  title="RDM training combined: right (actions)", fontsize=1.)
        analysis.make_mds(average_rdm_left, name+"left", labels=labels, title="MDS training combined: left (goals)", pattern=pattern)
        analysis.make_mds(average_rdm_right, name+"right", labels=labels, title="MDS training combined: right (actions)", pattern=pattern)


def make_targets_ari(seq_ari, blanks):
    target0 = utils.str_to_onehot(seq_ari[1], task.output_symbols)
    target1 = utils.str_to_onehot(seq_ari[2], task.output_symbols)
    target2 = utils.str_to_onehot(seq_ari[3], task.output_symbols)
    target3 = utils.str_to_onehot(seq_ari[4], task.output_symbols)
    target4 = utils.str_to_onehot(seq_ari[5], task.output_symbols)
    target5 = utils.str_to_onehot(seq_ari[6], task.output_symbols)

    targets = []
    if blanks:
        targets.append(task.Target(target0, task.goal_target_ari[0]))
        targets.append(task.Target(None, None))
        targets.append(task.Target(target1, task.goal_target_ari[1]))
        targets.append(task.Target(None, None))
        targets.append(task.Target(target2, task.goal_target_ari[2]))
        targets.append(task.Target(None, None))
        targets.append(task.Target(target3, task.goal_target_ari[3]))
        targets.append(task.Target(None, None))
        targets.append(task.Target(target4, task.goal_target_ari[4]))
        targets.append(task.Target(None, None))
        targets.append(task.Target(target5, task.goal_target_ari[-1]))
    else:
        targets.append(task.Target(target0, task.goal_target_ari[0]))
        targets.append(task.Target(target1, task.goal_target_ari[1]))
        targets.append(task.Target(target2, task.goal_target_ari[2]))
        targets.append(task.Target(target3, task.goal_target_ari[3]))
        targets.append(task.Target(target4, task.goal_target_ari[4]))
        targets.append(task.Target(utils.str_to_onehot(seq_ari[6], task.output_symbols), task.goal_target_ari[-1]))
    return targets


# Blanks
def ff_ari(nnet, seq_ari, blanks):
    if blanks:
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
    else:
        nnet.feedforward(utils.str_to_onehot(seq_ari[0], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[1], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[2], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[3], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[4], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_ari[5], task.input_symbols))


def ff_bev(nnet, seq_bev, blanks):
    if blanks:
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
    else:
        nnet.feedforward(utils.str_to_onehot(seq_bev[0], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[1], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[2], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[3], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[4], task.input_symbols))
        nnet.feedforward(utils.str_to_onehot(seq_bev[5], task.input_symbols))


def make_targets_bev(seq_bev, blanks):
    targets = []
    if blanks:
        targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.output_symbols), None))
        targets.append(task.Target(None, task.goal_target_bev[0]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.output_symbols), None))
        targets.append(task.Target(None, task.goal_target_bev[1]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.output_symbols), None))
        targets.append(task.Target(None, task.goal_target_bev[2]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.output_symbols), None))
        targets.append(task.Target(None, task.goal_target_bev[3]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.output_symbols), None))
        targets.append(task.Target(None, task.goal_target_bev[4]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.output_symbols), None))
    else:
        targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.output_symbols), task.goal_target_bev[0]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.output_symbols), task.goal_target_bev[1]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.output_symbols), task.goal_target_bev[2]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.output_symbols), task.goal_target_bev[3]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.output_symbols), task.goal_target_bev[4]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.output_symbols), None))
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
    else:
        raise NotImplementedError("only starts are ari and bev")

def make_targets_all(seq_bev, seq_ari, start):
    target0 = utils.str_to_onehot(seq_ari[1], task.output_symbols)
    target1 = utils.str_to_onehot(seq_ari[2], task.output_symbols)
    target2 = utils.str_to_onehot(seq_ari[3], task.output_symbols)
    target3 = utils.str_to_onehot(seq_ari[4], task.output_symbols)
    target4 = utils.str_to_onehot(seq_ari[5], task.output_symbols)
    target5 = utils.str_to_onehot(seq_ari[6], task.output_symbols)

    targets = []
    if start == task.START_BEV:
        targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.output_symbols), task.goal_target_ari[0]))
        targets.append(task.Target(target0, task.goal_target_bev[0]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.output_symbols), task.goal_target_ari[1]))
        targets.append(task.Target(target1, task.goal_target_bev[1]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.output_symbols), task.goal_target_ari[2]))
        targets.append(task.Target(target2, task.goal_target_bev[2]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.output_symbols), task.goal_target_ari[3]))
        targets.append(task.Target(target3, task.goal_target_bev[3]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.output_symbols), task.goal_target_ari[4]))
        targets.append(task.Target(target4, task.goal_target_bev[4]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.output_symbols), task.goal_target_ari[5]))
        targets.append(task.Target(target5, task.goal_target_bev[5]))
    elif start == task.START_ARI:
        targets.append(task.Target(target0, task.goal_target_bev[0]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[1], task.output_symbols), task.goal_target_ari[0]))
        targets.append(task.Target(target1, task.goal_target_bev[1]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[2], task.output_symbols), task.goal_target_ari[1]))
        targets.append(task.Target(target2, task.goal_target_bev[2]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[3], task.output_symbols), task.goal_target_ari[2]))
        targets.append(task.Target(target3, task.goal_target_bev[3]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[4], task.output_symbols), task.goal_target_ari[3]))
        targets.append(task.Target(target4, task.goal_target_bev[4]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[5], task.output_symbols), task.goal_target_ari[4]))
        targets.append(task.Target(target5, task.goal_target_bev[5]))
        targets.append(task.Target(utils.str_to_onehot(seq_bev[6], task.output_symbols), task.goal_target_ari[5]))
    return targets


def test_network_all(model):
    sequence_length = len(make_targets_all(task.beverage_seqs[0], task.arithmetic_seqs[0], start=task.START_BEV)) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.

    for start in task.STARTS:
        for idx, seq_bev in enumerate(task.beverage_seqs):
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
    accuracy_totals /= len(task.arithmetic_seqs_easy)*len(task.beverage_seqs)*len(task.STARTS)
    accuracy_fullseqs /= len(task.arithmetic_seqs_easy)*len(task.beverage_seqs)*len(task.STARTS)
    return hidden_activation, accuracy_totals, accuracy_fullseqs


def test_network_ari(model, blanks):
    sequence_length = len(make_targets_ari(task.arithmetic_seqs[0], blanks))  # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for seq_ari in task.arithmetic_seqs:
        model.new_episode()
        ff_ari(model, seq_ari, blanks)
        context = [c.numpy().flatten() for c in model.h_context]
        hidden_activation.append(context)

        # Now also test whether the model was correct or not: compare targets with actual outputs
        targets = make_targets_ari(seq_ari, blanks)
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


def test_network_bev(model, blanks):
    sequence_length = len(make_targets_bev(task.beverage_seqs[0], blanks)) # wasteful but works
    hidden_activation = []
    accuracy_totals = np.zeros(sequence_length)
    accuracy_fullseqs = 0.
    for idx, seq_bev in enumerate(task.beverage_seqs):
        model.new_episode()
        ff_bev(model, seq_bev, blanks)
        context = [c.numpy().flatten() for c in model.h_context]
        hidden_activation.append(context)

        # Now also test whether the model was correct or not: compare targets with actual outputs
        targets = make_targets_bev(seq_bev, blanks)
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