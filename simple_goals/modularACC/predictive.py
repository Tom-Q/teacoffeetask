# An ACC module that predicts the inner state of the task network.

# Architecture =
# 1. network that learns the CTT using RELU hidden layer
# 2. another network on top of that predicts RELU future activations
# 3. Other network should

from pnas import pnas2018task
from neural import optimizers, neuralnet as nn
from modularACC import task
import utils
import tensorflow as tf
import numpy as np
import scripts
import analysis

def stop_condition_behavior(iterations, model):
    return iterations > 10000

def stop_condition_acc(iterations, model):
    return iterations > 20000

def train(model=None, iterations=5000, l1reg=0.0, l2reg= 0.0,
          algorithm=optimizers.SGD,
          size_pfc=15, size_acc=200, learning_rate=None,
          stop_behavior=stop_condition_behavior, stop_acc=stop_condition_acc, ultra=False,
          learning_rate_acc=0.01,
          size_observation=len(pnas2018task.all_inputs),
          size_action=len(pnas2018task.all_outputs)):
    if model is None:
        if ultra:
            model = nn.UltraPredictiveACC(size_observation=size_observation, size_action=size_action, size_recurrent=size_pfc, size_acc=size_acc, algorithm=algorithm, initialization=utils.HE, learning_rate_acc=learning_rate_acc)
        else:
            model = nn.PredictiveACC(size_observation=size_observation, size_action=size_action, size_recurrent=size_pfc, size_acc=size_acc, algorithm=algorithm, initialization=utils.HE, learning_rate_acc=learning_rate_acc)
    num_episodes = iterations
    if learning_rate is not None:  # Else keep the model's learning rate
        model.learning_rate = learning_rate
    model.L1_regularization = l1reg
    model.L2_regularization = l2reg

    rng_avg_loss = 0.
    rng_avg_loss_acc = 0.
    rng_avg_actions = 0.
    rng_avg_sequence = 0.

    episode = 0
    while episode < iterations and not stop_acc(episode, model):
        model.new_episode()
        seqid = utils.idx_from_probabilities(pnas2018task.sequence_probabilities)

        sequence = pnas2018task.seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)

        # run the network
        with tf.GradientTape() as ptape, tf.GradientTape() as btape:
            #model.PredictionTape = ptape
            #model.BehaviorTape = btape
            for i in range(len(targets)):
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation, first_time_step=i == 0)
            tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            if not stop_behavior(episode, model):
                loss = model.train_behavior(btape, targets)
            else:
                loss = model.train_acc(ptape, iteration=episode)
        episode += 1

        if episode < 1000:
            speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
            rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
            rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
            rng_avg_sequence = utils.rolling_avg(rng_avg_sequence, ratios[0] == 1,
                                              speed)  # whole action sequence correct ?
            # Display on the console at regular intervals
            if (episode < 1000 and episode in [3 ** n for n in range(10)]) or \
               episode % 1000 == 0 or \
               episode + 1 == num_episodes:
                print("{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                        episode, rng_avg_loss, rng_avg_actions, rng_avg_sequence))
        else:
            ep = episode - 1000
            speed = 2. / (ep + 2) if ep < 1000 else 0.001  # enables more useful evaluations for early trials
            rng_avg_loss_acc = utils.rolling_avg(rng_avg_loss_acc, loss, speed)
            if (ep < 1000 and ep in [3 ** n for n in range(10)]) or \
               ep % 1000 == 0 or \
               ep + 1 == num_episodes:
                print("{0}: avg loss={1}".format(
                        episode, rng_avg_loss_acc))

    return model, rng_avg_sequence

def train_new_task(model=None, iterations=20000, l1reg=0.0, l2reg= 0.0,
          algorithm=optimizers.SGD,
          size_pfc=50, size_acc=50, learning_rate=None,
          stop_behavior=stop_condition_behavior, stop_acc=stop_condition_acc, ultra=False,
          learning_rate_acc=0.01,
          size_observation=len(task.all_inputs),
          size_action=len(task.all_outputs)):
    if model is None:
        if ultra:
            model = nn.UltraPredictiveACC(size_observation=size_observation, size_action=size_action, size_recurrent=size_pfc, size_acc=size_acc, algorithm=algorithm, initialization=utils.HE, learning_rate_acc=learning_rate_acc)
        else:
            model = nn.PredictiveACC(size_observation=size_observation, size_action=size_action, size_recurrent=size_pfc, size_acc=size_acc, algorithm=algorithm, initialization=utils.HE, learning_rate_acc=learning_rate_acc)
    num_episodes = iterations
    if learning_rate is not None:  # Else keep the model's learning rate
        model.learning_rate = learning_rate
    model.L1_regularization = l1reg
    model.L2_regularization = l2reg

    rng_avg_loss = 0.
    rng_avg_loss_acc = 0.
    rng_avg_actions = 0.
    rng_avg_sequence = 0.

    episode = 0
    while episode < iterations and not stop_acc(episode, model):
        model.new_episode()

        inputs = []
        for input_step in task.input_seq:
            inputs.append(utils.str_to_multihot(input_step, task.all_inputs))

        intent_step1 = np.random.uniform(low=0., high=1., size=2)  # tea or coffee
        intent_step2 = np.random.uniform(low=0., high=1., size=3)  # water first, milk first, or sugar first
        if intent_step1[0] > intent_step1[1]:
            if intent_step2[0] > intent_step2[1]: target = task.coffee_water_first
            else:                                 target = task.coffee_water_second
        else:
            if intent_step2[0] > intent_step2[2]:  target = task.tea_water_first
            else:                                  target = task.tea_water_second

        # different targets for different intents
        targets = utils.liststr_to_onehot(target, task.all_outputs)

        # add the intents to the inputs tea with sugar coffee with milk
        utils.set_value(task.all_inputs, inputs[0], "coffee intent", intent_step1[0])
        utils.set_value(task.all_inputs, inputs[0], "tea intent", intent_step1[1])
        utils.set_value(task.all_inputs, inputs[1], "water first intent", intent_step2[0])
        utils.set_value(task.all_inputs, inputs[1], "milk first intent", intent_step2[1])
        utils.set_value(task.all_inputs, inputs[1], "sugar first intent", intent_step2[2])

        # run the network
        with tf.GradientTape() as ptape, tf.GradientTape() as btape:
            #model.PredictionTape = ptape
            #model.BehaviorTape = btape
            for i in range(len(targets)):
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation, first_time_step=i == 0)
            tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            if not stop_behavior(episode, model):
                loss = model.train_behavior(btape, targets)
            else:
                loss = model.train_acc(ptape, iteration=episode)
        episode += 1

        if episode < 10000:
            speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
            rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
            rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
            rng_avg_sequence = utils.rolling_avg(rng_avg_sequence, ratios[0] == 1, speed)  # whole action sequence correct ?
            # Display on the console at regular intervals
            if (episode < 1000 and episode in [3 ** n for n in range(10)]) or \
               episode % 1000 == 0 or \
               episode + 1 == num_episodes:
                print("{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                        episode, rng_avg_loss, rng_avg_actions, rng_avg_sequence))
        else:
            ep = episode - 10000
            speed = 2. / (ep + 2) if ep < 10000 else 0.001  # enables more useful evaluations for early trials
            rng_avg_loss_acc = utils.rolling_avg(rng_avg_loss_acc, loss, speed)
            if (ep < 10000 and ep in [3 ** n for n in range(10)]) or \
               ep % 10000 == 0 or \
               ep + 1 == num_episodes:
                print("{0}: avg loss={1}".format(
                        episode, rng_avg_loss_acc))

    return model, rng_avg_sequence


def accuracy_test(model):
    context_activation = []
    acc_activation = []
    all_choices = []

    # Run each sequence and test its accuracy
    for sequence in pnas2018task.seqs:
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
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation, first_time_step=i == 0)
                context_activation.append(model.context_activation)
                acc_activation.append(model.acc_activation)
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
    print(accuracy_totals)
    return acc_activation, context_activation, accuracy_totals


import rdm


def _process_activations(activations, rdm_type):
    # Turn into a list of simple vectors
    for k, tensor in enumerate(activations):
        activations[k] = tensor.numpy().reshape(-1)

    if rdm_type == rdm.SPEARMAN:
        rdmatrix = rdm.rdm_spearman(activations)
    elif rdm_type == rdm.MAHALANOBIS:
        rdmatrix = rdm.rdm_mahalanobis(activations)
    elif rdm_type == rdm.EUCLIDIAN:
        rdmatrix = rdm.rdm_euclidian(activations)
    elif rdm_type == rdm.CRAPPYNOBIS:
        rdmatrix = rdm.rdm_crappynobis(activations)
    else:
        raise ValueError("Only implemented rdm types are mahalanobis, spearman, euclidian")
    return rdmatrix


def make_rdm_multiple(name, num_networks, save_files=True, skips=[],
                      rdm_type=rdm.EUCLIDIAN, save_name=None):
    # Make one rdm for each network
    acc_rdmatrices = []
    context_rdmatrices = []
    for i in range(num_networks+len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        acc, context, _ = accuracy_test(model)

        acc_rdmatrices.append(_process_activations(acc, rdm_type))
        context_rdmatrices.append(_process_activations(context, rdm_type))

    avg_acc_matrix = sum(acc_rdmatrices)/len(acc_rdmatrices)
    avg_context_matrix = sum(context_rdmatrices)/len(context_rdmatrices)

    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        if save_name is None:
            save_name = name
        np.savetxt(save_name + "_acc_rdm.txt", avg_acc_matrix, delimiter="\t", fmt='%.2e')
        np.savetxt(save_name+"_context_rdm.txt", avg_context_matrix, delimiter="\t", fmt='%.2e')
    properties = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for j, action in enumerate(sequence[1:]):
            property = {}
            property["seq"] = str(i)
            property["step"] = str(j)
            property["action"] = action
            properties.append(property)

    rdmat_acc = rdm.rdm(properties, avg_acc_matrix)
    rdmat_context = rdm.rdm(properties, avg_context_matrix)
    return rdmat_acc, rdmat_context



def accuracy_test_new_task(model):
    context_activation = []
    acc_activation = []
    all_choices = []

    # Run each sequence and test its accuracy
    for sequence in task.seqs:
        inputs = []
        for input_step in task.input_seq:
            inputs.append(utils.str_to_multihot(input_step, task.all_inputs))

        if sequence == task.coffee_water_first:
            targets = utils.liststr_to_onehot(sequence, task.coffee_water_first)
            utils.set_value(task.all_inputs, inputs[0], "coffee intent", 1)
            utils.set_value(task.all_inputs, inputs[0], "tea intent", 0)
            utils.set_value(task.all_inputs, inputs[1], "water first intent", 1)
            utils.set_value(task.all_inputs, inputs[1], "milk first intent", 0)
            utils.set_value(task.all_inputs, inputs[1], "sugar first intent", 0)
        elif sequence == task.coffee_water_second:
            targets = utils.liststr_to_onehot(sequence, task.coffee_water_second)
            utils.set_value(task.all_inputs, inputs[0], "coffee intent", 1)
            utils.set_value(task.all_inputs, inputs[0], "tea intent", 0)
            utils.set_value(task.all_inputs, inputs[1], "water first intent", 0)
            utils.set_value(task.all_inputs, inputs[1], "milk first intent", 1)
            utils.set_value(task.all_inputs, inputs[1], "sugar first intent", 0)
        elif sequence == task.tea_water_first:
            targets = utils.liststr_to_onehot(sequence, task.tea_water_first)
            utils.set_value(task.all_inputs, inputs[0], "coffee intent", 0)
            utils.set_value(task.all_inputs, inputs[0], "tea intent", 1)
            utils.set_value(task.all_inputs, inputs[1], "water first intent", 1)
            utils.set_value(task.all_inputs, inputs[1], "milk first intent", 0)
            utils.set_value(task.all_inputs, inputs[1], "sugar first intent", 0)
        elif sequence == task.tea_water_second:
            targets = utils.liststr_to_onehot(sequence, task.tea_water_second)
            utils.set_value(task.all_inputs, inputs[0], "coffee intent", 0)
            utils.set_value(task.all_inputs, inputs[0], "tea intent", 1)
            utils.set_value(task.all_inputs, inputs[1], "water first intent", 0)
            utils.set_value(task.all_inputs, inputs[1], "milk first intent", 0)
            utils.set_value(task.all_inputs, inputs[1], "sugar first intent", 1)

        seq_choices = []
        all_choices.append(seq_choices)


        # set up the intentions: coffee, water first etc

        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            model.new_episode()
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation, first_time_step=i == 0)
                context_activation.append(model.context_activation)
                acc_activation.append(model.acc_activation)
            # Get some statistics about what was correct and what wasn't
            choice = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            model.h_action_collapsed.clear()
            seq_choices.append(choice)

    # Now evaluate accuracy:
    #accuracy_totals = np.zeros((len(task.seq1) - 1))
    #for i in range(len(all_choices)):
    #    targets = utils.liststr_to_onehot(task.seqs[i][1:], task.all_outputs)
    #    for j in range(len(targets)):
    #        if (all_choices[i][0][j] == targets[j]).all():
    #            accuracy_totals[j] += 1
    #accuracy_totals /= 4
    #print(accuracy_totals)
    return acc_activation, context_activation, None #, accuracy_totals


def make_rdm_multiple_new_task(name, num_networks, save_files=True, skips=[],
                      rdm_type=rdm.EUCLIDIAN, save_name=None):
    # Make one rdm for each network
    acc_rdmatrices = []
    context_rdmatrices = []
    for i in range(num_networks+len(skips)):
        if i in skips:
            continue
        model = utils.load_object(name, i)
        acc, context, _ = accuracy_test_new_task(model)

        acc_rdmatrices.append(_process_activations(acc, rdm_type))
        context_rdmatrices.append(_process_activations(context, rdm_type))

    avg_acc_matrix = sum(acc_rdmatrices)/len(acc_rdmatrices)
    avg_context_matrix = sum(context_rdmatrices)/len(context_rdmatrices)

    name=name.replace('.', '_')+'_'+rdm_type
    if save_files:
        if save_name is None:
            save_name = name
        np.savetxt(save_name + "_acc_rdm.txt", avg_acc_matrix, delimiter="\t", fmt='%.2e')
        np.savetxt(save_name+"_context_rdm.txt", avg_context_matrix, delimiter="\t", fmt='%.2e')
    properties = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for j, action in enumerate(sequence[1:]):
            property = {}
            property["seq"] = str(i)
            property["step"] = str(j)
            property["action"] = action
            properties.append(property)

    rdmat_acc = rdm.rdm(properties, avg_acc_matrix)
    rdmat_context = rdm.rdm(properties, avg_context_matrix)
    return rdmat_acc, rdmat_context