# An ACC module that predicts the inner state of the task network.

# Architecture =
# 1. network that learns the CTT using RELU hidden layer
# 2. another network on top of that predicts RELU future activations
# 3. Other network should

from pnas import pnas2018task
from neural import optimizers, neuralnet as nn
import utils
import tensorflow as tf
import numpy as np
import scripts
import analysis
import rdm

def stop_condition_behavior(iterations, model):
    return iterations > 1500

def stop_condition_acc(iterations, model):
    return iterations > 2000


def train(model=None, iterations=2000, l1reg=0.0, l2reg= 0.0, algorithm=optimizers.SGD,
          size_pfc=15, size_acc=15, learning_rate=None, stop_behavior=stop_condition_behavior, stop_acc=stop_condition_acc, learning_rate_acc=0.01):
    if model is None:
        model = nn.CriticACC(size_recurrent=size_pfc, size_acc=size_acc, algorithm=algorithm, initialization=utils.HE, learning_rate_acc=learning_rate_acc)
    num_episodes = iterations
    if learning_rate is not None:  # Else keep the model's learning rate
        model.learning_rate = learning_rate
    model.L1_regularization = l1reg
    model.L2_regularization = l2reg

    rng_avg_loss = 0.
    rng_avg_loss_acc = 0.
    rng_avg_actions = 0.
    rng_avg_sequence = 0.
    avg_reward = []
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
            # train the behavior network
            if not stop_behavior(episode, model):
                for i in range(len(targets)):
                    observation = inputs[i].reshape(1, -1)
                    model.feedforward(observation, first_time_step=i == 0)
                # detect if there's an error
                tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
                ratios = scripts.evaluate([tchoices], [targets])
                loss = model.train_behavior(btape, targets)
            else: # train the ACC critic
                for i in range(len(targets)):
                    if i == 0:
                        observation = inputs[i].reshape(1, -1)
                    else:
                        observation_output = model.h_action_collapsed[-1] #<-- but that is an output. How do I convert this to an input??
                        observation_str = utils.onehot_to_str(observation_output, pnas2018task.all_outputs)
                        observation = utils.str_to_onehot(observation_str, pnas2018task.all_inputs)
                    observation += np.random.normal(0, 0.3, observation.shape)
                    model.feedforward(observation, first_time_step=i == 0)

                # detect if there's an error - does the behavior correspond to any of the learned sequences?
                best_accuracy = False
                for seq in pnas2018task.seqs:
                    targets = utils.liststr_to_onehot(seq[1:], pnas2018task.all_outputs)
                    tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
                    ratios = scripts.evaluate([tchoices], [targets])
                    if np.average(ratios) == 1:
                        best_accuracy = True
                        break
                if best_accuracy:
                    reward = 1.
                else:
                    reward = 0.
                avg_reward.append(reward)
                loss = model.train_acc(ptape, reward)
                print(reward)
                print(model.h_outcome_prediction)
                print(loss)
        episode += 1

        if episode < 1500:
            speed = 2. / (episode + 2) if episode < 1500 else 0.001  # enables more useful evaluations for early trials
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
        elif episode == 1500:
            # reset everything for the critic
            rng_avg_loss = 0.
            rng_avg_actions = 0.
            rng_avg_sequence = 0.

        else: # critic
            """
            ep = episode - 1000
            speed = 2. / (ep + 2) if ep < 1000 else 0.001  # enables more useful evaluations for early trials
            rng_avg_loss_acc = utils.rolling_avg(rng_avg_loss_acc, loss, speed)
            if (ep < 1000 and ep in [3 ** n for n in range(10)]) or \
               ep % 1000 == 0 or \
               ep + 1 == num_episodes:
                print("{0}: avg loss={1}".format(
                        episode, rng_avg_loss_acc))
            """
            critic_episode = episode - 1500
            speed = 2. / (critic_episode + 2) if critic_episode < 1000 else 0.001  # enables more useful evaluations for early trials
            rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
            rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
            rng_avg_sequence = utils.rolling_avg(rng_avg_sequence, ratios[0] == 1,
                                              speed)  # whole action sequence correct ?
            # Display on the console at regular intervals
            if (critic_episode < 1000 and critic_episode in [3 ** n for n in range(10)]) or \
               critic_episode % 1000 == 0 or \
               critic_episode + 1 == num_episodes:
                print("{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                        critic_episode, rng_avg_loss, rng_avg_actions, rng_avg_sequence))
    print(np.average(avg_reward))
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



def rdm_data_collection(model, rdm_type):
    if rdm_type != rdm.EUCLIDIAN:
        raise(NotImplementedError("only works with euclidian for now"))

    context_activation = []
    acc_activation = []
    all_choices = []
    correct = []
    properties = []

    # Run each sequence and test its accuracy
    for i in range(25):
        for seqid, sequence in enumerate(pnas2018task.seqs):
            seq_choices = []
            all_choices.append(seq_choices)
            inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
            targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
            model.action = np.zeros((1, model.size_action), dtype=np.float32)
            with tf.GradientTape() as tape:
                model.new_episode()
                for j in range(len(targets)):
                    if j == 0:
                        observation = inputs[j].reshape(1, -1)
                    else:
                        observation_output = model.h_action_collapsed[-1]
                        observation_str = utils.onehot_to_str(observation_output, pnas2018task.all_outputs)
                        observation = utils.str_to_onehot(observation_str, pnas2018task.all_inputs)
                    observation += np.random.normal(0, 0.4, observation.shape)
                    model.feedforward(observation, first_time_step=j == 0)
                    acc_activation.append(model.acc_activation)

                # Get some statistics about what was correct and what wasn't
                #choice = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
                #model.h_action_collapsed.clear()
                #seq_choices.append(choice)

            # evaluate if this was a correct or error trial
            # detect if there's an error - does the behavior correspond to any of the learned sequences?
            best_accuracy = False
            for seq in pnas2018task.seqs:
                targets = utils.liststr_to_onehot(seq[1:], pnas2018task.all_outputs)
                tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
                ratios = scripts.evaluate([tchoices], [targets])
                if np.average(ratios) == 1:
                    best_accuracy = True
                    break
            if best_accuracy:
                reward = 1.
            else:
                reward = 0.
            for j in range(len(targets)):
                property = {}
                property["seq"] = str(seqid)
                property["step"] = str(j)
                property["correct"] = str(reward==1)
                properties.append(property)
    return acc_activation, properties

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
    rdmats = []
    for i in range(num_networks+len(skips)):
        print(i)
        if i in skips:
            continue
        model = utils.load_object(name, i)
        acc_activations, properties = rdm_data_collection(model, rdm_type)

        rdmat_acc = rdm.rdm(properties, vectors=acc_activations, type=rdm_type)
        def ignore(prop1, prop2):
            if prop1["seq"] != prop2["seq"]:
                return True
            return False
        #rdmat_acc.save("rdm_critic_acc_temp")
        rdmat_acc = rdmat_acc.average_values(preserve_keys=["step", "correct"], ignore_func=ignore)
        rdmat_acc.sort_by(("step", False), ("correct", True))
        rdmats.append(rdmat_acc)

    average_mat = rdmats[0].matrix
    for i in range(1, num_networks):
        average_mat += rdmats[i].matrix
    average_mat = average_mat / num_networks
    rdmats[0].matrix = average_mat
    return rdmats[0]