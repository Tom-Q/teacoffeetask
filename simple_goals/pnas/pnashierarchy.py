from pnas import pnas2018task
import utils
from neural import neuralnet as nn
import numpy as np
import tensorflow as tf
import scripts
import analysis
import matplotlib.pyplot as plt

def train_with_goals(noise=0, iterations=10000, learning_rate=0.1):
    model = nn.GoalNet(size_hidden=15, size_observation=7, size_action=8, size_goal1=2, size_goal2=0, recurrent_layer=nn.ELMAN)
    num_episodes = iterations
    model.learning_rate = learning_rate
    model.L2_regularization = 0.

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        seqid = utils.idx_from_probabilities(pnas2018task.sequence_probabilities)

        goal = pnas2018task.goals[seqid]
        sequence = pnas2018task.seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.new_episode()  # = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.goal1 = goal[0]
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                # Add noise
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            loss, _ = model.train_obsolete(targets, goal, None, tape)
        # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_goals = utils.rolling_avg(rng_avg_goals, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print(
                "{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_goals))
    return model


def train_hierarchical_nogoals(noise=0, iterations=10000, learning_rate=0.1, reg_strength=0.001, reg_increase="linear"):
    model = nn.GoalNet(size_hidden=15, size_observation=9, size_action=8, size_goal1=0, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = learning_rate
    model.L2_regularization = 0.

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        model.new_episode()
        seqid = utils.idx_from_probabilities(pnas2018task.sequence_probabilities)

        #goal = pnas2018task.goals[seqid]
        sequence = pnas2018task.seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            #model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            #model.goal1 = np.zeros_like(goal[0])
            for i in range(len(targets)):
                #model.action = np.zeros((1, model.size_action), dtype=np.float32)
                # Add noise
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            cols = model.size_hidden
            # Regularization in the hidden layer weights
            # Recurrent hidden to hidden connections
            extra_loss = utils.weight_regularization_calculator(model.hidden_layer.w,
                                                          [0, model.size_hidden], [0, cols],
                                                          reg_strength, reg_type="recurrent", reg_increase=reg_increase)
            # Prev action to hidden
            #extra_loss += weight_regularization_calculator(model.hidden_layer.w,
            #                                                     [model.size_hidden+9, model.size_hidden+9+model.size_action], [0, cols],
            #                                                     reg_strength, reg_type="input_right", reg_increase=reg_increase)
            # Prev goal to hidden
            #extra_loss += weight_regularization_calculator(model.hidden_layer.w,
            #                                                     [model.size_hidden+9+model.size_action, model.size_hidden+9+model.size_action+2], [0, cols],
            #                                                     reg_strength, reg_type="input_left", reg_increase=reg_increase)

            #Regularization in the output layers (goals and actions) weights
            # hidden to next action
            extra_loss += utils.weight_regularization_calculator(model.action_layer.w,
                                                           [0, model.size_hidden], [0, model.size_action],
                                                           reg_strength, reg_type="output_right", reg_increase=reg_increase)

            # Hidden to next goal
            #extra_loss += weight_regularization_calculator(model.goal1_layer.w,
            #                                                    [0, model.size_hidden], [0, model.size_action],
            #                                                     reg_strength, reg_type="output_left", reg_increase=reg_increase)

            # Regularization of the observation (only goes to the action side)
            #extra_loss += weight_regularization_calculator(model.hidden_layer.w,
            #                                                     [model.size_hidden, model.size_hidden+model.size_observation],
            #                                                     [0, cols],
            #                                                     reg_strength, reg_type="input_right", reg_increase=reg_increase)

            loss, _ = model.train_obsolete(targets, None, None, tape, extra_loss)
            #if(episode%100 == 0):
            #    print(loss.numpy()-extra_loss.numpy(), extra_loss.numpy())
        # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_goals = utils.rolling_avg(rng_avg_goals, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print(
                "{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_goals))
    return model

def train_hierarchical(noise=0, iterations=10000, learning_rate=0.1, reg_strength=0.001, reg_increase="linear", num_goals=2):
    #model = nn.NeuralNet(size_hidden=15, size_observation=9, size_action=8, size_goal1=2, size_goal2=0)
    model = nn.GoalNet(size_hidden=15, size_observation=9, size_action=8, size_goal1=num_goals, size_goal2=0)
    num_episodes = iterations
    model.learning_rate = learning_rate
    model.L2_regularization = 0.

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.

    for episode in range(num_episodes):
        model.new_episode()
        seqid = utils.idx_from_probabilities(pnas2018task.sequence_probabilities)
        goal = pnas2018task.goals[seqid]
        sequence = pnas2018task.seqs[seqid]
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        #model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            #model.context = np.float32(np.abs(np.random.randint(0, 2, (1, model.size_hidden))-0.1))#np.zeros((1, model.size_hidden), dtype=np.float32)
            #model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.goal1 = goal[0] #np.zeros_like(goal[0]) ##
            for i in range(len(targets)):
                #model.action = np.zeros((1, model.size_action), dtype=np.float32)
                # Add noise
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)

            # Get some statistics about what was correct and what wasn't
            tchoices = np.array(model.h_action_collapsed).reshape((-1, len(targets[0])))
            ratios = scripts.evaluate([tchoices], [targets])
            # Train model, record loss.
            cols = model.size_hidden
            # Regularization in the hidden layer weights
            # Recurrent hidden to hidden connections
            extra_loss = utils.weight_regularization_calculator(model.hidden_layer.w,
                                                          [0, model.size_hidden], [0, cols],
                                                          reg_strength, reg_type="recurrent", reg_increase=reg_increase)
            # Prev action to hidden
            # extra_loss += weight_regularization_calculator(model.hidden_layer.w,
            #                                               [model.size_hidden+9, model.size_hidden+9+model.size_action],
            #                                               [0, cols],
            #                                               reg_strength, reg_type="input_right", reg_increase=reg_increase)
            # Prev goal to hidden
            extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                          [model.size_hidden+9+model.size_action, model.size_hidden+9+model.size_action+num_goals],
                                                          [0, cols],
                                                          reg_strength, reg_type="input_left", reg_increase=reg_increase)

            # SWITCHED OUTPUT LEFT AND OUTPUT RIGHT.
            #Regularization in the output layers (goals and actions) weights
            # hidden to next action
            extra_loss += utils.weight_regularization_calculator(model.action_layer.w,
                                                           [0, model.size_hidden], [0, model.size_action],
                                                           reg_strength, reg_type="output_right", reg_increase=reg_increase)
            # Hidden to next goal
            extra_loss += utils.weight_regularization_calculator(model.goal1_layer.w,
                                                           [0, model.size_hidden], [0, model.size_action],
                                                           reg_strength, reg_type="output_left", reg_increase=reg_increase)

            # Regularization of the observation (only goes to the action side)
            #extra_loss += weight_regularization_calculator(model.hidden_layer.w,
            #                                                     [model.size_hidden, model.size_hidden+model.size_observation],
            #                                                     [0, cols],
            #                                                     reg_strength, reg_type="input_right", reg_increase=reg_increase)

            loss, _ = model.train_obsolete(targets, goal, None, tape, extra_loss)

        # Monitor progress using rolling averages.
        speed = 2. / (episode + 2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
        rng_avg_goals = utils.rolling_avg(rng_avg_goals, ratios[0] == 1, speed)  # whole action sequence correct ?
        # Display on the console at regular intervals
        if (episode < 1000 and episode in [3 ** n for n in range(50)]) or episode % 1000 == 0 \
                or episode + 1 == num_episodes:
            print("{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    episode, rng_avg_loss, rng_avg_actions, rng_avg_goals))
    return model

def make_rdm_multiple_hierarchy(name, num_networks, title="-", save_files=True, file_save_name=None, skips=[]):
    if file_save_name == None:
        file_save_name = name
    # Make one rdm for each network
    rdmatrices_left = []
    rdmatrices_right = []
    for i in range(num_networks + len(skips)):
        # Skip number
        if skips is not None and i in skips:
           continue
        model = utils.load_object(name, i)
        hidden = accuracy_test_reg_hierarchy(model, i)

        # Turn a list of tensors into a list of np vectors
        for i, tensor in enumerate(hidden):
            hidden[i] = tensor.numpy().reshape(-1)

        # Now cut that in two and make an RDM for each
        cutoff = int(len(hidden[0])//2)
        left_units = [vector[:cutoff] for vector in hidden]
        rdm_left= analysis.rdm_spearman(left_units)
        rdmatrices_left.append(rdm_left)

        right_units = [vector[cutoff:] for vector in hidden]
        rdm_right = analysis.rdm_spearman(right_units)
        rdmatrices_right.append(rdm_right)

    # Do the same processing for each side (low level/left and high_level/right)
    for side in [[rdmatrices_left, "_goals"], [rdmatrices_right, "_actions"]]:
        # Now average over all matrices
        avg_matrix = None
        for matrix in side[0]:
            if avg_matrix is None:
                avg_matrix = matrix
            else:
                avg_matrix += matrix
        avg_matrix = avg_matrix / num_networks
        side_name = file_save_name+side[1]
        np.savetxt(side_name+"_rdm_mat"+utils.datestr()+".txt", avg_matrix, delimiter="\t", fmt='%.2e')
        labels = []
        for i, sequence in enumerate(pnas2018task.seqs):
            for action in sequence[1:]:
                labels.append(str(i)+'_'+action)
        analysis.plot_rdm(avg_matrix, labels, title+side_name + " spearman rho matrix")
        if save_files:
            plt.savefig(side_name+'_rdm'+utils.datestr())
        plt.clf()

        mdsy = analysis.mds(avg_matrix)
        for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
            analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
        plt.title(title+side_name)
        if save_files:
            plt.savefig(side_name + '_mds'+utils.datestr())
        plt.clf()

def make_rdm_multiple_hierarchy_nogoals(name, num_networks, title="-", save_files=True, file_save_name=None, cutoff=None):
    if file_save_name == None:
        file_save_name = name
    # Make one rdm for each network
    rdmatrices_left = []
    rdmatrices_right = []
    for i in range(num_networks):
        if i == 19:
            continue
        model = utils.load_object(name, i)
        hidden = accuracy_test_reg_hierarchy_nogoals(model, i)

        # Turn a list of tensors into a list of np vectors
        for i, tensor in enumerate(hidden):
            hidden[i] = tensor.numpy().reshape(-1)

        # Now cut that in two and make an RDM for each
        if cutoff is None:
            cutoff = int(len(hidden[0])//2)
        left_units = [vector[:cutoff] for vector in hidden]
        rdm_left = analysis.rdm_spearman(left_units)
        rdmatrices_left.append(rdm_left)

        right_units = [vector[cutoff:] for vector in hidden]
        rdm_right = analysis.rdm_spearman(right_units)
        rdmatrices_right.append(rdm_right)

    # Do the same processing for each side (low level/left and high_level/right)
    for side in [[rdmatrices_left, "_goals"], [rdmatrices_right, "_actions"]]:
        # Now average over all matrices
        avg_matrix = None
        for matrix in side[0]:
            if avg_matrix is None:
                avg_matrix = matrix
            else:
                avg_matrix += matrix
        avg_matrix = avg_matrix / num_networks
        side_name = file_save_name+side[1]
        np.savetxt(side_name+"_rdm_mat"+utils.datestr()+".csv", avg_matrix, delimiter=",")
        labels = []
        for i, sequence in enumerate(pnas2018task.seqs):
            for action in sequence[1:]:
                labels.append(str(i)+'_'+action)
        analysis.plot_rdm(avg_matrix, labels, title+side_name + " spearman rho matrix")
        if save_files:
            plt.savefig(side_name+'_rdm'+utils.datestr())
        plt.clf()

        mdsy = analysis.mds(avg_matrix)
        for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
            analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
        plt.title(title+side_name)
        if save_files:
            plt.savefig(side_name + '_mds'+utils.datestr())
        plt.clf()

import copy
def accuracy_test_with_goals(model, gain=[1, 1, 1, 1]):
    hidden_activation = []
    all_choices = []
    for j, sequence in enumerate(pnas2018task.seqs):
        goal = pnas2018task.goals[j]
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            #model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.new_episode()
            model.goal1 = copy.deepcopy(goal[0])
            # Reset the previous action
            for i in range(len(targets)):
                model.goal1 *= gain[j]
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
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
    return hidden_activation

def accuracy_test_reg_hierarchy(model, model_num=None):
    hidden_activation = []
    all_choices = []
    for j, sequence in enumerate(pnas2018task.seqs):
        goal = pnas2018task.goals[j]
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            model.goal1 = goal[0]#np.zeros_like(goal[0]) #
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
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
    if model_num is not None:
        print(model_num, accuracy_totals)
    else:
        print(accuracy_totals)
    return hidden_activation

def accuracy_test_reg_hierarchy_nogoals(model, model_num=None):
    hidden_activation = []
    all_choices = []
    for j, sequence in enumerate(pnas2018task.seqs):
        #goal = goals[j]
        seq_choices = []
        all_choices.append(seq_choices)
        inputs = utils.liststr_to_onehot(sequence[:-1], pnas2018task.all_inputs)
        targets = utils.liststr_to_onehot(sequence[1:], pnas2018task.all_outputs)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)
        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            #model.goal1 = np.zeros_like(goal[0])
            # Reset the previous action
            for i in range(len(targets)):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                observation = inputs[i].reshape(1, -1)
                model.feedforward(observation)
                hidden_activation.append(model.context)
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
    if model_num is not None:
        print(model_num, accuracy_totals)
    else:
        print(accuracy_totals)
    return hidden_activation

def make_rdm_and_mds_reg_hierarchy(name):
    model = utils.load_object(name, 1)  # eg 'noise_test2'
    hidden = accuracy_test_reg_hierarchy(model)
    # Turn a list of tensors into a list of np vectors
    for i, tensor in enumerate(hidden):
        hidden[i] = tensor.numpy().reshape(-1)

    # Now cut that in two
    left_units = [vector[:len(hidden)/2] for vector in hidden]
    right_units = [vector[len(hidden)/2:] for vector in hidden]

    # Make the labels for the rdms
    labels = []
    for i, sequence in enumerate(pnas2018task.seqs):
        for action in sequence[1:]:
            labels.append(str(i) + '_' + action)
    for side in [[left_units, "left units"], [right_units, "right_units"]]:
        rdm = analysis.rdm_spearman(side[0])
        analysis.plot_rdm(rdm, labels, "Spearman rho matrix"+side[1])

        #for i in range(4):
        #    mdsy = analysis.mds(side[0][6*i:6*i+6])
        #    analysis.plot_mds_points(mdsy, range(len(mdsy)), labels=labels[6*i:6*i+6])

        mdsy = analysis.mds(side[0])
        for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
            analysis.plot_mds_points(mdsy[6*i:6*i+6], range(6), labels=labels[6*i:6*i+6], style=style)


# Special request from clay: model that averages rdms of different types
def make_rdm_multiple_special(name1, name2, num_networks, file_save_name, title, skips1=[], skips2=[]):
    mats = []
    for name, skips in [(name1, skips1), (name2, skips2)]:
        print(name)
        # Make one rdm for each network
        rdmatrices_left = []
        rdmatrices_right = []
        for i in range(num_networks+len(skips)):
            # Skip number
            if skips is not None and i in skips:
               continue
            model = utils.load_object(name, i)
            if name == name1:  # Have to do the goal ones first.
                hidden = accuracy_test_reg_hierarchy(model, i)
            else:
                hidden = accuracy_test_reg_hierarchy_nogoals(model, i)

            # Turn a list of tensors into a list of np vectors
            for i, tensor in enumerate(hidden):
                hidden[i] = tensor.numpy().reshape(-1)

            # Now cut that in two and make an RDM for each
            cutoff = int(len(hidden[0])//2)
            left_units = [vector[:cutoff] for vector in hidden]
            rdm_left= analysis.rdm_spearman(left_units)
            rdmatrices_left.append(rdm_left)

            right_units = [vector[cutoff:] for vector in hidden]
            rdm_right = analysis.rdm_spearman(right_units)
            rdmatrices_right.append(rdm_right)

        # Do the same processing for each side (low level/left and high_level/right)
        for side in [rdmatrices_left, rdmatrices_right]:  # goals, then actions
            # Now average over all matrices
            avg_matrix = None
            for matrix in side:
                if avg_matrix is None:
                    avg_matrix = matrix
                else:
                    avg_matrix += matrix
            avg_matrix = avg_matrix / num_networks
            mats.append(avg_matrix)

    # Now average the average matrices
    rdmatrices_left = (mats[0] + mats[2]) / 2
    rdmatrices_right = (mats[1] + mats[3]) / 2

    # Do the same processing for each side (low level/left and high_level/right)
    for side in [[rdmatrices_left, "_goals"], [rdmatrices_right, "_actions"]]:
        matrix = side[0]
        side_name = file_save_name+side[1]
        np.savetxt(side_name+"_rdm_mat"+utils.datestr()+".txt", matrix, delimiter="\t", fmt='%.2e')
        labels = []
        for i, sequence in enumerate(pnas2018task.seqs):
            for action in sequence[1:]:
                labels.append(str(i)+'_'+action)
        analysis.plot_rdm(matrix, labels, title+side_name + " spearman rho matrix")
        plt.savefig(side_name+'_rdm'+utils.datestr())
        plt.clf()

        mdsy = analysis.mds(matrix)
        for i, style in enumerate(['ro-', 'b|--', 'gx-.', 'k_:']):
            analysis.plot_mds_points(mdsy[6 * i:6 * i + 6], range(6), labels=labels[6 * i:6 * i + 6], style=style)
        plt.title(title+side_name)
        plt.savefig(side_name + '_mds'+utils.datestr())
        plt.clf()

