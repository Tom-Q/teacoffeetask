import utils
import numpy as np
import tensorflow as tf
from goalenv import environment, task
import neuralnet as nn
import scripts
from termcolor import colored
import copy
import analysis
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import time


def _sequence_to_text(seq):
    top_goal_line = '\tGoals 1: '
    top_goal_counter = 1
    mid_goal_line = '\tGoals 2: '
    mid_goal_counter = 1
    action_line = '\tActions: '
    action_line += '1: ' + seq.targets[0].action_str + ' - '
    for j, target in enumerate(seq.targets[1:]):
        idx = j + 1
        prev = j
        # Check that the new goal is different from the previous one
        if target.goal1_str == seq.targets[prev].goal1_str and idx < seq.length - 1:
            top_goal_counter += 1
        else:
            # Print the previous action
            top_goal_line += str(idx - top_goal_counter + 1) + ':' + seq.targets[prev].goal1_str + ' x' + str(
                top_goal_counter) + ' - '
            top_goal_counter = 1
        # Check that the new goal is different from the previous one
        if target.goal2_str == seq.targets[prev].goal2_str and idx < seq.length - 1:
            mid_goal_counter += 1
        else:
            # Print the previous action
            mid_goal_line += str(idx - mid_goal_counter + 1) + ':' + seq.targets[prev].goal2_str + ' x' + str(
                mid_goal_counter) + ' - '
            mid_goal_counter = 1
        action_line += str(idx + 1) + ':' + target.action_str + ' - '
    line = top_goal_line + ';\n ' + mid_goal_line + ';\n ' + action_line
    return line


def _print_sequences_stats(sequences):
    unique_output_sequences = []
    counters = []
    for output_sequence in sequences:
        unique = True
        for i, sequence in enumerate(unique_output_sequences):
            if sequence.equals(output_sequence):#_sequence_equals(sequence, output_sequence):
                unique = False
                counters[i]+=1
        if unique:
            unique_output_sequences.append(output_sequence)
            counters.append(1)

    # Sort sequences and counters, starting with most frequent:
    zipped = zip(counters, unique_output_sequences)

    unique_output_sequences = [x for _, x in sorted(zipped, reverse=True, key=lambda pair: pair[0])]
    counters = sorted(counters, reverse=True)

    # Now display each unique sequence, but converted back to text:
    for i, seq in enumerate(unique_output_sequences):
        line = str(counters[i]) + ": "
        # Check if it's one of the target sequences
        is_target = False
        for target_sequence in task.sequences_list:
            if unique_output_sequences[i].equals(target_sequence): #_sequence_equals(unique_output_sequences[i], target_sequence.get_actions_one_hot()):
                line += "(TARGET: "+target_sequence.name+")\n"
                is_target = True
                break
        if not is_target:
            line += "(NOT A TARGET)\n"
        line += _sequence_to_text(seq)
        print(line)
    print(" \n")


def _print_stats_per_sequence(sequence_ids, outputs_per_sequence):
    for i, id in enumerate(sequence_ids):
        target_sequence = task.sequences_list[id]
        # Print sequence name and number + correct sequence:
        print("Sequence {0} ({1})\n{2}".format(id, task.sequences_list[i].name, _sequence_to_text(target_sequence)))
        print("Erroneous sequences (actions):")
        # For each instance, check whether it is 100% correct
        num_correct = 0
        for output_sequence in outputs_per_sequence[i]:
            # Check whether the outputs match the targets.
            correct = False
            if target_sequence.equals(output_sequence): #_sequence_equals(output_sequence, target_sequence.get_actions_one_hot()):
                correct = True
            else:
                for alt_solution in task.sequences_list[id].alt_solutions:
                    if target_sequence.equals(alt_solution): #_sequence_equals(output_sequence, alt_solution.get_actions_one_hot()):
                        correct = True
                        break
            if correct:
                num_correct += 1
        print("{0}/{1} correct".format(num_correct, len(outputs_per_sequence[i])))
        _print_sequences_stats(outputs_per_sequence[i])


def analyse_test_data(test_data, do_rdm=False, do_error_analysis=False):
    sequence_ids = range(len(test_data))

    ######################################################################################
    # Analysis 2: for each sequence, how many were right/wrong and print wrong sequences #
    ######################################################################################
    #Flatten the last level of outputs per sequence
    outputs_no_noise_step_distinction = []
    for seq_outputs in test_data:
        outputs_no_noise_step_distinction.append(utils.flatten_onelevel(seq_outputs))

    _print_stats_per_sequence(sequence_ids, outputs_no_noise_step_distinction)

    ####################################################################################################
    # Analysis 3: Noise. How long after the noise did the first problem occur? Was that step a switch? #
    ####################################################################################################
    if do_error_analysis:
        # Need to store actions per error state as well as per sequence - done
        # Need to detect at what step the first error is made
        for i, seq in enumerate(test_data):
            target_sequence = task.sequences_list[i]
            all_targets = [target_sequence.get_actions_one_hot()]
            for alt_solution in target_sequence.alt_solutions:
                all_targets.append(alt_solution.get_actions_one_hot())
            for noise_step, trials in enumerate(seq):
                first_error = [0] * target_sequence.length
                for trial in trials:
                    for j, action in enumerate(trial.get_actions_one_hot()):
                        error = 1
                        # It's only an error if it's an error for every target...
                        # I guess this misses situations in which the two sequences are tangled together. UGH!!
                        for target in all_targets:
                            if np.array_equal(action, target[j]):
                                error *= 0
                        if error:
                            first_error[j] += 1
                            break
                # Need to color the step at which noise is introduced.
                print("Target sequence " + str(i) + " (" + target_sequence.name + ") noise at step " + str(noise_step) + ": errors = " + str(first_error))

    # Analysis 4: RDM, MDS, and T-SNE
    # 1. RDM - from EVERY SEQUENCE
    # 1.a. Make a list of every sequence step. That means flattening everything into a list of steps.

    outputs_sequences_flat = utils.flatten_onelevel(outputs_no_noise_step_distinction)
    activations = [seq.activations for seq in outputs_sequences_flat]
    activations_flat = utils.flatten_onelevel(activations)  # all this list wrangling is ugly as hell
    if do_rdm:  # Do the RDM and MDS
        for i, tensor in enumerate(activations_flat):
            activations_flat[i] = tensor.numpy().reshape(-1)

        # Generate the RDM... That's actually very expensive computationally
        rdmatrix = analysis.rdm_spearman(activations_flat)

        print("rdm done")
        # Generate the MDS from the RDM.
        mdsy = analysis.mds(rdmatrix)
        # Display the MDS! Ugh!
        # Make labels!!
        print("mds done")

        mdsy_idx = 0
        colors = mcolors.CSS4_COLORS
        for i, seq in enumerate(outputs_sequences_flat):
            labels = []
            for j in range(seq.length):
                labels.append("seq "+ str(i) + ": " + str(j+1))
            analysis.plot_mds_points(mdsy[mdsy_idx:mdsy_idx + seq.length], range(seq.length), labels=labels, style=list(colors.values())[(i)%len(colors)])
            mdsy_idx += seq.length
        plt.title("MDS - ")
        plt.show()
        plt.clf()

    #T-SNE:
    print("Generating t-SNE...")
    activations = np.concatenate(activations_flat, axis=0)
    start = time.time()
    tsne = TSNE() #all default
    tsne_results = tsne.fit_transform(X=activations)
    end = time.time()
    print(end - start)
    print("...Done")
    results_pd = pd.DataFrame(tsne_results, columns=["x", "y"])
    #df_subset['tsne-2d-one'] = tsne_results[:, 0]
    #df_subset['tsne-2d-two'] = tsne_results[:, 1]
    #x = tsne_results[:, 0]
    #y = tsne_results[:, 1]
    print("plotting:")
    plt.figure(figsize=(16, 10))
    # Color points corresponding to sequence 1 in red, color points corresponding to action "add to mug" in blue,
    # Color points corresponding to subgoal "stir" in green
    x = tsne_results[:,0]
    y = tsne_results[:,1]
    plt.plot(x, y, ',k')

    # Add the TSNE plot points to the sequences.
    counter = 0
    for sequence in outputs_sequences_flat:
        sequence.additional_info = tsne_results[counter:counter+sequence.length, :]
        counter += sequence.length

    # Now plot whatever fancy shit we want.
    for seqid, sequence in enumerate(outputs_sequences_flat):
        if seqid == 100:
            # Plot the whole sequence in red
            plt.plot(sequence.additional_info[:, 0], sequence.additional_info[:, 1], '-r')
        for index, target in enumerate(sequence.targets):
            if target.goal2_str == "g_2_stir":
                plt.plot(sequence.additional_info[index, 0], sequence.additional_info[index, 1], 'xg')
            if target.action_str == "a_take":
                plt.plot(sequence.additional_info[index, 0], sequence.additional_info[index, 1], '+m')
            #if target.goal1_str == "g_1_make_coffee":
            #    plt.plot(sequence.additional_info[index, 0], sequence.additional_info[index, 1], 'ob')


    #sns.scatterplot(
    #    x="x", y="y",
    #    hue="y",
    #    #palette=sns.color_palette("hls", 10),
    #    data=results_pd,
    #    legend="full",
    #    alpha=0.3
    #)
    plt.savefig("tsne")


def generate_test_data(model, sequence_ids, noise_per_step=True, noise=0., goal1_noise=0., goal2_noise=0., num_tests=10, goals=False,
                       initialization="uniform", verbose=False):
    # This runs a hundred version of the model with different random initializations
    env = environment.GoalEnv()
    outputs_per_sequence = []

    # Find the max sequence length
    target_actions_sequences = []
    sequences = []
    max_length = 0
    for seqid in sequence_ids:
        sequence = task.sequences_list[seqid]
        sequences.append(sequence)
        target_actions_sequences.append(sequence.get_actions_one_hot())
        if max_length < sequence.length:
            max_length = sequence.length
    max_length += 10  # add a bit of a margin for additions

    for seqid, sequence in enumerate(sequences):
        print("testing sequence: "+str(seqid))
        outputs_per_noise_step = []
        num_noise_steps = sequence.length if noise_per_step else 1
        for noise_step in range(num_noise_steps):
            outputs = []
            for i in range(num_tests):
                # Initialize the sequence.
                init_state = sequence.initial_state
                # Set up the current state to be 0.
                init_state.current.o_ddairy_first = 0
                env.reinitialize(init_state)
                model.action = np.zeros((1, model.size_action), dtype=np.float32)

                # run the network
                with tf.GradientTape() as tape:
                    # Initialize context with random/uniform values.
                    if initialization == 'uniform':
                        model.context = np.random.uniform(0.01, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
                        model.action = np.zeros_like(sequence.targets[0].action_one_hot)
                        if goals:
                            model.goal1 = np.zeros_like(sequence.targets[0].goal1_one_hot)
                            model.goal2 = np.zeros_like(sequence.targets[0].goal2_one_hot)

                    output_sequence = task.BehaviorSequence(sequence.initial_state)
                    output_sequence.activations = []
                    for j in range(max_length):
                        # Add noise to context layer
                        if j == noise_step or not noise_per_step:
                            model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                            model.goal1 += np.float32(np.random.normal(0., goal1_noise, size=(1, model.size_goal1)))
                            model.goal2 += np.float32(np.random.normal(0., goal2_noise, size=(1, model.size_goal2)))
                        observation = env.observe()
                        model.feedforward(observation)
                        output_sequence.activations.append(model.context)
                        # if there's an impossible action, ignore it and continue.
                        next_state = copy.deepcopy(env.state.next)
                        try:
                            env.do_action(model.h_action_wta[-1])
                        except environment.ActionException as error:  # The action doesn't make sense and cannot be executed in the environment.
                            if verbose:
                                print(error)
                            # reset the state when an impossible action is attempted.
                            env.state.next = next_state
                        action_str = utils.onehot_to_str(model.h_action_wta[-1], environment.GoalEnvData.actions_list)
                        if action_str in environment.TERMINAL_ACTIONS:
                            break  # we said "done", we're done.

                    # Get some statistics about the sequences actually observed. Is it a recognized sequence? If not,
                    # What kind of mistake appeared?
                    output_sequence.set_targets(model.h_goal1_wta, model.h_goal2_wta, model.h_action_wta)
                    outputs.append(output_sequence)
                    model.clear_history()
            outputs_per_noise_step.append(outputs)
        outputs_per_sequence.append(outputs_per_noise_step)
    return outputs_per_sequence


def train(model=None, goals=False, num_iterations=50000, learning_rate=0.01, L2_reg=0.00001, noise=0., sequences=None):
    if sequences is None:
        sequences = [0]
    env = environment.GoalEnv()
    if model is None:
        if goals:
            model = nn.NeuralNet(size_hidden=50, size_observation=29, size_action=19,
                                 size_goal1=len(environment.GoalEnvData.goals1_list),
                                 size_goal2=len(environment.GoalEnvData.goals2_list),
                                 algorithm=nn.RMSPROP, learning_rate=learning_rate, initialization="uniform")
        else:
            model = nn.NeuralNet(size_hidden=50, size_observation=29, size_action=19,  size_goal1=len(sequences), size_goal2=0,
                                 algorithm=nn.RMSPROP, learning_rate=learning_rate, initialization="uniform")

    model.L2_regularization = L2_reg

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_fullseq = 0.
    rng_avg_goals1 = 0.
    rng_avg_goals2 = 0.

    for iteration in range(num_iterations):
        seqid = np.random.choice(sequences)
        sequence = task.sequences_list[seqid]
        env.reinitialize(sequence.initial_state)
        #if np.random.random() > 0.5:
        #    env.state.current.set_field("o_sequence"+str(seqid+1), 1)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)

        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.random.uniform(0.01, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
            # Set up the prior actions and goals to what they OUGHT to be
            model.action = np.zeros_like(sequence.targets[0].action_one_hot)
            model.goal1 = np.zeros_like(sequence.targets[0].goal1_one_hot)
            model.goal2 = np.zeros_like(sequence.targets[0].goal2_one_hot)
            # Alternative: zeros
            #model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            for i, target in enumerate(sequence.targets):
                # Add noise to context layer
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = env.observe()
                model.feedforward(observation)
                env.do_action(target.action_one_hot)
                # Set the correct recurrent units:
                model.goal1 = target.goal1_one_hot
                model.goal2 = target.goal2_one_hot
                model.action = target.action_one_hot

            # Get some statistics about the percentage of correct behavior
            actions = np.array(model.h_action_wta).reshape((-1, environment.GoalEnvData.num_actions))
            target_actions = sequence.get_actions_one_hot()
            ratio_actions = scripts.ratio_correct(actions, target_actions)
            if goals:
                goals1 = np.array(model.h_goal1_wta).reshape((-1, environment.GoalEnvData.num_goals1))
                target_goals1 = sequence.get_goals1_one_hot()
                ratio_goals1 = scripts.ratio_correct(goals1, target_goals1)

                goals2 = np.array(model.h_goal2_wta).reshape((-1, environment.GoalEnvData.num_goals2))
                target_goals2 = sequence.get_goals2_one_hot()
                ratio_goals2 = scripts.ratio_correct(goals2, target_goals2)

            # Train model, record loss.
            loss = model.train(sequence.targets, tape)

        # Monitor progress using rolling averages.
        full_sequence = int(ratio_actions == 1)
        speed = 2. / (iteration + 2) if iteration < 1000 else 0.001  # enables more useful evaluations for early trials
        rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
        rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratio_actions, speed)
        rng_avg_fullseq = utils.rolling_avg(rng_avg_fullseq, full_sequence, speed)
        if goals:
            rng_avg_goals1 = utils.rolling_avg(rng_avg_goals1, ratio_goals1, speed)  # whole action sequence correct ?
            rng_avg_goals2 = utils.rolling_avg(rng_avg_goals2, ratio_goals2, speed)
        # Display on the console at regular intervals
        if (iteration < 1000 and iteration in [3 ** n for n in range(50)]) or iteration % 1000 == 0 \
                or iteration + 1 == num_iterations:
            print("{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    iteration, rng_avg_loss, rng_avg_actions, rng_avg_fullseq, rng_avg_goals1, rng_avg_goals2))
    return model