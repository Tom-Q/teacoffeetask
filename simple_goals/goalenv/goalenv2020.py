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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import time
import random


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
            # Print the previous time-step
            top_goal_line += str(idx - top_goal_counter + 1) + ':' + seq.targets[prev].goal1_str + ' x' + str(
                top_goal_counter) + ' - '
            top_goal_counter = 1
        # Check that the new goal is different from the previous one
        if target.goal2_str == seq.targets[prev].goal2_str and idx < seq.length - 1:
            mid_goal_counter += 1
        else:
            # Print the previous time-step
            mid_goal_line += str(idx - mid_goal_counter + 1) + ':' + seq.targets[prev].goal2_str + ' x' + str(
                mid_goal_counter) + ' - '
            mid_goal_counter = 1
        action_line += str(idx + 1) + ':' + target.action_str + ' - '
    #line = top_goal_line + ';\n ' + mid_goal_line + ';\n ' + action_line
    line = action_line
    return line


VERBOSE = False
def get_sequences_stats(sequences, include_goals=True):
    num_is_a_target = 0
    unique_output_sequences = []
    for output_sequence in sequences:
        unique = True
        for i, sequence in enumerate(unique_output_sequences):
            if sequence.equals(output_sequence):
                unique = False
                sequence.clones.append(output_sequence)
        if unique:
            unique_output_sequences.append(output_sequence)
            output_sequence.clones = []

    # Sort sequences and counters and seqids, starting with most frequent:
    unique_output_sequences = sorted(unique_output_sequences, reverse=True,
                                     key=lambda unique_seq: len(unique_seq.clones))

    # Now display each unique sequence, but converted back to text:
    for i, seq in enumerate(unique_output_sequences):
        line = ""

        # Check if it's one of the target sequences
        is_target = False
        for target_sequence in task.sequences_list:
            if unique_output_sequences[i].equals(target_sequence, include_goals=include_goals):
                line += "TARGET: "+target_sequence.name
                is_target = True
                num_is_a_target += 1+len(seq.clones)
                break
        if not is_target:
            line += "NOT A TARGET"
        line += ", " + str(1 + len(seq.clones)) + " " + str([seq.id] + [clone.id for clone in seq.clones]) + " " + ":\n"
        line += _sequence_to_text(seq)
        if VERBOSE:
            print(line)
    if VERBOSE:
        print(" \n")
    return num_is_a_target


def stats_per_sequence(sequence_ids, outputs_per_sequence, goals=True):
    total_num = 0
    total_correct_seq = 0
    total_subseq_error = 0
    total_is_a_target = 0
    for i, id in enumerate(sequence_ids):
        target_sequence = task.sequences_list[id]
        # Print sequence name and number + correct sequence:
        if VERBOSE:
            print("Sequence {0} ({1})\n{2}".format(id, task.sequences_list[i].name, _sequence_to_text(target_sequence)))
        # For each instance, check whether it is 100% correct
        num_correct = 0
        num_replaced = num_omitted = num_added = num_repeated = num_more_frequent = 0
        for output_sequence in outputs_per_sequence[i]:
            # Check whether the outputs match the targets.
            correct = False
            if target_sequence.equals(output_sequence, include_goals=goals):
                correct = True
            else:
                for alt_solution in task.sequences_list[id].alt_solutions:
                    if alt_solution.equals(output_sequence, include_goals=goals):
                        correct = True
                        break
            #if correct:
            #    num_correct += 1
            #else: # incorrect
            #    replaced, omitted, added, repeated, more_frequent = target_sequence.subsequence_analysis(output_sequence)
            #    num_replaced += replaced
            #    num_omitted += omitted
            #    num_added += added
            #    num_repeated += repeated
            #    num_more_frequent += more_frequent
        total_seqs = len(outputs_per_sequence[i])
        #if VERBOSE:
        #    print(("{0}/{1} correct.\n" +
        #          "\tInter-subsequence errors: {2}\n"+
        #          "\tSubsequence errors: {3}\n"+
        #          "\tOmissions: {4}\n"+
        #          "\tAdditions: {5}\n"+
        #          "\tRepetitions: {6}\n"+
        #          "\tMore frequent: {7}\n").format(num_correct, total_seqs, total_seqs - num_correct - num_replaced,
        #                                           num_replaced, num_omitted, num_added, num_repeated, num_more_frequent))
        #total_correct_seq += num_correct
        #total_subseq_error += num_replaced
        #total_num += total_seqs

        if VERBOSE:
            print("Erroneous sequences (actions):")
        total_is_a_target += get_sequences_stats(outputs_per_sequence[i], goals)
    #total_fullseq_errors = total_is_a_target - total_correct_seq
    #total_error = total_num - total_correct_seq
    #total_action_errors = total_error - total_subseq_error
    #print("Overall totals: {0}/{1} correct. {2} errors, of which:\n Action errors:{3}\n Subsequence errors: {4}\n Full sequence errors {5}\n".format(
    #    total_correct_seq, total_num, total_num - total_correct_seq, total_action_errors, total_subseq_error - total_fullseq_errors, total_fullseq_errors
    #))


def compute_last_step_loss(model, target, include_regularization=False):
    loss = 0
    loss += tf.nn.softmax_cross_entropy_with_logits(target.action_one_hot, model.h_action_softmax[-1])
    if target.goal1_one_hot is not None:
        loss += tf.nn.softmax_cross_entropy_with_logits(target.goal1_one_hot, model.h_goal1_softmax[-1])
    if target.goal2_one_hot is not None:
        loss += tf.nn.softmax_cross_entropy_with_logits(target.goal2_one_hot, model.h_goal2_softmax[-1])
    if include_regularization:
        loss += model.L2_regularization * sum([tf.reduce_sum(weights**2) for weights in model.all_weights])
    return loss.numpy()[0]


def generate_test_data(model, sequence_ids, noise=0., goal1_noise=0., goal2_noise=0., num_tests=10, goals=False,
                       initialization="uniform", verbose=False,
                       noise_per_step=True, disruption_per_step=False,
                       single_step_noise=None):
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

    seq_counter = 0
    for seqid, sequence in enumerate(sequences):
        if verbose:
            print("testing sequence: "+str(seqid))
        outputs_per_noise_step = []
        num_runs = sequence.length if (noise_per_step or disruption_per_step) else 1

        for noise_step in range(num_runs):
            if single_step_noise is not None:
                noise_step = single_step_noise
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
                    model.new_episode()
                    # Initialize context with random/uniform values.
                    if initialization == 'uniform':
                        model.context = np.random.uniform(0.01, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
                        model.action = np.zeros_like(sequence.targets[0].action_one_hot)
                        if goals:
                            model.goal1 = np.zeros_like(sequence.targets[0].goal1_one_hot)
                            model.goal2 = np.zeros_like(sequence.targets[0].goal2_one_hot)
                    elif initialization == 'seminormal':
                        model.context = np.random.normal(0.0, 0.1, (1, model.size_hidden)).astype(dtype=np.float32)
                        model.context[model.context < 0.0] = 0.  # it's impossible for a sigmoid activation to be <0
                        model.action = np.zeros_like(sequence.targets[0].action_one_hot)
                        if goals:
                            model.goal1 = np.zeros_like(sequence.targets[0].goal1_one_hot)
                            model.goal2 = np.zeros_like(sequence.targets[0].goal2_one_hot)

                    output_sequence = task.BehaviorSequence(sequence.initial_state)
                    output_sequence.activations = []
                    output_sequence.losses = []
                    for j in range(max_length):
                        # Add noise to context layer
                        if j == noise_step:
                            if noise_per_step or single_step_noise:
                                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                                if goals:
                                    model.goal1 += np.float32(np.random.normal(0., goal1_noise, size=(1, model.size_goal1)))
                                    model.goal2 += np.float32(np.random.normal(0., goal2_noise, size=(1, model.size_goal2)))
                            if disruption_per_step:
                                # Change the state
                                env.state = disrupt_state(env.state, initial_state=sequence.initial_state, #mode=REINITIALIZE)
                                                          mode=HOLD_RANDOM_OBJECT)
                        observation = env.observe()
                        model.feedforward(observation)

                        if j < len(sequence.targets):  # after that it's not defined.
                            target = sequence.targets[j] if goals else sequence.targets_nogoals[j]
                            loss = compute_last_step_loss(model, target, include_regularization=False)
                            output_sequence.losses.append(loss)
                        else:
                            output_sequence.losses.append(None)  # loss is undefined

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
                    output_sequence.id = seq_counter
                    output_sequence.target_seq_id = seqid
                    output_sequence.noise_step = noise_step
                    seq_counter += 1
                    outputs.append(output_sequence)
                    model.clear_history()
            outputs_per_noise_step.append(outputs)
        outputs_per_sequence.append(outputs_per_noise_step)
    return outputs_per_sequence


HOLD_RANDOM_OBJECT = 1
REINITIALIZE = 2
def disrupt_state(state, initial_state, mode=HOLD_RANDOM_OBJECT):
    if mode == HOLD_RANDOM_OBJECT:
        c = state.current
        n = state.next
        if c.o_held_nothing == 0:
            # Drop what we're holding
            c.reset_held()
            n.reset_held()
        else:
            c.reset_held()
            n.reset_held()
            # Hold something at random among the 7 possible objects
            val = random.randint(1, 7)
            if val == 1:
                c.o_held_coffee_jar = 1
                n.o_held_coffee_jar = 1
            elif val == 2:
                c.o_held_sugar_cube= 1
                n.o_held_sugar_cube= 1
            elif val == 3:
                c.o_held_milk_carton = 1
                n.o_held_milk_carton = 1
            elif val == 4:
                c.o_held_cream_carton = 1
                n.o_held_cream_carton = 1
            elif val == 5:
                c.o_held_teabag = 1
                n.o_held_teabag = 1
            elif val == 6:
                c.o_held_mug = 1
                n.o_held_mug = 1
            elif val == 7:
                c.o_held_spoon = 1
                n.o_held_spoon = 1
            c.o_held_nothing = 0
            n.o_held_nothing = 0
        return state
    elif mode == REINITIALIZE:
        return initial_state


def analyse_test_data(test_data, goals=True, do_rdm=False, mds_sequences=None, mds_range=None):
    sequence_ids = range(len(test_data))

    ######################################################################################
    # Analysis 1: for each sequence, how many were right/wrong and print wrong sequences #
    ######################################################################################
    #Flatten the last level of outputs per sequence
    outputs_no_noise_step_distinction = []
    for seq_outputs in test_data:
        outputs_no_noise_step_distinction.append(utils.flatten_onelevel(seq_outputs))

    stats_per_sequence(sequence_ids, outputs_no_noise_step_distinction, goals=goals)
    outputs_sequences_flat = utils.flatten_onelevel(outputs_no_noise_step_distinction)

    ##################################################################################################
    # Analysis 2: Noise. How long after the noise did the first error occur? Was that step a switch? #
    ##################################################################################################
    steps_noise_to_error = 0
    num_errors = 0
    error_on_noise = 0
    error_on_transition = 0

    num_replaced = 0
    num_omitted = 0
    num_added = 0
    num_repeated = 0
    num_more_frequent = 0
    num_is_a_target = 0

    total_trials = 0
    total_correct_seq = 0

    for i, seq in enumerate(test_data):
        target_sequence = task.sequences_list[i]
        for noise_step, trials in enumerate(seq):
            first_error = [0] * target_sequence.length
            for trial in trials:
                all_targets = [target_sequence] + target_sequence.alt_solutions
                trial.first_error = None  # This trial is error-free
                for j, action in enumerate(trial.get_actions_one_hot()):
                    error = True
                    # It's only an error if it's an error for every target...
                    # BUT this misses situations in which the two sequences are somehow tangled together.
                    # Solution: as soon as a target sequence no longer matches, remove it from the list of targets.
                    invalid = []
                    for k, target in enumerate(all_targets):
                        if np.array_equal(action, target.get_actions_one_hot()[j]):
                            error = False
                        else:
                            invalid.append(k)

                    if error:
                        first_error[j] += 1
                        trial.first_error = j  # Add a first_error attribute to the sequence.
                        # stats on average distance between noise and error.
                        steps_noise_to_error += trial.first_error - noise_step
                        num_errors += 1
                        # Is the error precisely on the noise step?
                        if trial.first_error == noise_step:
                            error_on_noise += 1
                        # Is the error on a transition step?
                        if all_targets[0].first_error_on_transition(trial):
                            error_on_transition += 1

                        # This stuff doesn't actually work. Forget it. Only useful bit is "num_replaced"
                        replaced, omitted, added, repeated, more_frequent, is_a_target = all_targets[0].subsequence_analysis(
                            trial)
                        num_replaced += replaced
                        num_omitted += omitted
                        num_added += added
                        num_repeated += repeated
                        num_more_frequent += more_frequent
                        num_is_a_target += is_a_target

                    # remove the no longer valid alternatives.
                    for inv in reversed(invalid):
                        all_targets.pop(inv)
                    if error:  # If we found an error we can terminate here
                        break
                if trial.first_error is None:
                    total_correct_seq += 1
                total_trials += 1

            # Need to color the step at which noise is introduced.
            #if VERBOSE:
            #    print("Target sequence " + str(i) + " (" + target_sequence.name + ") noise at step " + str(noise_step) + ": errors = " + str(first_error))
    total_fullseq_errors = num_is_a_target
    total_error = total_trials - total_correct_seq
    total_subseq_error = num_replaced
    total_action_errors = total_error - total_subseq_error
    # There's still a bug in the error counts.
    print(
        "Overall totals: {0}/{1} correct. {2} errors, of which:\n Action errors:{3}\n Subsequence errors: {4}\n Full sequence errors {5}\n".format(
            total_correct_seq,
            total_trials,
            total_trials - total_correct_seq,
            total_action_errors, # action errors.
            total_subseq_error - total_fullseq_errors, # Are all full sequence errors ALSO subsequence errors?
            total_fullseq_errors
        ))

    print("Overall stats:\n errors on noise step: {0}\n errors on transition: {1}\n average steps noise->error: {2}\n".format(
        error_on_noise, error_on_transition, steps_noise_to_error/num_errors if num_errors>0 else -1
    ))

    ####################
    # Analysis 3: Loss #
    ####################
    # Compute average loss, and average loss at first error and on subsequent points
    counter_total = 0
    loss_total = 0

    counter_first_error = 0
    first_error_loss_total = 0

    noise_loss_total = 0
    counter_noise = 0

    prior_to_noise_loss_total = 0
    counter_prior_to_noise = 0

    loss_after_noise_no_error = np.zeros(55)
    loss_after_noise_error = np.zeros(55)
    loss_before_error = np.zeros(55)
    counter_loss_before_error = np.zeros(55)
    counter_loss_after_noise_no_error = np.zeros(55)
    counter_loss_after_noise_error = np.zeros(55)

    for sequence in outputs_sequences_flat:
        for i, loss in enumerate(sequence.losses):
            if loss is not None:
                loss_total += sequence.losses[i]
                counter_total += 1
                if sequence.noise_step is not None and i < sequence.noise_step:
                    prior_to_noise_loss_total += sequence.losses[i]
                    counter_prior_to_noise+=1
        if sequence.first_error is not None:
            first_error_loss_total += sequence.losses[sequence.first_error]
            counter_first_error += 1
        # noise step can be > sequence length when the sequence terminated early
        if sequence.noise_step is not None and sequence.noise_step < sequence.length:
            noise_loss_total += sequence.losses[sequence.noise_step]
            counter_noise += 1

            if sequence.first_error is None: # No error occurred.
                # Record loss between noise and error
                i = 0
                while sequence.noise_step + i < sequence.length:
                    loss = sequence.losses[sequence.noise_step+i]
                    loss_after_noise_no_error[i] += loss
                    counter_loss_after_noise_no_error[i] += 1
                    i += 1
            elif sequence.first_error is not None: # an error occurred.
                i = 0
                while sequence.noise_step + i < sequence.length and sequence.noise_step + i <= sequence.first_error:
                    loss = sequence.losses[sequence.noise_step+i]
                    loss_after_noise_error[i] += loss
                    counter_loss_after_noise_error[i] += 1
                    i += 1
                # Also record loss between error and noise (going backwards)
                i = sequence.first_error
                j = 0
                while i >= sequence.noise_step:
                    loss = sequence.losses[i]
                    loss_before_error[j] += loss
                    counter_loss_before_error[j] += 1
                    i -= 1
                    j += 1
    # Compute avgs
    loss_after_noise_error = np.divide(loss_after_noise_error, counter_loss_after_noise_error, out=np.zeros_like(loss_after_noise_error),
                                  where=counter_loss_after_noise_error!=0)
    loss_after_noise_no_error = np.divide(loss_after_noise_no_error, counter_loss_after_noise_no_error,
                                     out=np.zeros_like(loss_after_noise_no_error),
                                     where=counter_loss_after_noise_no_error!=0)
    loss_before_error = np.divide(loss_before_error, counter_loss_before_error,
                                  out=np.zeros_like(loss_before_error),
                                  where=counter_loss_before_error!=0)

    loss_avg = loss_total/counter_total
    loss_first_error_avg = first_error_loss_total / counter_first_error if counter_first_error != 0 else None
    loss_prior_noise_avg = prior_to_noise_loss_total / counter_prior_to_noise if counter_prior_to_noise != 0 else None
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print("Loss avg prior to noise: " + str(loss_prior_noise_avg))
    print("Loss on and after noise (no error):" + str(loss_after_noise_no_error))
    print("Loss on and after noise (error):" + str(loss_after_noise_error))
    print("Loss prior to and on error:" + str(loss_before_error))
    #print("LOSS:\n average loss={0:.2f}\n loss on first error={1:.2f}\n loss on noise={2:.2f} (+1: {3:.2f})\n loss_good_seq={4}\n loss_before_noise={5}".format(
    #                                                                            loss_avg,
    #                                                                            loss_first_error_avg,
    #                                                                            loss_prior_noise_avg))
    if not do_rdm:
        return None, test_data, num_errors

    ###################################
    # Analysis 4: RDM, MDS, and T-SNE #
    ###################################
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
            length = seq.length
            if mds_range is not None and length > mds_range:
                length = mds_range
            if i in mds_sequences:
                labels = []
                for j in range(length):
                    labels.append("seq "+ str(i) + ": " + str(j+1))
                analysis.plot_mds_points(mdsy[mdsy_idx:mdsy_idx + length], range(length), labels=None,#labels,
                                         style=list(colors.values())[(i)%len(colors)])
                mdsy_idx += length
        plt.title("MDS")
        plt.show()
        plt.clf()

    #T-SNE:
    print("Generating t-SNE...")
    activations = np.concatenate(activations_flat, axis=0)
    start = time.time()
    tsne = TSNE(perplexity=50.0, n_iter=3000)  # Defaults are perplexity=30 and n_iter=1000.
    # Reducing perplexity captures global structure better. increasing n_iter makes sure we've converged.
    tsne_results = tsne.fit_transform(X=activations)
    print("KL divergence="+str(tsne.kl_divergence_))
    end = time.time()
    print(end - start)
    print("...Done")
    #df_subset['tsne-2d-one'] = tsne_results[:, 0]
    #df_subset['tsne-2d-two'] = tsne_results[:, 1]
    #x = tsne_results[:, 0]
    #y = tsne_results[:, 1]
    return tsne_results, test_data, num_errors  # Test data is enriched during analysis (first error step)


def plot_tsne(tsne_results, test_data, tsne_goals=False, tsne_subgoals=False, tsne_actions=False, tsne_sequences=False,
              tsne_errors=False, tsne_sequence=[], tsne_sequence_interval=[], filename="tsne", annotate=False):
    # Flatten test data
    outputs_no_noise_step_distinction = []
    for seq_outputs in test_data:
        outputs_no_noise_step_distinction.append(utils.flatten_onelevel(seq_outputs))
    outputs_sequences_flat = utils.flatten_onelevel(outputs_no_noise_step_distinction)

    plt.figure(figsize=(16, 10))
    # Color points corresponding to sequence 1 in red, color points corresponding to action "add to mug" in blue,
    # Color points corresponding to subgoal "stir" in green
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    plt.plot(x, y, ',k')

    # Add the TSNE plot points to the sequences.
    counter = 0
    for sequence in outputs_sequences_flat:
        sequence.tsne_coords = tsne_results[counter:counter + sequence.length, :]
        counter += sequence.length

    colors = ["aquamarine", "blueviolet", "coral",
              "crimson", "gold", "hotpink",
              "olivedrab", "royalblue", "sienna",
              "springgreen", "tomato", "skyblue",
              "limegreen", "darkorchid", "brown",
              "darkseagreen", "goldenrod", "khaki",
              "plum", "maroon", "slateblue", "navyblue",
              "forestgreen", "red", "gray", "black"]
    # Now plot whatever fancy shit we want. First the points, errors etc.
    for seqid, sequence in enumerate(outputs_sequences_flat):
        for index, target in enumerate(sequence.targets):
            idx = len(colors)-1  # black
            if tsne_errors:
                if sequence.first_error is None or index < sequence.first_error:
                    plt.plot(sequence.tsne_coords[index, 0], sequence.tsne_coords[index, 1], color="forestgreen", marker=',')
                elif index == sequence.first_error:
                    plt.plot(sequence.tsne_coords[index, 0], sequence.tsne_coords[index, 1], color="red", marker='.')
                elif index > sequence.first_error:
                    plt.plot(sequence.tsne_coords[index, 0], sequence.tsne_coords[index, 1], color="gray", marker=',')
            else:
                if tsne_goals:
                    idx = environment.GoalEnvData.goals1_list.index(target.goal1_str)
                elif tsne_subgoals:
                    idx = environment.GoalEnvData.goals2_list.index(target.goal2_str)
                elif tsne_actions:
                    idx = environment.GoalEnvData.actions_list.index(target.action_str)
                elif tsne_sequences:
                    idx = sequence.target_seq_id
                plt.plot(sequence.tsne_coords[index, 0], sequence.tsne_coords[index, 1], color=colors[idx], marker=',')
    # Then the sequences. In that order because we want the sequences on top.
    for seqid, sequence in enumerate(outputs_sequences_flat):
        if seqid in tsne_sequence:
            if not tsne_sequence_interval:  # test if list is empty
                tsne_sequence_interval = [0, sequence.length]
            seq_start=tsne_sequence_interval[0]
            seq_end=tsne_sequence_interval[1]
            # Plot the whole sequence
            plt.plot(sequence.tsne_coords[seq_start:seq_end, 0], sequence.tsne_coords[seq_start:seq_end, 1], linestyle='-', linewidth=1.,
                     color=colors[tsne_sequence.index(seqid)])
            # Special points get markers
            # Start
            plt.plot(sequence.tsne_coords[seq_start, 0], sequence.tsne_coords[seq_start, 1],
                     marker='>', color=colors[tsne_sequence.index(seqid)])
            # Noise in
            if sequence.noise_step is not None and seq_start < sequence.noise_step < seq_end:
                plt.plot(sequence.tsne_coords[sequence.noise_step, 0], sequence.tsne_coords[sequence.noise_step, 1],
                         marker='v', color=colors[tsne_sequence.index(seqid)])
            # Finish
            plt.plot(sequence.tsne_coords[seq_end-1, 0], sequence.tsne_coords[seq_end-1, 1],
                     marker='o', color=colors[tsne_sequence.index(seqid)])
            # If there's an error, put a red X there
            if sequence.first_error is not None and seq_start < sequence.first_error < seq_end:
                plt.plot(sequence.tsne_coords[sequence.first_error, 0], sequence.tsne_coords[sequence.first_error, 1],
                          marker='x', color='red')
            if annotate:
                # Annotate every point with the corresponding action (and subgoal? and goal?).
                for i, coords in enumerate(sequence.tsne_coords[seq_start:seq_end, :]):
                    target = sequence.targets[seq_start + i]
                    text = str(seq_start+i+1) #+ ':' + target.goal1_str[4:] + '/' + target.goal2_str[4:] + '/' + target.action_str[2:]
                    plt.annotate(text, coords)
    # Generate legends

    patches = []
    if tsne_goals:
        for index, label in enumerate(environment.GoalEnvData.goals1_list):
            patches.append(mpatches.Patch(color=colors[index], label=label[2:]))
    elif tsne_subgoals:
        for index, label in enumerate(environment.GoalEnvData.goals2_list):
            patches.append(mpatches.Patch(color=colors[index], label=label[2:]))
    elif tsne_actions:
        for index, label in enumerate(environment.GoalEnvData.actions_list):
            patches.append(mpatches.Patch(color=colors[index], label=label[2:]))
    elif tsne_errors:
        for index, label in enumerate(["Beyond first mistake", "First mistake", "Before first mistake"]):
            patches.append(mpatches.Patch(color=colors[-2-index], label=label))
    elif tsne_sequences:
        for index, sequence in enumerate(task.sequences_list):
            patches.append(mpatches.Patch(color=colors[index], label=sequence.name))
    if tsne_sequence:
        for seqid in tsne_sequence:
            patches.append(mlines.Line2D([], [], color=colors[tsne_sequence.index(seqid)], label='Sequence '+str(seqid)))
        patches.append(mlines.Line2D([], [], color="black", marker="o", label="Final action"))
        patches.append(mlines.Line2D([], [], color="black", marker=">", label="First action"))
        patches.append(mlines.Line2D([], [], color="black", marker="v", label="Noise injection"))
        patches.append(mlines.Line2D([], [], linestyle='', color="red", marker="x", label="First error"))

    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    # sns.scatterplot(
    #    x="x", y="y",
    #    hue="y",
    #    #palette=sns.color_palette("hls", 10),
    #    data=results_pd,
    #    legend="full",
    #    alpha=0.3
    # )
    plt.savefig(filename)


# used to identify perfect accuracy to stop the training
def stop_condition(model, noise=0., goal1_noise=0., goal2_noise=0., goals=True, num_tests=10,
                   sequence_ids=range(21), noise_per_step=False,
                   disruption_per_step=False, initialization=nn.SEMINORMAL, do_rdm=False):
    test_data = generate_test_data(model, noise=noise,
                                   goal1_noise=goal1_noise, goal2_noise=goal2_noise,
                                   goals=goals, num_tests=num_tests,
                                   sequence_ids=sequence_ids,
                                   noise_per_step=noise_per_step,
                                   disruption_per_step=disruption_per_step,
                                   initialization=initialization)
    tsne_results, test_data, total_errors = analyse_test_data(test_data, do_rdm=do_rdm, goals=goals)
    return total_errors == 0


def train(stop_params, model, goals=False,
          noise=0., sequences=None,
          context_initialization=nn.ZEROS):
    # Example models:
    #        model = nn.ElmanGoalNet(size_hidden=50, size_observation=29, size_action=19,
    #                                size_goal1=len(environment.GoalEnvData.goals1_list),
    #                                size_goal2=len(environment.GoalEnvData.goals2_list),
    #                                algorithm=nn.RMSPROP, learning_rate=learning_rate, initialization="uniform",
    #        model = nn.ElmanGoalNet(size_hidden=50, size_observation=29, size_action=19,
    #                                size_goal1=0,
    #                                size_goal2=0,
    #                                algorithm=nn.RMSPROP, learning_rate=learning_rate, initialization="uniform")

    if sequences is None:
        sequences = [0]
    env = environment.GoalEnv()

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_fullseq = 0.
    rng_avg_goals1 = 0.
    rng_avg_goals2 = 0.

    iteration = 0
    while not stop_params.is_time_to_stop(model, iteration):
        seqid = np.random.choice(sequences)
        sequence = task.sequences_list[seqid]
        env.reinitialize(sequence.initial_state)

        # run the network
        with tf.GradientTape() as tape:
            model.new_episode()
            # Initialize context with random/uniform values.
            if model.nonlinearity not in [nn.SIGMOID, nn.TANH, nn.RELU]:
                raise(NotImplementedError("Only sigmoid, tanh, and ReLu activation functions are supported"))
            if context_initialization == nn.UNIFORM:
                if model.nonlinearity == nn.SIGMOID:
                    model.context = np.random.uniform(0.01, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
                elif model.nonlinearity == nn.TANH:
                    model.context = np.random.uniform(-0.99, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
                elif model.nonlinearity == nn.RELU:
                    # 1.0 is an arbitrary limit, but we can't be uniform over infty
                    model.context = np.random.uniform(0.0, 1., (1, model.size_hidden)).astype(dtype=np.float32)
            elif context_initialization == nn.NORMAL:
                if model.nonlinearity == nn.SIGMOID or model.nonlinearity == nn.RELU:
                    raise(Exception("Normal initialization incompatible with SIGMOID or RELU nonlinearity"))
                model.context = np.random.normal(0.01, 0.1, (1, model.size_hidden)).astype(dtype=np.float32)
            elif context_initialization == nn.SEMINORMAL:
                if model.nonlinearity == nn.TANH:
                    raise(Exception("seminormal initialization incompatible with tanh nonlinearity"))
                model.context = np.random.normal(0.0, 0.1, (1, model.size_hidden)).astype(dtype=np.float32)
                model.context[model.context < 0.01] = 0.  # it's impossible for a sigmoid activation to be <0
            elif context_initialization == nn.ZEROS:
                model.context = np.zeros((1, model.size_hidden)).astype(dtype=np.float32)

            # Set up the prior actions and goals to what they OUGHT to be
            model.action = np.zeros_like(sequence.targets[0].action_one_hot)
            if goals:
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
                if goals:
                    model.goal1 = copy.deepcopy(target.goal1_one_hot)
                    model.goal2 = copy.deepcopy(target.goal2_one_hot)
                model.action = copy.deepcopy(target.action_one_hot)

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
                targets = sequence.targets

            else: # No goals: set target goals to None
                targets = copy.deepcopy(sequence.targets)
                for target in targets:
                    target.goal1_str = None
                    target.goal2_str = None
            # Train model, record loss.
            loss = model.train(tape, targets)

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
        if (iteration < 1000 and iteration in [3 ** n for n in range(50)]) or iteration % 1000 == 0:
            print("{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}".format(
                    iteration, rng_avg_loss, rng_avg_actions, rng_avg_fullseq, rng_avg_goals1, rng_avg_goals2))

        iteration += 1
    return model


