import utils
import numpy as np
from numpy import linalg
import tensorflow as tf
from goalenv import environment, task
from neural import neuralnet as nn, layers
import scripts
import copy
import analysis
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
import time
import random
import rdm

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
    # action loss
    action_loss = tf.nn.softmax_cross_entropy_with_logits(target.action_one_hot, model.h_action_activation[-1])
    total_loss = action_loss
    goal_loss = action_loss * 0. #just so it's
    if target.goal1_one_hot is not None:
        goal1_loss = tf.nn.softmax_cross_entropy_with_logits(target.goal1_one_hot, model.h_goal1_activation[-1])
        total_loss += goal1_loss
        goal_loss += goal1_loss
    if target.goal2_one_hot is not None:
        goal2_loss = tf.nn.softmax_cross_entropy_with_logits(target.goal2_one_hot, model.h_goal2_activation[-1])
        total_loss += goal2_loss
        goal_loss += goal2_loss
    if include_regularization:
        regularization_loss = model.L2_regularization * sum([tf.reduce_sum(weights**2) for weights in model.all_weights])
        total_loss += regularization_loss

    return total_loss.numpy()[0], action_loss.numpy()[0], goal_loss.numpy()[0]


def generate_test_data(model, sequence_ids, noise=0., goal1_noise=0., goal2_noise=0., num_tests=10, goals=False,
                       initialization=utils.SEMINORMAL, verbose=False,
                       goal_multiplier=1,
                       noise_per_step=True,
                       disruption_per_step=False,
                       noise_per_step_to_input=False,
                       switch_sequence=None,
                       switch_goal1=None, #otherwise, should be (list[timesteps], ndarray[new goal]])
                       switch_goal2=None, #likewise
                       lesion_goal1_units=False,
                       lesion_goal2_units=False,
                       lesion_action_units=False,
                       lesion_observation_units=False,
                       single_step_noise=None,
                       clamped_goals=False,
                       constant_noise=0.,
                       constant_noise_to_input=0.,
                       noise_per_step_to_hidden=0.,
                       hidden_goal_multiplier=1,
                       gain_multiplier = 1.,
                       gain_multiplier_from=0,
                       gain_multiplier_to=None):
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
    seq_to_test = 0
    for seqid, sequence in zip(sequence_ids, sequences):
        if verbose:
            print("testing sequence: "+str(seqid))
        outputs_per_noise_step = []
        num_runs = 1 #sequence.length if (noise_per_step or disruption_per_step or noise_per_step_to_input or noise_per_step_to_hidden) else 1

        for run in range(num_runs):
            if single_step_noise is not None:
                noise_step = single_step_noise
            elif noise_per_step or disruption_per_step or noise_per_step_to_input or noise_per_step_to_hidden:
                noise_step = run
            else:
                noise_step = 0
            outputs = []
            for i in range(num_tests):
                noise_step = np.random.randint(0, sequence.length)
                sequence_solutions = [alt.targets for alt in sequence.alt_solutions] + [sequence.targets]
                # Initialize the sequence.
                init_state = sequence.initial_state
                # Set up the current state to be 0.
                #init_state.current.o_ddairy_first = 0  #what's this?? Why bother? This makes no sense!!
                #Just use the correct instructions all the time, and remove the alternative sequences. What is wrong with me.
                env.reinitialize(init_state)
                model.action = np.zeros((1, model.size_action), dtype=np.float32)

                # run the network
                with tf.GradientTape() as tape:
                    model.new_episode()
                    # Initialize context with random/uniform values.
                    if initialization == utils.UNIFORM:
                        model.context = np.random.uniform(0.01, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
                        model.action = np.zeros_like(sequence.targets[0].action_one_hot)
                        if goals:
                            model.goal1 = np.zeros_like(sequence.targets[0].goal1_one_hot)
                            model.goal2 = np.zeros_like(sequence.targets[0].goal2_one_hot)
                    elif initialization == utils.SEMINORMAL:
                        model.context = np.random.normal(0.0, 0.1, (1, model.size_hidden)).astype(dtype=np.float32)
                        model.context[model.context < 0.0] = 0.  # it's impossible for a sigmoid/relu activation to be <0
                        model.action = np.zeros_like(sequence.targets[0].action_one_hot)
                        if goals:
                            model.goal1 = np.zeros_like(sequence.targets[0].goal1_one_hot)
                            model.goal2 = np.zeros_like(sequence.targets[0].goal2_one_hot)
                    elif initialization == utils.ZERO_INIT:
                        model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
                        model.action = np.zeros_like(sequence.targets[0].action_one_hot)
                        if goals:
                            model.goal1 = np.zeros_like(sequence.targets[0].goal1_one_hot)
                            model.goal2 = np.zeros_like(sequence.targets[0].goal2_one_hot)
                    else:
                        raise NotImplementedError("Initialization method not implemented")

                    output_sequence = task.BehaviorSequence(sequence.initial_state)
                    output_sequence.activations = []
                    output_sequence.losses = [] #total losses
                    output_sequence.losses_actions = []
                    output_sequence.losses_goals = []
                    for j in range(max_length):
                        observation = env.observe()
                        # Do this before adding noise, to avoid multiplying the noise.
                        #if not isinstance(model.context, np.ndarray):
                        #    model.context = model.context.numpy()  # Doesn't matter here, we're not going to backpropagate thru that.
                        #model.context[0, :model.context.shape[1]//2] *= hidden_goal_multiplier
                        if goals and j >= noise_step and j <= noise_step + 2:
                            model.goal1 *= goal_multiplier
                            model.goal2 *= goal_multiplier

                        # Add noise to context layer
                        if j == noise_step:
                            if noise_per_step or single_step_noise:
                                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                                if goals:
                                    model.goal1 += np.float32(np.random.normal(0., goal1_noise, size=(1, model.size_goal1)))
                                    model.goal2 += np.float32(np.random.normal(0., goal2_noise, size=(1, model.size_goal2)))
                            if noise_per_step_to_input:
                                observation += np.float32(np.random.normal(0., noise, size=(1, model.size_observation)))
                            if disruption_per_step:
                                # Change the state
                                env.state = disrupt_state(env.state, initial_state=sequence.initial_state,
                                                          #mode=NOISIFY)
                                                          mode=FLASHBANG)
                                                          #mode=REINITIALIZE)
                                                          #mode=HOLD_RANDOM_OBJECT)
                        # constant level of noise
                        model.context += np.float32(np.random.normal(0., constant_noise, size=(1, model.size_hidden)))
                        observation += np.float32(np.random.normal(0., constant_noise_to_input, size=(1, model.size_observation)))

                        if seq_to_test == switch_sequence or switch_sequence is None:
                            if switch_goal1 is not None and j in switch_goal1[0]:
                                model.goal1 = copy.deepcopy(switch_goal1[1]) * goal_multiplier
                            if switch_goal2 is not None and j in switch_goal2[0]:
                                model.goal2 = copy.deepcopy(switch_goal2[1]) * goal_multiplier

                        if clamped_goals and j>0 and j<=len(sequence.targets):
                            # Take into account the alternative sequences.
                            error = True
                            # Check whether there's any goal that matches.
                            for sol in sequence_solutions:
                                if np.all(model.goal1 == sol[j - 1].goal1_one_hot) and \
                                   np.all(model.goal2 == sol[j - 1].goal2_one_hot):
                                    error = False
                            # If nothing matches...
                            if error:
                                #print("error")
                                # reset goals. Had overflows so I'm guessing this led to goal1_one_hot being multiplied
                                # which over many sequences would lead to a crazy big goal. Which obviously would make all results wrong.
                                model.goal1 = copy.deepcopy(sequence_solutions[0][j-1].goal1_one_hot)
                                model.goal2 = copy.deepcopy(sequence_solutions[0][j-1].goal2_one_hot)

                            # Remove any incorrect sequences: meaning any where the current goal1 and goal2 don't match.
                            sequence_solutions[:] = [seq for seq in sequence_solutions
                                                     if (np.all(seq[j-1].goal1_one_hot == model.goal1) and
                                                         np.all(seq[j-1].goal2_one_hot == model.goal2))]


                        if lesion_goal1_units:
                            model.goal1 *= 0.
                        if lesion_goal2_units:
                            model.goal2 *= 0.
                        if lesion_action_units:
                            model.action *= 0.
                        if lesion_observation_units:
                            observation *= 0.

                        if j == noise_step:
                            model.feedforward(observation, noise_to_hidden=noise_per_step_to_hidden+constant_noise)#, gain_multiplier=gain_multiplier,
                                              #gain_multiplier_from=gain_multiplier_from,
                                              #gain_multiplier_to=gain_multiplier_to)
                        else:
                            model.feedforward(observation, noise_to_hidden=constant_noise)

                        if j < len(sequence.targets):  # after that it's not defined.
                            target = sequence.targets[j] if goals else sequence.targets_nogoals[j]
                            loss, action_loss, goal_loss = compute_last_step_loss(model, target, include_regularization=False)
                            output_sequence.losses.append(loss)
                            output_sequence.losses_actions.append(action_loss)
                            output_sequence.losses_goals.append(goal_loss)
                        else:
                            output_sequence.losses.append(None)  # loss is undefined
                            output_sequence.losses_actions.append(None)
                            output_sequence.losses_goals.append(None)

                        output_sequence.activations.append(model.context)
                        # if there's an impossible action, ignore it and continue.
                        next_state = copy.deepcopy(env.state.next)
                        try:
                            env.do_action(model.h_action_collapsed[-1])

                        except environment.ActionException as error:  # The action doesn't make sense and cannot be executed in the environment.
                            if verbose:
                                print(error)
                            # maintain the current state when an impossible action is attempted
                            env.state.next = next_state
                        action_str = utils.onehot_to_str(model.h_action_collapsed[-1], environment.GoalEnvData.actions_list)
                        if action_str in environment.TERMINAL_ACTIONS:
                            break  # we said "done", we're done.

                    # Get some statistics about the sequences actually observed. Is it a recognized sequence? If not,
                    # What kind of mistake appeared?
                    output_sequence.set_targets(model.h_goal1_collapsed, model.h_goal2_collapsed, model.h_action_collapsed)
                    output_sequence.id = seq_counter
                    output_sequence.target_seq_id = seqid
                    output_sequence.noise_step = noise_step
                    seq_counter += 1
                    outputs.append(output_sequence)
                    model.clear_history()
            outputs_per_noise_step.append(outputs)
        outputs_per_sequence.append(outputs_per_noise_step)
        seq_to_test +=1
    return outputs_per_sequence


HOLD_RANDOM_OBJECT = 1
REINITIALIZE = 2
NOISIFY = 3
FLASHBANG = 4
def disrupt_state(state, initial_state, mode=HOLD_RANDOM_OBJECT):
    c = state.current
    n = state.next
    if mode == HOLD_RANDOM_OBJECT:
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
    elif mode == NOISIFY:
        raise NotImplementedError()
    elif mode == FLASHBANG: # Set all fields to 1
        c._set_fields('o_f', np.ones(15))
        c._set_fields('o_h', np.ones(8))
        n._set_fields('o_f', np.ones(15))
        n._set_fields('o_h', np.ones(8))
        return state


error_testing_labels = ["total sequences", "correct", "incorrect",
                       "action errors", "subseq errors", "fullseq errors",
                       "steps noise->error",
                       "loss avg before noise"] + \
                       ["loss on and after noise (no error)"] + [str(i) for i in range(1, 55)] + \
                       ["loss on and after noise (no error ACTIONS)"] + [str(i) for i in range(1, 55)] + \
                       ["loss on and after noise (no error GOALS)"] + [str(i) for i in range(1, 55)] + \
                       ["loss on and after noise (error)"] + [str(i) for i in range(1, 55)] + \
                       ["loss on and after noise (error ACTIONS)"] + [str(i) for i in range(1, 55)] + \
                       ["loss on and after noise (error GOALS)"] + [str(i) for i in range(1, 55)] + \
                       ["loss error->noise (reverse time)"] + [str(i) for i in range(1, 55)] + \
                       ["loss noise min1", "loss on noise"]

# RDM sorts
NONE = "none"
GOAL = "goal"
SUBGOAL = "subgoal"
ACTION = "action"
def analyse_test_data(test_data, goals=True, do_rdm=False, rdm_sort=NONE, do_tsne=False, do_loss=False,
                      mds_sequences=None, mds_range=None, noise_steps=None, one_rdm=True,
                      do_special_representations=True, do_dimensionality=True,
                      verbose=False, append_to_file=None):
    sequence_ids = range(len(test_data))


    if append_to_file is not None:
        myfile = open(append_to_file, "a")

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
    goal_errors = 0

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
                all_targets_goals = [target_sequence] + target_sequence.alt_solutions
                trial.first_error = None  # This trial is error-free
                prev_goal_error = False
                for j, action in enumerate(trial.get_actions_one_hot()):
                    # TODO: Remove this
                    # Just for the sake of this one figure
                    #if target_sequence.alt_solutions[0] not in all_targets:
                    #    break

                    error = True
                    # It's only an error if it's an error for every target...
                    # BUT this misses situations in which the two sequences are somehow tangled together.
                    # Solution: as soon as a target sequence no longer matches, remove it from the list of targets.
                    invalid = []
                    for k, target in enumerate(all_targets):
                        if np.array_equal(action, target.get_actions_one_hot()[j]):
                            # No error in action, but there might be a goal error.
                            error = False
                        else:
                            invalid.append(target)

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
                    for inv in invalid:
                        all_targets.remove(inv)
                        if inv in all_targets_goals:
                            all_targets_goals.remove(inv)
                    #for inv in reversed(invalid):
                    #    all_targets.pop(inv)

                    # Look for goal errors too... Are the goals compatible with any current goals?
                    if goals:
                        invalid = []
                        goal_error = True
                        for k, target in enumerate(all_targets_goals):
                            if np.array_equal(trial.get_goals1_one_hot()[j], target.get_goals1_one_hot()[j]) and \
                                    np.array_equal(trial.get_goals2_one_hot()[j], target.get_goals2_one_hot()[j]):
                                goal_error = False
                            if goal_error:
                                invalid.append(target)
                        for inv in invalid:
                            all_targets_goals.remove(inv)
                        if goal_error and not prev_goal_error:  # Only count the goal error once per sequence
                            goal_errors += 1
                            prev_goal_error = True

                    if error:  # If we found an error we can terminate here
                        break

                if trial.first_error is None:
                    total_correct_seq += 1
                total_trials += 1

            # Need to color the step at which noise is introduced.
            if VERBOSE:
                print("Target sequence " + str(i) + " (" + target_sequence.name + ") noise at step " + str(noise_step) + ": errors = " + str(first_error))
    total_fullseq_errors = num_is_a_target
    total_error = total_trials - total_correct_seq
    total_subseq_error = num_replaced
    total_action_errors = total_error - total_subseq_error
    # There's still a bug in the error counts.
    if verbose:
        print(
            "Overall totals: {0}/{1} correct. {2} errors, of which:\n Action errors:{3}\n Subsequence errors: {4}\n Full sequence errors {5},\n Goal errors {6}\n".format(
                total_correct_seq,
                total_trials,
                total_trials - total_correct_seq,
                total_action_errors, # action errors.
                total_subseq_error - total_fullseq_errors, # Are all full sequence errors ALSO subsequence errors?
                total_fullseq_errors,
                goal_errors
            ))

        print("Overall stats:\n errors on noise step: {0}\n errors on transition: {1}\n average steps noise->error: {2}\n".format(
            error_on_noise, error_on_transition, steps_noise_to_error/num_errors if num_errors>0 else -1
        ))
    else:
        print("Overall totals: {0}/{1} correct, goal errors={2}.".format(total_correct_seq, total_trials, goal_errors))
    #return [total_correct_seq, total_action_errors, total_subseq_error - total_fullseq_errors, total_fullseq_errors, goal_errors]
    if append_to_file is not None:
        myfile.write("{0};{2};{3};{4};{5};{6};\n".format(
                total_correct_seq,
                total_trials - total_correct_seq,
                total_action_errors, # action errors.
                total_subseq_error - total_fullseq_errors, # Are all full sequence errors ALSO subsequence errors?
                total_fullseq_errors,
                goal_errors,
                error_on_transition
            ))
        myfile.close()

    error_testing_results = None
    if do_loss:
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
        loss_after_noise_no_error_goals = np.zeros(55)
        loss_after_noise_no_error_actions = np.zeros(55)
        loss_after_noise_error = np.zeros(55)
        loss_after_noise_error_goals = np.zeros(55)
        loss_after_noise_error_actions = np.zeros(55)
        loss_before_error = np.zeros(55)

        loss_noise_min1 = 0
        loss_on_noise = 0
        counter_noise = 0
        counter_noise_min1 = 0

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

            if sequence.noise_step is not None and len(sequence.losses) > sequence.noise_step:
                if sequence.noise_step > 0 and len(sequence.losses) > sequence.noise_step - 1:
                    loss_noise_min1 += sequence.losses[sequence.noise_step - 1]
                    counter_noise_min1 += 1
                loss_on_noise += sequence.losses[sequence.noise_step]
                counter_noise += 1

            if noise_steps is not None:
                if sequence.first_error is None or (sequence.first_error is not None and sequence.first_error - sequence.noise_step != noise_steps):
                    continue  # Next sequence
                else:
                    print('1 seq')

            if sequence.first_error is not None:
                first_error_loss_total += sequence.losses[sequence.first_error]
                counter_first_error += 1

            # noise step can be > sequence length when the sequence terminated early.
            if sequence.noise_step is not None and sequence.noise_step < sequence.length:
                noise_loss_total += sequence.losses[sequence.noise_step]
                counter_noise += 1

                if sequence.first_error is None: # No error occurred.
                    # Record loss between noise and error
                    i = 0
                    while sequence.noise_step + i < sequence.length:
                        id = sequence.noise_step+i
                        try:
                            loss = sequence.losses[id]
                            loss_goals = sequence.losses_goals[id]
                            loss_actions = sequence.losses_actions[id]
                            loss_after_noise_no_error[i] += loss
                            loss_after_noise_no_error_goals[i] += loss_goals
                            loss_after_noise_no_error_actions[i] += loss_actions
                        except:
                            print("what what")
                            raise Exception()
                        counter_loss_after_noise_no_error[i] += 1
                        i += 1
                elif sequence.first_error is not None: # an error occurred.
                    i = 0
                    while sequence.noise_step + i < sequence.length and sequence.noise_step + i <= sequence.first_error:
                        id = sequence.noise_step+i
                        loss = sequence.losses[id]
                        loss_goals = sequence.losses_goals[id]
                        loss_actions = sequence.losses_actions[id]

                        loss_after_noise_error[i] += loss
                        loss_after_noise_error_goals[i] += loss_goals
                        loss_after_noise_error_actions[i] += loss_actions

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
        loss_on_noise = loss_on_noise / counter_noise if counter_noise != 0 else 0
        loss_noise_min1 = loss_noise_min1 / counter_noise_min1 if counter_noise_min1 != 0 else 0
        loss_after_noise_error = np.divide(loss_after_noise_error, counter_loss_after_noise_error, out=np.zeros_like(loss_after_noise_error),
                                      where=counter_loss_after_noise_error!=0)
        loss_after_noise_no_error = np.divide(loss_after_noise_no_error, counter_loss_after_noise_no_error,
                                         out=np.zeros_like(loss_after_noise_no_error),
                                         where=counter_loss_after_noise_no_error!=0)

        loss_after_noise_error_actions = np.divide(loss_after_noise_error_actions, counter_loss_after_noise_error, out=np.zeros_like(loss_after_noise_error_actions),
                                      where=counter_loss_after_noise_error!=0)
        loss_after_noise_no_error_actions = np.divide(loss_after_noise_no_error_actions, counter_loss_after_noise_no_error,
                                         out=np.zeros_like(loss_after_noise_no_error_actions),
                                         where=counter_loss_after_noise_no_error!=0)

        loss_after_noise_error_goals = np.divide(loss_after_noise_error_goals, counter_loss_after_noise_error, out=np.zeros_like(loss_after_noise_error_goals),
                                      where=counter_loss_after_noise_error!=0)
        loss_after_noise_no_error_goals = np.divide(loss_after_noise_no_error_goals, counter_loss_after_noise_no_error,
                                         out=np.zeros_like(loss_after_noise_no_error_goals),
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

        # Return results
        error_testing_results = [total_trials, total_correct_seq, total_trials - total_correct_seq,  # errors
                total_action_errors, # action errors.
                total_subseq_error - total_fullseq_errors, # subsequence errors
                total_fullseq_errors, # full sequence errors
                steps_noise_to_error/num_errors if num_errors>0 else -1,  # average step to noise->error
                loss_prior_noise_avg] + \
                loss_after_noise_no_error.tolist() +\
                loss_after_noise_no_error_actions.tolist() +\
                loss_after_noise_no_error_goals.tolist() +\
                loss_after_noise_error.tolist() +\
                loss_after_noise_error_actions.tolist() +\
                loss_after_noise_error_goals.tolist() +\
                loss_before_error.tolist() +\
                [loss_noise_min1, loss_on_noise]

    #print("LOSS:\n average loss={0:.2f}\n loss on first error={1:.2f}\n loss on noise={2:.2f} (+1: {3:.2f})\n loss_good_seq={4}\n loss_before_noise={5}".format(
    #                                                                            loss_avg,
    #                                                                            loss_first_error_avg,
    #                                                                            loss_prior_noise_avg))
    #if not do_rdm:
    #    return None, test_data, num_errors, error_testing_results

    ###################################
    # Analysis 4: RDM, MDS, and T-SNE #
    ###################################
    # 1. RDM - from EVERY SEQUENCE
    # 1.a. Make a list of every sequence step. That means flattening everything into a list of steps.

    outputs_sequences_flat = utils.flatten_onelevel(outputs_no_noise_step_distinction)
    activations = [seq.activations for seq in outputs_sequences_flat]
    activations_flat = utils.flatten_onelevel(activations)  # all this list wrangling is ugly as hell
    activations_flat_rdm = []
    properties_rdm = []
    if do_rdm:  # Do the RDM and MDS
        labels = []
        for seqid, seq in enumerate(outputs_sequences_flat):
            error = "true" if seq.first_error is not None else "false"
            for targetid, target in enumerate(seq.targets):
                label = ""
                label += "Seq: " + str(seqid)
                label += " - " + str(targetid + 1)
                label += ": " + target.goal1_str + '-' + target.goal2_str + '-' + target.action_str
                labels.append(label)
                dict = {
                       "seq":str(seqid),
                       "step": str(targetid+1),
                       "goal1":target.goal1_str,
                       "goal2":target.goal2_str,
                       "action":target.action_str,
                       "error": error}
                properties_rdm.append(dict)

        if one_rdm:
            for i, tensor in enumerate(activations_flat):
                activations_flat_rdm.append(tensor.numpy().reshape(-1))

            # Generate the RDM... That's actually very expensive computationally
            #rdmatrix = rdm.rdm_euclidian(activations_flat_rdm)
            my_rdm = rdm.rdm(properties_rdm, vectors=activations_flat_rdm, type=rdm.EUCLIDIAN)
            #targets = utils.flatten_onelevel([sequence.targets for sequence in outputs_sequences_flat])
            #rdmatrix, labels, _ = reorder_rdm(rdmatrix, labels, targets, mode=rdm_sort)

            my_rdm.save(labels=labels, filename="rdm_goals", title="kitchen env goals: euclidian matrix", image=True, csv=True, figsize=50, fontsize=0.5)
            plt.clf()
        else: #two rdms
            activations_goals = []
            activations_actions = []
            for i, tensor in enumerate(activations_flat):
                activations = tensor.numpy().reshape(-1)
                activations_goals.append(activations[0:len(activations)//2])
                activations_actions.append(activations[len(activations)//2:])

            for x in [(activations_goals, "goals"), (activations_actions, "actions")]:
                # Generate the RDM... That's actually very expensive computationally
                rdmatrix = rdm.rdm_euclidian(x[0])
                targets = utils.flatten_onelevel([sequence.targets for sequence in outputs_sequences_flat])
                rdmatrix, labels, _ = reorder_rdm(rdmatrix, labels, targets, rdm_sort)
                rdm.save(rdmatrix, labels=labels, filename="rdm_side"+x[1],
                                  title="kitchen env goals gradient: euclidian matrix - "+x[1], image=True, csv=True, figsize=50,
                                  fontsize=0.5)
                plt.clf()
        print("rdm done")
        # Generate the MDS from the RDM.
        mdsy = analysis.mds(my_rdm.matrix)
        # Display the MDS! Ugh!
        # Make labels!!
        print("mds done")

        mdsy_idx = 0
        colors = mcolors.CSS4_COLORS
        for i, seq in enumerate(outputs_sequences_flat):
            length = seq.length
            if mds_range is not None and length > mds_range:
                length = mds_range
            if mds_sequences is not None and i in mds_sequences:
                labels = []
                for j in range(length):
                    labels.append("seq "+ str(i) + ": " + str(j+1))
                analysis.plot_mds_points(mdsy[mdsy_idx:mdsy_idx + length], range(length), labels=None,#labels,
                                         style=list(colors.values())[(i)%len(colors)], fontsize=32)
                mdsy_idx += length
        plt.title("MDS")
        #plt.show()
        plt.savefig("mds")
        plt.clf()

    if do_dimensionality:
        # Turn activations into a matrix or matrices
        matrices = []
        for tensor_list in activations:
            vect_list = []
            # Stack those as vectors into a numpy matrix
            for tensor in tensor_list:
                vect_list.append(tensor.numpy())
            matrix = np.concatenate(vect_list, axis=0)
            matrices.append(matrix)  #matrix = matrix[:, 0:10]
            #matrix = np.transpose(matrix)
            #print(matrix)
            # compute the rank with various tolerances
            #print("rank per tolerance:")
            #for tolerance in [0.01, 0.1, 1.0, 1.5, 3, 5, 10., 15, 30, 100]:
            #    print("tol={0}, rank={1}".format(tolerance, np.linalg.matrix_rank(matrix, tol=tolerance)))
        # Now do this for all the sequences combined
        matrix = np.concatenate(matrices, axis=0)
        average = np.average(matrix)
        #print("rank per tolerance (combined):")
        #for tolerance in [0.01, 0.1, 1.0, 1.5, 3, 5, 10., 15, 30, 100]:
        #    print("tol={0}, rank={1}".format(tolerance*average, np.linalg.matrix_rank(matrix, tol=tolerance*average)))
        #print("average activation value")
        #print(average)
        #np.
        # Compute rank of activation matrix for a given sequence.
        #np.linalg.matrix_rank(mat, tol=0.1)
        #print dimensionality
        # Total - all sequences
        # Per sequence
        # Over a specific subsequence??

    if do_tsne:
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
    else:
        tsne_results = None

    return tsne_results, test_data, num_errors, error_testing_results, goal_errors, activations_flat_rdm, properties_rdm  # Test data is enriched during analysis (first error step)


def reorder_rdm(rdm, labels, targets, mode=ACTION):
    if mode == ACTION:
        steps = [target.action_str for target in targets]
    elif mode == SUBGOAL:
        steps = [target.goal2_str for target in targets]
    elif mode == GOAL:
        steps = [target.goal1_str for target in targets]
    elif mode == GOAL + SUBGOAL + ACTION:
        # Sort first by action, then by subgoal. That way actions are grouped by which subgoal they match.
        rdm, labels, targets = reorder_rdm(rdm, labels, targets, mode=ACTION)  # In principle resorting over actions, ESPECIALLY with a fixed labels
        rdm, labels, targets = reorder_rdm(rdm, labels, targets, mode=SUBGOAL)
        return reorder_rdm(rdm, labels, targets, mode=GOAL)
    else: # nothing to do
        return rdm

    rdm = rdm.tolist()
    # Sort the labels and targets so we can keep track of what goes where
    # essential to use the lambda as otherwise sorted will sort first by key 1, then by key 2, which causes problems
    # (sort is no longer stable which means we can get different permutations for the rows vs. the columns of the RDM!!)
    labels = [label for _, label in sorted(zip(steps, labels), key=lambda zipped:zipped[0])]
    targets = [target for _, target in sorted(zip(steps, targets), key=lambda zipped: zipped[0])]
    # Sort for the rows
    rdm = [rdmrow for _, rdmrow in sorted(zip(steps, rdm), key=lambda zipped:zipped[0])]
    # transpose the list-rdm
    rdm = np.transpose(np.array(rdm)).tolist()
    # Sort again for the columns
    rdm = [rdmrow for _, rdmrow in sorted(zip(steps, rdm), key=lambda zipped:zipped[0])]
    # put the rdm back in 2d ndarray form
    rdm = np.array(rdm)
    # No need to retranspose again since an rdm is symmetrical anyway
    return rdm, labels, targets


def plot_tsne(tsne_results, test_data, tsne_goals=False, tsne_subgoals=False, tsne_actions=False, tsne_sequences=False,
              tsne_errors=False, tsne_sequence=[], tsne_sequence_interval=[], filename="tsne", annotate=False, save_txt=False):
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

    # Save tsne points in csv, one line for x, one line for y
    if save_txt:
        np.savetxt(filename+"_tsne.txt", tsne_results, delimiter="\t", fmt='%.2e')

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
                seq_start = 0
                seq_end = sequence.length
            else:
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


NUM_TIMESTEPS = 899
# used to identify perfect accuracy to stop the training
BONUS_ITERATIONS = 0
def stop_condition(model, noise=0., goal1_noise=0., goal2_noise=0., goals=True, num_tests=1,
                   sequence_ids=range(21), noise_per_step=False,
                   disruption_per_step=False, context_initialization=utils.SEMINORMAL, do_rdm=False):
    global BONUS_ITERATIONS
    test_data = generate_test_data(model, noise=noise,
                                   goal1_noise=goal1_noise, goal2_noise=goal2_noise,
                                   goals=goals, num_tests=num_tests,
                                   sequence_ids=sequence_ids,
                                   noise_per_step=noise_per_step,
                                   disruption_per_step=disruption_per_step,
                                   initialization=context_initialization,
                                   verbose=False)
    tsne_results, test_data, total_errors, _, goal_errors, _, _ = analyse_test_data(test_data, do_rdm=do_rdm, goals=goals, verbose=False)
    #return total_errors + goal_errors < NUM_TIMESTEPS / 100  # (8 errors or less, or 1% of errors). TODO triple check
    no_errors = total_errors + goal_errors == 0
    if no_errors and BONUS_ITERATIONS == 1:
        BONUS_ITERATIONS = 0
        return True
    elif no_errors:
        BONUS_ITERATIONS = 1
    else:
        BONUS_ITERATIONS = 0
    return False

def train(stop_params, model, goals=False,
          noise=0., sequences=None,
          context_initialization=nn.ZEROS,
          gradient=False,
          reg_strength=0.001,
          reg_increase="linear"):
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
            if model.nonlinearity not in [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu]:
                raise(NotImplementedError("Only sigmoid, tanh, and ReLu activation functions are supported"))
            if context_initialization == nn.UNIFORM:
                if model.nonlinearity == tf.nn.sigmoid:
                    model.context = np.random.uniform(0.01, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
                elif model.nonlinearity == tf.nn.tanh:
                    model.context = np.random.uniform(-0.99, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
                elif model.nonlinearity == tf.nn.relu:
                    # 1.0 is an arbitrary limit, but we can't be uniform over infty
                    model.context = np.random.uniform(0.0, 1., (1, model.size_hidden)).astype(dtype=np.float32)
            elif context_initialization == utils.NORMAL:
                if model.nonlinearity == tf.nn.sigmoid or model.nonlinearity == tf.nn.relu:
                    raise(Exception("Normal initialization incompatible with SIGMOID or RELU nonlinearity"))
                model.context = np.random.normal(0.01, 0.1, (1, model.size_hidden)).astype(dtype=np.float32)
            elif context_initialization == utils.SEMINORMAL:
                if model.nonlinearity == tf.nn.tanh:
                    raise(Exception("seminormal initialization incompatible with tanh nonlinearity"))
                model.context = np.random.normal(0.0, 0.1, (1, model.size_hidden)).astype(dtype=np.float32)
                model.context[model.context < 0.01] = 0.  # it's impossible for a sigmoid activation to be <0
            elif context_initialization == nn.ZEROS:
                model.context = np.zeros((1, model.size_hidden)).astype(dtype=np.float32)

            # Set up the prior actions and goals to what they OUGHT to be. Which is just zeros here.
            model.action = np.zeros_like(sequence.targets[0].action_one_hot)
            if goals:
                model.goal1 = np.zeros_like(sequence.targets[0].goal1_one_hot)
                model.goal2 = np.zeros_like(sequence.targets[0].goal2_one_hot)
            # Run the entire sequence
            for i, target in enumerate(sequence.targets):
                # Add noise to context layer
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                # Observe and act
                observation = env.observe()
                model.feedforward(observation)
                # Ignore the selected action. Do the correct action anyway.
                env.do_action(target.action_one_hot)
                # Ignore the selected goal units. Use the correct recurrent units:
                if goals:
                    model.goal1 = copy.deepcopy(target.goal1_one_hot)
                    model.goal2 = copy.deepcopy(target.goal2_one_hot)
                model.action = copy.deepcopy(target.action_one_hot)

            # Get some statistics about the percentage of correct behavior
            actions = np.array(model.h_action_collapsed).reshape((-1, environment.GoalEnvData.num_actions))
            target_actions = sequence.get_actions_one_hot()
            ratio_actions = scripts.ratio_correct(actions, target_actions)
            if goals:
                goals1 = np.array(model.h_goal1_collapsed).reshape((-1, environment.GoalEnvData.num_goals1))
                target_goals1 = sequence.get_goals1_one_hot()
                ratio_goals1 = scripts.ratio_correct(goals1, target_goals1)

                goals2 = np.array(model.h_goal2_collapsed).reshape((-1, environment.GoalEnvData.num_goals2))
                target_goals2 = sequence.get_goals2_one_hot()
                ratio_goals2 = scripts.ratio_correct(goals2, target_goals2)
                targets = sequence.targets

            else: # No goals: set target goals to None
                targets = copy.deepcopy(sequence.targets)
                for target in targets:
                    target.goal1_str = None
                    target.goal2_str = None

            extra_loss = 0
            if goals and gradient:
                # Train model, record loss.
                #cols = model.size_hidden
                size_input_to_hidden = model.size_hidden + model.size_goal1 + model.size_goal2 + model.size_action + model.size_observation
                size_output = model.size_goal1 + model.size_goal2 + model.size_action
                # Regularization in the hidden layer weights
                # Recurrent hidden to hidden connections
                #print("recurrent")
                extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                              [0, model.size_hidden], [0, model.size_hidden],
                                                              reg_strength, reg_type="recurrent", reg_increase=reg_increase)

                # Prev action to hidden
                #print("action-->hidden")
                extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                               [model.size_hidden+model.size_observation, model.size_hidden + model.size_observation+model.size_action],
                                                               [0, model.size_hidden],
                                                               reg_strength, reg_type="input_right", reg_increase=reg_increase)
                # prev subgoal to hidden
                #print("subgoal-->hidden")
                extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                                     [model.size_hidden + model.size_observation + model.size_action,
                                                                      model.size_hidden + model.size_observation + model.size_action + model.size_goal2],
                                                                     [0, model.size_hidden],
                                                                     reg_strength, reg_type="input_left",
                                                                     reg_increase=reg_increase)#,
                                                                     #middle=0.25)
                # Prev goal to hidden
                #print("goal-->hidden")
                extra_loss += utils.weight_regularization_calculator(model.hidden_layer.w,
                                                              [model.size_hidden + model.size_observation + model.size_action + model.size_goal2,
                                                               model.size_hidden + model.size_observation + model.size_action + model.size_goal2 + model.size_goal1],
                                                              [0, model.size_hidden],
                                                              reg_strength, reg_type="input_left", reg_increase=reg_increase)

                # SWITCHED OUTPUT LEFT AND OUTPUT RIGHT.
                # Regularization in the output layers (goals and actions) weights
                # Layer looks like:
                # [goals.....subgoals......|..................actions]
                # hidden to next action
                #print("hidden->action")
                extra_loss += utils.weight_regularization_calculator(model.action_layer.w,
                                                               [0, model.size_hidden], [0, model.size_action],
                                                               reg_strength, reg_type="output_right", reg_increase=reg_increase)

                # hidden to next subgoal
                #print("hidden->subgoal")
                extra_loss += utils.weight_regularization_calculator(model.goal2_layer.w,
                                                               [0, model.size_hidden], [0, model.size_goal2],
                                                               reg_strength, reg_type="output_left",
                                                               reg_increase=reg_increase)#, middle=0.25)

                # Hidden to next goal
                #print("hidden->goal")
                extra_loss += utils.weight_regularization_calculator(model.goal1_layer.w,
                                                               [0, model.size_hidden], [0, model.size_goal1],
                                                               reg_strength, reg_type="output_left", reg_increase=reg_increase)

            # Regularization of the observation (only goes to the action side)
            #extra_loss += weight_regularization_calculator(model.hidden_layer.w,
            #                                                     [model.size_hidden, model.size_hidden+model.size_observation],
            #                                                     [0, cols],
            #                                                     reg_strength, reg_type="input_right", reg_increase=reg_increase)

            # Train model, record loss.
            loss = model.train(tape, targets, extra_loss=extra_loss)

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


