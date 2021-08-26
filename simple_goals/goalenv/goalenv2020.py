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
    line = top_goal_line + ';\n ' + mid_goal_line + ';\n ' + action_line
    return line


def _print_sequences_stats(sequences):
    unique_output_sequences = []
    #counters = []
    #seqids=[]
    for output_sequence in sequences:
        unique = True
        for i, sequence in enumerate(unique_output_sequences):
            if sequence.equals(output_sequence):
                unique = False
                sequence.clones.append(output_sequence)
                #counters[i]+=1
                #seqids[i].append(output_sequence.id)
        if unique:
            unique_output_sequences.append(output_sequence)
            output_sequence.clones = []
            #counters.append(1)
            #seqids.append([output_sequence.id])

    # Sort sequences and counters and seqids, starting with most frequent:
    #zipped = zip(counters, unique_output_sequences)
    #unique_output_sequences = [x for _, x in sorted(zipped, reverse=True, key=lambda pair: pair[0])]
    #counters = sorted(counters, reverse=True)
    unique_output_sequences = sorted(unique_output_sequences, reverse=True,
                                     key=lambda unique_seq: len(unique_seq.clones))

    # Now display each unique sequence, but converted back to text:
    for i, seq in enumerate(unique_output_sequences):
        line = ""

        # Check if it's one of the target sequences
        is_target = False
        for target_sequence in task.sequences_list:
            if unique_output_sequences[i].equals(target_sequence):
                line += "TARGET: "+target_sequence.name
                is_target = True
                break
        if not is_target:
            line += "NOT A TARGET"
        line += ", " + str(1 + len(seq.clones)) + " " + str([seq.id] + [clone.id for clone in seq.clones]) + " " + ":\n"
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


# TODO: put that in the model. I'm only putting it here cause I can't be bothered to add this to the already trained model.
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
                       noise_per_step=True, disruption_per_step=False):
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

    # Set up model learning rate and L2 regularization to compute loss.
    model.learning_rate = 0.
    model.L2_regularization = 0.

    seq_counter = 0
    for seqid, sequence in enumerate(sequences):
        print("testing sequence: "+str(seqid))
        outputs_per_noise_step = []
        num_runs = sequence.length if (noise_per_step or disruption_per_step) else 1
        for noise_step in range(num_runs):
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
                        model.context = np.random.normal(0.01, 0.1, (1, model.size_hidden)).astype(dtype=np.float32)
                        model.context[model.context < 0.01] = 0.  # it's impossible for a sigmoid activation to be <0
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
                            if noise_per_step:
                                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                                model.goal1 += np.float32(np.random.normal(0., goal1_noise, size=(1, model.size_goal1)))
                                model.goal2 += np.float32(np.random.normal(0., goal2_noise, size=(1, model.size_goal2)))
                            if disruption_per_step:
                                # Change the state
                                env.state = disrupt_state(env.state, initial_state=sequence.initial_state, #mode=REINITIALIZE)
                                                          mode=HOLD_RANDOM_OBJECT)
                        observation = env.observe()
                        model.feedforward(observation)

                        if j < len(sequence.targets):  # after that it's not defined.
                            loss = compute_last_step_loss(model, sequence.targets[j], include_regularization=False)
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
    outputs_sequences_flat = utils.flatten_onelevel(outputs_no_noise_step_distinction)
    activations = [seq.activations for seq in outputs_sequences_flat]
    activations_flat = utils.flatten_onelevel(activations)  # all this list wrangling is ugly as hell

    ##################################################################################################
    # Analysis 3: Noise. How long after the noise did the first error occur? Was that step a switch? #
    ##################################################################################################
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
                    trial.first_error = None  # This trial is error-free
                    for j, action in enumerate(trial.get_actions_one_hot()):
                        error = 1
                        # It's only an error if it's an error for every target...
                        # I guess this misses situations in which the two sequences are tangled together. UGH!!
                        for target in all_targets:
                            if np.array_equal(action, target[j]):
                                error *= 0
                        if error:
                            first_error[j] += 1
                            trial.first_error = j  # Add a first_error attribute to the sequence.
                            break

                # Need to color the step at which noise is introduced.
                print("Target sequence " + str(i) + " (" + target_sequence.name + ") noise at step " + str(noise_step) + ": errors = " + str(first_error))


    ####################
    # Analysis 4: Loss #
    ####################
    # Compute average loss, and average loss at first error and on subsequent points
    counter_total = 0
    loss_total = 0

    counter_first_error = 0
    first_error_loss_total = 0

    noise_loss_total = 0
    counter_noise = 0

    prior_to_error_loss_total = 0
    counter_prior_to_error = 0

    prior_to_noise_loss_total = 0
    counter_prior_to_noise = 0
    for sequence in outputs_sequences_flat:
        for i, loss in enumerate(sequence.losses):
            if loss is not None:
                #sequence.losses[i] = sequence.losses[i].numpy()[0]
                loss_total += sequence.losses[i]
                counter_total += 1
                if sequence.first_error is not None and i < sequence.first_error:
                    prior_to_error_loss_total += sequence.losses[i]
                    counter_prior_to_error+=1
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
    loss_avg = loss_total/counter_total
    loss_first_error_avg = first_error_loss_total / counter_first_error
    loss_noise_avg = noise_loss_total / counter_noise
    loss_prior_error_avg = prior_to_error_loss_total / counter_prior_to_error
    loss_prior_noise_avg = prior_to_noise_loss_total / counter_prior_to_noise
    print("average loss={0}, loss on first error={1}, loss on noise={2}, loss_good_seq={3}, loss_before_noise={4}".format(loss_avg,
                                                                                loss_first_error_avg,
                                                                                loss_noise_avg,
                                                                                loss_prior_error_avg,
                                                                                loss_prior_noise_avg))

    ###################################
    # Analysis 5: RDM, MDS, and T-SNE #
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
    return tsne_results, test_data # Test data is enriched during analysis (first error step)

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


def train(model=None, goals=False, num_iterations=50000, learning_rate=0.01, L2_reg=0.00001, noise=0., sequences=None):
    if sequences is None:
        sequences = [0]
    env = environment.GoalEnv()
    if model is None:
        if goals:
            model = nn.ElmanGoalNet(size_hidden=50, size_observation=29, size_action=19,
                                    size_goal1=len(environment.GoalEnvData.goals1_list),
                                    size_goal2=len(environment.GoalEnvData.goals2_list),
                                    algorithm=nn.RMSPROP, learning_rate=learning_rate, initialization="uniform",
                                    last_action_inputs=True)
        else:
            model = nn.ElmanGoalNet(size_hidden=50, size_observation=29, size_action=19, size_goal1=len(sequences), size_goal2=0,
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

        # run the network
        with tf.GradientTape() as tape:
            model.new_episode()
            # Initialize context with random/uniform values.
            model.context = np.random.uniform(0.01, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
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

            # Train model, record loss.
            loss = model.train(tape, sequence.targets, goals)


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


