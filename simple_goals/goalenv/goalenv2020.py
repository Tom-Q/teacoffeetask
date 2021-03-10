import utils
import numpy as np
import tensorflow as tf
from goalenv import environment, task
import neuralnet as nn
import scripts
from termcolor import colored
import copy

def _sequence_equals(sequence1, sequence2):
    if len(sequence1) == len(sequence2):
        return all(np.array_equal(sequence1[i], sequence2[i]) for i in range(len(sequence1)))
    return False

# This runs the accuracy test by allowing all sequences to complete, with the environment reacting organically.
# The report is a % of sequences that match one of the original sequences +
# print-out of all targets + all observed sequences with a count
def accuracy_test_botvinick(model, sequence_ids, noise=0, noise_step=None, num_tests=10, goals=False, initialization="uniform", verbose=True):
    # This runs a hundred version of the model with different random initializations
    env = environment.GoalEnv()
    output_actions = []


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
        #for field_set in [False, True]:  # Try both with the field set and with the field not set.
        for i in range(num_tests):
            # Initialize the sequence.
            env.reinitialize(copy.deepcopy(sequence.initial_state))
            #if field_set:
            #    env.state.current.set_field("o_sequence"+str(seqid+1), 1)
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
                for j in range(max_length):
                    # Add noise to context layer
                    if j == noise_step or noise_step is None:
                        model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                    observation = env.observe()
                    model.feedforward(observation)
                    # if there's an impossible action, ignore it and continue.
                    next_state = copy.deepcopy(env.state.next)
                    try:
                        env.do_action(model.h_action_wta[-1])
                    except environment.ActionException as error:  # The action doesn't make sense and cannot be executed in the environment.
                        print(error)
                        # reset the state when an impossible action is attempted.
                        env.state.next = next_state
                    action_str = utils.onehot_to_str(model.h_action_wta[-1], environment.GoalEnvData.actions_list)
                    if action_str == environment.TERMINAL_ACTION:
                        break  # we said "done", we're done.

                # Get some statistics about the sequences actually observed. Is it a recognized sequence? If not,
                # What kind of mistake appeared?
                actions = np.array(model.h_action_wta).reshape((-1, environment.GoalEnvData.num_actions))
                output_actions.append(actions)
                model.clear_history()

    # Make a list of all observed sequences
    unique_output_sequences = []
    counters = []
    for output_sequence in output_actions:
        unique = True
        for i, sequence in enumerate(unique_output_sequences):
            if _sequence_equals(sequence, output_sequence):
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
    for i in range(len(unique_output_sequences)):
        line = str(counters[i]) + ": "
        # Check if it's one of the target sequences
        for j, target_sequence in enumerate(target_actions_sequences):
            if _sequence_equals(unique_output_sequences[i], target_sequence):
                line += "(target sequence "+str(j)+") "

        for action in unique_output_sequences[i]:
            line += utils.onehot_to_str(action, environment.GoalEnvData.actions_list) + " - "
        print(line+"\n")


def accuracy_test(model, sequences, noise=0, goals=False):
    # Collection actions
    actions_output_sequences = []
    env = environment.GoalEnv()
    for seqid in sequences:
        #for field_set in [False, True]: # Try both with the field set and with the field not set.
        sequence = task.sequences_list[seqid]
        env.reinitialize(sequence.initial_state)
        #if field_set:
        #    env.state.current.set_field("o_sequence"+str(seqid+1), 1)
        model.action = np.zeros((1, model.size_action), dtype=np.float32)

        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            model.context = np.random.uniform(0.01, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
            # Alternative: zeros
            #model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            for i, target in enumerate(sequence.targets):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                # Add noise to context layer
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = env.observe()
                model.feedforward(observation)
                env.do_action(target.action_one_hot)

            # Get some statistics about the percentage of correct behavior
            actions = np.array(model.h_action_wta).reshape((-1, environment.GoalEnvData.num_actions))
            target_actions = sequence.get_actions_one_hot()
            actions_output_sequences.append([actions, target_actions])
            model.clear_history()

    # Now display all the actions and target actions, but converted back to text.
    for i in range(len(actions_output_sequences)):
        if i % 2 == 0:
            print("Sequence " + str(i+1) + " (no sequence identifier):")
        else:
            print("Sequence " + str(i+1) + " (with sequence identifier):")
        actions = actions_output_sequences[i][0]
        targets = actions_output_sequences[i][1]

        # Convert the arrays to lists of action names
        action_list = []
        target_list = []
        for i in range(len(actions)):
            action_list.append(utils.onehot_to_str(actions[i, :], environment.GoalEnvData.actions_list))
            target_list.append(utils.onehot_to_str(targets[i, :], environment.GoalEnvData.actions_list))

        # print the action and target and count the wrong ones
        total_wrong = 0
        for i in range(len(target_list)):
            if target_list[i] == action_list[i]:
                c = 'green'
            else:
                c = 'red'
            print(target_list[i] + "->" + colored(action_list[i], c) + ";", end=" ")
        print("\n")

def accuracy_test_allow_all():
    pass

def train(model=None, goals=False, num_iterations=50000, learning_rate=0.01, L2_reg=0.00001, noise=0., sequences=None):
    if sequences is None:
        sequences = [0]
    env = environment.GoalEnv()
    if model is None:
        if goals:
            model = nn.NeuralNet(size_hidden=50, size_observation=28, size_action=18,
                                 size_goal1=len(environment.GoalEnvData.goals1_list),
                                 size_goal2=len(environment.GoalEnvData.goals2_list),
                                 algorithm=nn.RMSPROP, learning_rate=learning_rate, initialization="uniform")
        else:
            model = nn.NeuralNet(size_hidden=50, size_observation=28, size_action=18,  size_goal1=len(sequences), size_goal2=0,
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
        env.reinitialize(copy.deepcopy(sequence.initial_state))
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