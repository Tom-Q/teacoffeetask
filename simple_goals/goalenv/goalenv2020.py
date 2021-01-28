import utils
import numpy as np
import tensorflow as tf
from goalenv import task
import neuralnet as nn
import scripts
from termcolor import colored


def accuracy_test_botvinick(model, sequences, noise=0, goals=False, initialization="uniform"):
    # This runs a hundred version of the model with different random initializations
    env = task.GoalEnv()
    actions_output_sequences = []

    for seqid in sequences:
        for i in range(50):
            for field_set in [False, True]:  # Try both with the field set and with the field not set.
                sequence = env.sequences[seqid]
                sequence.initialize()
                if field_set:
                    env.state.current.set_field("o_sequence"+str(seqid+1), 1)
                model.action = np.zeros((1, model.size_action), dtype=np.float32)

                # run the network
                with tf.GradientTape() as tape:
                    # Initialize context with random/uniform values.
                    if initialization == 'uniform':
                        model.context = np.random.uniform(0.01, 0.99, (1, model.size_hidden)).astype(dtype=np.float32)
                    for i, target in enumerate(sequence.targets):
                        model.action = np.zeros((1, model.size_action), dtype=np.float32)
                        # Add noise to context layer
                        model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                        observation = env.observe()
                        model.feedforward(observation)
                        env.do_action(model.h_action_wta)

                    # Get some statistics about the sequences actually observed. Is it a recognized sequence? If not,
                    # What kind of mistake appeared?
                    actions = np.array(model.h_action_wta).reshape((-1, task.GoalEnvData.num_actions))
                    target_actions = sequence.get_actions_one_hot()
                    actions_output_sequences.append([actions, target_actions])
                    model.clear_history()


def accuracy_test(model, sequences, noise=0, goals=False):
    # Collection actions
    actions_output_sequences = []
    env = task.GoalEnv()
    for seqid in sequences:
        for field_set in [False, True]: # Try both with the field set and with the field not set.
            sequence = env.sequences[seqid]
            sequence.initialize()
            if field_set:
                env.state.current.set_field("o_sequence"+str(seqid+1), 1)
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
                actions = np.array(model.h_action_wta).reshape((-1, task.GoalEnvData.num_actions))
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
            action_list.append(utils.onehot_to_str(actions[i, :], task.GoalEnvData.actions_list))
            target_list.append(utils.onehot_to_str(targets[i, :], task.GoalEnvData.actions_list))

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

def train(model=None, goals=False, num_iterations=50000, learning_rate=0.01,
          L2_reg=0.0000001, noise=0., sequences=None):
    if sequences is None:
        sequences = [0]
    env = task.GoalEnv()
    if model is None:
        if not goals:
            model = nn.NeuralNet(size_hidden=50, size_observation=29, size_action=17,  size_goal1=0, size_goal2=0,
                                 algorithm=nn.RMSPROP, learning_rate=learning_rate, initialization="uniform")
        #TODO: add goal model initialization.
    model.L2_regularization = L2_reg

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_fullseq = 0.
    rng_avg_goals1 = 0.
    rng_avg_goals2 = 0.

    for iteration in range(num_iterations):
        seqid = np.random.choice(sequences)
        sequence = env.sequences[seqid]
        sequence.initialize()
        if np.random.random() > 0.5:
            env.state.current.set_field("o_sequence"+str(seqid+1), 1)
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
            actions = np.array(model.h_action_wta).reshape((-1, task.GoalEnvData.num_actions))
            target_actions = sequence.get_actions_one_hot()
            ratio_actions = scripts.ratio_correct(actions, target_actions)
            if goals:
                goals1 = np.array(model.h_goal1_wta).reshape((-1, task.GoalEnvData.num_goals1))
                target_goals1 = sequence.get_actions_one_hot()
                ratio_goals1 = scripts.ratio_correct(goals1, target_goals1)

                goals2 = np.array(model.h_goal2_wta).reshape((-1, task.GoalEnvData.num_goals1))
                target_goals2 = sequence.get_actions_one_hot()
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