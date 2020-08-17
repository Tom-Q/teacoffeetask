# A file to bring together all the not-so-streamlined code that goes into making figures and experiments.
import state
import teacoffeeenv as tce
import tensorflow as tf
import numpy as np
import utils
import neuralnet as nn
import analysis
import matplotlib.pyplot as plt


def evaluate(values, targets):
    """
    :param values: list of arrays, or 2D array
    :param targets: list of arrays, or 2D array
    :return: ratio of values that are identical to their counterparts in targets
    """
    ratios = []
    for i in range(len(values)):
        ratios.append(ratio_correct(values[i], targets[i]))
    return ratios


def ratio_correct(choices, targets):
    correct = 0.
    for i in range(len(choices)):
        if (choices[i] == targets[i]).all():
            correct += 1.
    return correct / len(choices)


def train_supervised(model, num_episodes):
    env = tce.TeaCoffeeEnv()

    rng_avg_loss = 0.
    rng_avg_actions = 0.
    rng_avg_goals = 0.
    rng_avg_goal1 = 0.
    rng_avg_goal2 = 0.
    rng_avg_action1 = 0.

    goal_list = ["g_1_make_coffee", "g_1_make_tea",
                 "g_2_add_grounds", "g_2_add_cream",
                 "g_2_add_sugar", "g_2_drink", "g_2_dip_teabag"]
    for episode in range(num_episodes):
        goal = goal_list[episode % len(goal_list)]  # Cycle through the goals
        targets = tce.target_list[goal]  # Get the target actions and goals
        env.state = state.State(tce.TeaCoffeeData())  # Reinitialize the state
        env.state.current.set_field(goal, 1.)  # Set the goalS as active

        # run the network
        with tf.GradientTape() as tape:
            # Initialize context with random/uniform values.
            #model.context = np.zeros((1, model.size_hidden))
            model.context = np.float32(np.random.uniform(0.01, 0.99, (1, model.size_hidden)))
            targets_onehot = [[], [], []]  # actions, goal1s, goal2s
            for i, target in enumerate(targets):
                # Set up the input to be the correct actions and goals
                targets_onehot[0].append(utils.str_to_onehot(targets[i][0], tce.TeaCoffeeData.actions_list))
                targets_onehot[1].append(utils.str_to_onehot(targets[i][1], tce.TeaCoffeeData.goals1_list))
                targets_onehot[2].append(utils.str_to_onehot(targets[i][2], tce.TeaCoffeeData.goals2_list))
                model.action, model.goal1, model.goal2 = [targets_onehot[j][-1] for j in range(3)]

                observation = env.observe()
                model.feedforward(observation)
                env.do_action(target[0])  # Transition the MDP according to the *target* action, not the chosen action!

            # Get some statistics about what was correct and what wasn't
            ratios = evaluate([model.h_action_wta, model.h_goal1_wta, model.h_goal2_wta],
                              targets_onehot)
            rng_avg_action1 = utils.rolling_avg(rng_avg_action1, ratio_correct([model.h_action_wta[0]], [targets_onehot[0][0]]), 2./(episode+2) if episode < 1000 else 0.001)
            # Train model, record loss.
            loss = model.train(targets_onehot[0], targets_onehot[1], targets_onehot[2], tape)

            # Monitor progress using rolling averages.
            speed = 2./(episode+2) if episode < 1000 else 0.001  # enables more useful evaluations for early trials
            rng_avg_loss = utils.rolling_avg(rng_avg_loss, loss, speed)
            rng_avg_actions = utils.rolling_avg(rng_avg_actions, ratios[0], speed)
            rng_avg_goals = utils.rolling_avg(rng_avg_goals, ratios[0] == 1, speed)  # whole action sequence correct ?
            rng_avg_goal1 = utils.rolling_avg(rng_avg_goal1, ratios[1], speed)
            rng_avg_goal2 = utils.rolling_avg(rng_avg_goal2, ratios[2], speed)
            # Display on the console at regular intervals
            if (episode < 1000 and episode in [3**n for n in range(50)]) or episode % 1000 == 0 \
                               or episode+1 == num_episodes:
                print("{0}: avg loss={1}, \tactions={2}, \tfull_sequence={3}\tgoal1={4}\tgoal2={5}\tfirst_action={6}".format(
                      episode, rng_avg_loss, rng_avg_actions, rng_avg_goals, rng_avg_goal1, rng_avg_goal2, rng_avg_action1))


def train_and_save(num_models, name, num_episodes):
    for i in range(num_models):
        model = nn.NeuralNet()
        train_supervised(model, num_episodes)
        utils.save_object(name, model)
        print('Trained and saved model #{0} of {1}\n'.format(i+1, num_models))


def get_model_hidden_activations(model):
    env = tce.TeaCoffeeEnv()
    # Get the target actions and goals
    hidden = []
    for goal in tce.goal_list:
        targets = tce.target_list[goal]
        model.context = np.random.uniform(0.01, 0.1, (1, model.size_hidden))
        model.action = utils.str_to_onehot(targets[0][0], tce.TeaCoffeeData.actions_list)
        model.goal1 = utils.str_to_onehot(targets[0][1], tce.TeaCoffeeData.goals1_list)
        model.goal2 = utils.str_to_onehot(targets[0][2], tce.TeaCoffeeData.goals2_list)
        for i, target in enumerate(targets):
            observation = env.observe()
            model.feedforward(observation)
            hidden.append(model.context.numpy().flatten())
            env.do_action(target[0])
    return hidden




def run_models_with_noise(models, noise):
    goal_list = ["g_1_make_coffee", "g_1_make_tea",
                 "g_2_add_grounds", "g_2_add_cream",
                 "g_2_add_sugar", "g_2_drink", "g_2_dip_teabag"]
    all_action_targets_str = utils.flatten_onelevel([tce.action_list[goal] for goal in goal_list])
    all_action_targets = [utils.str_to_onehot(action, tce.TeaCoffeeData.actions_list) for action in all_action_targets_str]

    # Gather actual outputs
    all_outputs = []
    for model in models:
        all_outputs.append(run_model_with_noise(model, noise))
    # Now check which actions and which goals they got correctly
    num_bad_actions = np.zeros(len(all_action_targets))
    for output in all_outputs:
        for i, action in enumerate(output):
            if (all_action_targets[i] != action).any():
                num_bad_actions[i] += 1
            print(all_action_targets_str[i], utils.onehot_to_str(action, tce.TeaCoffeeData.actions_list))

    return num_bad_actions


def run_model_with_noise(model, noise_level):
    env = tce.TeaCoffeeEnv()
    # Get the target actions and goals
    outputs_actions = []
    goal_list = ["g_1_make_coffee", "g_1_make_tea",
                 "g_2_add_grounds", "g_2_add_cream",
                 "g_2_add_sugar", "g_2_drink", "g_2_dip_teabag"]
    for goal in goal_list:
        targets = tce.target_list[goal]
        env.state = state.State(tce.TeaCoffeeData())  # Reinitialize the state
        env.state.current.set_field(goal, 1.)  # Set the goalS as active
        model.context = np.random.uniform(0.01, 0.1, (1, model.size_hidden))
        for i, target in enumerate(targets):
            model.action = utils.str_to_onehot(targets[i][0], tce.TeaCoffeeData.actions_list)
            model.goal1 = utils.str_to_onehot(targets[i][1], tce.TeaCoffeeData.goals1_list)
            model.goal2 = utils.str_to_onehot(targets[i][2], tce.TeaCoffeeData.goals2_list)
            model.context += noise_level * np.random.normal(size=tf.shape(model.context))
            observation = env.observe()
            model.feedforward(observation)
            env.do_action(target[0])
            outputs_actions.append(model.action)
    return outputs_actions


def basic_tests_teacoffeeenv():
    env = tce.TeaCoffeeEnv()

    print(tce.action_list["g_1_make_coffee"])

    env.do_action("a_take_coffee_pack")
    env.do_action("a_open")
    env.do_action("a_pour")
    env.do_action("a_put_down")
    env.do_action("a_take_spoon")
    env.do_action("a_stir")

    tce.test_goals()


def make_rdm_and_mds(name):
    model = utils.load_object(name)  # eg 'noise_test2'
    hidden = get_model_hidden_activations(model)
    rdmatrix = analysis.rdm_spearman(hidden)
    labels = []
    for goal in tce.action_list:
        for action in tce.action_list[goal]:
            labels.append(goal+'_'+action)
    analysis.show_rdm(rdmatrix, labels, "Spearman rho matrix")
    mdsy = analysis.mds(hidden)
    analysis.plot_mds_points(mdsy, range(len(mdsy)), labels=labels)



def plot_effects_of_noise(name):
    models = utils.load_objects(name, 10)
    action_errors = []
    action_errors.append(run_models_with_noise(models, 0.0))
    action_errors.append(run_models_with_noise(models, 1.))
    action_errors.append(run_models_with_noise(models, 10.))
    action_errors.append(run_models_with_noise(models, 1000))
    colors = ['g', 'b', 'k', 'r']
    for i in range(len(action_errors)):
        plt.plot(range(len(action_errors[i])), action_errors[i], colors[i] + 'x-')
    plt.show()

