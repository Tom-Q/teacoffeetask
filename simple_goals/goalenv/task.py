from goalenv import environment as env, state
import numpy as np
import utils


class Target(object):
    def __init__(self, action, goal1 = None, goal2 = None):
        self.action_str = action
        self.goal1_str = goal1
        self.goal2_str = goal2

    @property
    def action_str(self):
        return self._action_str

    @action_str.setter
    def action_str(self, new_action_str):
        self._action_str = new_action_str
        if new_action_str is None:
            self._action_one_hot = None
        else:
            self._action_one_hot = utils.str_to_onehot(new_action_str, env.GoalEnvData.actions_list)

    @property
    def goal1_str(self):
        return self._goal1_str

    @goal1_str.setter
    def goal1_str(self, new_goal1_str):
        self._goal1_str = new_goal1_str
        if new_goal1_str is None:
            self._goal1_one_hot = None
        else:
            self._goal1_one_hot = utils.str_to_onehot(new_goal1_str, env.GoalEnvData.goals1_list)

    @property
    def goal2_str(self):
        return self._goal2_str

    @goal2_str.setter
    def goal2_str(self, new_goal2_str):
        self._goal2_str = new_goal2_str
        if new_goal2_str is None:
            self._goal2_one_hot = None
        else:
            self._goal2_one_hot = utils.str_to_onehot(new_goal2_str, env.GoalEnvData.goals2_list)

    @property
    def action_one_hot(self):
        return self._action_one_hot

    @property
    def goal1_one_hot(self):
        return self._goal1_one_hot

    @property
    def goal2_one_hot(self):
        return self._goal2_one_hot


import copy

class BehaviorSequence(object):
    def __init__(self, initial_state, targets=None):
        self.targets = targets
        self.initial_state = copy.deepcopy(initial_state)
        self._actions_list = [t.action_one_hot for t in self.targets]
        self._goals1_list = [t.goal1_one_hot for t in self.targets]
        self._goals2_list = [t.goal2_one_hot for t in self.targets]

    def _get_one_hot(self, elements_list, num_elements):
        return np.array(elements_list, dtype=float).reshape((-1, num_elements))

    def get_actions_one_hot(self):
        return self._get_one_hot(self._actions_list, env.GoalEnvData.num_actions)

    def get_goals1_one_hot(self):
        return self._get_one_hot(self._goals1_list, env.GoalEnvData.num_goals1)

    def get_goals2_one_hot(self):
        return self._get_one_hot(self._goals2_list, env.GoalEnvData.num_goals2)

    def _get_inputs_one_hot(self, elements_list):
        # Add a zero element at the beginning and delete the last element (which only serves as a target)
        return np.zeros_like(elements_list[0]) + elements_list[:-1]

    def get_actions_inputs_one_hot(self):
        return self._get_inputs_one_hot(self._actions_list)

    def get_goal1s_inputs_one_hot(self):
        return self._get_inputs_one_hot(self._goals1_list)

    def get_goal2s_inputs_one_hot(self, zeroed=False):
        return self._get_inputs_one_hot(self._goals2_list)


def _make_targets(list_topgoals, list_midgoals, list_actions):
    if len(list_topgoals) != len(list_midgoals) or len(list_topgoals) != len(list_actions):
        raise ValueError("There must be as many top goals, mid goals, and actions")
    return [Target(goal1=topgoal, goal2=midgoal, action=action)
            for topgoal, midgoal, action in zip(list_topgoals, list_midgoals, list_actions)]


default_state = state.State(env.GoalEnvData())

#                                            ####################                                                      #
# -------------------------------------------- COFFEE SEQUENCES -------------------------------------------------------#
#                                            ####################                                                      #
################
# BLACK COFFEE #
################
actions_coffee = [  # Grounds+close cupboard
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffee = ["g_1_make_coffee"] * len(actions_coffee)
midgoals_coffee = ["g_2_add_grounds"] * 12 + ["g_2_stir"] * 6 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffee, midgoals_coffee, actions_coffee)
sequence_coffee = BehaviorSequence(default_state, targets)

actions_coffeesugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # add sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffeesugar = ["g_1_make_coffee"] * len(actions_coffeesugar)
midgoals_coffeesugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 4 + \
                       ["g_2_stir"] * 6 + ["g_2_clean_up"] * 2 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffeesugar, midgoals_coffeesugar, actions_coffeesugar)
sequence_coffeesugar = BehaviorSequence(default_state, targets)

actions_coffee2sugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # add sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # add another sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffee2sugar = ["g_1_make_coffee"] * len(actions_coffee2sugar)
midgoals_coffee2sugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 8 + \
                        ["g_2_stir"] * 6 + ["g_2_clean_up"] * 2 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffee2sugar, midgoals_coffee2sugar, actions_coffee2sugar)
sequence_coffee2sugar = BehaviorSequence(default_state, targets)

#####################
# COFFEE WITH CREAM #
#####################
actions_coffeecream = [  # grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # close cupboard
    "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # cream
    "a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffeecream = ["g_1_make_coffee"] * len(actions_coffeecream)
midgoals_coffeecream = ["g_2_add_grounds"] * 12 + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 9 + \
                        ["g_2_stir"] * 6 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffeecream, midgoals_coffeecream, actions_coffeecream)
sequence_coffeecream = BehaviorSequence(default_state, targets)

# Sugar then cream
actions_coffeesugarcream = [  # grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # cream
    "a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffeesugarcream = ["g_1_make_coffee"] * len(actions_coffeesugarcream)
midgoals_coffeesugarcream = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 4 + ["g_2_stir"] * 6 + \
                            ["g_2_clean_up"] * 2 + ["g_2_add_cream"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffeesugarcream, midgoals_coffeesugarcream, actions_coffeesugarcream)
sequence_coffeesugarcream = BehaviorSequence(default_state, targets)

actions_coffee2sugarcream = [  # grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # second sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # cream
    "a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffee2sugarcream = ["g_1_make_coffee"] * len(actions_coffee2sugarcream)
midgoals_coffee2sugarcream = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 8 + ["g_2_stir"] * 6 + \
                            ["g_2_clean_up"] * 2 + ["g_2_add_cream"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffee2sugarcream, midgoals_coffee2sugarcream, actions_coffee2sugarcream)
sequence_coffee2sugarcream = BehaviorSequence(default_state, targets)

# Cream then sugar
actions_coffeecreamsugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Cream
    "a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffeecreamsugar = ["g_1_make_coffee"] * len(actions_coffeecreamsugar)
midgoals_coffeecreamsugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 9 + ["g_2_stir"] * 6 +\
                            ["g_2_add_sugar"] * 4 + ["g_2_stir"] * 6 + ["g_2_clean_up"] * 2 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffeecreamsugar, midgoals_coffeecreamsugar, actions_coffeecreamsugar)
sequence_coffeecreamsugar = BehaviorSequence(default_state, targets)

actions_coffeecream2sugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Cream
    "a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Another sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffeecream2sugar = ["g_1_make_coffee"] * len(actions_coffeecream2sugar)
midgoals_coffeecream2sugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 9 + ["g_2_stir"] * 6 +\
                            ["g_2_add_sugar"] * 8 + ["g_2_stir"] * 6 + ["g_2_clean_up"] * 2 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffeecream2sugar, midgoals_coffeecream2sugar, actions_coffeecream2sugar)
sequence_coffeecream2sugar = BehaviorSequence(default_state, targets)

####################
# COFFEE WITH MILK #
####################
#Initial state has no cream! Hence the fallback on milk.
nocream_state = state.State(env.GoalEnvData())
nocream_state.h_cream_present = 0
actions_coffeemilk = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Close cupboard
    "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffeemilk = ["g_1_make_coffee"] * len(actions_coffeemilk)
midgoals_coffeemilk = ["g_2_add_grounds"] * 12 + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 +\
                      ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffeemilk, midgoals_coffeemilk, actions_coffeemilk)
sequence_coffeemilk = BehaviorSequence(default_state, targets)

# Sugar then milk
actions_coffeesugarmilk = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffeesugarmilk = ["g_1_make_coffee"] * len(actions_coffeesugarmilk)
midgoals_coffeesugarmilk = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6  + ["g_2_add_sugar"] * 4 + ["g_2_stir"] * 6\
                           +["g_2_clean_up"] * 2 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffeesugarmilk, midgoals_coffeesugarmilk, actions_coffeesugarmilk)
sequence_coffeesugarmilk = BehaviorSequence(default_state, targets)

actions_coffee2sugarmilk = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Second sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffee2sugarmilk = ["g_1_make_coffee"] * len(actions_coffee2sugarmilk)
midgoals_coffee2sugarmilk = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6  + ["g_2_add_sugar"] * 8 + ["g_2_stir"] * 6\
                           +["g_2_clean_up"] * 2 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffee2sugarmilk, midgoals_coffee2sugarmilk, actions_coffee2sugarmilk)
sequence_coffee2sugarmilk = BehaviorSequence(default_state, targets)

# milk then sugar
actions_coffeemilksugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffeemilksugar = ["g_1_make_coffee"] * len(actions_coffeemilksugar)
midgoals_coffeemilksugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + \
                           ["g_2_add_sugar"] * 4 + ["g_2_stir"] * 6 +["g_2_clean_up"] * 2 +  ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffeemilksugar, midgoals_coffeemilksugar, actions_coffeemilksugar)
sequence_coffeemilksugar = BehaviorSequence(default_state, targets)


actions_coffeemilk2sugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_coffeemilk2sugar = ["g_1_make_coffee"] * len(actions_coffeemilk2sugar)
midgoals_coffeemilk2sugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + \
                           ["g_2_add_sugar"] * 8 + ["g_2_stir"] * 6 +["g_2_clean_up"] * 2 +  ["g_2_drink"] * 6
targets = _make_targets(topgoals_coffeemilk2sugar, midgoals_coffeemilk2sugar, actions_coffeemilk2sugar)
sequence_coffeemilk2sugar = BehaviorSequence(default_state, targets)

#                                            #################                                                         #
# -------------------------------------------- TEA SEQUENCES ----------------------------------------------------------#
#                                            #################                                                         #
############
# JUST TEA #
############
actions_tea = [  # dip teabag+close cupboard
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_tea = ["g_1_make_tea"] * len(actions_tea)
midgoals_tea = ["g_2_infuse_tea"] * 8 + ["g_2_stir"] * 6 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_tea, midgoals_tea, actions_tea)
sequence_tea = BehaviorSequence(default_state, targets)

actions_teasugar = [  # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # add sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_teasugar = ["g_1_make_tea"] * len(actions_teasugar)
midgoals_teasugar = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 4 + \
                       ["g_2_stir"] * 6 + ["g_2_clean_up"] * 2 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_teasugar, midgoals_teasugar, actions_teasugar)
sequence_teasugar = BehaviorSequence(default_state, targets)

actions_tea2sugar = [   # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # add sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # add another sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_tea2sugar = ["g_1_make_tea"] * len(actions_tea2sugar)
midgoals_tea2sugar = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 8 + \
                        ["g_2_stir"] * 6 + ["g_2_clean_up"] * 2 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_tea2sugar, midgoals_tea2sugar, actions_tea2sugar)
sequence_tea2sugar = BehaviorSequence(default_state, targets)

#################
# TEA WITH MILK #
#################

actions_teamilk = [   # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_teamilk = ["g_1_make_tea"] * len(actions_teamilk)
midgoals_teamilk = ["g_2_infuse_tea"] * 8 + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 +\
                      ["g_2_drink"] * 6
targets = _make_targets(topgoals_teamilk, midgoals_teamilk, actions_teamilk)
sequence_teamilk = BehaviorSequence(default_state, targets)

# Sugar then milk
actions_teasugarmilk = [   # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_teasugarmilk = ["g_1_make_tea"] * len(actions_teasugarmilk)
midgoals_teasugarmilk = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6  + ["g_2_add_sugar"] * 4 + ["g_2_stir"] * 6\
                           +["g_2_clean_up"] * 2 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_teasugarmilk, midgoals_teasugarmilk, actions_teasugarmilk)
sequence_teasugarmilk = BehaviorSequence(default_state, targets)

actions_tea2sugarmilk = [   # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Second sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_tea2sugarmilk = ["g_1_make_tea"] * len(actions_tea2sugarmilk)
midgoals_tea2sugarmilk = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6  + ["g_2_add_sugar"] * 8 + ["g_2_stir"] * 6\
                           +["g_2_clean_up"] * 2 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_tea2sugarmilk, midgoals_tea2sugarmilk, actions_tea2sugarmilk)
sequence_tea2sugarmilk = BehaviorSequence(default_state, targets)

# milk then sugar
actions_teamilksugar = [   # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_teamilksugar = ["g_1_make_tea"] * len(actions_teamilksugar)
midgoals_teamilksugar = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + \
                           ["g_2_add_sugar"] * 4 + ["g_2_stir"] * 6 +["g_2_clean_up"] * 2 +  ["g_2_drink"] * 6
targets = _make_targets(topgoals_teamilksugar, midgoals_teamilksugar, actions_teamilksugar)
sequence_teamilksugar = BehaviorSequence(default_state, targets)


actions_teamilk2sugar = [   # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
topgoals_teamilk2sugar = ["g_1_make_tea"] * len(actions_teamilk2sugar)
midgoals_teamilk2sugar = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + \
                           ["g_2_add_sugar"] * 8 + ["g_2_stir"] * 6 +["g_2_clean_up"] * 2 + ["g_2_drink"] * 6
targets = _make_targets(topgoals_teamilk2sugar, midgoals_teamilk2sugar, actions_teamilk2sugar)
sequence_teamilk2sugar = BehaviorSequence(default_state, targets)

sequences_list =\
        [sequence_coffee, sequence_coffeesugar, sequence_coffee2sugar,
        sequence_coffeecream,
        sequence_coffeesugarcream, sequence_coffee2sugarcream,
        sequence_coffeecreamsugar, sequence_coffeecream2sugar,
        sequence_coffeemilk,
        sequence_coffeesugarmilk, sequence_coffee2sugarmilk,
        sequence_coffeemilksugar, sequence_coffeemilk2sugar,

        sequence_tea, sequence_teasugar, sequence_tea2sugar,
        sequence_teamilk,
        sequence_teamilksugar, sequence_teamilk2sugar,
        sequence_teasugarmilk, sequence_tea2sugarmilk
        ]


