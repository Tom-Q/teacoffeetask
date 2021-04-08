from goalenv import environment as env, state
import numpy as np
import utils
import copy
import random


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


class BehaviorSequence(object):
    def __init__(self, initial_state, targets=None, name="no-name"):
        self.additional_info = None
        self.targets = targets
        self._initial_state = copy.deepcopy(initial_state)  # just as a safety...
        if targets is not None:
            self._actions_list = [t.action_one_hot for t in self.targets]
            self._goals1_list = [t.goal1_one_hot for t in self.targets]
            self._goals2_list = [t.goal2_one_hot for t in self.targets]
            self.length = len(self._actions_list)
        else:
            self.length = 0
        self.alt_solutions = []
        self.name = name

    def equals(self, behavior_sequence):
        if len(self.targets) != len(behavior_sequence.targets):
            return False
        for i in range(len(self.targets)):
            if self.targets[i].action_str != behavior_sequence.targets[i].action_str or\
               self.targets[i].goal1_str != behavior_sequence.targets[i].goal1_str or\
               self.targets[i].goal2_str != behavior_sequence.targets[i].goal2_str:
                return False
        return True

    @property
    def initial_state(self):
        state = copy.deepcopy(self._initial_state)
        # Special case for dairy first: 50% chance to just be 0.
        # The idea is to enforce the match between actions and goals.
        if state.current.o_ddairy_first == 1 or state.current.o_ddairy_first == -1 and random.random() > 0.5:
            state.current.o_ddairy_first = 0
            state.next.o_ddairy_first = 0
        return state

    def set_targets(self, list_goals1, list_goals2, list_actions):
        self._actions_list = copy.deepcopy(list_actions)
        self._goals1_list = copy.deepcopy(list_goals1)
        self._goals2_list = copy.deepcopy(list_goals2)

        if len(list_actions) != len(list_goals1) or len(list_actions)!= len(list_goals2):
            raise(ValueError("All target lists must have the same length"))

        self.targets = []
        for i in range(len(list_actions)):
            self.targets.append(Target(utils.onehot_to_str(list_actions[i], env.GoalEnvData.actions_list),
                                       utils.onehot_to_str(list_goals1[i], env.GoalEnvData.goals1_list),
                                       utils.onehot_to_str(list_goals2[i], env.GoalEnvData.goals2_list)))

        self.length = len(self._actions_list)


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


#                                            ####################                                                      #
# -------------------------------------------- COFFEE SEQUENCES -------------------------------------------------------#
#                                            ####################                                                      #
################
# BLACK COFFEE #
################
actions_coffee = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # close cupboard
    "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffee = ["g_1_make_coffee"] * len(actions_coffee)
midgoals_coffee = ["g_2_add_grounds"] * 11 + ["g_2_clean_up"] * 1 + ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffee, midgoals_coffee, actions_coffee)
data = env.GoalEnvData(o_dcoffee=1)
sequence_coffee = BehaviorSequence(state.State(data), targets, name="coffee black")

actions_coffeesugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # add sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffeesugar = ["g_1_make_coffee"] * len(actions_coffeesugar)
midgoals_coffeesugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 4 + \
                       ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffeesugar, midgoals_coffeesugar, actions_coffeesugar)
data = env.GoalEnvData(o_dcoffee=1, o_dsugar=1)
sequence_coffeesugar = BehaviorSequence(state.State(data), targets, name="coffee sugar")

actions_coffee2sugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # add sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # add another sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffee2sugar = ["g_1_make_coffee"] * len(actions_coffee2sugar)
midgoals_coffee2sugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 8 + \
                        ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 +  ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffee2sugar, midgoals_coffee2sugar, actions_coffee2sugar)
data = env.GoalEnvData(o_dcoffee=1, o_dsugar=1, o_dextrasugar=1)
sequence_coffee2sugar = BehaviorSequence(state.State(data), targets, name="coffee 2 sugars")

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
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffeecream = ["g_1_make_coffee"] * len(actions_coffeecream)
midgoals_coffeecream = ["g_2_add_grounds"] * 11 + ["g_2_clean_up"] * 1 + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 9 + \
                        ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffeecream, midgoals_coffeecream, actions_coffeecream)
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1)
sequence_coffeecream = BehaviorSequence(state.State(data), targets, name="coffee cream")

# Sugar then cream
actions_coffeesugarcream = [  # grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # cream
    "a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffeesugarcream = ["g_1_make_coffee"] * len(actions_coffeesugarcream)
midgoals_coffeesugarcream = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 4 + ["g_2_clean_up"] * 2 +\
                            ["g_2_stir"] * 6 + ["g_2_add_cream"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffeesugarcream, midgoals_coffeesugarcream, actions_coffeesugarcream)
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_ddairy_first=-1)
sequence_coffeesugarcream = BehaviorSequence(state.State(data), targets, name="coffee sugar cream")

actions_coffee2sugarcream = [  # grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # second sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # cream
    "a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffee2sugarcream = ["g_1_make_coffee"] * len(actions_coffee2sugarcream)
midgoals_coffee2sugarcream = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 8 + ["g_2_clean_up"] * 2 \
                             + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffee2sugarcream, midgoals_coffee2sugarcream, actions_coffee2sugarcream)
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1, o_ddairy_first=-1)
sequence_coffee2sugarcream = BehaviorSequence(state.State(data), targets, name="coffee 2 sugar cream")

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
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffeecreamsugar = ["g_1_make_coffee"] * len(actions_coffeecreamsugar)
midgoals_coffeecreamsugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 9 + ["g_2_stir"] * 6 +\
                            ["g_2_add_sugar"] * 4 + ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 +  ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffeecreamsugar, midgoals_coffeecreamsugar, actions_coffeecreamsugar)
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_ddairy_first=1)
sequence_coffeecreamsugar = BehaviorSequence(state.State(data), targets, name="coffee cream sugar")

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
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffeecream2sugar = ["g_1_make_coffee"] * len(actions_coffeecream2sugar)
midgoals_coffeecream2sugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 9 + ["g_2_stir"] * 6 +\
                            ["g_2_add_sugar"] * 8 + ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffeecream2sugar, midgoals_coffeecream2sugar, actions_coffeecream2sugar)
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1, o_ddairy_first=1)
sequence_coffeecream2sugar = BehaviorSequence(state.State(data), targets, name="coffee cream 2 sugar")

####################
# COFFEE WITH MILK #
####################
actions_coffeemilk = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Close cupboard
    "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # cream
    "a_fixate_fridge", "a_open",
    # no cream, have to get milk instead!
    "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffeemilk = ["g_1_make_coffee"] * len(actions_coffeemilk)
midgoals_coffeemilk = ["g_2_add_grounds"] * 11 + ["g_2_clean_up"] * 1 + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 2 + ["g_2_add_milk"] * 7 +\
                      ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffeemilk, midgoals_coffeemilk, actions_coffeemilk)
#Initial state has no cream! Hence the fallback on milk.
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, h_cream_present=0)
sequence_coffeemilk = BehaviorSequence(state.State(data), targets, name="coffee milk")

# Sugar then milk
actions_coffeesugarmilk = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # cream
    "a_fixate_fridge", "a_open",
    # Milk
    "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffeesugarmilk = ["g_1_make_coffee"] * len(actions_coffeesugarmilk)
midgoals_coffeesugarmilk = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 4 + ["g_2_clean_up"] * 2 \
                           + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 2 + ["g_2_add_milk"] * 7 + ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffeesugarmilk, midgoals_coffeesugarmilk, actions_coffeesugarmilk)
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, h_cream_present=0, o_ddairy_first=-1)
sequence_coffeesugarmilk = BehaviorSequence(state.State(data), targets, name="coffee sugar milk")

actions_coffee2sugarmilk = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Second sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Cream
    "a_fixate_fridge", "a_open",
    # Milk
    "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffee2sugarmilk = ["g_1_make_coffee"] * len(actions_coffee2sugarmilk)
midgoals_coffee2sugarmilk = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6  + ["g_2_add_sugar"] * 8 + ["g_2_clean_up"] * 2\
                            + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 2 + ["g_2_add_milk"] * 7 + ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffee2sugarmilk, midgoals_coffee2sugarmilk, actions_coffee2sugarmilk)
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1, h_cream_present=0, o_ddairy_first=-1)
sequence_coffee2sugarmilk = BehaviorSequence(state.State(data), targets, name="coffee 2 sugar milk")

# milk then sugar
actions_coffeemilksugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # cream
    "a_fixate_fridge", "a_open",
    # milk
    "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffeemilksugar = ["g_1_make_coffee"] * len(actions_coffeemilksugar)
midgoals_coffeemilksugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 2 + ["g_2_add_milk"] * 7 + ["g_2_stir"] * 6 + \
                           ["g_2_add_sugar"] * 4 + ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffeemilksugar, midgoals_coffeemilksugar, actions_coffeemilksugar)
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, h_cream_present=0, o_ddairy_first=1)
sequence_coffeemilksugar = BehaviorSequence(state.State(data), targets, name="coffee milk sugar")


actions_coffeemilk2sugar = [  # Grounds
    "a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
    "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # cream
    "a_fixate_fridge", "a_open",
    # milk
    "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink coffee
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_coffeemilk2sugar = ["g_1_make_coffee"] * len(actions_coffeemilk2sugar)
midgoals_coffeemilk2sugar = ["g_2_add_grounds"] * 11 + ["g_2_stir"] * 6 + ["g_2_add_cream"] * 2 + ["g_2_add_milk"] * 7 + ["g_2_stir"] * 6 + \
                           ["g_2_add_sugar"] * 8 + ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 + ["g_2_drink_coffee"] * 6
targets = _make_targets(topgoals_coffeemilk2sugar, midgoals_coffeemilk2sugar, actions_coffeemilk2sugar)
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1, h_cream_present=0, o_ddairy_first=1)
sequence_coffeemilk2sugar = BehaviorSequence(state.State(data), targets, name="coffee milk 2 sugars")

#                                            #################                                                         #
# -------------------------------------------- TEA SEQUENCES ----------------------------------------------------------#
#                                            #################                                                         #
############
# JUST TEA #
############
actions_tea = [  # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug", "a_add_to_mug",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # drink
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_tea"]
topgoals_tea = ["g_1_make_tea"] * len(actions_tea)
midgoals_tea = ["g_2_infuse_tea"] * 6 + ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 + ["g_2_drink_tea"] * 6
targets = _make_targets(topgoals_tea, midgoals_tea, actions_tea)
data = env.GoalEnvData(o_dtea=1)
sequence_tea = BehaviorSequence(state.State(data), targets, name="tea")

actions_teasugar = [  # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # add sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink tea
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_tea"]
topgoals_teasugar = ["g_1_make_tea"] * len(actions_teasugar)
midgoals_teasugar = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 4 + \
                       ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 +  ["g_2_drink_tea"] * 6
targets = _make_targets(topgoals_teasugar, midgoals_teasugar, actions_teasugar)
data = env.GoalEnvData(o_dtea=1, o_dsugar=1)
sequence_teasugar = BehaviorSequence(state.State(data), targets, name="tea sugar")

actions_tea2sugar = [   # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # add sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # add another sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink tea
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_tea"]
topgoals_tea2sugar = ["g_1_make_tea"] * len(actions_tea2sugar)
midgoals_tea2sugar = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 8 + ["g_2_clean_up"] * 2 + \
                        ["g_2_stir"] * 6 + ["g_2_drink_tea"] * 6
targets = _make_targets(topgoals_tea2sugar, midgoals_tea2sugar, actions_tea2sugar)
data = env.GoalEnvData(o_dtea=1, o_dsugar=1, o_dextrasugar=1)
sequence_tea2sugar = BehaviorSequence(state.State(data), targets, name="tea 2 sugar")

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
    # Drink tea
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_tea"]
topgoals_teamilk = ["g_1_make_tea"] * len(actions_teamilk)
midgoals_teamilk = ["g_2_infuse_tea"] * 6 + ["g_2_clean_up"] *2 + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 +\
                      ["g_2_drink_tea"] * 6
targets = _make_targets(topgoals_teamilk, midgoals_teamilk, actions_teamilk)
data = env.GoalEnvData(o_dtea=1, o_ddairy=1)
sequence_teamilk = BehaviorSequence(state.State(data), targets, name="tea milk")

# Sugar then milk
actions_teasugarmilk = [   # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink tea
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_tea"]
topgoals_teasugarmilk = ["g_1_make_tea"] * len(actions_teasugarmilk)
midgoals_teasugarmilk = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6  + ["g_2_add_sugar"] * 4 +["g_2_clean_up"] * 2 \
                           + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink_tea"] * 6
targets = _make_targets(topgoals_teasugarmilk, midgoals_teasugarmilk, actions_teasugarmilk)
data = env.GoalEnvData(o_dtea=1, o_ddairy=1, o_dsugar=1, o_ddairy_first=-1)
sequence_teasugarmilk = BehaviorSequence(state.State(data), targets, name="tea sugar milk")

actions_tea2sugarmilk = [   # dip teabag
    "a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug",
    "a_add_to_mug",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Second sugar
    "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug",
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Milk
    "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close",
    # Stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink tea
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_tea"]
topgoals_tea2sugarmilk = ["g_1_make_tea"] * len(actions_tea2sugarmilk)
midgoals_tea2sugarmilk = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6 + ["g_2_add_sugar"] * 8 +["g_2_clean_up"] * 2 \
                           + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + ["g_2_drink_tea"] * 6
targets = _make_targets(topgoals_tea2sugarmilk, midgoals_tea2sugarmilk, actions_tea2sugarmilk)
data = env.GoalEnvData(o_dtea=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1, o_ddairy_first=-1)
sequence_tea2sugarmilk = BehaviorSequence(state.State(data), targets, name="tea 2 sugar milk")

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
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink tea
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_tea"]
topgoals_teamilksugar = ["g_1_make_tea"] * len(actions_teamilksugar)
midgoals_teamilksugar = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + \
                           ["g_2_add_sugar"] * 4 + ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 + ["g_2_drink_tea"] * 6
targets = _make_targets(topgoals_teamilksugar, midgoals_teamilksugar, actions_teamilksugar)
data = env.GoalEnvData(o_dtea=1, o_ddairy=1, o_dsugar=1, o_ddairy_first=1)
sequence_teamilksugar = BehaviorSequence(state.State(data), targets, name="tea milk sugar")


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
    # Close cupboard
    "a_fixate_cupboard", "a_close",
    # stir
    "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
    # Drink tea
    "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_tea"]
topgoals_teamilk2sugar = ["g_1_make_tea"] * len(actions_teamilk2sugar)
midgoals_teamilk2sugar = ["g_2_infuse_tea"] * 6 + ["g_2_stir"] * 6 + ["g_2_add_milk"] * 9 + ["g_2_stir"] * 6 + \
                           ["g_2_add_sugar"] * 8 + ["g_2_clean_up"] * 2 + ["g_2_stir"] * 6 + ["g_2_drink_tea"] * 6
targets = _make_targets(topgoals_teamilk2sugar, midgoals_teamilk2sugar, actions_teamilk2sugar)
data = env.GoalEnvData(o_dtea=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1, o_ddairy_first=1)
sequence_teamilk2sugar = BehaviorSequence(state.State(data), targets, name="tea milk 2 sugars")

# Some sequences admit alternative, valid solutions (interchanging the order of sugar and dairy)
sequence_coffeesugarcream.alt_solutions.append(sequence_coffeecreamsugar)
sequence_coffeecreamsugar.alt_solutions.append(sequence_coffeesugarcream)
sequence_coffee2sugarcream.alt_solutions.append(sequence_coffeecream2sugar)
sequence_coffeecream2sugar.alt_solutions.append(sequence_coffee2sugarcream)

sequence_coffeesugarmilk.alt_solutions.append(sequence_coffeemilksugar)
sequence_coffeemilksugar.alt_solutions.append(sequence_coffeesugarmilk)
sequence_coffee2sugarmilk.alt_solutions.append(sequence_coffeemilk2sugar)
sequence_coffeemilk2sugar.alt_solutions.append(sequence_coffee2sugarmilk)

sequence_teasugarmilk.alt_solutions.append(sequence_teamilksugar)
sequence_teamilksugar.alt_solutions.append(sequence_teasugarmilk)
sequence_tea2sugarmilk.alt_solutions.append(sequence_teamilk2sugar)
sequence_teamilk2sugar.alt_solutions.append(sequence_tea2sugarmilk)

sequences_list =\
        [sequence_coffee, sequence_coffeesugar, sequence_coffee2sugar,  # 0, 1, 2
        sequence_coffeecream, # 3
        sequence_coffeesugarcream, sequence_coffee2sugarcream, # 4, 5
        sequence_coffeecreamsugar, sequence_coffeecream2sugar, # 6, 7
        sequence_coffeemilk, # 8
        sequence_coffeesugarmilk, sequence_coffee2sugarmilk, # 9, 10
        sequence_coffeemilksugar, sequence_coffeemilk2sugar, # 11, 12

        sequence_tea, sequence_teasugar, sequence_tea2sugar, # 13, 14, 15
        sequence_teamilk, # 16
        sequence_teamilksugar, sequence_teamilk2sugar,  # 17, 18
        sequence_teasugarmilk, sequence_tea2sugarmilk  # 19, 20
        ]


