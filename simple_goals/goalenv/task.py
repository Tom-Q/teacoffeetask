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
    def __init__(self, initial_state, targets=None, name="no-name", frequency=None):
        self.additional_info = None #placeholder for any analysis I wanna add to this
        self.targets = targets
        self.targets_nogoals = []
        self._initial_state = copy.deepcopy(initial_state)  # just as a safety...
        self.frequency = frequency  # used only for subsequences
        if targets is not None:
            self._actions_list = [t.action_one_hot for t in self.targets]
            self._goals1_list = [t.goal1_one_hot for t in self.targets]
            self._goals2_list = [t.goal2_one_hot for t in self.targets]
            self.length = len(self._actions_list)
            # A copy of targets without goals
            for target in targets:
                nogoal_target = copy.deepcopy(target)
                nogoal_target.goal1_str = None
                nogoal_target.goal2_str = None
                self.targets_nogoals.append(nogoal_target)
        else:
            self.length = 0
        self.alt_solutions = []
        self.name = name

    def equals(self, behavior_sequence, include_goals=True):
        if len(self.targets) != len(behavior_sequence.targets):
            return False

        for i in range(len(self.targets)):
            if self.targets[i].action_str != behavior_sequence.targets[i].action_str:
                return False
        # We consider sequences equals even if the goals differ.
        #    if include_goals and (self.targets[i].goal1_str != behavior_sequence.targets[i].goal1_str or
        #       self.targets[i].goal2_str != behavior_sequence.targets[i].goal2_str):
        #        return False
        return True

    def first_error_on_transition(self, behavior_sequence):
        # Check if errors occurred on a transition, meaning when the goal changed in the behavior sequence
        for i in range(len(self.targets)):
            if self.targets[i].action_str != behavior_sequence.targets[i].action_str:
                if i == 0 or self.targets[i-1].goal2_str != self.targets[i].goal2_str:
                    return True
                else:
                    return False
        return False

    def subsequence_analysis(self, behavior_sequence):
        omitted = added = repeated = more_frequent = False
        replaced, original, replacement, index = self._subsequence_replaced(behavior_sequence)
        if replaced:
            # a sequence is omitted if the replacement matches the next subsequence
            # identify next subsequence
            next_subsequence = self.identify_subseq(index + len(original.targets))
            # verify that the next subsequence and the replacement are identical
            omitted = next_subsequence is not None and next_subsequence.equals(replacement)
            if not omitted:
                next_subsequence = behavior_sequence.identify_subseq(index + len(replacement.targets))
                if next_subsequence is not None:
                    added = original.equals(next_subsequence)
                    if added:
                        prev_subsequence = self.identify_subseq(index - len(original.targets))
                        if prev_subsequence is not None:
                            repeated = prev_subsequence.equals(replacement)
            more_frequent = original.frequency < replacement.frequency

        is_a_target = False
        for sequence in sequences_list:
            if sequence.equals(behavior_sequence, include_goals=False):
                is_a_target = True
        return replaced, omitted, added, repeated, more_frequent, is_a_target

    def _subsequence_replaced(self, behavior_sequence):
        # Check for any complete subsequence swaps. So we have a target sequence (self), an actual sequence (behavior_sequence)
        # and a whole range of potential subsequences that might have gotten swapped with. We're going to look for the first error,
        # And then check if that error corresponds to any other subsequence. This can also be an omission or a repeat.
        if not hasattr(behavior_sequence, 'first_error') or\
                (hasattr(behavior_sequence, 'first_error') and behavior_sequence.first_error is None):
            raise Exception("This should not happen. first error should have been set by this point.")

        # Decompose the sequence according to goals. So we look for goal transitions. One exception, 2 sugars.
        # We tackle that separately.
        prev_target_goal = None
        prev_goal_idx = None
        for i in range(0, behavior_sequence.first_error + 1):
            if self.targets[i].goal2_str != prev_target_goal:
                prev_goal_idx = i
                prev_target_goal = self.targets[i].goal2_str
        if prev_target_goal == "g_2_add_sugar" and behavior_sequence.first_error - prev_goal_idx >= 4:
            prev_goal_idx += 4

        for subsequence in subsequences:
            identical = True
            for j, target in enumerate(subsequence.targets):
                if target.action_str != behavior_sequence.targets[prev_goal_idx+j].action_str:
                    identical = False
                    break
            if identical:
                target_ss = self.identify_subseq(prev_goal_idx)
                if target_ss is None:
                    raise Exception("Couldn't identify the target subsequence - this shouldn't be possible.")
                replacement_ss = subsequence
                return True, target_ss, replacement_ss, prev_goal_idx
        return False, None, None, None

    def identify_subseq(self, idx):
        for subsequence in subsequences:
            if len(self.targets[idx:]) < len(subsequence.targets):
                continue  # can't be this one, there's too few actions left.
            identical = True
            for i in range(len(subsequence.targets)):
                if subsequence.targets[i].action_str != self.targets[idx+i].action_str:
                    identical = False
                    break
            if identical:
                return subsequence
        return None


    @property
    def initial_state(self):
        state = copy.deepcopy(self._initial_state)
        # Special case for dairy first: 50% chance to just be 0.
        # The idea is to enforce the match between actions and goals.
        #if state.current.o_ddairy_first == 1 or state.current.o_ddairy_first == -1 and random.random() > 0.5:
        #    state.current.o_ddairy_first = 0
        #    state.next.o_ddairy_first = 0
        return state

    def set_targets(self, list_goals1, list_goals2, list_actions):
        self._actions_list = copy.deepcopy(list_actions)
        self._goals1_list = copy.deepcopy(list_goals1)
        self._goals2_list = copy.deepcopy(list_goals2)

        if len(list_actions) != len(list_goals1) or len(list_actions) != len(list_goals2):
            raise(ValueError("All target lists must have the same length"))

        self.targets = []
        self.targets_nogoals = []
        for i in range(len(list_actions)):
            self.targets.append(Target(utils.onehot_to_str(list_actions[i], env.GoalEnvData.actions_list),
                                       utils.onehot_to_str(list_goals1[i], env.GoalEnvData.goals1_list),
                                       utils.onehot_to_str(list_goals2[i], env.GoalEnvData.goals2_list)))
            self.targets_nogoals.append(Target(utils.onehot_to_str(list_actions[i], env.GoalEnvData.actions_list),
                                               None, None))

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


####################
#   SUBSEQUENCES   #
####################

# add grounds
actions_grounds = ["a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
                   "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down"]
topgoals_grounds = [None] * len(actions_grounds)
midgoals_grounds = ["g_2_add_grounds"]  * len(actions_grounds)
targets_grounds = _make_targets(topgoals_grounds, midgoals_grounds, actions_grounds)
subsequence_grounds = BehaviorSequence(None, targets_grounds, name="ss_add_grounds", frequency=13)

# infuse tea
actions_teabag = ["a_fixate_cupboard", "a_open", "a_fixate_teabags", "a_take", "a_fixate_mug", "a_add_to_mug"]
topgoals_teabag = [None] * len(actions_teabag)
midgoals_teabag = ["g_2_infuse_tea"] * len(actions_teabag)
targets_teabag = _make_targets(topgoals_teabag, midgoals_teabag, actions_teabag)
subsequence_teabag = BehaviorSequence(None, targets_teabag, name="ss_infuse_tea", frequency=8)

# clean up 1
actions_cleanup1 = ["a_close"]
topgoals_cleanup1 = [None] * len(actions_cleanup1)
midgoals_cleanup1 = ["g_2_clean_up"] * len(actions_cleanup1)
targets_cleanup1 = _make_targets(topgoals_cleanup1, midgoals_cleanup1, actions_cleanup1)
subsequence_cleanup1 = BehaviorSequence(None, targets_cleanup1, name="ss_cleanup1", frequency=21)

# clean up 2
actions_cleanup2 = ["a_fixate_cupboard", "a_close"]
topgoals_cleanup2 = [None] * len(actions_cleanup2)
midgoals_cleanup2 = ["g_2_clean_up"] * len(actions_cleanup2)
targets_cleanup2 = _make_targets(topgoals_cleanup2, midgoals_cleanup2, actions_cleanup2)
subsequence_cleanup2 = BehaviorSequence(None, targets_cleanup2, name="ss_cleanup2", frequency=21)

# milk1
actions_milk1 = ["a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close"]
topgoals_milk1 = [None] * len(actions_milk1)
midgoals_milk1 = ["g_2_clean_up"] * len(actions_milk1)
targets_milk1 = _make_targets(topgoals_milk1, midgoals_milk1, actions_milk1)
subsequence_milk1 = BehaviorSequence(None, targets_milk1, name="ss_milk1", frequency=10)

# milk2
actions_milk2 = ["a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close"]
topgoals_milk2 = [None] * len(actions_milk2)
midgoals_milk2 = ["g_2_clean_up"] * len(actions_milk2)
targets_milk2 = _make_targets(topgoals_milk2, midgoals_milk2, actions_milk2)
subsequence_milk2 = BehaviorSequence(None, targets_milk2, name="ss_milk2", frequency=10)

# cream 1
actions_cream1 = ["a_fixate_fridge", "a_open"]
topgoals_cream1 = [None] * len(actions_cream1)
midgoals_cream1 = ["g_2_add_cream"] * len(actions_cream1)
targets_cream1 = _make_targets(topgoals_cream1, midgoals_cream1, actions_cream1)
subsequence_cream1 = BehaviorSequence(None, targets_cream1, name="ss_cream1", frequency=10)

# cream 2
actions_cream2 = ["a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
    "a_fixate_fridge", "a_put_down", "a_close"]
topgoals_cream2 = [None] * len(actions_cream2)
midgoals_cream2 = ["g_2_add_cream"] * len(actions_cream2)
targets_cream2 = _make_targets(topgoals_cream2, midgoals_cream2, actions_cream2)
subsequence_cream2 = BehaviorSequence(None, targets_cream2, name="ss_cream2", frequency=10)

# sugar
actions_sugar = ["a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug"]
topgoals_sugar = [None] * len(actions_sugar)
midgoals_sugar = ["g_2_add_sugar"] * len(actions_sugar)
targets_sugar = _make_targets(topgoals_sugar, midgoals_sugar, actions_sugar)
subsequence_sugar = BehaviorSequence(None, targets_sugar, name="ss_sugar", frequency=22)

# stir
actions_stir = ["a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down"]
topgoals_stir = [None] * len(actions_stir)
midgoals_stir = ["g_2_stir"] * len(actions_stir)
targets_stir = _make_targets(topgoals_stir, midgoals_stir, actions_stir)
subsequence_stir = BehaviorSequence(None, targets_stir, name="ss_stir", frequency=52)

# Make special sequences that combine subsequence + stir
# sugar + stir
"""
actions_sugarstir = ["a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug", "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down"]
topgoals_sugarstir = [None] * len(actions_sugarstir)
midgoals_sugarstir = ["g_2_add_sugar"] * len(actions_sugar) + ["g_2_stir"] * len(actions_stir)
targets_sugarstir = _make_targets(topgoals_sugarstir, midgoals_sugarstir, actions_sugarstir)
subsequence_sugarstir = BehaviorSequence(None, targets_sugarstir, name="ss_sugarstir", frequency=XX)

# 2sugar + stir
actions_2sugarstir = ["a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug", "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug", "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down"]
topgoals_2sugarstir = [None] * len(actions_2sugarstir)
midgoals_2sugarstir = ["g_2_add_sugar"] * len(actions_sugar) * 2 + ["g_2_stir"] * len(actions_stir)
targets_2sugarstir = _make_targets(topgoals_2sugarstir, midgoals_2sugarstir, actions_2sugarstir)
subsequence_2sugarstir = BehaviorSequence(None, targets_2sugarstir, name="ss_2sugarstir", frequency=XX)
"""

# drink tea
actions_drinktea = ["a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_tea"]
topgoals_drinktea = [None] * len(actions_drinktea)
midgoals_drinktea = ["g_2_drink_tea"] * len(actions_drinktea)
targets_drinktea = _make_targets(topgoals_drinktea, midgoals_drinktea, actions_drinktea)
subsequence_drinktea = BehaviorSequence(None, targets_drinktea, name="ss_drinktea", frequency=8)

# drink coffee
actions_drinkcoffee = ["a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_good_coffee"]
topgoals_drinkcoffee = [None] * len(actions_drinkcoffee)
midgoals_drinkcoffee = ["g_2_drink_coffee"] * len(actions_drinkcoffee)
targets_drinkcoffee = _make_targets(topgoals_drinkcoffee, midgoals_drinkcoffee, actions_drinkcoffee)
subsequence_drinkcoffee = BehaviorSequence(None, targets_drinkcoffee, name="ss_drinkcoffee", frequency=13)

subsequences = [
    subsequence_cream1,
    subsequence_cream2,
    subsequence_milk1,
    subsequence_milk2,
    subsequence_stir,
    subsequence_sugar,
    subsequence_cleanup1,
    subsequence_cleanup2,
    subsequence_drinkcoffee,
    subsequence_drinktea,
    subsequence_grounds,
    subsequence_teabag
]

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
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1)#, o_ddairy_first=-1)
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
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1)#, o_ddairy_first=-1)
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
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1)#, o_ddairy_first=1)
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
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1)#, o_ddairy_first=1)
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
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, h_cream_present=0)#, o_ddairy_first=-1)
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
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1, h_cream_present=0)#, o_ddairy_first=-1)
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
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, h_cream_present=0)#, o_ddairy_first=1)
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
data = env.GoalEnvData(o_dcoffee=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1, h_cream_present=0)#, o_ddairy_first=1)
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
data = env.GoalEnvData(o_dtea=1, o_ddairy=1, o_dsugar=1)#, o_ddairy_first=-1)
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
data = env.GoalEnvData(o_dtea=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1)#, o_ddairy_first=-1)
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
data = env.GoalEnvData(o_dtea=1, o_ddairy=1, o_dsugar=1)#, o_ddairy_first=1)
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
data = env.GoalEnvData(o_dtea=1, o_ddairy=1, o_dsugar=1, o_dextrasugar=1)#, o_ddairy_first=1)
sequence_teamilk2sugar = BehaviorSequence(state.State(data), targets, name="tea milk 2 sugars")

# Some sequences admit alternative, valid solutions (interchanging the order of sugar and dairy)
# Except that's no longer true if there's an ordering instruction?!?
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


