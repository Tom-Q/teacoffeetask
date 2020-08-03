import numpy as np
import copy
import sys
import dataclasses as dc
from dataclasses import dataclass
from termcolor import colored
import typing
from abc import ABC
import state


@dataclass(repr=False)
class TeaCoffeeData(state.AbstractData):
    """
    Dataclasses offer a bunch of useful features for doing this,
    especially:
        * Easy conversion to dictionary
        * Self-generated constructor
    I'm using the variable names to define what they are. It would probably be more OO to use objects instead of fields
    and allow each field to have a name and a type... But this would be super bothersome to use ("Field.value")
    """
    # TODO: I'm not using these goals after all.
    # Goals: high-level
    g_1_none: int = 0.
    g_1_make_tea: int = 0.
    g_1_make_coffee: int = 0.
    # Goal: low-level
    g_2_none: int = 0.
    g_2_add_grounds: int = 0.
    g_2_add_sugar: int = 0.
    g_2_dip_teabag: int = 0.
    g_2_add_cream: int = 0.
    g_2_drink: int = 0.

    # Observable state
    # State of the drink
    o_brown: int = 0.
    o_clear: int = 1.
    o_light: int = 0.
    o_how_full: float = 1.

    # Held items
    o_sugar_pack: int = 0.
    o_coffee_pack: int = 0.
    o_teabag: int = 0.
    o_spoon: int = 0.
    o_cup: int = 0.
    o_carton: int = 0.
    o_nothing: int = 1.
    # Status of held object
    o_open: int = 0.

    # Non-observable state
    # Previously done things
    h_coffee_poured: int = 0.
    h_coffee_stirred: int = 0.
    h_sugar_poured: int = 0.
    h_sugar_stirred: int = 0.
    h_cream_poured: int = 0.
    h_cream_stirred: int = 0.
    h_tea_dipped: int = 0.

    # Action
    a_take_coffee_pack: int = 0.
    a_take_sugar_pack: int = 0.
    a_take_teabag: int = 0.
    a_take_carton: int = 0.
    a_take_cup: int = 0.
    a_take_spoon: int = 0.
    a_pour: int = 0.
    a_open: int = 0.
    a_sip: int = 0.
    a_dip: int = 0.
    a_put_down: int = 0.
    a_stir: int = 0.

    # Special fields = class fields
    sorted_fields: typing.ClassVar[list]
    actions_list: typing.ClassVar[list]
    goals_list: typing.ClassVar[list]
    goals1_list: typing.ClassVar[list]
    goals2_list: typing.ClassVar[list]
    observations_list: typing.ClassVar[list]
    hiddens_list: typing.ClassVar[list]
    category_tuples: typing.ClassVar[list] = [('Goal', 'g_', 'green'),  ('Observable', 'o_', 'blue'),
                                              ('Hidden', 'h_', 'yellow'),  ('Action', 'a_', 'red')]

    def get_observable(self):
        # return a np array with only the observable fields, sorted alphabetically.
        return self._get_values("o_")

    def get_goals(self):
        return self._get_values("g_")

    def get_actions(self):
        return self._get_values("a_")

    def get_lvl1_goals(self):
        return self._get_values("g_1_")

    def get_lvl2_goals(self):
        return self._get_values("g_2_")

# Initialize the class variables... This has to be done outside the class definition
TeaCoffeeData.sorted_fields = sorted(dc.fields(TeaCoffeeData), key=lambda current_field: current_field.name)
TeaCoffeeData.actions_list = state.get_field_names('a_', TeaCoffeeData)
TeaCoffeeData.goals_list = state.get_field_names('g_', TeaCoffeeData)
TeaCoffeeData.goals1_list = state.get_field_names('g_1_', TeaCoffeeData)
TeaCoffeeData.goals2_list = state.get_field_names('g_2_', TeaCoffeeData)
TeaCoffeeData.observations_list = state.get_field_names('o_', TeaCoffeeData)
TeaCoffeeData.hiddens_list = state.get_field_names('h_', TeaCoffeeData)
TeaCoffeeData.all_list = [field.name for field in TeaCoffeeData.sorted_fields]


# This class implements the transition rules: what happens to the state when we go from one time-step to the next.
class TeaCoffeeEnv(object):
    def __init__(self):
        self.state = state.State(TeaCoffeeData())

    def do_action(self, action, verbose=False):
        if isinstance(action, str):  # Action is a string resembling a field name, like "a_take_coffee_pack"
            self.state.current.set_field(action, 1.)
        else:  # Action is a numpy one-hot array
            self.state.current.set_actions(action)  # The mapping is alphabetical
        if verbose:
            print(self.state.current)
        self.transition()

    def transition(self):
        # Check for any incoherent combinations
        self.sanity_check()

        # Make the code a bit more compact
        c = self.state.current
        n = self.state.next

        # Pick-up actions
        if c.o_nothing and (c.a_take_coffee_pack or c.a_take_sugar_pack or c.a_take_carton
                            or c.a_take_spoon or c.a_take_teabag or c.a_take_cup):
            n.o_nothing = 0.
            if c.a_take_coffee_pack: n.o_coffee_pack = 1.
            elif c.a_take_sugar_pack: n.o_sugar_pack = 1.
            elif c.a_take_carton: n.o_carton = 1.
            elif c.a_take_spoon: n.o_spoon = 1.
            elif c.a_take_teabag: n.o_teabag = 1.
            elif c.a_take_cup: n.o_cup = 1.
        # Pour
        if c.a_pour and c.o_open:  # not elif
            if c.o_coffee_pack:
                n.h_coffee_poured = 1.
                n.o_clear = 0.
                n.o_brown = 1.
            elif c.o_sugar_pack:
                n.h_sugar_poured = 1.
            elif c.o_carton:
                n.h_cream_poured = 1.
                n.o_clear = 0.
                n.o_light = 1.

        # Open
        elif c.a_open and (c.o_coffee_pack or c.o_sugar_pack or c.o_carton):
            n.o_open = 1.
        # Sip
        elif c.a_sip and c.o_cup:
            n.o_how_full = 0.
        # Dip
        elif c.a_dip and c.o_teabag:
            n.h_tea_dipped = 1.
        # Put down
        elif c.a_put_down:
            n.o_carton = n.o_spoon = n.o_coffee_pack = n.o_sugar_pack = n.o_cup = n.o_teabag = 0.
            n.o_nothing = 1.
            n.o_open = 0.
        # Stir
        elif c.a_stir:
            if c.h_coffee_poured: n.h_coffee_stirred = 1.
            if c.h_sugar_poured: n.h_sugar_stirred = 1.
            if c.h_cream_poured: n.h_cream_stirred = 1.

        self.state.next_time_step()

    def observe(self):
        return self.state.current.get_observable()

    def sanity_check(self):
        c = self.state.current
        # Exactly one action is executed at a time
        if np.sum(c.get_actions()) != 1.:
            raise Exception("Exactly one action must be executed at each step")
        if c.o_clear and (c.o_light or c.o_brown):
            raise Exception("The liquid cannot be both clear and colored")
        if (c.o_coffee_pack + c.o_sugar_pack + c.o_carton + c.o_teabag + c.o_spoon + c.o_cup + c.o_nothing) != 1:
            raise Exception("Can't hold two things at once (or something and nothing)")


#class TeaCoffeeSupervisor(object):
#    """
#   Sets (hard-coded) target goals and target actions for the TeaCoffeeEnv agent.
#    """
#    def __init__(self):
#TODO: This is a mess. e.g. why is this a class instance??
action_list = {}
goal_list = {}

# For each goal, set up a sequence of actions:
# Low-level goals
action_list["g_2_add_grounds"] = ["a_take_coffee_pack", "a_open", "a_pour", "a_put_down", "a_take_spoon", "a_stir"]
action_list["g_2_add_cream"] = ["a_put_down", "a_take_carton", "a_open", "a_pour", "a_put_down", "a_take_spoon", "a_stir"]
action_list["g_2_add_sugar"] = ["a_put_down", "a_take_sugar_pack", "a_open", "a_pour", "a_put_down", "a_take_spoon", "a_stir"]
action_list["g_2_drink"] = ["a_put_down", "a_take_cup", "a_sip"]
action_list["g_2_dip_teabag"] = ["a_take_teabag", "a_dip"]

# High level goals
action_list["g_1_make_coffee"] = action_list["g_2_add_grounds"] + action_list["g_2_add_cream"] + action_list["g_2_add_sugar"] + action_list["g_2_drink"]
action_list["g_1_make_tea"] = action_list["g_2_dip_teabag"] + action_list["g_2_add_sugar"] + action_list["g_2_drink"]

# More complicated sequence of goals and subgoals:
for low_level_goal in TeaCoffeeData.goals2_list:
    if low_level_goal is not "g_2_none":
        goal_list[low_level_goal] = [["g_1_none", low_level_goal]] * len(action_list[low_level_goal])

goal_list["g_1_make_coffee"] = [["g_1_make_coffee", "g_2_add_grounds"]] * 6 + \
                               [["g_1_make_coffee", "g_2_add_cream"]] * 7 + \
                               [["g_1_make_coffee", "g_2_add_sugar"]] * 7 + \
                               [["g_1_make_coffee", "g_2_drink"]] * 3
goal_list["g_1_make_tea"] = [["g_1_make_tea", "g_2_dip_teabag"]] * 2 + \
                            [["g_1_make_tea", "g_2_add_sugar"]] * 7 + \
                            [["g_1_make_tea", "g_2_drink"]] * 3

# Put all of this in a single "targets" list.
target_list = {}
for goal in action_list:
    targets = []
    for step in range(len(action_list[goal])):
        targets.append([action_list[goal][step]] + goal_list[goal][step])
    target_list[goal] = targets


def test_goals(self):
    test_env = TeaCoffeeEnv()
    for goal in self.action_list:
        # Reset the state.
        test_env.state.reset()
        print("Goal: " + goal)
        for action in self.action_list[goal]:
            test_env.do_action(action, verbose=True)
        print(test_env.state.current)
        print("\n")

