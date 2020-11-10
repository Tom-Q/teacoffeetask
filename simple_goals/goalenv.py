import numpy as np
import copy
import sys
import dataclasses as dc
from dataclasses import dataclass
from termcolor import colored
import typing
from abc import ABC
import state
import utils
import neuralnet as nn
import tensorflow as tf
import scripts

# Location constants for objects
FRIDGE = 1
CUPBOARD = 2
TABLE = 3
HELD = 4


class ActionException(Exception):
    pass


@dataclass(repr=False)
class GoalEnvData(state.AbstractData):
    """
    Dataclasses offer a bunch of useful features for doing this,
    especially:
        * Easy conversion to dictionary
        * Self-generated constructor
    I'm using the variable names to define what they are. It would probably be more OO to use objects instead of fields
    and allow each field to have a name and a type... But this would be super bothersome to use ("Field.value")
    """

    # Observable state
    # Sequence indicators / goals
    o_sequence1: int = 0
    o_sequence2: int = 0
    o_sequence3: int = 0
    o_sequence4: int = 0
    o_sequence5: int = 0
    o_sequence6: int = 0

    # Containers
    o_fix_container_door_open: int = 0
    o_fix_cupboard: int = 0
    o_fix_fridge: int = 0
    o_fix_table: int = 0
    # cupboard contents
    o_fix_coffee_jar: int = 0
    o_fix_teabags: int = 0
    o_fix_sugar_box: int = 0
    # fridge contents
    o_fix_milk_carton: int = 0
    o_fix_cream_carton: int = 0
    # table contents
    o_fix_mug: int = 0
    o_fix_mug_full: int = 0
    o_fix_mug_milky: int = 0
    o_fix_mug_dark: int = 0
    o_fix_spoon: int = 0

    # Is the fixated object open (carton/jar/box)?
    o_fix_obj_open: int = 0

    # Held objects
    o_held_coffee_jar: int = 0
    o_held_sugar_cube: int = 0
    o_held_milk_carton: int = 0
    o_held_cream_carton: int = 0
    o_held_teabag: int = 0
    o_held_mug: int = 0
    o_held_spoon: int = 0
    o_held_nothing: int = 1

    # Non-observable state
    # Previously done things
    h_coffee_poured: int = 0
    h_coffee_stirred: int = 0
    h_sugar_poured: int = 0
    h_sugar_stirred: int = 0
    h_cream_poured: int = 0
    h_cream_stirred: int = 0
    h_milk_poured: int = 0
    h_milk_stirred: int = 0
    h_tea_dipped: int = 0

    # Other non-observable things
    # Ingredient availability
    h_mug_full: int = 1
    h_coffee_present: int = 1
    h_tea_present: int = 1
    h_milk_present: int = 1
    h_cream_present: int = 1
    h_sugar_present: int = 1
    # Location of objects
    h_location_coffee_jar: int = CUPBOARD
    h_location_milk_carton: int = FRIDGE
    h_location_cream_carton: int = FRIDGE
    h_location_spoon: int = TABLE
    h_location_mug: int = TABLE
    # Objects open?
    h_coffee_jar_open: int = 0
    h_milk_carton_open: int = 1
    h_cream_carton_open: int = 1
    h_sugar_box_open: int = 1
    h_teabags_box_open: int = 1
    h_fridge_open = 0
    h_cupboard_open = 0

    # Action
    a_fixate_cupboard: int = 0
    a_fixate_fridge: int = 0
    a_fixate_table: int = 0
    a_fixate_coffee_jar: int = 0
    a_fixate_sugar_box: int = 0
    a_fixate_mug: int = 0
    a_fixate_spoon: int = 0
    a_fixate_milk: int = 0
    a_fixate_cream: int = 0
    a_open: int = 0
    a_close: int = 0
    a_take: int = 0
    a_put_down: int = 0
    a_add_to_mug: int = 0
    a_stir: int = 0
    a_sip: int = 0
    a_say_done: int = 0

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
    # More practical things to have
    num_percepts: int = 0
    num_actions: int = 0
    num_goals1: int = 0
    num_goals2: int = 0

    def is_fixate_action(self):
        return np.sum(self._get_values("a_fixate")) > 0

    def reset_fixated(self):
        self._set_fields_to_value("o_fix", 0)

    def reset_held(self):
        self._set_fields_to_value("o_held", 0)
        self.o_held_nothing = 1

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
GoalEnvData.sorted_fields = sorted(dc.fields(GoalEnvData), key=lambda current_field: current_field.name)
GoalEnvData.actions_list = state.get_field_names('a_', GoalEnvData)
GoalEnvData.goals_list = state.get_field_names('g_', GoalEnvData)
GoalEnvData.goals1_list = state.get_field_names('g_1_', GoalEnvData)
GoalEnvData.goals2_list = state.get_field_names('g_2_', GoalEnvData)
GoalEnvData.observations_list = state.get_field_names('o_', GoalEnvData)
GoalEnvData.hiddens_list = state.get_field_names('h_', GoalEnvData)
GoalEnvData.all_list = [field.name for field in GoalEnvData.sorted_fields]
GoalEnvData.num_percepts = len(GoalEnvData.observations_list)
GoalEnvData.num_actions = len(GoalEnvData.actions_list)
GoalEnvData.num_goals1 = len(GoalEnvData.goals1_list)
GoalEnvData.num_goals2 = len(GoalEnvData.goals2_list)


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
            self._action_one_hot = utils.str_to_onehot(new_action_str, GoalEnvData.actions_list)

    @property
    def goal1_str(self):
        return self._goal1_str

    @goal1_str.setter
    def goal1_str(self, new_goal1_str):
        self._goal1_str = new_goal1_str
        if new_goal1_str is None:
            self._goal1_one_hot = None
        else:
            self._goal1_one_hot = utils.str_to_onehot(new_goal1_str, GoalEnvData.goals1_list)

    @property
    def goal2_str(self):
        return self._goal2_str

    @goal2_str.setter
    def goal2_str(self, new_goal2_str):
        self._goal2_str = new_goal2_str
        if new_goal2_str is None:
            self._goal2_one_hot = None
        else:
            self._goal2_one_hot = utils.str_to_onehot(new_goal2_str, GoalEnvData.goals2_list)

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
    def __init__(self, initialization_routine, targets=None):
        self.targets = targets
        self.initialize = initialization_routine

    # TODO: Factor all this code
    def get_actions_one_hot(self):
        actions_list = [target.action_one_hot for target in self.targets]
        return np.array(actions_list, dtype=float).reshape((-1, GoalEnvData.num_actions))

    def get_goals1_one_hot(self):
        goals1_list = [target.goal1_one_hot for target in self.targets]
        return np.array(goals1_list, dtype=float).reshape((-1, GoalEnvData.num_goals1))

    def get_goals2_one_hot(self):
        goals2_list = [target.goal2_one_hot for target in self.targets]
        return np.array(goals2_list, dtype=float).reshape((-1, GoalEnvData.num_goals2))

    def get_actions_inputs_one_hot(self, zeroed=False):
        if zeroed:
            return np.zeros_like(self.get_actions_one_hot())
        else:
            actions_list = [target.goal1_one_hot for target in self.targets]
            # Add a zero action at the beginning and delete the last action (which only serves as a target)
            actions_list = np.zeros_like(actions_list[0]) + actions_list[:-1]
            return actions_list

    def get_goal1s_inputs_one_hot(self, zeroed=False):
        if zeroed:
            return np.zeros_like(self.get_goals1_one_hot())
        else:
            goals1_list = [target.goal1_one_hot for target in self.targets]
            # Add a zero action at the beginning and delete the last action (which only serves as a target)
            goals1_list = np.zeros_like(goals1_list[0]) + goals1_list[:-1]
            return goals1_list

    def get_goal2s_inputs_one_hot(self, zeroed=False):
        if zeroed:
            return np.zeros_like(self.get_goals2_one_hot())
        else:
            goals2_list = [target.goal2_one_hot for target in self.targets]
            # Add a zero action at the beginning and delete the last action (which only serves as a target)
            goals2_list = np.zeros_like(goals2_list[0]) + goals2_list[:-1]
            return goals2_list


class GoalEnv(state.Environment):
    def __init__(self):
        super().__init__()
        self.state = state.State(GoalEnvData())
        self.sequences = self.initialize_sequences()

    def do_action(self, action, verbose=False):
        if isinstance(action, str):  # Action is a string resembling a field name, like "a_take_coffee_pack"
            self.state.current.set_field(action, 1.)
        else:  # Action is a numpy one-hot array
            self.state.current.set_actions(action)  # The mapping is alphabetical
        if verbose:
            print(self.state.current)
        self.transition()

    def _can_fix(self, item_location, item_present):
        if not item_present:
            return False
        if item_location == FRIDGE: return self.state.current.h_fridge_open
        if item_location == CUPBOARD: return self.state.current.h_cupboard_open
        if item_location == TABLE or item_location == HELD: return True

    def transition(self):
        # Check for any incoherent combinations
        self.sanity_check()

        # Make the code a bit more compact
        c = self.state.current
        n = self.state.next

        # 1. Fixation
        if c.is_fixate_action():
            # If a new fixation action takes place, reset all fixated inputs to 0
            n.reset_fixated()
        # 1. Go through fixation actions
        if c.a_fixate_cupboard:
            n.o_fix_cupboard = 1.
            n.o_fix_container_door_open = c.h_cupboard_open
            if c.h_cupboard_open:
                n.o_fix_coffee_jar = c.h_coffee_present * int(c.h_location_coffee_jar == CUPBOARD)
                n.o_fix_milk_carton = c.h_milk_present * int(c.h_location_milk_carton == CUPBOARD)
                n.o_fix_cream_carton = c.h_cream_present * int(c.h_location_cream_carton == CUPBOARD)
                n.o_fix_spoon = int(c.h_location_spoon == CUPBOARD)
                n.o_fix_mug = int(c.h_location_spoon == CUPBOARD)
                n.o_fix_teabags = c.h_tea_present
                n.o_fix_sugar_box = c.h_sugar_present
        elif c.a_fixate_fridge:
            n.o_fix_fridge = 1.
            n.o_fix_container_door_open = c.h_fridge_open
            if c.h_fridge_open:
                n.o_fix_coffee_jar = c.h_coffee_present * int(c.h_location_coffee_jar == FRIDGE)
                n.o_fix_milk_carton = c.h_milk_present * int(c.h_location_milk_carton == FRIDGE)
                n.o_fix_cream_carton = c.h_cream_present * int(c.h_location_cream_carton == FRIDGE)
                n.o_fix_spoon = (int(c.h_location_spoon == FRIDGE))
                n.o_fix_mug = (int(c.h_location_spoon == FRIDGE))
        elif c.a_fixate_table:
            n.o_fix_table = 1.
            n.o_fix_spoon = int(c.h_location_spoon == TABLE)
            n.o_fix_mug = int(c.h_location_spoon == TABLE)
            n.o_fix_coffee_jar = c.h_coffee_present * int(c.h_location_coffee_jar == TABLE)
            n.o_fix_milk_carton = c.h_milk_present * int(c.h_location_milk_carton == TABLE)
            n.o_fix_cream_carton = c.h_cream_present * int(c.h_location_cream_carton == TABLE)
        elif c.a_fixate_coffee_jar:
            if self._can_fix(n.h_location_coffee_jar, n.h_coffee_present):
                n.o_fix_coffee_jar = 1.
                n.o_fix_obj_open = c.h_coffee_jar_open
            else:
                raise ActionException("Impossible action: coffee jar can't be fixated")
        elif c.a_fixate_sugar_box:
            if self._can_fix(CUPBOARD, n.h_sugar_present):
                n.o_fix_sugar_box = 1.
                n.o_fix_obj_open = c.h_sugar_box_open
            else:
                raise ActionException("Impossible action: sugar box can't be fixated")
        elif c.a_fixate_mug:
            if self._can_fix(n.h_location_mug, True):
                n.o_fix_mug = 1.
                n.o_fix_mug_full = c.h_mug_full
                n.o_fix_mug_milky = max(c.h_cream_poured, c.h_milk_poured)
                n.o_fix_mug_dark = max(c.h_tea_dipped, c.h_coffee_poured)
            else:
                raise ActionException("Impossible action: mug can't be fixated")
        elif c.a_fixate_spoon:
            if self._can_fix(n.h_location_spoon, True):
                n.o_fix_spoon = 1.
            else:
                raise ActionException("Impossible action: spoon can't be fixated")
        elif c.a_fixate_milk:
            if self._can_fix(n.h_location_milk_carton, n.h_milk_present):
                n.o_fix_milk_carton = 1.
                n.o_fix_obj_open = c.h_milk_carton_open
            else:
                raise ActionException("Impossible action: milk can't be fixated")
        elif c.a_fixate_cream:
            if self._can_fix(n.h_location_cream_carton, n.h_cream_present):
                n.o_fix_cream_carton = 1.
                n.o_fix_obj_open = c.h_cream_carton_open
            else:
                raise ActionException("Impossible action: cream can't be fixated")
        # 2. Other actions
        elif c.a_take:
            # Holds the single fixated object if possible, otherwise nothing happens
            if c.o_held_nothing and c.o_fix_cupboard == 0 and c.o_fix_fridge == 0 and c.o_fix_table == 0:
                if c.o_fix_coffee_jar:
                    n.o_held_nothing = 0
                    n.o_held_coffee_jar = 1
                    n.h_location_coffee = HELD
                elif c.o_fix_teabags:
                    n.o_held_nothing = 0.
                    n.o_held_teabag = 1
                elif c.o_fix_sugar_box:
                    n.o_held_sugar_cube = 1
                    n.o_held_nothing = 0.
                elif c.o_fix_milk_carton:
                    n.o_held_nothing = 0.
                    n.o_held_milk_carton = 1
                    n.h_location_milk_carton = HELD
                elif c.o_fix_cream_carton:
                    n.o_held_nothing = 0.
                    n.o_held_cream_carton = 1
                    n.h_location_cream_carton = HELD
                elif c.o_fix_spoon:
                    n.o_held_nothing = 0.
                    n.o_held_spoon = 1.
                    n.h_location_spoon = HELD
                elif c.o_fix_mug:
                    n.o_held_nothing = 0.
                    n.o_held_mug = 1
                    n.h_location_mug = HELD
                else:
                    raise ActionException("Impossible action: object fixated can't be held")
            else:
                raise ActionException("Impossible action: either holding something already, or object can't be held")
        elif c.a_open:
            # if we're looking at something closed, opens it. Otherwise, nothing happens
            if (c.o_fix_fridge or c.o_fix_cupboard) and not c.o_fix_container_door_open:
                if c.o_fix_fridge:
                    n.h_fridge_open = 1
                    n.o_fix_container_door_open = 1
                    n.o_fix_milk_carton = c.h_milk_present * (int(c.h_location_milk_carton==FRIDGE))
                    n.o_fix_cream_carton = c.h_cream_present * (int(c.h_location_cream_carton==FRIDGE))
                elif c.o_fix_cupboard:
                    n.h_cupboard_open = 1
                    n.o_fix_container_door_open = 1
                    n.o_fix_coffee_jar = c.h_coffee_present * (int(c.h_location_coffee_jar == CUPBOARD))
                    n.o_fix_sugar_box = c.h_sugar_present
                    n.o_fix_teabags = c.h_tea_present
                else:
                    raise ActionException("Something went wrong in the open action")
            elif not c.o_fix_obj_open:
                # Open the object
                n.o_fix_obj_open = 1
                # Keep track that it is now open
                if c.o_fix_sugar_box: n.h_sugar_box_open = 1
                elif c.o_fix_coffee_jar: n.h_coffee_jar_open = 1
                elif c.o_fix_teabags: n.h_teabags_box_open = 1
                elif c.o_fix_milk: n.h_milk_carton_open = 1
                elif c.o_fix_cream: n.h_cream_carton_open = 1
                else: raise ActionException("Impossible action: object can't be opened")
            else:
                raise ActionException("Impossible action: object fixated can't be opened")
        elif c.a_close:
            if c.o_fix_container_door_open:
                if c.o_fix_fridge:
                    n.h_fridge_open = 0
                    n.o_fix_container_door_open = 0
                    n.o_fix_milk_carton = 0
                    n.o_fix_cream_carton = 0
                else: #if c.o_fix_cupboard:
                    n.h_cupboard_open = 0
                    n.o_fix_container_door_open = 0
                    n.o_fix_coffee_jar = 0
                    n.o_fix_sugar_box = 0
                    n.o_fix_teabags = 0
            elif c.o_fix_obj_open:
                # Close the object
                n.o_fix_obj_open = 0
                # Keep track that it is now closed
                if c.o_fix_sugar_box: n.h_sugar_box_open = 0
                elif c.o_fix_coffee_jar: n.h_coffee_jar_open = 0
                elif c.o_fix_teabags: n.h_teabags_box_open = 0
                elif c.o_fix_milk: n.h_milk_carton_open = 0
                elif c.o_fix_cream: n.h_cream_carton_open = 0
                else: raise ActionException("Impossible action: object fixated can't be closed")
            else:
                raise ActionException("Impossible action: object fixated can't be closed")
        elif c.a_put_down:
            if not c.o_held_nothing:
                # Put the item in the fridge
                new_loc = None
                if c.o_fix_fridge and c.o_fix_container_door_open:
                    new_loc = FRIDGE
                elif c.o_fix_cupboard and c.o_fix_container_door_open:
                    new_loc = CUPBOARD
                elif c.o_fix_table:
                    new_loc = TABLE
                else:
                    raise ActionException("Impossible action: nowhere to put object down")

                n.reset_held()
                if c.o_held_coffee_jar: n.h_location_coffee_jar = new_loc
                elif c.o_held_cream_carton: n.h_location_cream_carton = new_loc
                elif c.o_held_milk_carton: n.h_location_milk_carton = new_loc
                elif c.o_held_spoon: n.h_location_spoon = new_loc
                elif c.o_held_mug: n.h_location_mug = new_loc
                elif c.o_held_teabag: pass # teabag disappears
                elif c.o_held_sugar_cube: pass # sugar cube disappears
                else: raise ActionException("Impossible action: object can't be put down")
            else:
                raise ActionException("Impossible Action: already holding something")
        elif c.a_add_to_mug:
            if c.o_fix_mug:
                if c.o_held_cream_carton:
                    n.h_cream_poured = 1.
                    n.o_fix_mug_milky = 1.
                elif c.o_held_milk_carton:
                    n.h_milk_poured = 1.
                    n.o_fix_mug_milky = 1.
                elif c.o_held_sugar_cube:
                    n.h_sugar_poured = 1.
                    n.reset_held() # We're no longer holding a sugar cube
                elif c.o_held_coffee_jar:
                    n.h_coffee_poured = 1.
                    n.o_fix_mug_dark = 1.
                elif c.o_held_teabag:
                    n.h_tea_dipped = 1.
                    n.o_fix_mug_dark = 1.
                else: ActionException("Impossible Action: object cannot be poured in the mug")
            else:
                raise ActionException("Impossible Action: pouring in something other than the mug")
        elif c.a_stir:
            if c.o_held_spoon and c.o_fix_mug:
                if c.h_cream_poured: n.h_cream_stirred = 1
                if c.h_milk_poured: n.h_milk_stirred = 1
                if c.h_tea_dipped: n.h_tea_stirred = 1
                if c.h_coffee_poured: n.h_coffee_stirred = 1
                if c.h_sugar_poured: n.h_sugar_poured = 1
            else:
                raise ActionException("Impossible Action: can't stir without holding a spoon and looking at the mug")
        elif c.a_sip:
            if c.o_held_mug:
                n.h_mug_full = 0.
                # Reset mug
                if c.o_fix_mug:
                    n.o_fix_mug_full = 0.
                    n.o_fix_mug_milky = 0.
                    n.o_fix_mug_dark = 0.
                n.h_cream_poured = 0
                n.h_milk_poured = 0
                n.h_coffee_poured = 0
                n.h_sugar_poured = 0
                n.h_tea_dipped = 0
                n.h_cream_stirred = 0
                n.h_milk_stirred = 0
                n.h_coffee_stirred = 0
                n.h_sugar_stirred = 0
                n.h_tea_stirred = 0
            else:
                raise ActionException("Impossible action: drinking without holding the mug")
        elif c.a_say_done:
            pass  # Not sure if this serves a purpose

        # Activate the transition!!
        self.state.next_time_step()

    def observe(self):
        return self.state.current.get_observable()

    def sanity_check(self):
        c = self.state.current
        # Exactly one action is executed at a time
        if np.sum(c.get_actions()) != 1.:
            raise Exception("Exactly one action must be executed at each step")
        if (c.o_held_coffee_jar + c.o_held_sugar_cube + c.o_held_milk_carton + c.o_held_cream_carton +
            c.o_held_teabag + c.o_held_spoon + c.o_held_mug + c.o_held_nothing) != 1:
            raise Exception("Can't hold two things at once (or something and nothing)")

    def reinitialize(self):
        self.state = state.State(GoalEnvData())

    def initialize_sequences(self):
        # Action sequence 1: black coffee.
        actions_sequence1 = ["a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
                             "a_add_to_mug", "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
                             "a_close",
                             "a_fixate_spoon", "a_take", "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down",
                             "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
        sequence1 = BehaviorSequence(self.reinitialize, [Target(action=action) for action in actions_sequence1])

        # Action sequence 2: coffee with sugar
        actions_sequence2 = ["a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
                             "a_add_to_mug",  "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down",
                             "a_fixate_spoon",  "a_take", "a_fixate_mug", "a_stir"]
        """
                              ,
                             "a_fixate_table", "a_put_down", "a_fixate_sugar_box", "a_take", "a_fixate_mug",
                             "a_add_to_mug", "a_fixate_spoon", "a_take",
                             "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down", "a_fixate_cupboard", "a_close",
                             "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]"""
        sequence2 = BehaviorSequence(self.reinitialize, [Target(action=action) for action in actions_sequence2])

        # Action sequence 3: coffee with sugar and milk (in this order)
        actions_sequence3 = ["a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
                             "a_add_to_mug",
                             "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down", "a_fixate_spoon",
                             "a_take", "a_fixate_mug", "a_stir",
                             "a_fixate_table", "a_put_down",
                             "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug", "a_fixate_spoon", "a_take",
                             "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down", "a_fixate_cupboard", "a_close",
                             "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
                             "a_fixate_fridge", "a_put_down", "a_close", "a_fixate_spoon", "a_take", "a_fixate_mug",
                             "a_stir",
                             "a_fixate_table", "a_put_down",
                             "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
        sequence3 = BehaviorSequence(self.reinitialize, [Target(action=action) for action in actions_sequence3])

        # Action sequence 4: coffee with milk and sugar (in this order)
        actions_sequence4 = ["a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
                             "a_add_to_mug",
                             "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down", "a_fixate_spoon",
                             "a_take", "a_fixate_mug", "a_stir",
                             "a_fixate_table", "a_put_down",
                             "a_fixate_fridge", "a_open", "a_fixate_milk", "a_take", "a_fixate_mug", "a_add_to_mug",
                             "a_fixate_fridge", "a_put_down", "a_close", "a_fixate_spoon", "a_take", "a_fixate_mug",
                             "a_stir",
                             "a_fixate_table", "a_put_down",
                             "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug", "a_fixate_spoon", "a_take",
                             "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down", "a_fixate_cupboard", "a_close",
                             "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
        sequence4 = BehaviorSequence(self.reinitialize, [Target(action=action) for action in actions_sequence4])

        # Action sequence 5: coffee with cream and sugar (in this order)
        actions_sequence5 = ["a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
                             "a_add_to_mug",
                             "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down", "a_fixate_spoon",
                             "a_take", "a_fixate_mug", "a_stir",
                             "a_fixate_table", "a_put_down",
                             "a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
                             "a_fixate_fridge", "a_put_down", "a_close", "a_fixate_spoon", "a_take", "a_fixate_mug",
                             "a_stir",
                             "a_fixate_table", "a_put_down",
                             "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug", "a_fixate_spoon", "a_take",
                             "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down", "a_fixate_cupboard", "a_close",
                             "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
        sequence5 = BehaviorSequence(self.reinitialize, [Target(action=action) for action in actions_sequence5])

        # Action sequence 6: coffee with sugar and cream (in this order)
        actions_sequence6 = ["a_fixate_cupboard", "a_open", "a_fixate_coffee_jar", "a_take", "a_open", "a_fixate_mug",
                             "a_add_to_mug",
                             "a_fixate_coffee_jar", "a_close", "a_fixate_cupboard", "a_put_down", "a_fixate_spoon",
                             "a_take", "a_fixate_mug", "a_stir",
                             "a_fixate_table", "a_put_down",
                             "a_fixate_sugar_box", "a_take", "a_fixate_mug", "a_add_to_mug", "a_fixate_spoon", "a_take",
                             "a_fixate_mug", "a_stir", "a_fixate_table", "a_put_down", "a_fixate_cupboard", "a_close",
                             "a_fixate_fridge", "a_open", "a_fixate_cream", "a_take", "a_fixate_mug", "a_add_to_mug",
                             "a_fixate_fridge", "a_put_down", "a_close", "a_fixate_spoon", "a_take", "a_fixate_mug",
                             "a_stir",
                             "a_fixate_table", "a_put_down",
                             "a_fixate_mug", "a_take", "a_sip", "a_fixate_table", "a_put_down", "a_say_done"]
        sequence6 = BehaviorSequence(self.reinitialize, [Target(action=action) for action in actions_sequence6])
        return [sequence1, sequence2, sequence3, sequence4, sequence5, sequence6]

    def test_environment(self):
        for i, sequence in enumerate(self.sequences):
            self.reinitialize()
            print("\nSequence number " + str(i+1))
            for target in sequence.targets:
                self.do_action(target.action_str, verbose=True)


def train(model = None, goals=False, num_iterations=50000, learning_rate=0.001, L2_reg = 0.000001, noise = 0., sequences=None):
    if sequences is None:
        sequences = [0]
    env = GoalEnv()
    if model is None:
        if not goals:
            model = nn.NeuralNet(size_hidden=50, size_observation=29, size_action=17,  size_goal1=0, size_goal2=0,
                                 algorithm=nn.RMSPROP, learning_rate=learning_rate)
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
            model.context = np.zeros((1, model.size_hidden), dtype=np.float32)
            for i, target in enumerate(sequence.targets):
                model.action = np.zeros((1, model.size_action), dtype=np.float32)
                # Add noise to context layer
                model.context += np.float32(np.random.normal(0., noise, size=(1, model.size_hidden)))
                observation = env.observe()
                model.feedforward(observation)

            # Get some statistics about the percentage of correct behavior
            actions = np.array(model.h_action_wta).reshape((-1, GoalEnvData.num_actions))
            target_actions = sequence.get_actions_one_hot()
            ratio_actions = scripts.ratio_correct(actions, target_actions)
            if goals:
                goals1 = np.array(model.h_goal1_wta).reshape((-1, GoalEnvData.num_goals1))
                target_goals1 = sequence.get_actions_one_hot()
                ratio_goals1 = scripts.ratio_correct(goals1, target_goals1)

                goals2 = np.array(model.h_goal2_wta).reshape((-1, GoalEnvData.num_goals1))
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