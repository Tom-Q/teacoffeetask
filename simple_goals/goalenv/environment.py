import numpy as np
import dataclasses as dc
from dataclasses import dataclass
import typing
from goalenv import state
import utils

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
    a_fixate_teabags: int = 0
    a_open: int = 0
    a_close: int = 0
    a_take: int = 0
    a_put_down: int = 0
    a_add_to_mug: int = 0
    a_stir: int = 0
    a_sip: int = 0
    a_say_done: int = 0

    # Top-level goals
    g_1_make_coffee: int = 0
    g_1_make_tea: int = 0

    # Mid-level goals
    g_2_add_grounds: int = 0
    g_2_stir: int = 0
    g_2_add_sugar: int = 0
    g_2_infuse_tea: int = 0
    g_2_add_milk: int = 0
    g_2_add_cream: int = 0
    g_2_clean_up: int = 0
    g_2_drink: int = 0

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
TERMINAL_ACTION = "a_say_done"


class GoalEnv(state.Environment):
    def __init__(self):
        super().__init__()
        self.state = state.State(GoalEnvData())
        #self.sequences = self.initialize_sequences()

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
        elif c.a_fixate_teabags:
            if self._can_fix(CUPBOARD, n.h_tea_present):
                n.o_fix_teabags = 1.
                n.o_fix_obj_open = c.h_teabags_box_open
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
                elif c.o_fix_milk_carton: n.h_milk_carton_open = 1
                elif c.o_fix_cream_carton: n.h_cream_carton_open = 1
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
                elif c.o_fix_milk_carton: n.h_milk_carton_open = 0
                elif c.o_fix_cream_carton: n.h_cream_carton_open = 0
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
                    n.reset_held() # We're no longer holding the teabag
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

    def reinitialize(self, initial_state=None):
        if initial_state is None:
            initial_state = state.State(GoalEnvData())
        self.state = initial_state

    def test_environment(self, sequences):
        for i, sequence in enumerate(sequences):
            self.reinitialize()
            print("\nSequence number " + str(i + 1))
            for target in sequence.targets:
                self.do_action(target.action_str, verbose=True)
