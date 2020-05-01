# This is the coffee environment as originally proposed by Botvinick and Plaut (2004)
# Not everything is fully specified in Botvinick and Plaut, notably in terms of the internal state.
# The implementation encapsulate rule-based transition logic using a "state" class with "match" and "set" functions,
# in order to organize the code a little bit and make it easier and less error-prone to maintain.
import numpy as np
import warnings
import state


class CoffeeEnv(object):
    def __init__(self):
        self.fixated_keys = ["cup", "one_handle", "two_handles", "lid", "clear_liquid", "light_", "brown_liquid",
                             "carton", "open", "closed", "packet", "foil", "paper", "torn", "untorn", "spoon",
                             "teabag", "sugar"]
        self.held_keys = self.fixated_keys + ["nothing"]
        # Prefixes to avoid having to worry about duplicate keys
        self.fixated_keys = ['f_' + key for key in self.fixated_keys]
        self.held_keys = ['h_' + key for key in self.held_keys]
        self.fixate_action_keys = ["fixate_cup", "fixate_teabag", "fixate_coffee_pack",
                                   "fixate_spoon", "fixate_carton", "fixate_sugar", "fixate_sugar_bowl"]
        self.do_action_keys = ["pick_up", "put_down", "pour", "peel_open", "tear_open", "pull_open", "pull_off",
                               "scoop", "sip", "stir", "dip", "say_done"]
        self.actions_keys = self.fixate_action_keys + self.do_action_keys
        self.hidden_state_keys = ["sugar_bowl_open",
                                  "coffee_packet_torn", "sugar_packet_torn", "carton_open",
                                  "how_full",
                                  "clear_liquid", "brown_liquid", "light_",
                                  "sugar_bowl",
                                  "coffee_poured", "cream_poured", "sugar_poured",
                                  "coffee_stirred", "cream_stirred", "sugar_stirred", "tea_dipped",
                                  "bad_taste", "disaster", "done", "reward"]
        self.hidden_state_keys = ['s_' + key for key in self.hidden_state_keys]
        self.observable_keys = self.fixated_keys + self.held_keys
        self.all_keys = self.actions_keys + self.observable_keys + self.hidden_state_keys
        assert len(self.all_keys) == len(set(self.all_keys))  # Check that there are no duplicates
        self.state = state.State(self.all_keys, self.observable_keys)
        self.state.str_func = self.state_to_string

    def transition(self, action):
        if isinstance(action, str):
            self.state.set_current(**{action: 1.})
        else:  # assume ndarray
            self.state.set_current(**(self.array_to_dict(self.actions_keys, action)))  # Set the action

        s = self.state  # Just to avoid writing "self.state" a couple hundred times

        # Now apply the matching rules and corresponding consequences.
        # Python, by "design", doesn't have a neat switch case break statement which would allow us to neatly separate
        # the different actions
        if s["pick_up"] and s["h_nothing"]:
            # Whatever is being fixated is now held. The next line is inelegant but I'd rather hardcode this than
            # hardcode the relationship between fixated and held objects
            s.set(h_one_handle=s["f_one_handle"], h_clear_liquid=s["f_clear_liquid"],
                  h_light_=s["f_light_"], h_brown_liquid=s["f_brown_liquid"],
                  h_two_handles=s["f_two_handles"], h_lid=s["f_lid"], h_sugar=s["f_sugar"],
                  h_carton=s["f_carton"], h_open=s["f_open"], h_closed=s["f_closed"],
                  h_packet=s["f_packet"], h_foil=s["f_foil"], h_paper=s["f_paper"],
                  h_torn=s["f_torn"], h_untorn=s["f_untorn"],
                  h_spoon=s["f_spoon"], h_teabag=s["f_teabag"])
            # If something was picked:
            if sum(s.get(*self.fixated_keys)) > 0.:
                s.set(h_nothing=0.)
        elif s["put_down"]:
            s.set(**dict(zip(self.held_keys, [0.]*len(self.held_keys))))
            s.set(h_nothing=1.)
        elif s["pour"]:
            if s["h_packet"] and s["h_foil"] and s["h_torn"]:  # We're pouring coffee...
                if s["f_cup"] and s["f_one_handle"]:  # ... into the coffee cup
                    s.set(s_coffee_poured=1., s_brown_liquid=1., f_brown_liquid=1., f_clear_liquid=0., s_clear_liquid=0.)
                else:
                    s.set(s_disaster=1.)  # into something else
            elif (s["h_packet"] and s["h_paper"] and s["h_torn"]) or (s["h_spoon"] and s["h_sugar"]):  # Pouring sugar
                if s["h_spoon"]:
                    s.set(h_sugar=0.)  # if it's from a spoon, there's no more sugar there
                if s["f_cup"] and s["f_one_handle"]:
                    s.set(s_sugar_poured=1.)  # ... into the coffee cup
                elif s["f_cup"] and s["f_two_handle"] and s["f_sugar"]:
                    pass  # ... into the sugar bowl
                else:
                    s.set(s_disaster=1.)  # into something else
            elif s["h_carton"] and s["h_open"]:  # We're pouring cream
                if s["f_cup"] and s["f_one_handle"]:
                    s.set(s_cream_poured=1., s_light_=1., f_light_=1.)
                else:
                    s.set(s_disaster=1.)
            elif s["h_cup"]:  # we're pouring either the contents of the coffee cup or the sugar cup
                s.set(s_disaster=1.)
        elif s["peel_open"] and s["h_carton"] and s["h_closed"] and s["f_carton"]:
            s.set(f_open=1., h_open=1., f_closed=0., h_closed=0., s_carton_open=1.)
        elif s["pull_open"] and s["f_packet"] and s["h_packet"] and s["h_untorn"] and s["h_foil"]:
            s.set(h_torn=1., h_untorn=0., f_torn=1., f_untorn=0., s_coffee_packet_torn=1.)
        elif s["tear_open"] and s["f_packet"] and s["h_packet"] and s["h_untorn"] and s["h_paper"]:
            s.set(h_torn=1., h_untorn=0., f_torn=1., f_untorn=0., s_sugar_packet_torn=1.)
        elif s["pull_off"] and s["f_cup"] and s["f_two_handles"] and s["f_lid"] and s["h_nothing"]:
            s.set(s_sugar_bowl_open=1., f_sugar=1., f_lid=0., h_lid=1., h_nothing=0.)
        elif s["scoop"] and s["h_spoon"] and s["f_sugar"]:
            s.set(h_sugar=1.)
        elif s["sip"] == 1.:
            if s["f_cup"] and s["f_two_handles"]:
                s.set(s_disaster=1.)  # drinking from sugar bowl
            elif s["f_cup"] and s["f_one_handle"]:  # drinking from tea/coffee cup
                if s["s_how_full"] == 1.:
                    s.set(s_how_full=0.5)
                elif s["s_how_full"] == 0.5:
                    s.set(s_how_full=0., s_clear_liquid=0., s_brown_liquid=0., s_light_=0.)
                if s["s_coffee_stirred"] and s["s_sugar_stirred"] and s["s_cream_stirred"] and not s["s_bad_taste"]:
                    s.set(s_reward=1.)  # good coffee
                elif s["s_tea_dipped"] and s["s_sugar_stirred"] and not s["s_bad_taste"]:
                    s.set(s_reward=1.)  # good tea
        elif s["stir"] and s["h_spoon"] and s["f_cup"]:
            if s["f_two_handles"]:
                pass  # Stirring the sugar bowl
            elif s["f_one_handle"]:
                s.set(s_coffee_stirred=s["s_coffee_poured"], s_sugar_stirred=s["s_sugar_poured"],
                      s_cream_stirred=s["s_cream_poured"])
        elif s["dip"] and s["h_teabag"]:
            if s["f_one_handle"] and s["s_how_full"] > 0.:
                s.set(s_tea_dipped=1., s_brown_liquid=1., s_clear_liquid=0., f_brown_liquid=1., f_clear_liquid=0.)
        elif s["say_done"]:
            s.set(s_done=1.)
        # Checks whether the action is one of the fixate actions. Could also use match_one
        elif sum(s.get(*self.fixate_action_keys)) == 1.:
            # Reset all fixated inputs
            s.set(**dict(zip(self.fixated_keys, [0.]*len(self.fixated_keys))))
            # Now deal with each action separately
            if s["fixate_cup"]:
                s.set(f_cup=1., f_one_handle=1.,
                      f_clear_liquid=s["s_clear_liquid"],
                      f_brown_liquid=s["s_brown_liquid"],
                      f_light_=s["s_light_"])
            elif s["fixate_teabag"]:
                s.set(f_teabag=1.)
            elif s["fixate_coffee_pack"]:
                s.set(f_packet=1., f_foil=1.,
                      f_torn=s["s_coffee_packet_torn"], f_untorn=1.-s["s_coffee_packet_torn"])
            elif s["fixate_spoon"]:
                s.set(f_spoon=1.)
            elif s["fixate_carton"]:
                s.set(f_carton=1., f_open=s["s_carton_open"], f_closed=1.-s["s_carton_open"])
            elif s["fixate_sugar"]:
                if s["s_sugar_bowl"]:
                    s.set(f_cup=1., f_two_handles=1.,
                          f_sugar=s["s_sugar_bowl_open"], f_lid=1.-s["s_sugar_bowl_open"])
                else:  # sugar packets it is
                    s.set(f_packet=1., f_paper=1.,
                          f_torn=s["s_sugar_packet_torn"], f_untorn=1.-s["s_sugar_packet_torn"])
            elif s["fixate_sugar_bowl"] and s["s_sugar_bowl"]:
                # Note: attempting to fixate the sugar bowl when there is no sugar bowl means the agent is not fixating
                # anything at all anymore
                s.set(f_cup=1., f_two_handles=1.,
                      f_sugar=s["s_sugar_bowl_open"],
                      f_lid=1.-s["s_sugar_bowl_open"])
        else:
            # noinspection PyUnreachableCode
            if __debug__:
                warnings.warn("No action matched or an inapplicable action was attempted")
        # Checks:
        # 1. Stir after every ingredient;
        # 2. coffee XOR tea;
        # 3. coffee before sugar/cream
        # 4. tea before sugar
        if sum(s.get("s_coffee_poured", "s_cream_poured", "s_sugar_poured")) -\
           sum(s.get("s_coffee_stirred", "s_cream_stirred", "s_sugar_stirred")) > 1 or \
           (s["s_tea_dipped"] and (s["s_coffee_stirred"] or s["s_coffee_poured"])) or \
           (s["s_cream_stirred"] and not s["s_coffee_stirred"]) or \
           (s["s_sugar_stirred"] and not(s["s_tea_dipped"] or s["s_coffee_stirred"])):
            s.set(s_bad_taste=1.)

        s.transition()

    @staticmethod
    def array_to_dict(keys_ordered, array):
        return dict(zip(keys_ordered, array.tolist()))

    def _test_final_state_coffee(self):
        return not self.state["s_how_full"] and not self.state["s_bad_taste"] and \
               not self.state["s_disaster"] and self.state["s_cream_stirred"] and self.state["s_sugar_stirred"]

    def _test_final_state_tea(self):
        return not self.state["s_how_full"] and not self.state["s_bad_taste"] and \
               not self.state["s_disaster"] and self.state["s_sugar_stirred"]

    def test_sequences(self):
        while True:
            self.initialize()
            if self.state["s_sugar_bowl"]:
                break
        actions = ["fixate_coffee_pack", "pick_up", "pull_open", "fixate_cup", "pour", "fixate_spoon", "put_down",
                   "pick_up", "fixate_cup", "stir",
                   "fixate_sugar", "put_down", "pull_off", "fixate_spoon", "put_down", "pick_up", "fixate_sugar_bowl",
                   "scoop", "fixate_cup", "pour", "stir",
                   "fixate_carton", "put_down", "pick_up", "peel_open", "fixate_cup", "pour", "fixate_spoon",
                   "put_down", "pick_up", "fixate_cup", "stir",
                   "put_down", "pick_up", "sip", "sip", "say_done"]
        print("\nTest: Coffee (sugar bowl, cream)")
        for i, action in enumerate(actions):
            print("step {0:2d}: {1}. Action: {2}".format(i + 1, self.state, action))
            self.transition(action)
        print("Test passed:{0}".format(bool(self._test_final_state_coffee())))

        while True:
            self.initialize()
            if self.state["s_sugar_bowl"] == 0.:
                break
        actions = ["fixate_coffee_pack", "pick_up", "pull_open", "fixate_cup", "pour", "fixate_spoon", "put_down",
                   "pick_up", "fixate_cup", "stir",
                   "fixate_carton", "put_down", "pick_up", "peel_open", "fixate_cup", "pour", "fixate_spoon",
                   "put_down", "pick_up", "fixate_cup", "stir",
                   "fixate_sugar", "put_down", "pick_up", "tear_open", "fixate_cup", "pour", "fixate_spoon",
                   "put_down", "pick_up", "fixate_cup", "stir",
                   "put_down", "pick_up", "sip", "sip", "say_done"]
        print("\nTest: Coffee (cream, sugar-packet):")
        for i, action in enumerate(actions):
            print("step {0:2d}: {1}. Action: {2}".format(i + 1, self.state, action))
            self.transition(action)
        print("Test passed:{0}".format(bool(self._test_final_state_coffee())))

        while True:
            self.initialize()
            if self.state["s_sugar_bowl"] == 0.:
                break
        actions = ["fixate_teabag", "pick_up", "fixate_cup", "dip", "fixate_sugar", "put_down",
                   "pick_up", "tear_open", "fixate_cup", "pour", "fixate_spoon", "put_down", "pick_up",
                   "fixate_cup", "stir", "put_down", "pick_up", "sip", "sip", "say_done"]
        print("\nTest:Coffee (cream, sugar-packet):")
        for i, action in enumerate(actions):
            print("step {0:2d}: {1}. Action: {2}".format(i + 1, self.state, action))
            self.transition(action)
        print("Test passed: {0}".format(bool(self._test_final_state_tea())))

    def initialize(self):
        self.state.reset_to_zero()
        self.state.set(f_cup=1., f_one_handle=1., f_clear_liquid=1., h_nothing=1., s_how_full=1., s_clear_liquid=1.,
                       s_sugar_bowl=np.random.randint(2))
        self.state.transition()

    @staticmethod
    def state_to_string(state_dict):
        state_str = "Fixated:"
        for key in state_dict:
            if key.startswith("f_") and state_dict[key] != 0.:
                state_str += " " + key[2:]
        state_str += ". Held:"
        for key in state_dict:
            if key.startswith("h_") and state_dict[key] != 0.:
                state_str += " " + key[2:]
        state_str += ". State:"
        for key in state_dict:
            if key.startswith("s_") and state_dict[key] != 0.:
                if key == "s_how_full":
                    state_str += " {0}%full".format(int(100*state_dict[key]))
                else:
                    state_str += " " + key[2:]
        return state_str