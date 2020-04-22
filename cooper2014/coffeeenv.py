# This is the coffee environment as originally proposed by Botvinick and Plaut (2004)
# Not everything is fully specified in Botvinick and Plaut, notably in terms of the internal state.
import numpy as np
import copy


class CoffeeEnv(object):
    def __init__(self):
        # The carton is cream
        self.fixated_inputs_keys = ["cup", "one-handle", "two-handles", "lid", "clear-liquid", "light", "brown-liquid",
                                    "carton", "open", "closed", "packet", "foil", "paper", "torn", "untorn", "spoon",
                                    "teabag", "sugar"]
        self.held_inputs_keys = self.fixated_inputs_keys + ["nothing"]
        self.actions_keys = ["pick-up", "put-down", "pour", "peel-open", "tear-open", "pull-open", "pull-off", "scoop",
                             "sip", "stir", "dip", "say-done", "fixate-cup", "fixate-teabag", "fixate-coffee-pack",
                             "fixate-spoon", "fixate-carton", "fixate-sugar", "fixate-sugar-bowl"]
        self.hidden_state_keys = ["sugar-bowl-open",
                                  "full", "half-empty", "empty",
                                  "clear-liquid", "brown-liquid", "light-",
                                  "sugar-packet", "sugar-bowl",
                                  "coffee-poured", "cream-poured", "sugar-poured",
                                  "coffee-stirred", "cream-stirred", "sugar-stirred", "tea-dipped",
                                  "literally-undrinkable"]

        self.num_fixated = len(self.fixated_inputs_keys)
        self.num_held = len(self.held_inputs_keys)
        self.num_actions = len(self.actions_keys)
        self.num_states = len(self.hidden_state_keys)

        # Zero-valued dictionaries. Practical for resetting stuff.
        self.fixated_inputs_dic = dict(zip(self.fixated_inputs_keys, [0.] * self.num_fixated))
        self.held_inputs_dic = dict(zip(self.held_inputs_keys, [0.] * self.num_held))
        self.actions_dic = dict(zip(self.actions_keys, [0.] * self.num_actions))
        self.states_dic = dict(zip(self.hidden_state_keys, [0.] * self.num_states))

        self.state = copy.deepcopy(self.states_dic)

    def transition(self, observation, action):
        """
        :param observation: numpy array encoding the observation
        :param state: dictionary encoding the hidden state
        :param action: numpy array encoding the action
        :return: next_observation (numpy array), reward
        """
        fix_dict, held_dict = self._array_to_dicts_obs(observation)
        action_dict = self._action_array_to_dict(action)
        next_fix, next_held, self.state, reward = self._transition_rules(fix_dict, held_dict, self.state, action_dict)
        return self._dicts_to_array_obs(next_fix, next_held), reward

    def set_up_coffee(self):
        self.state = copy.deepcopy(self.states_dic)
        self.state["clear-liquid"] = 1.
        self.state["full"] = 1.
        if np.random.randint(2):
            self.state["sugar-packet"] = 1.
            self.state["sugar-bowl"] = 0.
        else:
            self.state["sugar-packet"] = 0.
            self.state["sugar-bowl"] = 1.

        fixated = copy.deepcopy(self.fixated_inputs_dic)
        fixated["cup"] = 1.
        fixated["one-handle"] = 1.
        fixated["clear-liquid"] = 1.
        held = copy.deepcopy(self.held_inputs_dic)
        held["nothing"] = 1.
        return self._dicts_to_array_obs(fixated, held)

    def set_up_tea(self):
        return self.set_up_coffee()  # For now no difference

    def _dicts_to_array_obs(self, fixated, held):
        # np.array(fixated.values() + held.values()) # Not safe: dict.values does not preserve order
        return np.array([fixated[key] for key in self.fixated_inputs_keys] +
                        [held[key] for key in self.held_inputs_keys])

    def _array_to_dicts_obs(self, observation):
        observation = observation.tolist()
        fix_list = observation[:self.num_fixated]
        fix_dict = dict(zip(self.fixated_inputs_keys, fix_list))

        held_list = observation[self.num_fixated:]
        held_dict = dict(zip(self.held_inputs_keys, held_list))
        return fix_dict, held_dict

    def _action_array_to_dict(self, action_array):
        return dict(zip(self.actions_keys, action_array.tolist()))

    @staticmethod
    def _key_typo_check(allowed_keys, actual_keys):
        for key in actual_keys:
            if key not in allowed_keys:
                raise Exception("Key typo: {0}".format(key))

    # This contains the transition rules in a readable format for editing readability etc...
    def _transition_rules(self, fix, held, state, action):
        """
        Inefficient implementation but I find this less ugly to work with than hard-coding indexes
        into a bunch of consts and working directly with arrays... matter of taste?
        :param fix: fixated_input dictionary
        :param held: held_input dictionary
        :param state: hidden_state dictionary
        :param action: action dictionary
        :return: next_fix, next_held, next_state, next_action as dictionaries
        """
        next_state = copy.deepcopy(state)
        next_held = copy.deepcopy(held)
        next_fix = copy.deepcopy(fix)

        # This includes all the rules
        if action["fixate-cup"]:
            next_fix = copy.deepcopy(self.fixated_inputs_dic)  # reset to 0.
            next_fix["cup"] = 1.
            next_fix["one-handle"] = 1.
            if state["clear-liquid"]:
                next_fix["clear-liquid"] = 1.
            elif state["brown-liquid"]:
                next_fix["brown-liquid"] = 1.
            elif state["empty"]:
                next_fix["empty"] = 1.
            elif state["light-"]:
                next_fix["light-"] = 1.
            else:
                raise Exception("This shouldn't happen")
        elif action["fixate-teabag"]:
            next_fix = copy.deepcopy(self.fixated_inputs_dic)  # reset to 0.
            next_fix["teabag"] = 1.
        elif action["fixate-coffee-pack"]:
            next_fix = copy.deepcopy(self.fixated_inputs_dic)  # reset to 0.
            next_fix["packet"] = 1.
            next_fix["foil"] = 1.
            next_fix["untorn"] = 1.
        elif action["fixate-spoon"]:
            next_fix = copy.deepcopy(self.fixated_inputs_dic)  # reset to 0.
            next_fix["spoon"] = 1.
        elif action["fixate-carton"]:
            next_fix = copy.deepcopy(self.fixated_inputs_dic)  # reset to 0.
            next_fix["carton"] = 1.
            next_fix["closed"] = 1.
        elif action["fixate-sugar-bowl"]:
            if state["sugar-bowl"]:
                next_fix = copy.deepcopy(self.fixated_inputs_dic)  # reset to 0.
                if state["sugar-bowl-open"]:
                    next_fix["cup"] = 1.
                    next_fix["two-handles"] = 1.
                    next_fix["sugar"] = 1.
                else: #action failed - sugar bowl not yet opened
                    next_fix["cup"] = 1.
                    next_fix["two-handles"] = 1.
                    next_fix["lid"] = 1.
        elif action["fixate-sugar"]:
            if state["sugar-packet"]:
                next_fix = copy.deepcopy(self.fixated_inputs_dic)  # reset to 0.
                next_fix["packet"] = 1.
                next_fix["paper"] = 1.
                next_fix["untorn"] = 1.
            elif state["sugar-bowl"]:
                if not state["sugar-bowl-open"]:
                    next_fix = copy.deepcopy(self.fixated_inputs_dic)  # reset to 0.
                    next_fix["cup"] = 1.
                    next_fix["two-handles"] = 1.
                    next_fix["lid"] = 1.
            else:
                raise Exception("this shouldn't happen")
        elif action["pick-up"]:
            if held["nothing"]:
                next_held["nothing"] = 0.  # reset to 0.
                if fix["spoon"]:
                    next_held["spoon"] = 1.
                elif fix["packet"]:
                    next_held["packet"] = 1.
                    next_held["untorn"] = 1.
                    if fix["foil"]:
                        next_held["foil"] = 1.
                    elif fix["paper"]:
                        next_held["paper"] = 1.
                    else:
                        raise Exception("This shouldn't happen")
                    next_held["untorn"] = 1.
                elif fix["cup"]:
                    next_held["cup"] = 1.
                    next_held["one-handle"] = 1.
                    if state["clear-liquid"]:
                        next_held["clear-liquid"] = 1.
                    elif state["brown-liquid"]:
                        next_held["brown-liquid"] = 1.
                    elif state["empty"]:
                        next_held["empty"] = 1.
                    elif state["light-"]:
                        next_held["light-"] = 1.
                    else:
                        raise Exception("This shouldn't happen")
                elif fix["teabag"]:
                    next_held["teabag"] = 1.
                elif fix["carton"]:
                    next_held["carton"] = 1.
                    next_held["closed"] = 1.
                else: # Something is being fixed that can't be held
                    next_held["nothing"] = 1.
        elif action["put-down"]: # Drop whatever is being held
            # This resets some objects (packets) which are assumed thrown away.
            next_held = copy.deepcopy(self.held_inputs_dic) # reset to 0.
            next_held["nothing"] = 1.
        elif action["pour"]:  # if holding something pourable and fixating the cup, pour it in the cup
            if fix["cup"] and fix["one-handle"]:
                if held["spoon"] and held["sugar"]:
                    next_state["sugar-poured"] = 1.
                    next_held["sugar"] = 0.
                elif held["packet"] and held["paper"] and held["torn"]:
                    next_state["sugar-poured"] = 1.
                elif held["packet"] and held["foil"] and held["torn"]:
                    if next_state["clear-liquid"]:
                        next_state["clear-liquid"] = 0.
                        next_state["brown-liquid"] = 1.
                        next_state["coffee-poured"] = 1.
                    if fix["clear-liquid"]:
                        next_fix = copy.deepcopy(fix)  # again can't be too strong
                        next_fix["clear-liquid"] = 0.
                        next_fix["brown-liquid"] = 1.
                elif held["carton"] and held["open"]:
                    next_state["light-"] = 1.
                    next_fix["light-"] = 1.
                    next_state["cream-poured"] = 1.
        elif action["peel-open"]:
            if held["carton"] and held["closed"]:
                next_held["open"] = 1.
                next_held["closed"] = 0.
        elif action["tear-open"]:
            if held["packet"] and held["paper"] and held["untorn"]:
                next_held["torn"] = 1.
                next_held["untorn"] = 0.
        elif action["pull-open"]:
            if held["packet"] and held["foil"] and held["untorn"]:
                next_held["untorn"] = 0.
                next_held["torn"] = 1.
        elif action["pull-off"]:
            if held["nothing"] and fix["cup"] and fix["two-handles"] and fix["lid"]:
                next_held["nothing"] = 0.
                next_held["lid"] = 1.
                next_fix["lid"] = 0.
                next_fix["sugar"] = 1.
        elif action["scoop"]:
            if held["spoon"] and fix["cup"] and fix["two-handles"] and fix["sugar"]:
                held["sugar"] = 1.
        elif action["sip"]:
            if held["cup"] and held["one-handle"]:
                if state["half-empty"]:
                    next_state["half-empty"] = 0.
                    next_state["empty"] = 1.
                    next_state["light-"] = state["brown-liquid"] = state["clear-liquid"] = 0.
                elif state["full"]:
                    next_state["full"] = 0.
                    next_state["half-empty"] = 1.
        elif action["stir"]:
            if held["spoon"] and fix["cup"] and fix["one-handle"]:
                if state["coffee-poured"]:
                    next_state["coffee-stirred"] = 1.
                    next_state["coffee-poured"] = 0.
                if state["sugar-poured"]:
                    next_state["sugar-stirred"] = 1.
                    next_state["sugar-poured"] = 0.
                if state["cream-poured"]:
                    next_state["cream-stirred"] = 1.
                    next_state["cream-poured"] = 0.
        elif action["dip"]:
            if held["teabag"]:
                next_state["tea-dipped"] = 1.
                next_state["brown-liquid"] = 1.
        elif action["say-done"]:
            # Check whether successful?
            pass

        # Crimes against hot beverages checks
        if state["coffee-poured"] + state["cream-poured"] + state["sugar-poured"] >= 2.:
            next_state["literally-undrinkable"] = 1.  # Can't stir more than one thing at a time
        if (state["coffee-stirred"] or state["coffee-poured"]) and state["tea-dipped"]:
            next_state["literally-undrinkable"] = 1.  # Ew.
        if state["sugar-poured"] and not (state["coffee-stirred"] or state["tea-dipped"]):
            next_state["literally-undrinkable"] = 1.  # Coffee/tea goes before sugar
        if state["cream-poured"] and not state["coffee-stirred"]:
            next_state["literally-undrinkable"] = 1.  # Cream goes after coffee

        # Typo check: did I mistype a key?
        if __debug__:
            for (allowed, actual) in zip([self.hidden_state_keys, self.fixated_inputs_keys, self.held_inputs_keys],
                                         [next_state.keys(), next_fix.keys(), next_held.keys()]):
                self._key_typo_check(allowed, actual)

        # If this is an MDP at some point I'll set up some reward scheme.
        reward = 0.
        return next_fix, next_held, next_state, reward
