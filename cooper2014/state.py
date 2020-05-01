# Helper class for coding rule-based environments.
# Essentially, when something is set it applies to the next state, not to the current state.

import numpy as np
import copy
import sys

DEBUG = sys.gettrace()


class State(object):
    def __init__(self, keys, sorted_observable_keys):
        """
        :param keys: keys that are allowed for modification
        :param sorted_observable_keys: keys that can be observed. All these keys must be set for a state to be valid
        :param assume_continuity: Whether a new state is initialized as a copy of the previous state
        """
        # This means unless otherwise specified, the next state is like the current state
        self.allowed_keys = keys
        self.observable_keys = sorted_observable_keys
        self.current = dict()
        self.next = dict()
        self.str_func = None

    def __getitem__(self, key):
        self._check_keys(*[key])
        return self.current[key]

    def reset(self, new):
        """
        :param new: a brand new state. The old state is erased
        """
        self._check_keys(new)
        self.next = dict()
        for key in new:
            self.next[key] = new[key]

    def reset_to_zero(self):
        self.next = dict(zip(self.allowed_keys, [0.]*len(self.allowed_keys)))

    def set(self, **kwargs):
        """
        :param kwargs: example - state.set(variable1=value1, variable2=value2,...). This changes the next state,
        not the current state. Call transition() to switch to the next state.
        """
        self._check_keys(**kwargs)
        for key, value in kwargs.items():
            self.next[key] = value

    def set_current(self, **kwargs):
        """
        :param kwargs: example - state.set(variable1=value1, variable2=value2,...). This changes the current state..
        """
        self._check_keys(**kwargs)
        for key, value in kwargs.items():
            self.current[key] = value

    def get(self, *argv):
        """
        :param argv: keys for state variables. E.g. get("variable1", "variable2",...)
        :return: corresponding values; as a list if there's more than 1 argument
        """
        self._check_keys(*argv)
        # Return the requested arguments as
        if len(argv) == 1:
            return self.current[argv[0]]
        else:
            return [self.current[key] for key in argv]

    def observe(self):
        """
        :return: an ndarray of the observable values (say, to serve as input for a neural network),
        in the order specified by observable_keys
        """
        return np.array([self.current[key] for key in self.observable_keys])

    def transition(self):
        """
        Move one time-step forward: current state becomes next state, next state is reinitialized
        """
        self.current = self.next
        self.next = copy.deepcopy(self.current)
        self._check_state(self.current)

    def __str__(self):
        if not self.str_func:
            summary = ""
            for key in self.allowed_keys:
                if self.current[key]:  # only non zero keys
                    summary += key + "; "
            return summary
        else:
            return self.str_func(self.current)

    def _check_keys(self, *argv, **kwargs):
        if DEBUG:
            for key in kwargs:
                if key not in self.allowed_keys:
                    raise Exception("Unknown key: "+key)
            for key in argv:
                if key not in self.allowed_keys:
                    raise Exception("Unknown key: "+key)

    def _check_state(self, state):
        if DEBUG:
            for key in self.observable_keys:
                assert key in state