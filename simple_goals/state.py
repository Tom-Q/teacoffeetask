import numpy as np
import copy
import dataclasses as dc
from dataclasses import dataclass
from termcolor import colored
from abc import ABC

# Practical method for the dataclass.
def get_field_names(condition, class_type):
    return [field.name for field in class_type.sorted_fields if field.name.startswith(condition)]


# Base class for TeaCoffeeData - a data class with no data
@dataclass(repr=False)
class AbstractData(ABC):

    def __post_init__(self):
        # Enforce class attributes in child classes. I tried to do this using metaclasses but couldn't make it work.
        # Fortunately __post_init__ can be used for this.
        required_attributes = ["sorted_fields", "category_tuples"]
        for attribute in required_attributes:
            if not (hasattr(self, attribute)):
                raise Exception("Child classes of AbstractData must implement the class attribute " + attribute)


    def _get_values(self, condition):
        # Return a np array with only the fields satisfying the condition, sorted alphabetically.
        # 1. Get the fields
        as_dict = dc.asdict(self)
        field_values = []
        for field in self.sorted_fields:
            if field.name.startswith(condition):
                field_values.append(as_dict[field.name])

        # 2. Make a 2-d (row) np array
        return np.array(field_values, dtype=np.float32).reshape((1, -1))

    def set_field(self, field_name, value):
        # Just a wrapper over built-in setattr...
        setattr(self, field_name, value)

    def _set_fields(self, condition, nparray):
        nparray = nparray.flatten()

        # Get the fields
        fields = []
        for field in self.sorted_fields:  # This must be defined in the inheriting class
            if field.name.startswith(condition):
                field += field

        if len(fields) == nparray.size:
            raise Exception("The array size must match exactly the number of fields to fill-in")

        for i, field in enumerate(fields):
            self.set_field(field.name, nparray[i])

    def set_goals(self, nparray):
        self._set_fields("g_", nparray)

    def set_actions(self, nparray):
        self._set_fields("a_", nparray)

    def __str__(self):
        return self._stringify(colored=True)

    def __repr__(self):
        return self._stringify(colored=False)

    def _stringify(self, colored=True):
        """
        :return: A string representing the data (for debugging and printing)
        """
        non_zero_fields = []
        for key, val in dc.asdict(self).items():
            if val:
                non_zero_fields.append(key)

        # This must be implemented by the child class.
        # Example usage:  [('Action', 'a_', 'green'), ('Observable', 'o_', 'blue')]
        cats = self.category_tuples
        string = ''
        for cat in cats:
            if colored:
                string += self._str_category(non_zero_fields, cat[0], cat[1], cat[2])
            else:
                string += self._str_category(non_zero_fields, cat[0], cat[1])
        return string[:-2] + '.'

    @staticmethod
    def _str_category(field_names, category_name, field_identifier, color=None):
        if color is not None:
            string = colored(category_name, color) + ': '
        else:
            string = category_name + ': '
        count = 0
        for f in field_names:
            if f.startswith(field_identifier):
                string += f[2:] + ', '
                count += 1
        if count == 0:
            string += '(None); '
        else:
            string = string[:-2] + '; '
        return string


class State(object):
    """
    Just contains current and next state, and allows for transitioning from current to next.
    This is just to avoid any bug that could occur from updating the state "in place".
    """
    def __init__(self, data: AbstractData):
        self.current = copy.copy(data)
        self.next = copy.copy(data)

    def next_time_step(self):
        self.current = self.next
        self.next = copy.copy(self.current)

    def reset(self):
        self.current = self.current.__class__()  # To reinitialize to default values, make a new instance of the data.
        self.next = copy.copy(self.current)