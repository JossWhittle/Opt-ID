# Copyright 2017 Diamond Light Source
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


import typing
import numpy as np


class StringEmptyError(Exception):
    """
    Exception to throw when a string is empty.
    """

    def __str__(self):
        return 'string must have a non-empty value'


class StringTypeError(Exception):
    """
    Exception to throw when a string is not a string.
    """

    def __init__(self, element_value : typing.Any):
        super().__init__()
        self._element_value = element_value

    @property
    def element_value(self):
        return self._element_value

    def __str__(self):
        return f'parameter with value {self.element_value} is not a string: ' \
               f'expected {str}, observed {type(self.element_value)}'


def validate_string(value : str,
                    assert_non_empty : bool = True):
    """
    Tests whether a python strings has the desired properties. Raises an exception on invalid inputs.

    Parameters
    ----------
    value : str
        A tensor whose size and dtype we want to validate match expected values.

    assert_non_empty : bool
        If true assert that the string has a non-empty value.

    Returns
    -------
    If the string is valid and matches the properties then return the string to allow streamlined assignment.
    """

    if not isinstance(value, str):
        raise StringTypeError(element_value=value)

    if assert_non_empty:
        if len(value) == 0:
            raise StringEmptyError()

    # Return the string if it is valid
    return value

