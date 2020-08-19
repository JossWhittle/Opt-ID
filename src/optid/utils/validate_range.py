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

from optid.utils import Range


class ValidateRangeErrorBase(Exception):
    """
    Base Exception to inherit from for range errors.
    """


class ValidateRangeTypeError(ValidateRangeErrorBase):
    """
    Exception to throw when an axis range does not have the correct data types.
    """

    def __init__(self, dtype : typing.Any):
        super().__init__()
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __str__(self):
        return f'axis_range has incorrect types: expected {Range}, observed {self.dtype}'


class ValidateRangeElementTypeError(ValidateRangeErrorBase):
    """
    Exception to throw when an axis range does not have the correct data types.
    """

    def __init__(self, axis_range : Range):
        super().__init__()
        self._axis_range = axis_range

    @property
    def axis_range(self):
        return self._axis_range

    def __str__(self):
        return f'axis_range has incorrect types: expected (float, float, int), observed ' \
               f'({type(self.axis_range.min)}, {type(self.axis_range.max)}, {type(self.axis_range.steps)})'


class ValidateRangeBoundaryError(ValidateRangeErrorBase):
    """
    Exception to throw when an axis range has a maximum value less than its minimum value.
    """

    def __init__(self, axis_range : Range):
        super().__init__()
        self._axis_range = axis_range

    @property
    def axis_range(self):
        return self._axis_range

    def __str__(self):
        return f'axis_range has a maximum value less than its minimum value: ' \
               f'{self.axis_range.min} > {self.axis_range.max}'


class ValidateRangeStepsError(ValidateRangeErrorBase):
    """
    Exception to throw when an axis range has a step count less than or equal to zero.
    """

    def __init__(self, axis_range : Range):
        super().__init__()
        self._axis_range = axis_range

    @property
    def axis_range(self):
        return self._axis_range

    def __str__(self):
        return f'axis_range must have a positive step count greater than zero: {self.axis_range.steps}'


class ValidateRangeSingularityError(ValidateRangeErrorBase):
    """
    Exception to throw when an axis range has a maximum value equal to its minimum value but more than one step.
    """

    def __init__(self, axis_range : Range):
        super().__init__()
        self._axis_range = axis_range

    @property
    def axis_range(self):
        return self._axis_range

    def __str__(self):
        return f'axis_range has a maximum value equal to its minimum value but has multiple steps: ' \
               f'({self.axis_range.min} == {self.axis_range.max}) but steps is {self.axis_range.steps}'


def validate_range(axis_range : Range):
    """
    Tests whether a given range specification represents a valid positive monotonic range.
    Raises an exception on invalid range inputs.

    Parameters
    ----------
    axis_range : Range(min : float, max : float, steps : int)
        A range specification linearly transitioning between min and max over a given number steps.

    Returns
    -------
    If the range is valid and matches the expected properties then return the range to allow
    streamlined assignment.
    """

    if not isinstance(axis_range, Range):
        raise ValidateRangeTypeError(dtype=type(axis_range))

    if (not isinstance(axis_range.min, (float, int))) or \
       (not isinstance(axis_range.max, (float, int))) or \
       (not isinstance(axis_range.steps, int)):
        raise ValidateRangeElementTypeError(axis_range=axis_range)

    if axis_range.steps <= 0:
        raise ValidateRangeStepsError(axis_range=axis_range)

    if axis_range.min > axis_range.max:
        raise ValidateRangeBoundaryError(axis_range=axis_range)

    if (axis_range.min == axis_range.max) and (axis_range.steps != 1):
        raise ValidateRangeSingularityError(axis_range=axis_range)

    # Return the range if it is valid
    return axis_range
