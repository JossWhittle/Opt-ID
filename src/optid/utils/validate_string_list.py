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


import optid


class ValidateStringListErrorBase(Exception):
    """
    Base Exception to inherit from for string list errors.
    """


class ValidateStringListTypeError(ValidateStringListErrorBase):
    """
    Exception to throw when a string list is not a list.
    """

    def __init__(self, dtype):
        super().__init__()
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __str__(self):
        return f'string list is not a list: expected {list}, observed {self.dtype}'


class ValidateStringListEmptyError(ValidateStringListErrorBase):
    """
    Exception to throw when a string list does not have any elements.
    """

    def __str__(self):
        return 'list must have at least one element'


class ValidateStringListShapeError(ValidateStringListErrorBase):
    """
    Exception to throw when a string list does not have the expected length.
    """

    def __init__(self, expected_shape : int, observed_shape : int):
        super().__init__()
        self._expected_shape = expected_shape
        self._observed_shape = observed_shape

    @property
    def expected_shape(self):
        return self._expected_shape

    @property
    def observed_shape(self):
        return self._observed_shape

    def __str__(self):
        return f'list does not have the expected number of values: ' \
               f'expected ({self.expected_shape},), observed ({self.observed_shape},)'


class ValidateStringListElementTypeError(ValidateStringListErrorBase):
    """
    Exception to throw when a string list element is not a string.
    """

    def __init__(self, element_index : int, element_value : typing.Any):
        super().__init__()
        self._element_index = element_index
        self._element_value = element_value

    @property
    def element_index(self):
        return self._element_index

    @property
    def element_value(self):
        return self._element_value

    def __str__(self):
        return f'list element at index {self.element_index} with value {self.element_value} is not a string: ' \
               f'expected {str}, observed {type(self.element_value)}'


class ValidateStringListElementEmptyError(ValidateStringListErrorBase):
    """
    Exception to throw when a string list element an empty string.
    """

    def __init__(self, element_index : int):
        super().__init__()
        self._element_index = element_index

    @property
    def element_index(self):
        return self._element_index

    def __str__(self):
        return f'list element at index {self.element_index} is an empty string'


class ValidateStringListElementUniquenessError(ValidateStringListErrorBase):
    """
    Exception to throw when a string list contains duplicate elements.
    """

    def __str__(self):
        return 'list elements must all be unique strings'


def validate_string_list(values : optid.types.ListStrings,
                         shape : typing.Optional[int] = None,
                         assert_non_empty_list : bool = True,
                         assert_non_empty_strings : bool = True,
                         assert_unique_strings : bool = False):
    """
    Tests whether a python list of strings tensor has the number of elements expected, and that the string elements
    have values matching the given criteria. Raises an exception on invalid list inputs.

    Parameters
    ----------
    values : np.ndarray
        A list of string whose size and elements we want to validate match expected values.

    shape : None or int
        If shape is an int then assert the list has exactly that many elements.

    assert_non_empty_list : bool
        If true assert that the python list of strings contains at least one string.

    assert_non_empty_strings : bool
        If true assert that all string elements are non empty strings.

    assert_unique_strings : bool
        If true assert that all string elements have unique non-overlapping values.

    Returns
    -------
    If the list is valid and matches the expected length and properties then return the list to allow
    streamlined assignment.
    """

    if not isinstance(values, list):
        raise ValidateStringListTypeError(dtype=type(values))

    if shape is not None:
        if len(values) != shape:
            raise ValidateStringListShapeError(expected_shape=shape, observed_shape=len(values))

    if assert_non_empty_list:
        if len(values) == 0:
            raise ValidateStringListEmptyError()

    for index, value in enumerate(values):
        if type(value) != str:
            raise ValidateStringListElementTypeError(element_index=index, element_value=value)

        if assert_non_empty_strings:
            if len(value) == 0:
                raise ValidateStringListElementEmptyError(element_index=index)

    if assert_unique_strings:
        if len(set(values)) != len(values):
            raise ValidateStringListElementUniquenessError()

    # Return the list if it is valid
    return values

