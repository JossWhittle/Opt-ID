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


class ValidateMagnetGenomeErrorBase(Exception):
    """
    Base Exception to inherit from for magnet permutation errors.
    """


class ValidateMagnetGenomePermutationDuplicateError(ValidateMagnetGenomeErrorBase):
    """
    Exception to throw when a magnet permutation contains duplicate indices.
    """

    def __str__(self):
        return 'permutations must not contain duplicate magnet indices'


class ValidateMagnetGenomePermutationBoundaryError(ValidateMagnetGenomeErrorBase):
    """
    Exception to throw when a magnet permutation contains magnet indices that are either negative or greater than
    the size of the permutation.
    """

    def __init__(self, permutation_size : int, min_index : int, max_index : int):
        super().__init__()
        self._permutation_size = permutation_size
        self._min_index = min_index
        self._max_index = max_index

    @property
    def permutation_size(self):
        return self._permutation_size

    @property
    def min_index(self):
        return self._min_index

    @property
    def max_index(self):
        return self._max_index

    def __str__(self):
        return f'permutation must have all indices within the range [0, {self.permutation_size-1}]: ' \
               f'minimum index {self.min_index}, maximum index {self.max_index}'
