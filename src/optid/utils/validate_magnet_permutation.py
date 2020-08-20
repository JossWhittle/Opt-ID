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
import nptyping as npt
import numpy as np

from optid.utils import validate_tensor


class ValidateMagnetPermutationErrorBase(Exception):
    """
    Base Exception to inherit from for magnet permutation errors.
    """


class ValidateMagnetPermutationDuplicateError(ValidateMagnetPermutationErrorBase):
    """
    Exception to throw when a magnet permutation contains duplicate indices.
    """

    def __str__(self):
        return 'permutations must not contain duplicate magnet indices'


class ValidateMagnetPermutationBoundaryError(ValidateMagnetPermutationErrorBase):
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


class ValidateMagnetPermutationFlipError(ValidateMagnetPermutationErrorBase):
    """
    Exception to throw when a magnet permutation contains invalid flip values.
    """

    def __init__(self, min_flip : int, max_flip : int):
        super().__init__()
        self._min_flip = min_flip
        self._max_flip = max_flip

    @property
    def min_flip(self):
        return self._min_flip

    @property
    def max_flip(self):
        return self._max_flip

    def __str__(self):
        return f'permutation must have all flip values within the range [0, 1]: ' \
               f'minimum flip {self.min_flip}, maximum flip {self.max_flip}'


def validate_magnet_permutation(magnet_permutation : npt.NDArray[(typing.Any, 2), npt.Int]):
    """
    Tests whether a given numpy tensor represents a valid magnet permutation of ordering and flips.

    Parameters
    ----------
    magnet_permutation : int tensor (M, 2)
            A tensor of integer value pairs representing MagnetSet index [0, M-1] in the first column
            and flip state [0, 1] in the second column.

    Returns
    -------
    If the tensor is valid and matches the expected shape and type then return the tensor to allow
    streamlined assignment.
    """

    magnet_permutation = validate_tensor(magnet_permutation, shape=(None, 2), dtype=np.integer)

    count = magnet_permutation.shape[0]
    permutation, flips = magnet_permutation[:, 0], magnet_permutation[:, 1]

    if len(set(permutation.tolist())) != count:
        raise ValidateMagnetPermutationDuplicateError()

    if np.any((permutation < 0) | (permutation >= count)):
        raise ValidateMagnetPermutationBoundaryError(permutation_size=count,
                                                     min_index=permutation.min(),
                                                     max_index=permutation.max())

    if not np.all((flips == 0) | (flips == 1)):
        raise ValidateMagnetPermutationFlipError(min_flip=flips.min(), max_flip=flips.max())

    # Return the tensor if it is valid
    return magnet_permutation
