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


class ValidateMagnetCutoutsErrorBase(Exception):
    """
    Base Exception to inherit from for magnet cutout errors.
    """


class ValidateMagnetCutoutsOverlapError(ValidateMagnetCutoutsErrorBase):
    """
    Exception to throw when a magnet cutout extends outside the size of the magnet.
    """

    def __init__(self,
                 magnet_cutout : npt.NDArray[(2, 3), npt.Float],
                 magnet_size : npt.NDArray[(3,), npt.Float]):
        super().__init__()
        self._magnet_size = magnet_size
        self._magnet_cutout = magnet_cutout

    @property
    def magnet_size(self):
        return self._magnet_size

    @property
    def magnet_cutout(self):
        return self._magnet_cutout

    def __str__(self):
        return f'cutout extends outside the size of the magnet: magnet_size {self.magnet_size}, ' \
               f'magnet_cutout position {self.magnet_cutout[0]} size {self.magnet_cutout[1]}'


def validate_magnet_cutouts(magnet_cutouts : npt.NDArray[(typing.Any, 2, 3), npt.Float],
                            magnet_size : npt.NDArray[(3,), npt.Float]):
    """
    Tests whether a given numpy tensor has the number of dimensions and shape matching a shape pattern, and that
    the dtype matches an expected dtype. Raises an exception on invalid tensor inputs.

    Parameters
    ----------
    magnet_cutouts : float tensor (C, 2, 3)
        A tensor of C pairs of 3-dim float vectors of shape (C, 2, 3) representing the constant position offset
        and size for all magnet cutout regions. These regions are applied to the magnet in its identity orientation
        before it is transformed by a MagnetSlots direction matrix.

    magnet_size : float tensor (3,)
            A single 3-dim float vector representing the constant size for all magnets in this set.

    Returns
    -------
    If the tensor is valid and matches the expected shape and type then return the tensor to allow
    streamlined assignment.
    """

    magnet_cutouts = validate_tensor(magnet_cutouts, shape=(None, 2, 3))
    magnet_size = validate_tensor(magnet_size, shape=(3,))

    for magnet_cutout in magnet_cutouts:
        bln_corner = magnet_cutout[0]
        trf_corner = magnet_cutout[0] + magnet_cutout[1]

        if np.any(bln_corner < 0) or np.any(trf_corner > magnet_size):
            raise ValidateMagnetCutoutsOverlapError(magnet_cutout=magnet_cutout, magnet_size=magnet_size)

    # Return the tensor if it is valid
    return magnet_cutouts
