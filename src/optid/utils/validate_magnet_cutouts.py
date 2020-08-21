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


class ValidateMagnetCutoutsBoundaryError(ValidateMagnetCutoutsErrorBase):
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
               f'magnet_cutout bottom-left-near corner {self.magnet_cutout[0]} ' \
               f'top-right-far corner {self.magnet_cutout[1]}'


class ValidateMagnetCutoutsOverlapError(ValidateMagnetCutoutsErrorBase):
    """
    Exception to throw when two magnet cutouts overlap.
    """

    def __init__(self, magnet_cutouts : npt.NDArray[(2, 2, 3), npt.Float]):
        super().__init__()
        self._magnet_cutouts = magnet_cutouts

    @property
    def magnet_cutouts(self):
        return self._magnet_cutouts

    def __str__(self):
        return f'cutouts overlap their areas of influence: ' \
               f'cutout A bottom-left-near corner {self.magnet_cutouts[0, 0]} ' \
               f'top-right-far corner {self.magnet_cutouts[0, 1]}, ' \
               f'cutout B bottom-left-near corner {self.magnet_cutouts[1, 0]} ' \
               f'top-right-far corner {self.magnet_cutouts[1, 1]}, '


class ValidateMagnetCutoutsSizeError(ValidateMagnetCutoutsErrorBase):
    """
    Exception to throw when a magnet cutout has an invalid size.
    """

    def __init__(self, magnet_cutout : npt.NDArray[(2, 3), npt.Float]):
        super().__init__()
        self._magnet_cutout = magnet_cutout

    @property
    def magnet_cutout(self):
        return self._magnet_cutout

    def __str__(self):
        return f'cutout shape either has zero or negative size in at least one axis: ' \
               f'bottom-left-near corner {self.magnet_cutout[0]} ' \
               f'top-right-far corner {self.magnet_cutout[1]}, size {self.magnet_cutout[1] - self.magnet_cutout[0]}'


def validate_magnet_cutouts(magnet_cutouts : npt.NDArray[(typing.Any, 2, 3), npt.Float],
                            magnet_size : npt.NDArray[(3,), npt.Float]):
    """
    Tests whether a given numpy tensor is a valid set of magnet cutouts for a magnet of a given size.
    Raises an exception on invalid tensor inputs.

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

    for a_index, a_cutout in enumerate(magnet_cutouts):
        if np.any(a_cutout[0] >= a_cutout[1]):
            raise ValidateMagnetCutoutsSizeError(magnet_cutout=a_cutout)

        if np.any(a_cutout[0] < 0) or np.any(a_cutout[1] > magnet_size):
            raise ValidateMagnetCutoutsBoundaryError(magnet_cutout=a_cutout, magnet_size=magnet_size)

        a_centre = np.mean(a_cutout, axis=0)
        a_extent = (a_centre - a_cutout[0])

        for b_index, b_cutout in enumerate(magnet_cutouts[(a_index + 1):]):

            b_centre = np.mean(b_cutout, axis=0)
            b_extent = (b_centre - b_cutout[0])

            if np.all(np.abs(b_centre - a_centre) < (a_extent + b_extent)):
                raise ValidateMagnetCutoutsOverlapError(
                    magnet_cutouts=magnet_cutouts[[a_index, (a_index + 1 + b_index)], ...])

    # Return the tensor if it is valid
    return magnet_cutouts
