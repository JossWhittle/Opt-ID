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
from optid.utils import validate_tensor, validate_string, validate_string_list
from optid.utils.logging import get_logger

logger = get_logger('optid.magnets.MagnetSlots')


class MagnetSlots:
    """
    Represents a set of magnet slots of a common magnet type across all usages within an insertion device.
    """

    def __init__(self,
                 mtype : str,
                 beams : optid.types.ListStrings,
                 slots : optid.types.ListStrings,
                 positions : optid.types.TensorPoints,
                 gap_vectors : optid.types.TensorVectors,
                 direction_matrices : optid.types.TensorMatrices):
        """
        Constructs a MagnetSlots instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        mtype : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        beams : list(str)
            A list of named identifiers for physical beams in the insertion device that each magnet of this type
            is placed in. Most devices will have two (PPM, CPMU) or four (APPLE) unique beam identifiers reused by
            multiple magnets. Magnets of a given type will most likely be present in all beams due to device symmetries.

        slots : list(str)
            A list of unique non-empty strings representing the named identifier for each physical magnet slot.
            Uniqueness of slot names must be guaranteed for slots with matching beam names.
            i.e. "{beam}:{slot}" must be unique.

        positions : float tensor (S, 3)
            A float tensor of S 3-dim positions for where to place each magnet slot within the device. Represents the
            centre of the AABB for the magnet slot.

        gap_vectors : float tensor (S, 3)
            A float tensor of S 3-dim unit vectors for the direction each slot should be moved in to represent
            phase (height) shimming. Usually these will be +Z and -Z axis vectors.

        direction_matrices : float tensor (S, 3, 3)
            A float tensor of 3x3 rotation matrices for what direction the magnet is transformed into, both geometry
            and field direction.
        """

        try:
            self._mtype = validate_string(mtype, assert_non_empty=True)
        except Exception as ex:
            logger.exception('mtype must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._beams = validate_string_list(beams, assert_non_empty_list=True, assert_non_empty_strings=True)

            # Number of magnet slots derived from number of beam names provided. All other inputs must be consistent.
            self._count = len(self.beams)
        except Exception as ex:
            logger.exception('beams must be a non-empty list of non-empty strings', exc_info=ex)
            raise ex

        try:
            self._slots = validate_string_list(slots, shape=self.count, assert_non_empty_list=True,
                                               assert_non_empty_strings=True)
        except Exception as ex:
            logger.exception('slots must be a non-empty list of non-empty strings', exc_info=ex)
            raise ex

        try:
            validate_string_list([f'{beam}:{slot}' for beam, slot in zip(beams, slots)], shape=self.count,
                                 assert_non_empty_list=True, assert_non_empty_strings=True, assert_unique_strings=True)
        except Exception as ex:
            logger.exception('"{beam}:{slot}" must be unique for all name pairs in beams and slots', exc_info=ex)
            raise ex

        try:
            self._positions = validate_tensor(positions, shape=(self.count, 3))
        except Exception as ex:
            logger.exception('positions must be a float tensor of shape (S, 3)', exc_info=ex)
            raise ex

        try:
            self._gap_vectors = validate_tensor(gap_vectors, shape=(self.count, 3))
        except Exception as ex:
            logger.exception('gap_vectors must be a float tensor of shape (S, 3)', exc_info=ex)
            raise ex

        try:
            self._direction_matrices = validate_tensor(direction_matrices, shape=(self.count, 3, 3))
        except Exception as ex:
            logger.exception('direction_matrices must be a float tensor of shape (S, 3, 3)', exc_info=ex)
            raise ex

    @property
    def mtype(self) -> str:
        return self._mtype

    @property
    def beams(self) -> optid.types.ListStrings:
        return self._beams

    @property
    def slots(self) -> optid.types.ListStrings:
        return self._slots

    @property
    def positions(self) -> optid.types.TensorPoints:
        return self._positions

    @property
    def gap_vectors(self) -> optid.types.TensorVectors:
        return self._gap_vectors

    @property
    def direction_matrices(self) -> optid.types.TensorMatrices:
        return self._direction_matrices

    @property
    def count(self) -> int:
        return self._count

    def save(self, file : optid.types.BinaryFileHandle):
        """
        Saves a MagnetSlots instance to a .magslots file.

        Parameters
        ----------
        file : str or open writable file handle
            A path to where a .magslots file should be created or overwritten, or an open writable file handle to
            a .magslots file.
        """

        logger.info('Saving magnet slots...')
        optid.utils.io.save(file, dict(
            mtype=self.mtype,
            beams=self.beams,
            slots=self.slots,
            positions=self.positions,
            gap_vectors=self.gap_vectors,
            direction_matrices=self.direction_matrices
        ))

    @staticmethod
    def from_file(file : optid.types.BinaryFileHandle) -> 'MagnetSlots':
        """
        Constructs a MagnetSlots instance from a .magslots file.

        Parameters
        ----------
        file : str or open file handle
            A path to a .magslots file or an open file handle to a .magslots file.

        Returns
        -------
        A MagnetSet instance with the desired values loaded from the .magslots file.
        """

        logger.info('Loading magnet slots...')
        return MagnetSlots(**optid.utils.io.from_file(file))
