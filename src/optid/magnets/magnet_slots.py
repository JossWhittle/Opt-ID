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


import io
import typing
import nptyping as npt
import pickle
import numpy as np

import optid
from optid.utils import validate_tensor, validate_string, validate_string_list
from optid.errors import FileHandleError

logger = optid.utils.logging.get_logger('optid.magnets.MagnetSlots')


#                                     Matrix
Flip_Matrix_Type        = npt.NDArray[(3, 3), npt.Float]
#                                     Vector
Size_Type               = npt.NDArray[(3,), npt.Float]
#                                     Magnets,     Position
Positions_Type          = npt.NDArray[(typing.Any, 3), npt.Float]
#                                     Magnets,     Matrix
Direction_Matrices_Type = npt.NDArray[(typing.Any, 3, 3), npt.Float]


class MagnetSlots:
    """
    Represents a set of magnet slots. Magnet slots are characterized by beam name, position, 3x3 direction matrix,
    and flip vector. All magnet slots share a constant magnet size and type name allowing magnets from a MagnetSet
    class to be placed in arbitrary orders and flips via a MagnetPermutation.
    """

    def __init__(self,
                 magnet_type : str,
                 beams : typing.List[str],
                 slots : typing.List[str],
                 positions : Positions_Type,
                 direction_matrices : Direction_Matrices_Type,
                 size : Size_Type,
                 flip_matrix : Flip_Matrix_Type):
        """
        Constructs a MagnetSlots instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        magnet_type : str
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
            A float tensor of 3-dim positions for where to place each magnet slot within the device.

        direction_matrices : float tensor (S, 3, 3)
            A float tensor of 3x3 rotation matrices for what direction the magnet is transformed into, both geometry
            and field direction.

        size : float tensor (3,)
            A float tensor of a single 3-dim size for the reference magnet that is common to all slots.

        flip_matrix : float tensor (3, 3)
            A float matrix of shape (3, 3) representing the flips that would be applied to a magnet in order to swap
            the direction of its minor axis vectors while keeping its major (easy) axis vector.
        """

        try:
            self._magnet_type = validate_string(magnet_type, assert_non_empty=True)
        except Exception as ex:
            logger.exception('magnet_type must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._beams = validate_string_list(beams, assert_non_empty_list=True, assert_non_empty_strings=True)

            # Number of magnet slots derived from number of beam names provided. All other inputs must be consistent.
            self._count = len(self.beams)
        except Exception as ex:
            logger.exception('beams must be a non-empty list of non-empty strings', exc_info=ex)
            raise ex

        try:
            self._slots = validate_string_list(slots, shape=self.count,
                                               assert_non_empty_list=True, assert_non_empty_strings=True)
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
            self._direction_matrices = validate_tensor(direction_matrices, shape=(self.count, 3, 3))
        except Exception as ex:
            logger.exception('direction_matrices must be a float tensor of shape (S, 3, 3)', exc_info=ex)
            raise ex

        try:
            self._size = validate_tensor(size, shape=(3,))
        except Exception as ex:
            logger.exception('size must be a float tensor of shape (3,)', exc_info=ex)
            raise ex

        try:
            self._flip_matrix = validate_tensor(flip_matrix, shape=(3, 3))
            self._flippable = not np.allclose(self.flip_matrix, np.eye(3, dtype=np.float32))
        except Exception as ex:
            logger.exception('flip_matrix must be a float tensor of shape (3, 3)', exc_info=ex)
            raise ex

    @property
    def magnet_type(self) -> str:
        return self._magnet_type

    @property
    def beams(self) -> typing.List[str]:
        return self._beams

    @property
    def slots(self) -> typing.List[str]:
        return self._slots

    @property
    def positions(self) -> Positions_Type:
        return self._positions

    @property
    def direction_matrices(self) -> Direction_Matrices_Type:
        return self._direction_matrices

    @property
    def size(self) -> Size_Type:
        return self._size

    @property
    def flip_matrix(self) -> Flip_Matrix_Type:
        return self._flip_matrix

    @property
    def flippable(self) -> bool:
        return self._flippable

    @property
    def count(self) -> int:
        return self._count

    def save(self, file : typing.Union[str, typing.BinaryIO]):
        """
        Saves a MagnetSlots instance to a .magslots file.

        Parameters
        ----------
        file : str or open writable file handle
            A path to where a .magslots file should be created or overwritten, or an open writable file handle to
            a .magslots file.
        """

        def write_file(file_handle : typing.BinaryIO):
            """
            Private helper function for writing data to a .magslots file given an already open file handle.

            Parameters
            ----------
            file_handle : open writable file handle
                An open writable file handle to a .magslots file.
            """

            # Pack members into .magslots file as a single tuple
            pickle.dump((self.magnet_type, self.beams, self.slots, self.positions,
                         self.direction_matrices, self.size, self.flip_matrix), file_handle)

            logger.info('Saved magnet slots to .magslots file handle')

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logger.info('Saving magnet slots to .magslots file handle')
            write_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .magslots file in a closure to ensure it gets closed on error
            with open(file, 'wb') as file_handle:
                logger.info('Saving magnet slots to .magslots file [%s]', file)
                write_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()

    @staticmethod
    def from_file(file : typing.Union[str, typing.BinaryIO]) -> 'MagnetSlots':
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

        def read_file(file_handle : typing.BinaryIO) -> 'MagnetSlots':
            """
            Private helper function for reading data from a .magslots file given an already open file handle.

            Parameters
            ----------
            file_handle : open file handle
                An open file handle to a .magslots file.

            Returns
            -------
            A MagnetSet instance with the desired values loaded from the .magslots file.
            """

            # Unpack members from .magslots file as a single tuple
            (magnet_type, beams, slots, positions, direction_matrices, size, flip_matrix) = pickle.load(file_handle)

            # Offload object construction and validation to the MagnetSlots constructor
            magnet_slots = MagnetSlots(magnet_type=magnet_type, beams=beams, slots=slots, positions=positions,
                                       direction_matrices=direction_matrices, size=size, flip_matrix=flip_matrix)

            logger.info('Loaded magnet slots [%s] with [%d] slots', magnet_type, magnet_slots.count)

            return magnet_slots

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logger.info('Loading magnet set from .magslots file handle')
            return read_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .magslots file in a closure to ensure it gets closed on error
            with open(file, 'rb') as file_handle:
                logger.info('Loading magnet set from .magslots file [%s]', file)
                return read_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()
