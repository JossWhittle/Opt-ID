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

from optid.utils import validate_tensor, validate_string, validate_string_list
from optid.errors import FileHandleError

import optid
logger = optid.utils.logging.get_logger('optid.magnets.MagnetSlots')


class MagnetSlots:
    """
    Represents a set of magnet slots. Magnet slots are characterized by beam name, position, 3x3 direction matrix,
    and flip vector. All magnet slots share a constant magnet size and type name allowing magnets from a MagnetSet
    class to be placed in arbitrary orders and flips via a MagnetPermutation.
    """

    def __init__(self,
                 magnet_type : str,
                 magnet_size : npt.NDArray[(3,), npt.Float],
                 magnet_beams : typing.List[str],
                 magnet_positions : npt.NDArray[(typing.Any, 3), npt.Float],
                 magnet_direction_matrices : npt.NDArray[(typing.Any, 3, 3), npt.Float],
                 magnet_flip_vectors : npt.NDArray[(typing.Any, 3), npt.Float]):
        """
        Constructs a MagnetSlots instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        magnet_type : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        magnet_size : np.ndarray
            A single 3-dim float vector representing the constant size for all magnets in this set.

        magnet_beams : list(str)
            A list of named identifiers for physical beams in the insertion device that each magnet of this type
            is placed in. Most devices will have two (PPM, CPMU) or four (APPLE) unique beam identifiers reused by
            multiple magnets. Magnets of a given type will most likely be present in all beams due to device symmetries.

        magnet_positions : np.ndarray
            A tensor of N 3-dim float vectors of shape (N, 3) representing the bottom left hand near corner of each
            magnet slot of this type.

        magnet_direction_matrices : np.ndarray
            A tensor of N 3x3 float matrices of shape (N, 3, 3) representing the set of orthogonal rotations that
            would be applied to a magnets primary coordinate system if its field were aligned w.r.t an X,Z,S
            coordinate system.

        magnet_flip_vectors : np.ndarray
            A tensor of N 3-dim float vectors of shape (N, 3) representing the set of flips that would be applied
            to a magnet in this slot in order to swap the direction of its minor axis vectors while keeping its
            primary (easy) axis vector aligned to the direction for this slow, as specified by the corresponding
            direction matrix.
        """

        try:
            self._magnet_type = validate_string(magnet_type, assert_non_empty=True)
        except Exception as ex:
            logger.exception('magnet_type must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._magnet_size = validate_tensor(magnet_size, shape=(3,))
        except Exception as ex:
            logger.exception('magnet_size must be a single 3-dim float vector', exc_info=ex)
            raise ex

        try:
            self._magnet_beams = validate_string_list(magnet_beams, assert_non_empty_list=True,
                                                                   assert_non_empty_strings=True)
        except Exception as ex:
            logger.exception('magnet_beams must be a non-empty list of non-empty strings', exc_info=ex)
            raise ex

        # Number of magnet slots derived from number of beam names provided. All other inputs must be consistent.
        self._count = len(self.magnet_beams)

        try:
            self._magnet_positions = validate_tensor(magnet_positions, shape=(self.count, 3))
        except Exception as ex:
            logger.exception('magnet_positions must be a float tensor of shape (N, 3)', exc_info=ex)
            raise ex

        try:
            self._magnet_direction_matrices = validate_tensor(magnet_direction_matrices, shape=(self.count, 3, 3))
        except Exception as ex:
            logger.exception('magnet_direction_matrices must be a float tensor of shape (N, 3, 3)', exc_info=ex)
            raise ex

        try:
            self._magnet_flip_vectors = validate_tensor(magnet_flip_vectors, shape=(self.count, 3))
        except Exception as ex:
            logger.exception('magnet_flip_vectors must be a float tensor of shape (N, 3)', exc_info=ex)
            raise ex

    @property
    def magnet_type(self):
        return self._magnet_type

    @property
    def magnet_size(self):
        return self._magnet_size

    @property
    def magnet_beams(self):
        return self._magnet_beams

    @property
    def magnet_positions(self):
        return self._magnet_positions

    @property
    def magnet_direction_matrices(self):
        return self._magnet_direction_matrices

    @property
    def magnet_flip_vectors(self):
        return self._magnet_flip_vectors

    @property
    def count(self):
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

            try:
                # Pack members into .magslots file as a single tuple
                pickle.dump((self.magnet_type, self.magnet_size,
                             self.magnet_beams, self.magnet_positions,
                             self.magnet_direction_matrices,
                             self.magnet_flip_vectors), file_handle)

                logger.info('Saved magnet slots to .magslots file handle')

            except Exception as ex:
                logger.exception('Failed to save magnet slots to .magslots file', exc_info=ex)
                raise ex

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

            try:
                # Unpack members from .magslots file as a single tuple
                (magnet_type, magnet_size, magnet_beams, magnet_positions,
                 magnet_direction_matrices, magnet_flip_vectors) = pickle.load(file_handle)

                # Offload object construction and validation to the MagnetSlots constructor
                magnet_slots = MagnetSlots(magnet_type=magnet_type, magnet_size=magnet_size,
                                           magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                                           magnet_direction_matrices=magnet_direction_matrices,
                                           magnet_flip_vectors=magnet_flip_vectors)

                logger.info('Loaded magnet slots [%s] with [%d] slots', magnet_type, len(magnet_beams))

            except Exception as ex:
                logger.exception('Failed to load magnet slots from .magslots file', exc_info=ex)
                raise ex

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
