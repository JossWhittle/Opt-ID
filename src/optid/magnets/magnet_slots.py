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

import optid
from optid.utils import validate_tensor, validate_magnet_cutouts, validate_string, validate_string_list
from optid.errors import FileHandleError

logger = optid.utils.logging.get_logger('optid.magnets.MagnetSlots')


class MagnetSlots:
    """
    Represents a set of magnet slots. Magnet slots are characterized by beam name, position, 3x3 direction matrix,
    and flip vector. All magnet slots share a constant magnet size and type name allowing magnets from a MagnetSet
    class to be placed in arbitrary orders and flips via a MagnetPermutation.
    """

    def __init__(self,
                 magnet_type : str,
                 size : npt.NDArray[(3,), npt.Float],
                 cutouts: npt.NDArray[(typing.Any, 2, 3), npt.Float],
                 beams : typing.List[str],
                 positions : npt.NDArray[(typing.Any, 3), npt.Float],
                 direction_matrices : npt.NDArray[(typing.Any, 3, 3), npt.Float],
                 flip_vectors : npt.NDArray[(typing.Any, 3), npt.Float]):
        """
        Constructs a MagnetSlots instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        magnet_type : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        size : float tensor (3,)
            A single 3-dim float vector representing the constant size for all magnets in this set.

        cutouts : float tensor (C, 2, 3)
            A tensor of C pairs of 3-dim float vectors of shape (C, 2, 3) representing the constant position offset
            and size for all magnet cutout regions. These regions are applied to the magnet in its identity orientation
            before it is transformed by a MagnetSlots direction matrix.

        beams : list(str)
            A list of named identifiers for physical beams in the insertion device that each magnet of this type
            is placed in. Most devices will have two (PPM, CPMU) or four (APPLE) unique beam identifiers reused by
            multiple magnets. Magnets of a given type will most likely be present in all beams due to device symmetries.

        positions : float tensor (S, 3)
            A tensor of S 3-dim float vectors of shape (S, 3) representing the bottom left hand near corner of each
            magnet slot of this type.

        direction_matrices : np.ndarray
            A tensor of S 3x3 float matrices of shape (S, 3, 3) representing the set of orthogonal rotations that
            would be applied to a magnets primary coordinate system if its field were aligned w.r.t an X,Z,S
            coordinate system.

        flip_vectors : np.ndarray
            A tensor of S 3-dim float vectors of shape (S, 3) representing the set of flips that would be applied
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
            self._size = validate_tensor(size, shape=(3,))
        except Exception as ex:
            logger.exception('size must be a single 3-dim float vector', exc_info=ex)
            raise ex

        try:
            self._cutouts = validate_magnet_cutouts(cutouts, size=self.size)
        except Exception as ex:
            logger.exception('cutouts must be a float tensor of shape (C, 2, 3) where cutout regions'
                             'do not extend outside the size of the magnet', exc_info=ex)
            raise ex

        try:
            self._beams = validate_string_list(beams, assert_non_empty_list=True,
                                                                   assert_non_empty_strings=True)
        except Exception as ex:
            logger.exception('beams must be a non-empty list of non-empty strings', exc_info=ex)
            raise ex

        # Number of magnet slots derived from number of beam names provided. All other inputs must be consistent.
        self._count = len(self.beams)

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
            self._flip_vectors = validate_tensor(flip_vectors, shape=(self.count, 3))
        except Exception as ex:
            logger.exception('flip_vectors must be a float tensor of shape (S, 3)', exc_info=ex)
            raise ex

    @property
    def magnet_type(self):
        return self._magnet_type

    @property
    def size(self):
        return self._size

    @property
    def cutouts(self):
        return self._cutouts

    @property
    def beams(self):
        return self._beams

    @property
    def positions(self):
        return self._positions

    @property
    def direction_matrices(self):
        return self._direction_matrices

    @property
    def flip_vectors(self):
        return self._flip_vectors

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

            # Pack members into .magslots file as a single tuple
            pickle.dump((self.magnet_type, self.size, self.cutouts,
                         self.beams, self.positions,
                         self.direction_matrices,
                         self.flip_vectors), file_handle)

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
            (magnet_type, size, cutouts, beams, positions,
                direction_matrices, flip_vectors) = pickle.load(file_handle)

            # Offload object construction and validation to the MagnetSlots constructor
            magnet_slots = MagnetSlots(magnet_type=magnet_type, size=size, cutouts=cutouts,
                                       beams=beams, positions=positions,
                                       direction_matrices=direction_matrices,
                                       flip_vectors=flip_vectors)

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
