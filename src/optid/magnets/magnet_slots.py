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
import logging
import pickle
import numpy as np


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

        self.magnet_type = magnet_type
        assert (len(magnet_type) > 0), \
               'must be non-empty string'

        self.magnet_size = magnet_size
        assert (self.magnet_size.shape == (3,)), \
               'must be a single 3-dim vector'
        assert np.issubdtype(self.magnet_size.dtype, np.floating), \
               'dtype must be a float'

        self.magnet_beams = magnet_beams
        assert (len(self.magnet_beams) > 0), \
               'must have at least one magnet beam name'
        assert all((len(name) > 0) for name in self.magnet_beams), \
               'all magnet beam names must be non-empty strings'

        self.magnet_positions = magnet_positions
        assert (self.magnet_positions.shape[0] > 0) and (self.magnet_positions.shape[1:] == (3,)) and \
               'must be a tensor of N 3-dim vectors of shape (N, 3)'
        assert np.issubdtype(self.magnet_positions.dtype, np.floating), \
               'dtype must be a float'

        self.magnet_direction_matrices = magnet_direction_matrices
        assert (self.magnet_direction_matrices.shape[0] > 0) and \
               (self.magnet_direction_matrices.shape[1:] == (3, 3)) and \
               'must be a tensor of N 3x3 matrices of shape (N, 3, 3)'
        assert np.issubdtype(self.magnet_direction_matrices.dtype, np.floating), \
               'dtype must be a float'

        self.magnet_flip_vectors = magnet_flip_vectors
        assert (self.magnet_flip_vectors.shape[0] > 0) and (self.magnet_flip_vectors.shape[1:] == (3,)) and \
               'must be a tensor of N 3-dim vectors of shape (N, 3,)'
        assert np.issubdtype(self.magnet_flip_vectors.dtype, np.floating), \
               'dtype must be a float'

        self.count = len(self.magnet_beams)
        assert (self.count == self.magnet_positions.shape[0]), \
               'must have the same number of beam names as magnet positions'
        assert (self.count == self.magnet_direction_matrices.shape[0]), \
               'must have the same number of beam names as magnet direction matrices'
        assert (self.count == self.magnet_flip_vectors.shape[0]), \
               'must have the same number of beam names as magnet flip vectors'

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
            pickle.dump((self.magnet_type, self.magnet_size,
                         self.magnet_beams, self.magnet_positions,
                         self.magnet_direction_matrices,
                         self.magnet_flip_vectors), file_handle)

            logging.info('Saved magnet slots to .magslots file handle')

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logging.info('Saving magnet slots to .magslots file handle')
            write_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .magslots file in a closure to ensure it gets closed on error
            with open(file, 'wb') as file_handle:
                logging.info('Saving magnet slots to .magslots file [%s]', file)
                write_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise AttributeError('file must be a string file path or a file handle to an already open file')

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
            (magnet_type, magnet_size, magnet_beams, magnet_positions,
             magnet_direction_matrices, magnet_flip_vectors) = pickle.load(file_handle)

            # Offload object construction and validation to the MagnetSlots constructor
            magnet_slots = MagnetSlots(magnet_type=magnet_type, magnet_size=magnet_size,
                                       magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                                       magnet_direction_matrices=magnet_direction_matrices,
                                       magnet_flip_vectors=magnet_flip_vectors)

            logging.info('Loaded magnet slots [%s] with [%d] slots', magnet_type, len(magnet_beams))
            return magnet_slots

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logging.info('Loading magnet set from .magslots file handle')
            return read_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .magslots file in a closure to ensure it gets closed on error
            with open(file, 'rb') as file_handle:
                logging.info('Loading magnet set from .magslots file [%s]', file)
                return read_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise AttributeError('file must be a string file path or a file handle to an already open file')
