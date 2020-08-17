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

from optid.utils import validate_tensor, validate_string, validate_string_list
from optid.errors import FileHandleError


class MagnetSet:
    """
    Represents a set of named candidate magnets of the same type and size with their measured field strength vectors.
    """

    def __init__(self,
                 magnet_type : str,
                 magnet_size : npt.NDArray[(3,), npt.Float],
                 magnet_names : typing.List[str],
                 magnet_field_vectors : npt.NDArray[(typing.Any, 3), npt.Float]):
        """
        Constructs a MagnetSet instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        magnet_type : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        magnet_size : np.ndarray
            A single 3-dim float vector representing the constant size for all magnets in this set.

        magnet_names : list(str)
            A list of unique non-empty strings representing the named identifier for each physical magnet as
            specified by the manufacturer or build team.

        magnet_field_vectors : np.ndarray
            A tensor of N 3-dim float vectors of shape (N, 3) representing the average magnetic field strength
            measurements for each magnet in this set.
        """

        try:
            self._magnet_type = validate_string(magnet_type, assert_non_empty=True)
        except Exception as ex:
            logging.exception('magnet_type must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._magnet_size = validate_tensor(magnet_size, shape=(3,))
        except Exception as ex:
            logging.exception('magnet_size must be a single 3-dim float vector', exc_info=ex)
            raise ex

        try:
            self._magnet_names = validate_string_list(magnet_names, assert_non_empty_list=True,
                                                                    assert_non_empty_strings=True,
                                                                    assert_unique_strings=True)
        except Exception as ex:
            logging.exception('magnet_names must be a non-empty list of non-empty and unique strings', exc_info=ex)
            raise ex

        # Number of magnets derived from number of names provided. All other inputs must be consistent.
        self._count = len(self.magnet_names)

        try:
            self._magnet_field_vectors = validate_tensor(magnet_field_vectors, shape=(self.count, 3))
        except Exception as ex:
            logging.exception('magnet_field_vectors must be a float tensor of shape (N, 3)', exc_info=ex)
            raise ex

    @property
    def magnet_type(self):
        return self._magnet_type

    @property
    def magnet_size(self):
        return self._magnet_size

    @property
    def magnet_names(self):
        return self._magnet_names

    @property
    def magnet_field_vectors(self):
        return self._magnet_field_vectors

    @property
    def count(self):
        return self._count

    def save(self, file : typing.Union[str, typing.BinaryIO]):
        """
        Saves a MagnetSet instance to a .magset file.

        Parameters
        ----------
        file : str or open writable file handle
            A path to where a .magset file should be created or overwritten, or an open writable file handle to
            a .magset file.
        """

        def write_file(file_handle : typing.BinaryIO):
            """
            Private helper function for writing data to a .magset file given an already open file handle.

            Parameters
            ----------
            file_handle : open writable file handle
                An open writable file handle to a .magset file.
            """

            # Pack members into .magset file as a single tuple
            pickle.dump((self.magnet_type, self.magnet_size,
                         self.magnet_names, self.magnet_field_vectors), file_handle)

            logging.info('Saved magnet set to .magset file handle')

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logging.info('Saving magnet set to .magset file handle')
            write_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .magset file in a closure to ensure it gets closed on error
            with open(file, 'wb') as file_handle:
                logging.info('Saving magnet set to .magset file [%s]', file)
                write_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()

    @staticmethod
    def from_file(file : typing.Union[str, typing.BinaryIO]) -> 'MagnetSet':
        """
        Constructs a MagnetSet instance from a .magset file.

        Parameters
        ----------
        file : str or open file handle
            A path to a .magset file or an open file handle to a .magset file.

        Returns
        -------
        A MagnetSet instance with the desired values loaded from the .magset file.
        """

        def read_file(file_handle : typing.BinaryIO) -> 'MagnetSet':
            """
            Private helper function for reading data from a .magset file given an already open file handle.

            Parameters
            ----------
            file_handle : open file handle
                An open file handle to a .magset file.

            Returns
            -------
            A MagnetSet instance with the desired values loaded from the .magset file.
            """

            # Unpack members from .magset file as a single tuple
            (magnet_type, magnet_size, magnet_names, magnet_field_vectors) = pickle.load(file_handle)

            # Offload object construction and validation to the MagnetSet constructor
            magnet_set = MagnetSet(magnet_type=magnet_type, magnet_size=magnet_size,
                                   magnet_names=magnet_names, magnet_field_vectors=magnet_field_vectors)

            logging.info('Loaded magnet set [%s] with [%d] magnets', magnet_type, len(magnet_names))
            return magnet_set

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logging.info('Loading magnet set from .magset file handle')
            return read_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .magset file in a closure to ensure it gets closed on error
            with open(file, 'rb') as file_handle:
                logging.info('Loading magnet set from .magset file [%s]', file)
                return read_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()

    @staticmethod
    def from_sim_file(magnet_type : str,
                      magnet_size : npt.NDArray[(3,), npt.Float],
                      file : typing.Union[str, typing.TextIO]) -> 'MagnetSet':
        """
        Constructs a MagnetSet instance using per magnet names and field vectors from a .sim file provided by
        the magnet manufacturer.

        Parameters
        ----------
        magnet_type : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        magnet_size : np.ndarray
            A single 3-dim float vector representing the constant size for all magnets in this set.

        file : str or open file handle
            A path to a .sim file or an open file handle to a .sim file containing per magnet names and field
            vectors as provided by the magnet manufacturer.

        Returns
        -------
        A MagnetSet instance with the desired values loaded from the .sim file.
        """

        def read_file(magnet_type : str,
                      magnet_size : npt.NDArray[(3,), npt.Float],
                      file_handle : typing.TextIO) -> 'MagnetSet':
            """
            Private helper function for reading data from a .sim file given an already open file handle.

            Parameters
            ----------
            magnet_type : str
                A non-empty string name for this magnet type that should be unique in the context of the full insertion
                device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

            magnet_size : np.ndarray
                A single 3-dim float vector representing the constant size for all magnets in this set.

            file_handle : open file handle
                An open file handle to a .sim file containing per magnet names and field
                vectors as provided by the magnet manufacturer.

            Returns
            -------
            A MagnetSet instance with the desired values loaded from the .sim file.
            """

            # Load the data into python lists
            magnet_names = []
            magnet_field_vectors = []

            for line_index, line in enumerate(file_handle):
                # Skip this line if it is blank
                line = line.strip()
                if len(line) == 0:
                    continue

                logging.debug('Line [%d] : [%s]', line_index, line)

                # Unpack and parse values for the current magnet
                name, field_x, field_z, field_s = line.split()
                magnet_names += [name]
                magnet_field_vectors += [(float(field_x), float(field_z), float(field_s))]

            # Convert python list of field vector tuples into numpy array with shape (N, 3)
            magnet_field_vectors = np.array(magnet_field_vectors, dtype=np.float32)

            # Offload object construction and validation to the MagnetSet constructor
            magnet_set = MagnetSet(magnet_type=magnet_type, magnet_size=magnet_size,
                                   magnet_names=magnet_names, magnet_field_vectors=magnet_field_vectors)

            logging.info('Loaded magnet set [%s] with [%d] magnets', magnet_type, len(magnet_names))
            return magnet_set

        if isinstance(file, io.TextIOWrapper):
            # Load directly from the already open file handle
            logging.info('Loading magnet set [%s] from open .sim file handle', magnet_type)
            return read_file(magnet_type=magnet_type, magnet_size=magnet_size, file_handle=file)

        elif isinstance(file, str):
            # Open the .sim file in a closure to ensure it gets closed on error
            with open(file, 'r') as file_handle:
                logging.info('Loading magnet set [%s] from .sim file [%s]', magnet_type, file)
                return read_file(magnet_type=magnet_type, magnet_size=magnet_size, file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()
