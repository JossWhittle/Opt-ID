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
import logging
import numpy as np


class MagnetSet:
    """
    Represents a set of named candidate magnets of the same type and size with their measured field strength vectors.
    """

    def __init__(self,
                 magnet_type : str,
                 magnet_size : np.ndarray,
                 magnet_names : typing.List[str],
                 magnet_field_vectors : np.ndarray):
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

        self.magnet_type = magnet_type
        assert (len(magnet_type) > 0), \
               'must be non-empty string'

        self.magnet_size = magnet_size
        assert (self.magnet_size.shape == (3,)), \
               'must be a single 3-dim vector'
        assert np.issubdtype(self.magnet_size.dtype, np.floating), \
               'dtype must be a float'

        self.magnet_names = magnet_names
        assert (len(self.magnet_names) > 0), \
               'must have at least one magnet name'
        assert all((len(name) > 0) for name in magnet_names), \
               'all magnet names must be non-empty strings'
        assert (len(set(self.magnet_names)) == len(self.magnet_names)), \
               'all magnet names must be unique strings'

        self.magnet_field_vectors = magnet_field_vectors
        assert (self.magnet_field_vectors.shape[0] > 0) and (self.magnet_field_vectors.shape[1:] == (3,)) and \
               'must be a tensor of N 3-dim vectors of shape (N, 3)'
        assert np.issubdtype(self.magnet_field_vectors.dtype, np.floating), \
               'dtype must be a float'

        self.count = len(magnet_names)
        assert (self.count == self.magnet_field_vectors.shape[0]), \
               'must have the same number of names as magnet field vectors'

    @staticmethod
    def from_sim_file(magnet_type : str,
                      magnet_size : np.ndarray,
                      sim_file_path : str) -> 'MagnetSet':
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

        sim_file_path : str
            A path to a .sim file containing per magnet names and field vectors as provided by the magnet
            manufacturer.

        Returns
        -------
        A MagnetSet instance with the desired values loaded from the .sim file.
        """

        logging.info('Loading magnet set [%s] from sim file [%s]', magnet_type, sim_file_path)

        with open(sim_file_path, 'r') as sim_file:
            # Load the data into python lists
            magnet_names = []
            magnet_field_vectors = []

            for line_index, line in enumerate(sim_file):
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

        logging.info('Loaded magnet set [%s] with [%d] magnets', magnet_type, len(magnet_names))

        return MagnetSet(magnet_type=magnet_type,
                         magnet_size=magnet_size,
                         magnet_names=magnet_names,
                         magnet_field_vectors=magnet_field_vectors)
