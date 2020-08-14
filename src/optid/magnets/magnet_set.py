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
        str : magnet_type
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        np.ndarray : magnet_size
            A single 3-dim float vector representing the constant size for all magnets in this set.

        list(str) : magnet_names
            A list of unique non-empty strings representing the named identifier for each physical magnet as
            specified by the manufacturer or build team.

        np.ndarray : magnet_field_vectors
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
        assert (len(self.magnet_names) > 0) and \
               all((len(name) > 0) for name in magnet_names) and \
               (len(set(self.magnet_names)) == len(self.magnet_names)), \
               'must all be unique non-empty strings'

        self.magnet_field_vectors = magnet_field_vectors
        assert (self.magnet_field_vectors.shape[0] > 0) and (self.magnet_field_vectors.shape[1:] == (3,)) and \
               'must be a tensor of N 3-dim vectors of shape (N, 3)'
        assert np.issubdtype(self.magnet_field_vectors.dtype, np.floating), \
               'dtype must be a float'

        self.count = len(magnet_names)
        assert (self.count == self.magnet_field_vectors.shape[0]), \
               'must have the same number of names as magnet field vectors'
