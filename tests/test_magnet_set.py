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


# Utility imports
import unittest
import numpy as np

# Test imports
from optid.magnets import MagnetSet


class MagnetSetTest(unittest.TestCase):
    """
    Tests the MagnetSet class can be imported and used correctly.
    """

    def test_constructor(self):
        """
        Tests the MagnetSet class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count = 16
        magnet_type = 'HH'
        magnet_size = np.random.uniform(size=(3,))
        magnet_names = [f'{index:03d}' for index in range(count)]
        magnet_field_vectors = np.random.uniform(size=(count, 3))

        # Construct MagnetSet instance
        magnet_set = MagnetSet(
            magnet_type=magnet_type,
            magnet_size=magnet_size,
            magnet_names=magnet_names,
            magnet_field_vectors=magnet_field_vectors)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_set.magnet_size, magnet_size))
        self.assertEqual(magnet_set.magnet_names, magnet_names)
        self.assertTrue(np.allclose(magnet_set.magnet_field_vectors, magnet_field_vectors))

    def test_constructor_raises_on_bad_parameters(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count = 16
        magnet_type = 'HH'
        magnet_size = np.random.uniform(size=(3,))
        magnet_names = [f'{index:03d}' for index in range(count)]
        magnet_field_vectors = np.random.uniform(size=(count, 3))

        # Assert constructor throws error from empty magnet type string
        self.assertRaises(Exception, MagnetSet,
                          magnet_type='',
                          magnet_size=magnet_size,
                          magnet_names=magnet_names,
                          magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from incorrectly shaped magnet size
        self.assertRaises(Exception, MagnetSet,
                          magnet_type=magnet_type,
                          magnet_size=np.random.uniform(size=(4,)),
                          magnet_names=magnet_names,
                          magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from empty name string in magnet names
        self.assertRaises(Exception, MagnetSet,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          magnet_names=['' if (index % 2 == 0) else f'{index:03d}' for index in range(count)],
                          magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from incorrectly shaped magnet field vectors
        self.assertRaises(Exception, MagnetSet,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          magnet_names=magnet_names,
                          magnet_field_vectors=np.random.uniform(size=(count, 4)))

        # Assert constructor throws error from magnet names and magnet field vectors being different lengths
        self.assertRaises(Exception, MagnetSet,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          magnet_names=magnet_names[:-1],
                          magnet_field_vectors=magnet_field_vectors)

