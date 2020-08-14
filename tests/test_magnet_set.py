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
import os
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
        magnet_type  = 'HH'
        magnet_size  = np.random.uniform(size=(3,))
        magnet_names = [f'{index+1:03d}' for index in range(count)]
        magnet_field_vectors = np.random.uniform(size=(count, 3))

        # Construct MagnetSet instance
        magnet_set = MagnetSet(magnet_type=magnet_type,
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
        magnet_type  = 'HH'
        magnet_size  = np.random.uniform(size=(3,))
        magnet_names = [f'{index+1:03d}' for index in range(count)]
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

        # Assert constructor throws error from empty list of name strings for magnet names
        self.assertRaises(Exception, MagnetSet,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          magnet_names=[],
                          magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from empty name string in magnet names
        self.assertRaises(Exception, MagnetSet,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          magnet_names=['' if (index == 1) else f'{index:03d}' for index in range(count)],
                          magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from non unique name strings in magnet names
        self.assertRaises(Exception, MagnetSet,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          magnet_names=['TEST' if (index % 2 == 0) else f'{index:03d}' for index in range(count)],
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

    def test_static_from_sim_file(self):
        """
        Tests the MagnetSet class can be constructed from a .sim file using the static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 'test_static_from_sim_file')

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        magnet_type   = 'HH'
        magnet_size   = np.random.uniform(size=(3,))

        # Expect to see four magnets with the following values in the example.sim file
        count = 4
        magnet_names = [f'{index+1:03d}' for index in range(count)]
        magnet_field_vectors = np.array([
            [ 0.003770334, -0.000352049, 1.339567917],
            [-0.007018214, -0.002714164, 1.344710227],
            [-0.004826321, -0.001714764, 1.342079598],
            [ 0.008846784, -0.003088993, 1.344698631],
        ], dtype=np.float32)

        # Construct MagnetSet instance
        magnet_set = MagnetSet.from_sim_file(magnet_type=magnet_type,
                                             magnet_size=magnet_size,
                                             sim_file_path=os.path.join(inputs_path, 'example.sim'))

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_set.magnet_size, magnet_size))
        self.assertEqual(magnet_set.magnet_names, magnet_names)
        self.assertTrue(np.allclose(magnet_set.magnet_field_vectors, magnet_field_vectors))

    def test_static_from_sim_file_raises_on_bad_sim_file(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect data from a .sim file.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 'test_static_from_sim_file_raises_on_bad_sim_file')

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        magnet_type   = 'HH'
        magnet_size   = np.random.uniform(size=(3,))

        # Assert from_sim_file throws error when sim file has a missing magnet name
        self.assertRaises(Exception, MagnetSet.from_sim_file,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          sim_file_path=os.path.join(inputs_path, 'example_missing_name.sim'))

        # Assert from_sim_file throws error when sim file has a missing magnet field vector
        self.assertRaises(Exception, MagnetSet.from_sim_file,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          sim_file_path=os.path.join(inputs_path, 'example_missing_field_vector.sim'))

        # Assert from_sim_file throws error when sim file has an invalid magnet field vector
        self.assertRaises(Exception, MagnetSet.from_sim_file,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          sim_file_path=os.path.join(inputs_path, 'example_invalid_field_vector.sim'))

        # Assert from_sim_file throws error when sim file has a duplicate magnet name
        self.assertRaises(Exception, MagnetSet.from_sim_file,
                          magnet_type=magnet_type,
                          magnet_size=magnet_size,
                          sim_file_path=os.path.join(inputs_path, 'example_duplicate_name.sim'))
