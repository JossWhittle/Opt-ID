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
import tempfile
import shutil
import inspect
import numpy as np

# Test imports
import optid
from optid.magnets import MagnetSet


class MagnetSetTest(unittest.TestCase):
    """
    Tests the MagnetSet class can be imported and used correctly.
    """

    @staticmethod
    def dummy_magnet_set_values():
        """
        Creates a set of constant test values used for constructing and comparing MagnetSet
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        count = 4
        magnet_type = 'HH'
        magnet_size = np.array([0.54348747, 0.05696444, 0.03594551], dtype=np.float32)
        magnet_names = [f'{index + 1:03d}' for index in range(count)]
        magnet_field_vectors = np.array([
            [0.003770334, -0.000352049, 1.339567917],
            [-0.007018214, -0.002714164, 1.344710227],
            [-0.004826321, -0.001714764, 1.342079598],
            [0.008846784, -0.003088993, 1.344698631],
        ], dtype=np.float32)

        return count, magnet_type, magnet_size, magnet_names, magnet_field_vectors

    def test_constructor(self):
        """
        Tests the MagnetSet class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Construct MagnetSet instance
        magnet_set = MagnetSet(magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_names=magnet_names, magnet_field_vectors=magnet_field_vectors)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_set.magnet_size, magnet_size))
        self.assertEqual(magnet_set.magnet_names, magnet_names)
        self.assertTrue(np.allclose(magnet_set.magnet_field_vectors, magnet_field_vectors))

    def test_constructor_raises_on_bad_parameters_magnet_type(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Assert constructor throws error from empty magnet type string
        self.assertRaisesRegex(optid.errors.StringEmptyError, '.*', MagnetSet,
                               magnet_type='',
                               magnet_size=magnet_size, magnet_names=magnet_names,
                               magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from magnet type not being a string
        self.assertRaisesRegex(optid.errors.StringTypeError, '.*', MagnetSet,
                               magnet_type=None,
                               magnet_size=magnet_size, magnet_names=magnet_names,
                               magnet_field_vectors=magnet_field_vectors)

    def test_constructor_raises_on_bad_parameters_magnet_size(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Assert constructor throws error from incorrectly shaped magnet size
        self.assertRaisesRegex(optid.errors.TensorShapeError, '.*', MagnetSet,
                               magnet_type=magnet_type,
                               magnet_size=np.random.uniform(size=(4,)),
                               magnet_names=magnet_names,
                               magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from incorrectly typed magnet size
        self.assertRaisesRegex(optid.errors.TensorTypeError, '.*', MagnetSet,
                               magnet_type=magnet_type,
                               magnet_size=magnet_size.astype(np.int32),
                               magnet_names=magnet_names,
                               magnet_field_vectors=magnet_field_vectors)

    def test_constructor_raises_on_bad_parameters_magnet_names(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Assert constructor throws error from wrong typed list of name strings for magnet names
        self.assertRaisesRegex(optid.errors.StringListTypeError, '.*', MagnetSet,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_names=None,
                               magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from empty list of name strings for magnet names
        self.assertRaisesRegex(optid.errors.StringListEmptyError, '.*', MagnetSet,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_names=[],
                               magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from empty name string in magnet names
        self.assertRaisesRegex(optid.errors.StringListElementEmptyError, '.*', MagnetSet,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_names=['' if (index == 1) else name
                                             for index, name in enumerate(magnet_names)],
                               magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from non unique name strings in magnet names
        self.assertRaisesRegex(optid.errors.StringListElementUniquenessError, '.*', MagnetSet,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_names=['TEST' if (index % 2 == 0) else name
                                             for index, name in enumerate(magnet_names)],
                               magnet_field_vectors=magnet_field_vectors)

        # Assert constructor throws error from magnet names and magnet field vectors being different lengths
        self.assertRaisesRegex(optid.errors.TensorShapeError, '.*', MagnetSet,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_names=magnet_names[:-1],
                               magnet_field_vectors=magnet_field_vectors)

    def test_constructor_raises_on_bad_parameters_magnet_field_vectors(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Assert constructor throws error from incorrectly shaped magnet field vectors
        self.assertRaisesRegex(optid.errors.TensorShapeError, '.*', MagnetSet,
                               magnet_type=magnet_type, magnet_size=magnet_size, magnet_names=magnet_names,
                               magnet_field_vectors=np.random.uniform(size=(count, 4)))

        # Assert constructor throws error from incorrectly shaped magnet field vectors
        self.assertRaisesRegex(optid.errors.TensorTypeError, '.*', MagnetSet,
                               magnet_type=magnet_type, magnet_size=magnet_size, magnet_names=magnet_names,
                               magnet_field_vectors=magnet_field_vectors.astype(np.int32))

    def test_save(self):
        """
        Tests the MagnetSet class can be saved to a .magset file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magset')

            # Construct MagnetSet instance
            magnet_set = MagnetSet(magnet_type=magnet_type, magnet_size=magnet_size,
                                   magnet_names=magnet_names, magnet_field_vectors=magnet_field_vectors)

            # Save the MagnetSet to the temporary directory
            magnet_set.save(file=tmp_file_path)

            # Throw away the local object and reload it from the temporary file
            magnet_set = MagnetSet.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_set.magnet_size, magnet_size))
        self.assertEqual(magnet_set.magnet_names, magnet_names)
        self.assertTrue(np.allclose(magnet_set.magnet_field_vectors, magnet_field_vectors))

    def test_save_open_file_handle(self):
        """
        Tests the MagnetSet class can be saved to a .magset file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magset')

            # Construct MagnetSet instance
            magnet_set = MagnetSet(magnet_type=magnet_type, magnet_size=magnet_size,
                                   magnet_names=magnet_names, magnet_field_vectors=magnet_field_vectors)

            with open(tmp_file_path, 'wb') as tmp_file_handle:
                # Save the MagnetSet to the temporary directory
                magnet_set.save(file=tmp_file_handle)

            # Throw away the local object and reload it from the temporary file
            magnet_set = MagnetSet.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_set.magnet_size, magnet_size))
        self.assertEqual(magnet_set.magnet_names, magnet_names)
        self.assertTrue(np.allclose(magnet_set.magnet_field_vectors, magnet_field_vectors))

    def test_save_raises_on_bad_parameters(self):
        """
        Tests the MagnetSet class save member function raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Construct MagnetSet instance
        magnet_set = MagnetSet(magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_names=magnet_names, magnet_field_vectors=magnet_field_vectors)

        # Attempt to save to a bad file parameter
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', magnet_set.save, file=None)

    def test_static_from_file(self):
        """
        Tests the MagnetSet class can be constructed from a .magset file using the static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Construct MagnetSet instance
        magnet_set = MagnetSet.from_file(file=os.path.join(inputs_path, 'example.magset'))

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_set.magnet_size, magnet_size))
        self.assertEqual(magnet_set.magnet_names, magnet_names)
        self.assertTrue(np.allclose(magnet_set.magnet_field_vectors, magnet_field_vectors))

    def test_static_from_file_open_file_handle(self):
        """
        Tests the MagnetSet class can be constructed from an open handle to a .magset file using the
        static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        with open(os.path.join(inputs_path, 'example.magset'), 'rb') as file_handle:
            # Construct MagnetSet instance
            magnet_set = MagnetSet.from_file(file=file_handle)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_set.magnet_size, magnet_size))
        self.assertEqual(magnet_set.magnet_names, magnet_names)
        self.assertTrue(np.allclose(magnet_set.magnet_field_vectors, magnet_field_vectors))

    def test_static_from_file_raises_on_bad_parameters(self):
        """
        Tests the MagnetSet class raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Attempt to load from to a bad file parameter
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', MagnetSet.from_file, file=None)

    def test_static_from_sim_file(self):
        """
        Tests the MagnetSet class can be constructed from a .sim file using the static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        # Construct MagnetSet instance
        magnet_set = MagnetSet.from_sim_file(magnet_type=magnet_type, magnet_size=magnet_size,
                                             file=os.path.join(inputs_path, 'example.sim'))

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_set.magnet_size, magnet_size))
        self.assertEqual(magnet_set.magnet_names, magnet_names)
        self.assertTrue(np.allclose(magnet_set.magnet_field_vectors, magnet_field_vectors))

    def test_static_from_sim_file_open_file_handle(self):
        """
        Tests the MagnetSet class can be constructed from an open handle to a .sim file using the static
        factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_names, magnet_field_vectors = self.dummy_magnet_set_values()

        with open(os.path.join(inputs_path, 'example.sim'), 'r') as file_handle:
            # Construct MagnetSet instance
            magnet_set = MagnetSet.from_sim_file(magnet_type=magnet_type, magnet_size=magnet_size, file=file_handle)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_set.magnet_size, magnet_size))
        self.assertEqual(magnet_set.magnet_names, magnet_names)
        self.assertTrue(np.allclose(magnet_set.magnet_field_vectors, magnet_field_vectors))

    def test_static_from_sim_file_raises_on_bad_parameters(self):
        """
        Tests the MagnetSet class raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        _, magnet_type, magnet_size, _, _ = self.dummy_magnet_set_values()

        # Assert from_sim_file throws error when file is not a string file path or an open file handle
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', MagnetSet.from_sim_file,
                               magnet_type=magnet_type, magnet_size=magnet_size, file=None)

    def test_static_from_sim_file_raises_on_bad_sim_file(self):
        """
        Tests the MagnetSet class raises an error when constructed with incorrect data from a .sim file.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        _, magnet_type, magnet_size, _, _ = self.dummy_magnet_set_values()

        # Assert from_sim_file throws error when sim file has a missing magnet name
        self.assertRaisesRegex(Exception, '.*', MagnetSet.from_sim_file,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               file=os.path.join(inputs_path, 'example_missing_name.sim'))

        # Assert from_sim_file throws error when sim file has a missing magnet field vector
        self.assertRaisesRegex(Exception, '.*', MagnetSet.from_sim_file,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               file=os.path.join(inputs_path, 'example_missing_field_vector.sim'))

        # Assert from_sim_file throws error when sim file has an invalid magnet field vector
        self.assertRaisesRegex(Exception, '.*', MagnetSet.from_sim_file,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               file=os.path.join(inputs_path, 'example_invalid_field_vector.sim'))

        # Assert from_sim_file throws error when sim file has a duplicate magnet name
        self.assertRaisesRegex(Exception, '.*', MagnetSet.from_sim_file,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               file=os.path.join(inputs_path, 'example_duplicate_name.sim'))
