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

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


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
        mtype = 'HH'

        reference_size = np.array([10, 10, 2], dtype=np.float32)
        reference_field_vector = np.array([0, 0, 1], dtype=np.float32)
        flip_matrix = np.eye(3).astype(np.float32)

        names = [f'{index + 1:03d}' for index in range(count)]
        sizes = np.ones((count, 3)) * reference_size[np.newaxis, ...]
        field_vectors = np.array([
            [0.003770334, -0.000352049, 1.339567917],
            [-0.007018214, -0.002714164, 1.344710227],
            [-0.004826321, -0.001714764, 1.342079598],
            [0.008846784, -0.003088993, 1.344698631],
        ], dtype=np.float32)

        return count, mtype, reference_size, reference_field_vector, flip_matrix, names, sizes, field_vectors

    def test_constructor(self):
        """
        Tests the MagnetSet class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        # Construct MagnetSet instance
        magnet_set = MagnetSet(mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector,
                               flip_matrix=flip_matrix, names=names, sizes=sizes,
                               field_vectors=field_vectors)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.mtype, mtype)
        self.assertTrue(np.allclose(magnet_set.reference_size, reference_size))
        self.assertTrue(np.allclose(magnet_set.reference_field_vector, reference_field_vector))
        self.assertTrue(np.allclose(magnet_set.flip_matrix, flip_matrix))
        self.assertEqual(magnet_set.names, names)
        self.assertTrue(np.allclose(magnet_set.sizes, sizes))
        self.assertTrue(np.allclose(magnet_set.field_vectors, field_vectors))

    def test_constructor_rescale(self):
        """
        Tests the MagnetSet class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        # Construct MagnetSet instance
        magnet_set = MagnetSet(mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector,
                               flip_matrix=flip_matrix, names=names, sizes=sizes,
                               field_vectors=field_vectors,
                               rescale_reference_field_vector=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.mtype, mtype)
        self.assertTrue(np.allclose(magnet_set.reference_size, reference_size))
        self.assertTrue(np.linalg.norm(magnet_set.reference_field_vector, axis=-1) >
                        np.linalg.norm(reference_field_vector, axis=-1))
        self.assertTrue(np.allclose(magnet_set.flip_matrix, flip_matrix))
        self.assertEqual(magnet_set.names, names)
        self.assertTrue(np.allclose(magnet_set.sizes, sizes))
        self.assertTrue(np.allclose(magnet_set.field_vectors, field_vectors))

    def test_constructor_raises_on_bad_parameters_mtype(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        fixed_params = dict(reference_size=reference_size, reference_field_vector=reference_field_vector,
                            flip_matrix=flip_matrix, names=names, sizes=sizes, field_vectors=field_vectors)

        self.assertRaisesRegex(optid.errors.ValidateStringEmptyError, '.*', MagnetSet, **fixed_params,
                               mtype='')

        self.assertRaisesRegex(optid.errors.ValidateStringTypeError, '.*', MagnetSet, **fixed_params,
                               mtype=None)

    def test_constructor_raises_on_bad_parameters_names(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        fixed_params = dict(mtype=mtype, reference_size=reference_size, reference_field_vector=reference_field_vector,
                            flip_matrix=flip_matrix, sizes=sizes, field_vectors=field_vectors)

        self.assertRaisesRegex(optid.errors.ValidateStringListTypeError, '.*', MagnetSet, **fixed_params,
                               names=None)

        self.assertRaisesRegex(optid.errors.ValidateStringListEmptyError, '.*', MagnetSet, **fixed_params,
                               names=[])

        self.assertRaisesRegex(optid.errors.ValidateStringListElementEmptyError, '.*', MagnetSet, **fixed_params,
                               names=['' if (index == 1) else name for index, name in enumerate(names)])

        self.assertRaisesRegex(optid.errors.ValidateStringListElementTypeError, '.*', MagnetSet, **fixed_params,
                               names=[None if (index == 1) else name for index, name in enumerate(names)])

        self.assertRaisesRegex(optid.errors.ValidateStringListElementUniquenessError, '.*', MagnetSet, **fixed_params,
                               names=['TEST' if (index % 2 == 0) else name for index, name in enumerate(names)])

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSet, **fixed_params,
                               names=names[:-1])

    def test_constructor_raises_on_bad_parameters_field_vectors(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        fixed_params = dict(mtype=mtype, reference_size=reference_size, reference_field_vector=reference_field_vector,
                            flip_matrix=flip_matrix, names=names, sizes=sizes)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSet, **fixed_params,
                               field_vectors=np.random.uniform(size=(count, 4)))

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetSet, **fixed_params,
                               field_vectors=field_vectors.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetSet, **fixed_params,
                               field_vectors=None)

    def test_constructor_raises_on_bad_parameters_reference_field_vector(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        fixed_params = dict(mtype=mtype, reference_size=reference_size,
                            flip_matrix=flip_matrix, names=names, sizes=sizes, field_vectors=field_vectors)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSet, **fixed_params,
                               reference_field_vector=np.random.uniform(size=(4,)))

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetSet, **fixed_params,
                               reference_field_vector=reference_field_vector.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetSet, **fixed_params,
                               reference_field_vector=None)

    def test_flippable(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        fixed_params = dict(mtype=mtype, reference_size=reference_size, reference_field_vector=reference_field_vector,
                            names=names, sizes=sizes, field_vectors=field_vectors)

        magnet_set = MagnetSet(**fixed_params, flip_matrix=np.ones((3, 3)))
        self.assertTrue(magnet_set.flippable)

        magnet_set = MagnetSet(**fixed_params, flip_matrix=np.eye(3))
        self.assertFalse(magnet_set.flippable)

    def test_constructor_raises_on_bad_parameters_reference_size(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        fixed_params = dict(mtype=mtype, reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                            names=names, sizes=sizes, field_vectors=field_vectors)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSet, **fixed_params,
                               reference_size=reference_size[:-1])

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetSet, **fixed_params,
                               reference_size=reference_size.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetSet, **fixed_params,
                               reference_size=None)

    def test_constructor_raises_on_bad_parameters_flip_matrix(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        fixed_params = dict(mtype=mtype, reference_size=reference_size, reference_field_vector=reference_field_vector,
                            names=names, sizes=sizes, field_vectors=field_vectors)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSet, **fixed_params,
                               flip_matrix=flip_matrix[:-1])

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetSet, **fixed_params,
                               flip_matrix=flip_matrix.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetSet, **fixed_params,
                               flip_matrix=None)

    def test_save(self):
        """
        Tests the MagnetSet class can be saved to a .magset file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magset')

            # Construct MagnetSet instance
            magnet_set = MagnetSet(mtype=mtype, reference_size=reference_size,
                                   reference_field_vector=reference_field_vector,
                                   flip_matrix=flip_matrix, names=names, sizes=sizes,
                                   field_vectors=field_vectors)

            # Save the MagnetSet to the temporary directory
            magnet_set.save(file=tmp_file_path)

            # Throw away the local object and reload it from the temporary file
            magnet_set = MagnetSet.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.mtype, mtype)
        self.assertTrue(np.allclose(magnet_set.reference_size, reference_size))
        self.assertTrue(np.allclose(magnet_set.reference_field_vector, reference_field_vector))
        self.assertTrue(np.allclose(magnet_set.flip_matrix, flip_matrix))
        self.assertEqual(magnet_set.names, names)
        self.assertTrue(np.allclose(magnet_set.sizes, sizes))
        self.assertTrue(np.allclose(magnet_set.field_vectors, field_vectors))

    def test_save_open_file_handle(self):
        """
        Tests the MagnetSet class can be saved to a .magset file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magset')

            # Construct MagnetSet instance
            magnet_set = MagnetSet(mtype=mtype, reference_size=reference_size,
                                   reference_field_vector=reference_field_vector,
                                   flip_matrix=flip_matrix, names=names, sizes=sizes,
                                   field_vectors=field_vectors)

            with open(tmp_file_path, 'wb') as tmp_file_handle:
                # Save the MagnetSet to the temporary directory
                magnet_set.save(file=tmp_file_handle)

            # Throw away the local object and reload it from the temporary file
            magnet_set = MagnetSet.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.mtype, mtype)
        self.assertTrue(np.allclose(magnet_set.reference_size, reference_size))
        self.assertTrue(np.allclose(magnet_set.reference_field_vector, reference_field_vector))
        self.assertTrue(np.allclose(magnet_set.flip_matrix, flip_matrix))
        self.assertEqual(magnet_set.names, names)
        self.assertTrue(np.allclose(magnet_set.sizes, sizes))
        self.assertTrue(np.allclose(magnet_set.field_vectors, field_vectors))

    def test_save_raises_on_bad_parameters(self):
        """
        Tests the MagnetSet class save member function raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        # Construct MagnetSet instance
        magnet_set = MagnetSet(mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector,
                               flip_matrix=flip_matrix, names=names, sizes=sizes,
                               field_vectors=field_vectors)

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
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        # Construct MagnetSet instance
        magnet_set = MagnetSet.from_file(file=os.path.join(inputs_path, 'example.magset'))

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.mtype, mtype)
        self.assertTrue(np.allclose(magnet_set.reference_size, reference_size))
        self.assertTrue(np.allclose(magnet_set.reference_field_vector, reference_field_vector))
        self.assertTrue(np.allclose(magnet_set.flip_matrix, flip_matrix))
        self.assertEqual(magnet_set.names, names)
        self.assertTrue(np.allclose(magnet_set.sizes, sizes))
        self.assertTrue(np.allclose(magnet_set.field_vectors, field_vectors))

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
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        with open(os.path.join(inputs_path, 'example.magset'), 'rb') as file_handle:
            # Construct MagnetSet instance
            magnet_set = MagnetSet.from_file(file=file_handle)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.mtype, mtype)
        self.assertTrue(np.allclose(magnet_set.reference_size, reference_size))
        self.assertTrue(np.allclose(magnet_set.reference_field_vector, reference_field_vector))
        self.assertTrue(np.allclose(magnet_set.flip_matrix, flip_matrix))
        self.assertEqual(magnet_set.names, names)
        self.assertTrue(np.allclose(magnet_set.sizes, sizes))
        self.assertTrue(np.allclose(magnet_set.field_vectors, field_vectors))

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
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        # Construct MagnetSet instance
        magnet_set = MagnetSet.from_sim_file(mtype=mtype, reference_size=reference_size,
                                             reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                                             file=os.path.join(inputs_path, 'example.sim'))

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.mtype, mtype)
        self.assertTrue(np.allclose(magnet_set.reference_size, reference_size))
        self.assertTrue(np.allclose(magnet_set.reference_field_vector, reference_field_vector))
        self.assertTrue(np.allclose(magnet_set.flip_matrix, flip_matrix))
        self.assertEqual(magnet_set.names, names)
        self.assertTrue(np.allclose(magnet_set.sizes, sizes))
        self.assertTrue(np.allclose(magnet_set.field_vectors, field_vectors))

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
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        with open(os.path.join(inputs_path, 'example.sim'), 'r') as file_handle:
            # Construct MagnetSet instance
            magnet_set = MagnetSet.from_sim_file(mtype=mtype, reference_size=reference_size,
                                                 reference_field_vector=reference_field_vector,
                                                 flip_matrix=flip_matrix, file=file_handle)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_set.count, count)
        self.assertEqual(magnet_set.mtype, mtype)
        self.assertTrue(np.allclose(magnet_set.reference_size, reference_size))
        self.assertTrue(np.allclose(magnet_set.reference_field_vector, reference_field_vector))
        self.assertTrue(np.allclose(magnet_set.flip_matrix, flip_matrix))
        self.assertEqual(magnet_set.names, names)
        self.assertTrue(np.allclose(magnet_set.sizes, sizes))
        self.assertTrue(np.allclose(magnet_set.field_vectors, field_vectors))

    def test_static_from_sim_file_raises_on_bad_parameters(self):
        """
        Tests the MagnetSet class raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        # Assert from_sim_file throws error when file is not a string file path or an open file handle
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', MagnetSet.from_sim_file,
                               mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector,
                               flip_matrix=flip_matrix, file=None)

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
        count, mtype, reference_size, reference_field_vector, flip_matrix, \
            names, sizes, field_vectors = self.dummy_magnet_set_values()

        self.assertRaisesRegex(Exception, '.*', MagnetSet.from_sim_file,
                               mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                               file=os.path.join(inputs_path, 'example_missing_name.sim'))

        self.assertRaisesRegex(Exception, '.*', MagnetSet.from_sim_file,
                               mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                               file=os.path.join(inputs_path, 'example_missing_field_vector.sim'))

        self.assertRaisesRegex(Exception, '.*', MagnetSet.from_sim_file,
                               mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                               file=os.path.join(inputs_path, 'example_invalid_field_vector.sim'))

        self.assertRaisesRegex(Exception, '.*', MagnetSet.from_sim_file,
                               mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                               file=os.path.join(inputs_path, 'example_duplicate_name.sim'))

