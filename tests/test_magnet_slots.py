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
from optid.magnets import MagnetSlots


class MagnetSlotsTest(unittest.TestCase):
    """
    Tests the MagnetSlots class can be imported and used correctly.
    """

    @staticmethod
    def dummy_magnet_slots_values():
        """
        Creates a set of constant test values used for constructing and comparing MagnetSlots
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        count = 4
        magnet_type  = 'HH'
        magnet_size  = np.array([10, 10, 2], dtype=np.float32)
        magnet_beams = [f'Q1' for index in range(count)]

        magnet_positions = []
        x_pos, z_pos, s_pos = (magnet_size[0] * -0.5), (-magnet_size[1]), 0
        for index in range(count):
            magnet_positions += [np.array([x_pos, z_pos, s_pos], dtype=np.float32)]
            s_pos += magnet_size[2]
        magnet_positions = np.stack(magnet_positions, axis=0)

        magnet_direction_matrices = np.repeat(np.eye(3, dtype=np.float32)[np.newaxis, ...], axis=0, repeats=count)
        magnet_flip_vectors = np.ones((count, 3), dtype=np.float32)

        return count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
               magnet_direction_matrices, magnet_flip_vectors

    def test_constructor(self):
        """
        Tests the MagnetSlots class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Construct MagnetSlots instance
        magnet_slots = MagnetSlots(magnet_type=magnet_type, magnet_size=magnet_size,
                                   magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                                   magnet_direction_matrices=magnet_direction_matrices,
                                   magnet_flip_vectors=magnet_flip_vectors)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_slots.count, count)
        self.assertEqual(magnet_slots.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_slots.magnet_size, magnet_size))
        self.assertEqual(magnet_slots.magnet_beams, magnet_beams)
        self.assertTrue(np.allclose(magnet_slots.magnet_positions, magnet_positions))
        self.assertTrue(np.allclose(magnet_slots.magnet_direction_matrices, magnet_direction_matrices))
        self.assertTrue(np.allclose(magnet_slots.magnet_flip_vectors, magnet_flip_vectors))

    def test_constructor_raises_on_bad_parameters_magnet_type(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Assert constructor throws error from empty magnet type string
        self.assertRaisesRegex(optid.errors.StringEmptyError, '.*', MagnetSlots,
                               magnet_type='', magnet_size=magnet_size,
                               magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

        # Assert constructor throws error from wrong typed magnet type string
        self.assertRaisesRegex(optid.errors.StringTypeError, '.*', MagnetSlots,
                               magnet_type=None, magnet_size=magnet_size,
                               magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

    def test_constructor_raises_on_bad_parameters_magnet_size(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Assert constructor throws error from incorrectly shaped magnet size
        self.assertRaisesRegex(optid.errors.TensorShapeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=np.random.uniform(size=(4,)),
                               magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

        # Assert constructor throws error from incorrectly typed magnet size
        self.assertRaisesRegex(optid.errors.TensorTypeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size.astype(np.int32),
                               magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

    def test_constructor_raises_on_bad_parameters_magnet_beams(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Assert constructor throws error from wrong typed list of name strings for magnet beams
        self.assertRaisesRegex(optid.errors.StringListTypeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_beams=None, magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

        # Assert constructor throws error from empty list of name strings for magnet beams
        self.assertRaisesRegex(optid.errors.StringListEmptyError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_beams=[], magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

        # Assert constructor throws error from empty name string in magnet beams
        self.assertRaisesRegex(optid.errors.StringListElementEmptyError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_beams=['' if (index == 1) else beam
                                             for index, beam in enumerate(magnet_beams)],
                               magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

    def test_constructor_raises_on_bad_parameters_magnet_positions(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Assert constructor throws error from incorrectly shaped magnet positions
        self.assertRaisesRegex(optid.errors.TensorShapeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size, magnet_beams=magnet_beams,
                               magnet_positions=np.random.uniform(size=(count, 4)),
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

        # Assert constructor throws error from magnet names and magnet positions being different lengths
        self.assertRaisesRegex(optid.errors.TensorShapeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_beams=magnet_beams, magnet_positions=magnet_positions[:-1],
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

        # Assert constructor throws error from incorrectly typed magnet positions
        self.assertRaisesRegex(optid.errors.TensorTypeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size, magnet_beams=magnet_beams,
                               magnet_positions=magnet_positions.astype(np.int32),
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors)

    def test_constructor_raises_on_bad_parameters_magnet_direction_matrices(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Assert constructor throws error from magnet names and magnet direction matrices being different lengths
        self.assertRaisesRegex(optid.errors.TensorShapeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices[:-1],
                               magnet_flip_vectors=magnet_flip_vectors)

        # Assert constructor throws error from incorrectly typed magnet direction matrices
        self.assertRaisesRegex(optid.errors.TensorTypeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices.astype(np.int32),
                               magnet_flip_vectors=magnet_flip_vectors)

    def test_constructor_raises_on_bad_parameters_magnet_flip_vectors(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Assert constructor throws error from magnet names and magnet flip vectors being different lengths
        self.assertRaisesRegex(optid.errors.TensorShapeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors[:-1])

        # Assert constructor throws error from incorrectly typed magnet flip vectors
        self.assertRaisesRegex(optid.errors.TensorTypeError, '.*', MagnetSlots,
                               magnet_type=magnet_type, magnet_size=magnet_size,
                               magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                               magnet_direction_matrices=magnet_direction_matrices,
                               magnet_flip_vectors=magnet_flip_vectors.astype(np.int32))

    def test_save(self):
        """
        Tests the MagnetSlots class can be saved to a .magslots file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magslots')

            # Construct MagnetSlots instance
            magnet_slots = MagnetSlots(magnet_type=magnet_type, magnet_size=magnet_size,
                                       magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                                       magnet_direction_matrices=magnet_direction_matrices,
                                       magnet_flip_vectors=magnet_flip_vectors)

            # Save the MagnetSlots to the temporary directory
            magnet_slots.save(file=tmp_file_path)

            # Throw away the local object and reload it from the temporary file
            magnet_slots = MagnetSlots.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

            # Assert object members have been correctly assigned
            self.assertEqual(magnet_slots.count, count)
            self.assertEqual(magnet_slots.magnet_type, magnet_type)
            self.assertTrue(np.allclose(magnet_slots.magnet_size, magnet_size))
            self.assertEqual(magnet_slots.magnet_beams, magnet_beams)
            self.assertTrue(np.allclose(magnet_slots.magnet_positions, magnet_positions))
            self.assertTrue(np.allclose(magnet_slots.magnet_direction_matrices, magnet_direction_matrices))
            self.assertTrue(np.allclose(magnet_slots.magnet_flip_vectors, magnet_flip_vectors))

    def test_save_open_file_handle(self):
        """
        Tests the MagnetSlots class can be saved to a .magslots file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magslots')

            # Construct MagnetSlots instance
            magnet_slots = MagnetSlots(magnet_type=magnet_type, magnet_size=magnet_size,
                                       magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                                       magnet_direction_matrices=magnet_direction_matrices,
                                       magnet_flip_vectors=magnet_flip_vectors)

            with open(tmp_file_path, 'wb') as tmp_file_handle:
                # Save the MagnetSlots to the temporary directory
                magnet_slots.save(file=tmp_file_handle)

            # Throw away the local object and reload it from the temporary file
            magnet_slots = MagnetSlots.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

            # Assert object members have been correctly assigned
            self.assertEqual(magnet_slots.count, count)
            self.assertEqual(magnet_slots.magnet_type, magnet_type)
            self.assertTrue(np.allclose(magnet_slots.magnet_size, magnet_size))
            self.assertEqual(magnet_slots.magnet_beams, magnet_beams)
            self.assertTrue(np.allclose(magnet_slots.magnet_positions, magnet_positions))
            self.assertTrue(np.allclose(magnet_slots.magnet_direction_matrices, magnet_direction_matrices))
            self.assertTrue(np.allclose(magnet_slots.magnet_flip_vectors, magnet_flip_vectors))

    def test_save_raises_on_bad_parameters(self):
        """
        Tests the MagnetSlots class save member function raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Construct MagnetSlots instance
        magnet_slots = MagnetSlots(magnet_type=magnet_type, magnet_size=magnet_size,
                                   magnet_beams=magnet_beams, magnet_positions=magnet_positions,
                                   magnet_direction_matrices=magnet_direction_matrices,
                                   magnet_flip_vectors=magnet_flip_vectors)

        # Attempt to save to a bad file parameter
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', magnet_slots.save, file=None)

    def test_static_from_file(self):
        """
        Tests the MagnetSlots class can be constructed from a .magslots file using the static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        # Construct MagnetSlots instance
        magnet_slots = MagnetSlots.from_file(file=os.path.join(inputs_path, 'example.magslots'))

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_slots.count, count)
        self.assertEqual(magnet_slots.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_slots.magnet_size, magnet_size))
        self.assertEqual(magnet_slots.magnet_beams, magnet_beams)
        self.assertTrue(np.allclose(magnet_slots.magnet_positions, magnet_positions))
        self.assertTrue(np.allclose(magnet_slots.magnet_direction_matrices, magnet_direction_matrices))
        self.assertTrue(np.allclose(magnet_slots.magnet_flip_vectors, magnet_flip_vectors))

    def test_static_from_file_open_file_handle(self):
        """
        Tests the MagnetSlots class can be constructed from an open handle to a .magslots file using the
        static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, magnet_type, magnet_size, magnet_beams, magnet_positions, \
                         magnet_direction_matrices, magnet_flip_vectors = self.dummy_magnet_slots_values()

        with open(os.path.join(inputs_path, 'example.magslots'), 'rb') as file_handle:
            # Construct MagnetSlots instance
            magnet_slots = MagnetSlots.from_file(file=file_handle)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_slots.count, count)
        self.assertEqual(magnet_slots.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_slots.magnet_size, magnet_size))
        self.assertEqual(magnet_slots.magnet_beams, magnet_beams)
        self.assertTrue(np.allclose(magnet_slots.magnet_positions, magnet_positions))
        self.assertTrue(np.allclose(magnet_slots.magnet_direction_matrices, magnet_direction_matrices))
        self.assertTrue(np.allclose(magnet_slots.magnet_flip_vectors, magnet_flip_vectors))

    def test_static_from_file_raises_on_bad_parameters(self):
        """
        Tests the MagnetSlots class raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Attempt to load from to a bad file parameter
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', MagnetSlots.from_file, file=None)
