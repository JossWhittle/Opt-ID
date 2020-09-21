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

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


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

        count = 8
        mtype = 'HH'
        beams = [f'B{((index % 2) + 1):d}' for index in range(count)]
        slots = [f'S{(((index - (index % 2)) // 2) + 1):03d}' for index in range(count)]
        
        positions = np.zeros((count, 3), dtype=np.float32)

        shim_vectors = np.zeros((count, 3), dtype=np.float32)
        shim_vectors[:, 1] = 1.0

        direction_matrices = np.zeros((count, 3, 3), dtype=np.float32)
        direction_matrices[:, ...] = np.eye(3, dtype=np.float32)[np.newaxis, ...]
        
        return count, mtype, beams, slots, positions, shim_vectors, direction_matrices

    def test_constructor(self):
        """
        Tests the MagnetSlots class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        # Construct MagnetSlots instance
        magnet_slots = MagnetSlots(mtype=mtype, beams=beams, slots=slots, positions=positions,
                                   shim_vectors=shim_vectors, direction_matrices=direction_matrices)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_slots.count, count)
        self.assertEqual(magnet_slots.mtype, mtype)
        self.assertEqual(magnet_slots.beams, beams)
        self.assertEqual(magnet_slots.slots, slots)
        self.assertTrue(np.allclose(magnet_slots.positions, positions))
        self.assertTrue(np.allclose(magnet_slots.shim_vectors, shim_vectors))
        self.assertTrue(np.allclose(magnet_slots.direction_matrices, direction_matrices))

    def test_constructor_raises_on_bad_parameters_mtype(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        fixed_params = dict(beams=beams, slots=slots, positions=positions,
                            shim_vectors=shim_vectors, direction_matrices=direction_matrices)

        # Assert constructor throws error from empty magnet type string
        self.assertRaisesRegex(optid.errors.ValidateStringEmptyError, '.*', MagnetSlots, **fixed_params,
                               mtype='')

        # Assert constructor throws error from wrong typed magnet type string
        self.assertRaisesRegex(optid.errors.ValidateStringTypeError, '.*', MagnetSlots, **fixed_params,
                               mtype=None)

    def test_constructor_raises_on_bad_parameters_beams(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        fixed_params = dict(mtype=mtype, slots=slots, positions=positions,
                            shim_vectors=shim_vectors, direction_matrices=direction_matrices)

        # Assert constructor throws error from wrong typed list of name strings for magnet beams
        self.assertRaisesRegex(optid.errors.ValidateStringListTypeError, '.*', MagnetSlots, **fixed_params,
                               beams=None)

        # Assert constructor throws error from empty list of name strings for magnet beams
        self.assertRaisesRegex(optid.errors.ValidateStringListEmptyError, '.*', MagnetSlots, **fixed_params,
                               beams=[])

        # Assert constructor throws error from empty name string in magnet beams
        self.assertRaisesRegex(optid.errors.ValidateStringListElementEmptyError, '.*', MagnetSlots, **fixed_params,
                               beams=['' if (index == 1) else beam for index, beam in enumerate(beams)])

        # Assert constructor throws error from wrong typed string in magnet beams
        self.assertRaisesRegex(optid.errors.ValidateStringListElementTypeError, '.*', MagnetSlots, **fixed_params,
                               beams=[None if (index == 1) else beam for index, beam in enumerate(beams)])

    def test_constructor_raises_on_bad_parameters_slots(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        fixed_params = dict(mtype=mtype, beams=beams, positions=positions,
                            shim_vectors=shim_vectors, direction_matrices=direction_matrices)

        self.assertRaisesRegex(optid.errors.ValidateStringListTypeError, '.*', MagnetSlots, **fixed_params,
                               slots=None)

        self.assertRaisesRegex(optid.errors.ValidateStringListShapeError, '.*', MagnetSlots, **fixed_params,
                               slots=[])

        self.assertRaisesRegex(optid.errors.ValidateStringListElementEmptyError, '.*', MagnetSlots, **fixed_params,
                               slots=['' if (index == 1) else slot for index, slot in enumerate(slots)])

        self.assertRaisesRegex(optid.errors.ValidateStringListElementTypeError, '.*', MagnetSlots, **fixed_params,
                               slots=[None if (index == 1) else slot for index, slot in enumerate(slots)])

        self.assertRaisesRegex(optid.errors.ValidateStringListElementUniquenessError, '.*', MagnetSlots, **fixed_params,
                               slots=[f'S000' for index in range(count)])



    def test_constructor_raises_on_bad_parameters_positions(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        fixed_params = dict(mtype=mtype, beams=beams, slots=slots,
                            shim_vectors=shim_vectors, direction_matrices=direction_matrices)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSlots, **fixed_params,
                               positions=positions[:-1])

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetSlots, **fixed_params,
                               positions=positions.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetSlots, **fixed_params,
                               positions=None)

    def test_constructor_raises_on_bad_parameters_shim_vectors(self):
        """
        Tests the MagnetSlots class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        fixed_params = dict(mtype=mtype, beams=beams, slots=slots, positions=positions,
                            direction_matrices=direction_matrices)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSlots, **fixed_params,
                               shim_vectors=shim_vectors[:-1])

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetSlots, **fixed_params,
                               shim_vectors=shim_vectors.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetSlots, **fixed_params,
                               shim_vectors=None)

    def test_constructor_raises_on_bad_parameters_direction_matrices(self):
        """
        Tests the MagnetSet class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        fixed_params = dict(mtype=mtype, beams=beams, slots=slots, positions=positions, shim_vectors=shim_vectors)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSlots, **fixed_params,
                               direction_matrices=direction_matrices[:-1])

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetSlots, **fixed_params,
                               direction_matrices=direction_matrices.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetSlots, **fixed_params,
                               direction_matrices=None)

    def test_save(self):
        """
        Tests the MagnetSlots class can be saved to a .magslots file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magslots')

            # Construct MagnetSlots instance
            magnet_slots = MagnetSlots(mtype=mtype, beams=beams, slots=slots, positions=positions,
                                       shim_vectors=shim_vectors, direction_matrices=direction_matrices)

            # Save the MagnetSlots to the temporary directory
            magnet_slots.save(file=tmp_file_path)

            # Throw away the local object and reload it from the temporary file
            magnet_slots = MagnetSlots.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_slots.count, count)
        self.assertEqual(magnet_slots.mtype, mtype)
        self.assertEqual(magnet_slots.beams, beams)
        self.assertEqual(magnet_slots.slots, slots)
        self.assertTrue(np.allclose(magnet_slots.positions, positions))
        self.assertTrue(np.allclose(magnet_slots.shim_vectors, shim_vectors))
        self.assertTrue(np.allclose(magnet_slots.direction_matrices, direction_matrices))

    def test_save_open_file_handle(self):
        """
        Tests the MagnetSlots class can be saved to a .magslots file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magslots')

            # Construct MagnetSlots instance
            magnet_slots = MagnetSlots(mtype=mtype, beams=beams, slots=slots, positions=positions,
                                       shim_vectors=shim_vectors, direction_matrices=direction_matrices)

            with open(tmp_file_path, 'wb') as tmp_file_handle:
                # Save the MagnetSlots to the temporary directory
                magnet_slots.save(file=tmp_file_handle)

            # Throw away the local object and reload it from the temporary file
            magnet_slots = MagnetSlots.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_slots.count, count)
        self.assertEqual(magnet_slots.mtype, mtype)
        self.assertEqual(magnet_slots.beams, beams)
        self.assertEqual(magnet_slots.slots, slots)
        self.assertTrue(np.allclose(magnet_slots.positions, positions))
        self.assertTrue(np.allclose(magnet_slots.shim_vectors, shim_vectors))
        self.assertTrue(np.allclose(magnet_slots.direction_matrices, direction_matrices))

    def test_save_raises_on_bad_parameters(self):
        """
        Tests the MagnetSlots class save member function raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        # Construct MagnetSlots instance
        magnet_slots = MagnetSlots(mtype=mtype, beams=beams, slots=slots, positions=positions,
                                   shim_vectors=shim_vectors, direction_matrices=direction_matrices)

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
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        # Construct MagnetSlots instance
        magnet_slots = MagnetSlots.from_file(file=os.path.join(inputs_path, 'example.magslots'))

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_slots.count, count)
        self.assertEqual(magnet_slots.mtype, mtype)
        self.assertEqual(magnet_slots.beams, beams)
        self.assertEqual(magnet_slots.slots, slots)
        self.assertTrue(np.allclose(magnet_slots.positions, positions))
        self.assertTrue(np.allclose(magnet_slots.shim_vectors, shim_vectors))
        self.assertTrue(np.allclose(magnet_slots.direction_matrices, direction_matrices))

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
        count, mtype, beams, slots, positions, shim_vectors, direction_matrices = self.dummy_magnet_slots_values()

        with open(os.path.join(inputs_path, 'example.magslots'), 'rb') as file_handle:
            # Construct MagnetSlots instance
            magnet_slots = MagnetSlots.from_file(file=file_handle)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_slots.count, count)
        self.assertEqual(magnet_slots.mtype, mtype)
        self.assertEqual(magnet_slots.beams, beams)
        self.assertEqual(magnet_slots.slots, slots)
        self.assertTrue(np.allclose(magnet_slots.positions, positions))
        self.assertTrue(np.allclose(magnet_slots.shim_vectors, shim_vectors))
        self.assertTrue(np.allclose(magnet_slots.direction_matrices, direction_matrices))

    def test_static_from_file_raises_on_bad_parameters(self):
        """
        Tests the MagnetSlots class raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Attempt to load from to a bad file parameter
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', MagnetSlots.from_file, file=None)
