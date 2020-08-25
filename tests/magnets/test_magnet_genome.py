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
from optid.magnets import MagnetGenome, MagnetSet

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class MagnetGenomeTest(unittest.TestCase):
    """
    Tests the MagnetGenome class can be imported and used correctly.
    """

    @staticmethod
    def dummy_magnet_genome_values():
        """
        Creates a set of constant test values used for constructing and comparing MagnetGenome
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        count = 4
        magnet_type = 'HH'
        magnet_permutation = np.arange(count).astype(np.int32)
        magnet_flips = (np.arange(count) % 2).astype(np.bool)
        rng_states = (np.random.RandomState(seed=1234), np.random.RandomState(seed=12345))

        return count, magnet_type, magnet_permutation, magnet_flips, rng_states

    @staticmethod
    def compare_rng_states(rng_a, rng_b):
        (a_name, a_state), a_params = rng_a.get_state()[:2], rng_a.get_state()[2:]
        (b_name, b_state), b_params = rng_b.get_state()[:2], rng_b.get_state()[2:]
        return (a_name == b_name) and np.all(a_state == b_state) and (a_params == b_params)

    def test_constructor(self):
        """
        Tests the MagnetGenome class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        # Construct MagnetGenome instance
        magnet_genome = MagnetGenome(magnet_type=magnet_type, magnet_permutation=magnet_permutation,
                                     magnet_flips=magnet_flips, rng_states=rng_states)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.count, count)
        self.assertEqual(magnet_genome.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_genome.magnet_permutation, magnet_permutation))
        self.assertTrue(np.allclose(magnet_genome.magnet_flips, magnet_flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

    def test_constructor_raises_on_bad_parameters_magnet_type(self):
        """
        Tests the MagnetGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        # Assert constructor throws error from empty magnet type string
        self.assertRaisesRegex(optid.errors.ValidateStringEmptyError, '.*', MagnetGenome,
                               magnet_type='', magnet_permutation=magnet_permutation,
                               magnet_flips=magnet_flips, rng_states=rng_states)

        # Assert constructor throws error from magnet type not being a string
        self.assertRaisesRegex(optid.errors.ValidateStringTypeError, '.*', MagnetGenome,
                               magnet_type=None, magnet_permutation=magnet_permutation,
                               magnet_flips=magnet_flips, rng_states=rng_states)

    def test_constructor_raises_on_bad_parameters_magnet_permutation(self):
        """
        Tests the MagnetGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetGenome,
                               magnet_type=magnet_type,
                               magnet_permutation=magnet_permutation[..., np.newaxis],
                               magnet_flips=magnet_flips, rng_states=rng_states)

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetGenome,
                               magnet_type=magnet_type,
                               magnet_permutation=magnet_permutation.astype(np.float32),
                               magnet_flips=magnet_flips, rng_states=rng_states)

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetGenome,
                               magnet_type=magnet_type,
                               magnet_permutation=None,
                               magnet_flips=magnet_flips, rng_states=rng_states)

        self.assertRaisesRegex(optid.errors.ValidateMagnetGenomePermutationDuplicateError, '.*', MagnetGenome,
                               magnet_type=magnet_type,
                               magnet_permutation=np.zeros((count,), dtype=np.int32),
                               magnet_flips=magnet_flips, rng_states=rng_states)

        self.assertRaisesRegex(optid.errors.ValidateMagnetGenomePermutationBoundaryError, '.*', MagnetGenome,
                               magnet_type=magnet_type,
                               magnet_permutation=(magnet_permutation - 1),
                               magnet_flips=magnet_flips, rng_states=rng_states)

        self.assertRaisesRegex(optid.errors.ValidateMagnetGenomePermutationBoundaryError, '.*', MagnetGenome,
                               magnet_type=magnet_type,
                               magnet_permutation=(magnet_permutation + 1),
                               magnet_flips=magnet_flips, rng_states=rng_states)

    def test_constructor_raises_on_bad_parameters_magnet_flips(self):
        """
        Tests the MagnetGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetGenome,
                               magnet_type=magnet_type,
                               magnet_permutation=magnet_permutation,
                               magnet_flips=magnet_flips[..., np.newaxis], rng_states=rng_states)

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetGenome,
                               magnet_type=magnet_type, magnet_permutation=magnet_permutation,
                               magnet_flips=magnet_flips.astype(np.float32), rng_states=rng_states)

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetGenome,
                               magnet_type=magnet_type, magnet_permutation=magnet_permutation,
                               magnet_flips=None, rng_states=rng_states)

    def test_constructor_raises_on_bad_parameters_rng_states(self):
        """
        Tests the MagnetGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        self.assertRaisesRegex(Exception, '.*', MagnetGenome,
                               magnet_type=magnet_type,
                               magnet_permutation=magnet_permutation,
                               magnet_flips=magnet_flips, rng_states=None)

        self.assertRaisesRegex(Exception, '.*', MagnetGenome,
                               magnet_type=magnet_type,
                               magnet_permutation=magnet_permutation,
                               magnet_flips=magnet_flips, rng_states=(None, None))

    def test_from_magnet_set(self):
        """
        Tests the MagnetGenome class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count = 4
        magnet_type = 'HH'
        magnet_names = [f'{index + 1:03d}' for index in range(count)]
        magnet_field_vectors = np.array([
            [0.003770334, -0.000352049, 1.339567917],
            [-0.007018214, -0.002714164, 1.344710227],
            [-0.004826321, -0.001714764, 1.342079598],
            [0.008846784, -0.003088993, 1.344698631],
        ], dtype=np.float32)

        # Construct MagnetSet instance
        magnet_set = MagnetSet(magnet_type=magnet_type, magnet_names=magnet_names,
                               magnet_field_vectors=magnet_field_vectors)

        # Construct MagnetGenome instance
        magnet_genome = MagnetGenome.from_magnet_set(magnet_set=magnet_set, seed=1234)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.count, count)
        self.assertEqual(magnet_genome.magnet_type, magnet_type)

    def test_from_magnet_genome(self):
        """
        Tests the MagnetGenome class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        # Construct MagnetGenome instance
        magnet_genome = MagnetGenome(magnet_type=magnet_type, magnet_permutation=magnet_permutation,
                                     magnet_flips=magnet_flips, rng_states=rng_states)

        # Construct MagnetGenome instance from a parent genome
        child_genome = MagnetGenome.from_magnet_genome(magnet_genome=magnet_genome)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.count, child_genome.count)
        self.assertEqual(magnet_genome.magnet_type, child_genome.magnet_type)
        self.assertTrue(np.allclose(magnet_genome.magnet_permutation, child_genome.magnet_permutation))
        self.assertTrue(np.allclose(magnet_genome.magnet_flips, magnet_flips))
        self.assertFalse(self.compare_rng_states(magnet_genome.rng_children, child_genome.rng_children))
        self.assertFalse(self.compare_rng_states(magnet_genome.rng_mutations, child_genome.rng_mutations))

    def test_save(self):
        """
        Tests the MagnetGenome class can be saved to a .maggenome file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.maggenome')

            # Construct MagnetGenome instance
            magnet_genome = MagnetGenome(magnet_type=magnet_type, magnet_permutation=magnet_permutation,
                                         magnet_flips=magnet_flips, rng_states=rng_states)

            # Save the MagnetGenome to the temporary directory
            magnet_genome.save(file=tmp_file_path)

            # Throw away the local object and reload it from the temporary file
            magnet_genome = MagnetGenome.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.count, count)
        self.assertEqual(magnet_genome.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_genome.magnet_permutation, magnet_permutation))
        self.assertTrue(np.allclose(magnet_genome.magnet_flips, magnet_flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

    def test_save_open_file_handle(self):
        """
        Tests the MagnetGenome class can be saved to a .maggenome file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.maggenome')

            # Construct MagnetGenome instance
            magnet_genome = MagnetGenome(magnet_type=magnet_type, magnet_permutation=magnet_permutation,
                                         magnet_flips=magnet_flips, rng_states=rng_states)

            with open(tmp_file_path, 'wb') as tmp_file_handle:
                # Save the MagnetGenome to the temporary directory
                magnet_genome.save(file=tmp_file_handle)

            # Throw away the local object and reload it from the temporary file
            magnet_genome = MagnetGenome.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.count, count)
        self.assertEqual(magnet_genome.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_genome.magnet_permutation, magnet_permutation))
        self.assertTrue(np.allclose(magnet_genome.magnet_flips, magnet_flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

    def test_save_raises_on_bad_parameters(self):
        """
        Tests the MagnetGenome class save member function raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        # Construct MagnetGenome instance
        magnet_genome = MagnetGenome(magnet_type=magnet_type, magnet_permutation=magnet_permutation,
                                     magnet_flips=magnet_flips, rng_states=rng_states)

        # Attempt to save to a bad file parameter
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', magnet_genome.save, file=None)

    def test_static_from_file(self):
        """
        Tests the MagnetGenome class can be constructed from a .maggenome file using the static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        # Construct MagnetGenome instance
        magnet_genome = MagnetGenome.from_file(file=os.path.join(inputs_path, 'example.maggenome'))

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.count, count)
        self.assertEqual(magnet_genome.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_genome.magnet_permutation, magnet_permutation))
        self.assertTrue(np.allclose(magnet_genome.magnet_flips, magnet_flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

    def test_static_from_file_open_file_handle(self):
        """
        Tests the MagnetGenome class can be constructed from an open handle to a .maggenome file using the
        static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, magnet_type, magnet_permutation, magnet_flips, rng_states = self.dummy_magnet_genome_values()

        with open(os.path.join(inputs_path, 'example.maggenome'), 'rb') as file_handle:
            # Construct MagnetGenome instance
            magnet_genome = MagnetGenome.from_file(file=file_handle)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.count, count)
        self.assertEqual(magnet_genome.magnet_type, magnet_type)
        self.assertTrue(np.allclose(magnet_genome.magnet_permutation, magnet_permutation))
        self.assertTrue(np.allclose(magnet_genome.magnet_flips, magnet_flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

    def test_static_from_file_raises_on_bad_parameters(self):
        """
        Tests the MagnetGenome class raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Attempt to load from to a bad file parameter
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', MagnetGenome.from_file, file=None)
