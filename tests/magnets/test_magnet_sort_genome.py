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
from optid.magnets import MagnetSortGenome, MagnetSet, MagnetSlots, MagnetSortLookup
from optid.utils import Range

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class MagnetSortGenomeTest(unittest.TestCase):
    """
    Tests the MagnetSortGenome class can be imported and used correctly.
    """

    @staticmethod
    def dummy_magnet_genome_values():
        """
        Creates a set of constant test values used for constructing and comparing MagnetSortGenome
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        count = 4
        mtype = 'HH'

        # MagnetSet
        reference_size = np.array([10, 10, 2], dtype=np.float32)
        reference_field_vector = np.array([0, 0, 1], dtype=np.float32)
        flip_matrix = np.ones((3, 3), dtype=np.float32)
        names = [f'{index + 1:03d}' for index in range(count)]
        sizes = np.ones((count, 3)) * reference_size[np.newaxis, ...]
        field_vectors = np.array([
            [0.003770334, -0.000352049, 1.339567917],
            [-0.007018214, -0.002714164, 1.344710227],
            [-0.004826321, -0.001714764, 1.342079598],
            [0.008846784, -0.003088993, 1.344698631],
        ], dtype=np.float32)

        magnet_set = MagnetSet(mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                               names=names, sizes=sizes, field_vectors=field_vectors)

        # MagnetSlots
        beams = [f'B{((index % 2) + 1):d}' for index in range(count)]
        slots = [f'S{(((index - (index % 2)) // 2) + 1):03d}' for index in range(count)]
        positions = np.zeros((count, 3), dtype=np.float32)
        shim_vectors = np.zeros((count, 3), dtype=np.float32)
        shim_vectors[:, 1] = 1.0
        direction_matrices = np.zeros((count, 3, 3), dtype=np.float32)
        direction_matrices[:, ...] = np.eye(3, dtype=np.float32)[np.newaxis, ...]

        magnet_slots = MagnetSlots(mtype=mtype, beams=beams, slots=slots, positions=positions,
                                   shim_vectors=shim_vectors, direction_matrices=direction_matrices)

        # MagnetSortLookup
        x_range = Range(-1, 1, 5)
        z_range = Range(-1, 1, 5)
        s_range = Range(-1, 1, 5)
        lookup = np.random.uniform(size=(count, x_range.steps, z_range.steps, s_range.steps, 3, 3))

        magnet_lookup = MagnetSortLookup(mtype=mtype, x_range=x_range,
                                         z_range=z_range, s_range=s_range, lookup=lookup)

        permutation = np.arange(count).astype(np.int32)
        flips = (np.arange(count) % 2).astype(np.bool)
        rng_states = (np.random.RandomState(seed=1234), np.random.RandomState(seed=12345))

        return count, mtype, permutation, flips, rng_states, magnet_set, magnet_slots, magnet_lookup

    @staticmethod
    def compare_rng_states(rng_a, rng_b):
        (a_name, a_state), a_params = rng_a.get_state()[:2], rng_a.get_state()[2:]
        (b_name, b_state), b_params = rng_b.get_state()[:2], rng_b.get_state()[2:]
        return (a_name == b_name) and np.all(a_state == b_state) and (a_params == b_params)

    def test_constructor(self):
        """
        Tests the MagnetSortGenome class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        # Construct MagnetSortGenome instance
        magnet_genome = MagnetSortGenome(mtype=mtype, permutation=permutation, flips=flips,
                                         rng_states=rng_states, magnet_set=magnet_set,
                                         magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.set_count, count)
        self.assertEqual(magnet_genome.slot_count, count)
        self.assertEqual(magnet_genome.mtype, mtype)
        self.assertTrue(np.allclose(magnet_genome.permutation, permutation))
        self.assertTrue(np.allclose(magnet_genome.flips, flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

        self.assertTrue(magnet_genome.magnet_set is magnet_set)
        self.assertTrue(magnet_genome.magnet_slots is magnet_slots)
        self.assertTrue(magnet_genome.magnet_lookup is magnet_lookup)

    def test_non_flippable(self):
        """
        Tests the MagnetSortGenome class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, mtype, _, _, _, \
            _, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        # MagnetSet
        reference_size = np.array([10, 10, 2], dtype=np.float32)
        reference_field_vector = np.array([0, 0, 1], dtype=np.float32)
        flip_matrix = np.eye(3, dtype=np.float32)
        names = [f'{index + 1:03d}' for index in range(count)]
        sizes = np.ones((count, 3)) * reference_size[np.newaxis, ...]
        field_vectors = np.array([
            [0.003770334, -0.000352049, 1.339567917],
            [-0.007018214, -0.002714164, 1.344710227],
            [-0.004826321, -0.001714764, 1.342079598],
            [0.008846784, -0.003088993, 1.344698631],
        ], dtype=np.float32)

        magnet_set = MagnetSet(mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                               names=names, sizes=sizes, field_vectors=field_vectors)

        magnet_genome = MagnetSortGenome.from_random(seed=1234, magnet_set=magnet_set,
                                                     magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.set_count, count)
        self.assertEqual(magnet_genome.slot_count, count)
        self.assertEqual(magnet_genome.mtype, mtype)
        self.assertTrue(np.all(~magnet_genome.flips))

        self.assertTrue(magnet_genome.magnet_set is magnet_set)
        self.assertTrue(magnet_genome.magnet_slots is magnet_slots)
        self.assertTrue(magnet_genome.magnet_lookup is magnet_lookup)

    def test_constructor_raises_on_bad_parameters_mtype(self):
        """
        Tests the MagnetSortGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        fixed_params = dict(permutation=permutation, flips=flips, rng_states=rng_states,
                            magnet_set=magnet_set, magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        # Assert constructor throws error from empty magnet type string
        self.assertRaisesRegex(optid.errors.ValidateStringEmptyError, '.*', MagnetSortGenome, **fixed_params,
                               mtype='')

        # Assert constructor throws error from magnet type not being a string
        self.assertRaisesRegex(optid.errors.ValidateStringTypeError, '.*', MagnetSortGenome, **fixed_params,
                               mtype=None)

    def test_constructor_raises_on_bad_parameters_permutation(self):
        """
        Tests the MagnetSortGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        fixed_params = dict(mtype=mtype, flips=flips, rng_states=rng_states,
                            magnet_set=magnet_set, magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSortGenome, **fixed_params,
                               permutation=permutation[..., np.newaxis])

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetSortGenome, **fixed_params,
                               permutation=permutation.astype(np.float32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetSortGenome, **fixed_params,
                               permutation=None)

        self.assertRaisesRegex(Exception, '.*', MagnetSortGenome,
                               **fixed_params, permutation=np.zeros((count,), dtype=np.int32))

        self.assertRaisesRegex(Exception, '.*', MagnetSortGenome,
                               **fixed_params, permutation=(permutation - 1))

        self.assertRaisesRegex(Exception, '.*', MagnetSortGenome,
                               **fixed_params, permutation=(permutation + 1))

    def test_constructor_raises_on_bad_parameters_flips(self):
        """
        Tests the MagnetSortGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        fixed_params = dict(mtype=mtype, permutation=permutation, rng_states=rng_states,
                            magnet_set=magnet_set, magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', MagnetSortGenome, **fixed_params,
                               flips=flips[..., np.newaxis])

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', MagnetSortGenome, **fixed_params,
                               flips=flips.astype(np.float32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', MagnetSortGenome, **fixed_params,
                               flips=None)

    def test_constructor_raises_on_bad_parameters_rng_states(self):
        """
        Tests the MagnetSortGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        fixed_params = dict(mtype=mtype, permutation=permutation, flips=flips,
                            magnet_set=magnet_set, magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        self.assertRaisesRegex(Exception, '.*', MagnetSortGenome, **fixed_params, rng_states=None)

        self.assertRaisesRegex(Exception, '.*', MagnetSortGenome, **fixed_params, rng_states=(None, None))

    def test_constructor_raises_on_bad_parameters_magnet_set(self):
        """
        Tests the MagnetSortGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        fixed_params = dict(mtype=mtype, permutation=permutation, flips=flips,
                            rng_states=rng_states, magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        self.assertRaisesRegex(Exception, '.*', MagnetSortGenome, **fixed_params, magnet_set=None)

    def test_constructor_raises_on_bad_parameters_magnet_slots(self):
        """
        Tests the MagnetSortGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        fixed_params = dict(mtype=mtype, permutation=permutation, flips=flips,
                            rng_states=rng_states, magnet_set=magnet_set, magnet_lookup=magnet_lookup)

        self.assertRaisesRegex(Exception, '.*', MagnetSortGenome, **fixed_params, magnet_slots=None)

    def test_constructor_raises_on_bad_parameters_magnet_lookup(self):
        """
        Tests the MagnetSortGenome class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        fixed_params = dict(mtype=mtype, permutation=permutation, flips=flips,
                            rng_states=rng_states, magnet_set=magnet_set, magnet_slots=magnet_slots)

        self.assertRaisesRegex(Exception, '.*', MagnetSortGenome, **fixed_params, magnet_lookup=None)

    def test_from_random(self):
        """
        Tests the MagnetSortGenome class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        # Construct MagnetSortGenome instance
        magnet_genome = MagnetSortGenome.from_random(seed=1234, magnet_set=magnet_set,
                                                 magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.set_count, count)
        self.assertEqual(magnet_genome.slot_count, count)
        self.assertEqual(magnet_genome.mtype, mtype)

    def test_from_magnet_genome(self):
        """
        Tests the MagnetSortGenome class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        # Construct MagnetSortGenome instance
        magnet_genome = MagnetSortGenome(mtype=mtype, permutation=permutation,
                                         flips=flips, rng_states=rng_states, magnet_set=magnet_set,
                                         magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        # Construct MagnetSortGenome instance from a parent genome
        child_genome = MagnetSortGenome.from_magnet_genome(magnet_genome=magnet_genome)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.set_count, child_genome.set_count)
        self.assertEqual(magnet_genome.mtype, child_genome.mtype)
        self.assertTrue(np.allclose(magnet_genome.permutation, child_genome.permutation))
        self.assertTrue(np.allclose(magnet_genome.flips, flips))
        self.assertFalse(self.compare_rng_states(magnet_genome.rng_children, child_genome.rng_children))
        self.assertFalse(self.compare_rng_states(magnet_genome.rng_mutations, child_genome.rng_mutations))

        self.assertTrue(magnet_genome.magnet_set is child_genome.magnet_set)
        self.assertTrue(magnet_genome.magnet_slots is child_genome.magnet_slots)
        self.assertTrue(magnet_genome.magnet_lookup is child_genome.magnet_lookup)

    def test_save(self):
        """
        Tests the MagnetSortGenome class can be saved to a .magsortgenome file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magsortgenome')

            # Construct MagnetSortGenome instance
            magnet_genome = MagnetSortGenome(mtype=mtype, permutation=permutation,
                                             flips=flips, rng_states=rng_states, magnet_set=magnet_set,
                                             magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

            # Save the MagnetSortGenome to the temporary directory
            magnet_genome.save(file=tmp_file_path)

            # Throw away the local object and reload it from the temporary file
            magnet_genome = MagnetSortGenome.from_file(file=tmp_file_path, magnet_set=magnet_set,
                                                       magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.set_count, count)
        self.assertEqual(magnet_genome.slot_count, count)
        self.assertEqual(magnet_genome.mtype, mtype)
        self.assertTrue(np.allclose(magnet_genome.permutation, permutation))
        self.assertTrue(np.allclose(magnet_genome.flips, flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

        self.assertTrue(magnet_genome.magnet_set is magnet_set)
        self.assertTrue(magnet_genome.magnet_slots is magnet_slots)
        self.assertTrue(magnet_genome.magnet_lookup is magnet_lookup)

    def test_save_open_file_handle(self):
        """
        Tests the MagnetSortGenome class can be saved to a .magsortgenome file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magsortgenome')

            # Construct MagnetSortGenome instance
            magnet_genome = MagnetSortGenome(mtype=mtype, permutation=permutation,
                                             flips=flips, rng_states=rng_states, magnet_set=magnet_set,
                                             magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

            with open(tmp_file_path, 'wb') as tmp_file_handle:
                # Save the MagnetSortGenome to the temporary directory
                magnet_genome.save(file=tmp_file_handle)

            # Throw away the local object and reload it from the temporary file
            magnet_genome = MagnetSortGenome.from_file(file=tmp_file_path, magnet_set=magnet_set,
                                                   magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.set_count, count)
        self.assertEqual(magnet_genome.slot_count, count)
        self.assertEqual(magnet_genome.mtype, mtype)
        self.assertTrue(np.allclose(magnet_genome.permutation, permutation))
        self.assertTrue(np.allclose(magnet_genome.flips, flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

        self.assertTrue(magnet_genome.magnet_set is magnet_set)
        self.assertTrue(magnet_genome.magnet_slots is magnet_slots)
        self.assertTrue(magnet_genome.magnet_lookup is magnet_lookup)

    def test_save_raises_on_bad_parameters(self):
        """
        Tests the MagnetSortGenome class save member function raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        # Construct MagnetSortGenome instance
        magnet_genome = MagnetSortGenome(mtype=mtype, permutation=permutation,
                                         flips=flips, rng_states=rng_states, magnet_set=magnet_set,
                                         magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        # Attempt to save to a bad file parameter
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', magnet_genome.save, file=None)

    def test_static_from_file(self):
        """
        Tests the MagnetSortGenome class can be constructed from a .magsortgenome file using the static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        # Construct MagnetSortGenome instance
        magnet_genome = MagnetSortGenome.from_file(file=os.path.join(inputs_path, 'example.magsortgenome'),
                                               magnet_set=magnet_set, magnet_slots=magnet_slots,
                                               magnet_lookup=magnet_lookup)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.set_count, count)
        self.assertEqual(magnet_genome.slot_count, count)
        self.assertEqual(magnet_genome.mtype, mtype)
        self.assertTrue(np.allclose(magnet_genome.permutation, permutation))
        self.assertTrue(np.allclose(magnet_genome.flips, flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

        self.assertTrue(magnet_genome.magnet_set is magnet_set)
        self.assertTrue(magnet_genome.magnet_slots is magnet_slots)
        self.assertTrue(magnet_genome.magnet_lookup is magnet_lookup)

    def test_static_from_file_open_file_handle(self):
        """
        Tests the MagnetSortGenome class can be constructed from an open handle to a .magsortgenome file using the
        static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        with open(os.path.join(inputs_path, 'example.magsortgenome'), 'rb') as file_handle:
            # Construct MagnetSortGenome instance
            magnet_genome = MagnetSortGenome.from_file(file=file_handle, magnet_set=magnet_set,
                                                   magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_genome.set_count, count)
        self.assertEqual(magnet_genome.slot_count, count)
        self.assertEqual(magnet_genome.mtype, mtype)
        self.assertTrue(np.allclose(magnet_genome.permutation, permutation))
        self.assertTrue(np.allclose(magnet_genome.flips, flips))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_children, rng_states[0]))
        self.assertTrue(self.compare_rng_states(magnet_genome.rng_mutations, rng_states[1]))

        self.assertTrue(magnet_genome.magnet_set is magnet_set)
        self.assertTrue(magnet_genome.magnet_slots is magnet_slots)
        self.assertTrue(magnet_genome.magnet_lookup is magnet_lookup)

    def test_static_from_file_raises_on_bad_parameters(self):
        """
        Tests the MagnetSortGenome class raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        count, mtype, permutation, flips, rng_states, \
            magnet_set, magnet_slots, magnet_lookup = self.dummy_magnet_genome_values()

        # Attempt to load from to a bad file parameter
        self.assertRaisesRegex(optid.errors.FileHandleError, '.*', MagnetSortGenome.from_file, file=None,
                               magnet_set=magnet_set, magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)
