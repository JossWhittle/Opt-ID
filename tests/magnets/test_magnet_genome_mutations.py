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
from optid.magnets import MagnetGenome, MagnetSet, MagnetSlots, MagnetLookup
from optid.utils import Range, validate_tensor

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class MagnetGenomeMutationTest(unittest.TestCase):
    """
    Tests the MagnetGenome class can be imported and used correctly.
    """

    @staticmethod
    def dummy_values():
        """
        Creates a set of constant test values used for constructing and comparing MagnetGenome
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        count = 4
        magnet_type = 'HH'

        # MagnetSet
        names = [f'{index + 1:03d}' for index in range(count)]
        field_vectors = np.array([
            [0.003770334, -0.000352049, 1.339567917],
            [-0.007018214, -0.002714164, 1.344710227],
            [-0.004826321, -0.001714764, 1.342079598],
            [0.008846784, -0.003088993, 1.344698631],
        ], dtype=np.float32)

        magnet_set = MagnetSet(magnet_type=magnet_type, names=names, field_vectors=field_vectors)

        # MagnetSlots
        beams = [f'B{((index % 2) + 1):d}' for index in range(count)]
        slots = [f'S{(((index - (index % 2)) // 2) + 1):03d}' for index in range(count)]
        flip_matrix = np.ones((3, 3), dtype=np.float32)
        flippable = True

        magnet_slots = MagnetSlots(magnet_type=magnet_type, beams=beams, slots=slots, flip_matrix=flip_matrix)

        # MagnetLookup
        x_range = Range(-1, 1, 5)
        z_range = Range(-1, 1, 5)
        s_range = Range(-1, 1, 5)
        lookup = np.random.uniform(size=(count, x_range.steps, z_range.steps, s_range.steps, 3, 3))

        magnet_lookup = MagnetLookup(magnet_type=magnet_type, x_range=x_range,
                                     z_range=z_range, s_range=s_range, lookup=lookup)

        magnet_genome = MagnetGenome.from_random(seed=1234, magnet_set=magnet_set,
                                                 magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        return count, magnet_type, names, field_vectors, \
               beams, slots, flip_matrix, flippable, \
               x_range, z_range, s_range, lookup, \
               magnet_set, magnet_slots, magnet_lookup, magnet_genome

    def test_flip(self):
        """
        Tests the MagnetGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index = 1

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        flip_old = magnet_genome.flips.copy()[index]
        bfield_old = magnet_genome.calculate_bfield()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # Apply mutation
        bfield_delta = magnet_genome.flip_mutation(index=index)

        validate_tensor(bfield_delta, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        flip_new = magnet_genome.flips.copy()[index]
        bfield_new = magnet_genome.calculate_bfield()

        validate_tensor(bfield_new, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        self.assertTrue((not flip_old) == flip_new)
        self.assertTrue(np.all(permutation_old == permutation_new))
        self.assertTrue(np.allclose((bfield_old + bfield_delta), bfield_new))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

    def test_random_flip(self):
        """
        Tests the MagnetGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        flips_old = magnet_genome.flips.copy()
        bfield_old = magnet_genome.calculate_bfield()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # Apply mutation
        bfield_delta = magnet_genome.random_flip_mutation()

        validate_tensor(bfield_delta, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        flips_new = magnet_genome.flips.copy()
        bfield_new = magnet_genome.calculate_bfield()

        validate_tensor(bfield_new, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        self.assertEqual(1, np.sum(flips_old != flips_new))
        self.assertTrue(np.all(permutation_old == permutation_new))
        self.assertTrue(np.allclose((bfield_old + bfield_delta), bfield_new))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

    def test_exchange(self):
        """
        Tests the MagnetGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index_a, index_b = 1, 3

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()[[index_a, index_b]]
        flips_old = magnet_genome.flips.copy()[[index_a, index_b]]
        bfield_old = magnet_genome.calculate_bfield()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # Apply mutation
        bfield_delta = magnet_genome.exchange_mutation(index_a=index_a, index_b=index_b)

        validate_tensor(bfield_delta, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()[[index_a, index_b]]
        flips_new = magnet_genome.flips.copy()[[index_a, index_b]]
        bfield_new = magnet_genome.calculate_bfield()

        validate_tensor(bfield_new, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        self.assertTrue(np.allclose(permutation_old, permutation_new[::-1]))
        self.assertTrue(np.allclose(flips_old, flips_new[::-1]))
        self.assertTrue(np.allclose((bfield_old + bfield_delta), bfield_new))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

    def test_random_exchange(self):
        """
        Tests the MagnetGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        flips_old = magnet_genome.flips.copy()
        bfield_old = magnet_genome.calculate_bfield()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # Apply mutation
        bfield_delta = magnet_genome.random_exchange_mutation()

        validate_tensor(bfield_delta, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        flips_new = magnet_genome.flips.copy()
        bfield_new = magnet_genome.calculate_bfield()

        validate_tensor(bfield_new, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        self.assertEqual(2, np.sum(flips_old != flips_new))
        self.assertEqual(2, np.sum(permutation_old != permutation_new))
        self.assertTrue(np.allclose((bfield_old + bfield_delta), bfield_new))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

    def test_random_mutations(self):
        """
        Tests the MagnetGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        num_mutations = 16

        # Current genome state
        bfield_old = magnet_genome.calculate_bfield()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # Apply mutation
        bfield_delta = sum(magnet_genome.random_mutation() for _ in range(num_mutations))

        validate_tensor(bfield_delta, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # New genome state
        bfield_new = magnet_genome.calculate_bfield()

        validate_tensor(bfield_new, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        self.assertTrue(np.allclose((bfield_old + bfield_delta), bfield_new))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

    def test_random_mutations_non_flippable(self):
        """
        Tests the MagnetGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, _, magnet_lookup, _ = self.dummy_values()

        # MagnetSlots
        beams = [f'B{((index % 2) + 1):d}' for index in range(count)]
        slots = [f'S{(((index - (index % 2)) // 2) + 1):03d}' for index in range(count)]
        flip_matrix = np.eye(3, dtype=np.float32)

        magnet_slots = MagnetSlots(magnet_type=magnet_type, beams=beams, slots=slots, flip_matrix=flip_matrix)

        magnet_genome = MagnetGenome.from_random(seed=1234, magnet_set=magnet_set,
                                                 magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        self.assertTrue(np.all(~magnet_genome.flips))

        num_mutations = 16

        # Current genome state
        bfield_old = magnet_genome.calculate_bfield()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # Apply mutation
        bfield_delta = sum(magnet_genome.random_mutation() for _ in range(num_mutations))

        validate_tensor(bfield_delta, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        # New genome state
        bfield_new = magnet_genome.calculate_bfield()

        validate_tensor(bfield_new, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        self.assertTrue(np.allclose((bfield_old + bfield_delta), bfield_new))
        self.assertFalse(np.allclose(bfield_old, bfield_new))
        self.assertTrue(np.all(~magnet_genome.flips))
