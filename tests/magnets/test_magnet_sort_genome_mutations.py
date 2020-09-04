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
from optid.magnets import MagnetSortGenome, MagnetSet, MagnetSlots, MagnetSortLookup
from optid.utils import Range, validate_tensor

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class MagnetSortGenomeMutationTest(unittest.TestCase):
    """
    Tests the MagnetSortGenome class can be imported and used correctly.
    """

    @staticmethod
    def dummy_values():
        """
        Creates a set of constant test values used for constructing and comparing MagnetSortGenome
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        count = 8
        magnet_type = 'HH'

        # MagnetSet
        names = [f'{index + 1:03d}' for index in range(count*2)]
        field_vectors = np.random.uniform(size=(count*2, 3))

        magnet_set = MagnetSet(magnet_type=magnet_type, names=names, field_vectors=field_vectors)

        # MagnetSlots
        beams = [f'B{((index % 2) + 1):d}' for index in range(count)]
        slots = [f'S{(((index - (index % 2)) // 2) + 1):03d}' for index in range(count)]
        positions = np.zeros((count, 3), dtype=np.float32)
        direction_matrices = np.zeros((count, 3, 3), dtype=np.float32)
        direction_matrices[:, ...] = np.eye(3, dtype=np.float32)[np.newaxis, ...]
        size = np.ones((3,), dtype=np.float32)
        flip_matrix = np.ones((3, 3), dtype=np.float32)
        flippable = True

        magnet_slots = MagnetSlots(magnet_type=magnet_type, beams=beams, slots=slots, positions=positions,
                                   direction_matrices=direction_matrices, size=size, flip_matrix=flip_matrix)

        # MagnetSortLookup
        x_range = Range(-1, 1, 5)
        z_range = Range(-1, 1, 5)
        s_range = Range(-1, 1, 5)
        lookup = np.random.uniform(size=(count, x_range.steps, z_range.steps, s_range.steps, 3, 3))

        magnet_lookup = MagnetSortLookup(magnet_type=magnet_type, x_range=x_range,
                                         z_range=z_range, s_range=s_range, lookup=lookup)

        magnet_genome = MagnetSortGenome.from_random(seed=1234, magnet_set=magnet_set,
                                                     magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        return count, magnet_type, names, field_vectors, \
               beams, slots, flip_matrix, flippable, \
               x_range, z_range, s_range, lookup, \
               magnet_set, magnet_slots, magnet_lookup, magnet_genome

    def test_flip(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index = 1

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        flips_old = magnet_genome.flips.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.flip_mutation(index=index)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        flips_new = magnet_genome.flips.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertEqual(1, np.sum(flips_old != flips_new))
        self.assertTrue(np.all(permutation_old == permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_bad_flip(self):
        """
        Tests the MagnetSortGenome class throws exception when flipping a magnet in an unused slot.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        self.assertRaisesRegex(Exception, '.*', magnet_genome.flip_mutation, index=(count + 1))

    def test_random_flip(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        flips_old = magnet_genome.flips.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.random_flip_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        flips_new = magnet_genome.flips.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertEqual(1, np.sum(flips_old != flips_new))
        self.assertTrue(np.all(permutation_old == permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_exchange(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index_a, index_b = 1, 3

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.exchange_mutation(index_a=index_a, index_b=index_b)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertEqual(2, np.sum(permutation_old != permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_bad_exchange(self):
        """
        Tests the MagnetSortGenome class throws exception when exchanging a bad pair of magnet slots.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        magnet_genome.exchange_mutation(index_a=1, index_b=2)
        magnet_genome.exchange_mutation(index_a=2, index_b=1)
        magnet_genome.exchange_mutation(index_a=1, index_b=(count + 1))
        magnet_genome.exchange_mutation(index_a=(count + 1), index_b=1)

        self.assertRaisesRegex(Exception, '.*', magnet_genome.exchange_mutation,
                               index_a=1, index_b=1)

        self.assertRaisesRegex(Exception, '.*', magnet_genome.exchange_mutation,
                               index_a=(count + 1), index_b=(count + 2))

    def test_random_exchange(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.random_exchange_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertEqual(2, np.sum(permutation_old != permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_insertion(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index_a, index_b = 0, 1

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.insertion_mutation(index_a=index_a, index_b=index_b)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertTrue(((index_b + 1) - index_a), np.sum(permutation_old != permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

        index_a, index_b = 1, (count + 1)

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.insertion_mutation(index_a=index_a, index_b=index_b)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertTrue(((index_b + 1) - index_a), np.sum(permutation_old != permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_bad_insertion(self):
        """
        Tests the MagnetSortGenome class throws exception when exchanging a bad pair of magnet slots.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        magnet_genome.insertion_mutation(index_a=1, index_b=2)
        magnet_genome.insertion_mutation(index_a=1, index_b=(count + 1))

        self.assertRaisesRegex(Exception, '.*', magnet_genome.insertion_mutation,
                               index_a=1, index_b=1)

        self.assertRaisesRegex(Exception, '.*', magnet_genome.insertion_mutation,
                               index_a=2, index_b=1)

        self.assertRaisesRegex(Exception, '.*', magnet_genome.insertion_mutation,
                               index_a=(count + 1), index_b=(count + 2))

    def test_random_insertion(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.random_insertion_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertTrue(np.sum(permutation_old != permutation_new) > 0)

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_random_mutations(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        num_mutations = 16

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutations
        for _ in range(num_mutations):
            magnet_genome.random_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertEqual(sorted(set(permutation_old.tolist())), sorted(set(permutation_new.tolist())))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_random_mutations_non_flippable(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            beams, slots, flip_matrix, flippable, \
            x_range, z_range, s_range, lookup, \
            magnet_set, _, magnet_lookup, _ = self.dummy_values()

        # MagnetSlots
        beams = [f'B{((index % 2) + 1):d}' for index in range(count)]
        slots = [f'S{(((index - (index % 2)) // 2) + 1):03d}' for index in range(count)]
        positions = np.zeros((count, 3), dtype=np.float32)
        direction_matrices = np.zeros((count, 3, 3), dtype=np.float32)
        direction_matrices[:, ...] = np.eye(3, dtype=np.float32)[np.newaxis, ...]
        size = np.ones((3,), dtype=np.float32)
        flip_matrix = np.eye(3, dtype=np.float32)

        magnet_slots = MagnetSlots(magnet_type=magnet_type, beams=beams, slots=slots, positions=positions,
                                   direction_matrices=direction_matrices, size=size, flip_matrix=flip_matrix)

        magnet_genome = MagnetSortGenome.from_random(seed=1234, magnet_set=magnet_set,
                                                     magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        self.assertTrue(np.all(~magnet_genome.flips))

        num_mutations = 16

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutations
        for _ in range(num_mutations):
            magnet_genome.random_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertEqual(sorted(set(permutation_old.tolist())), sorted(set(permutation_new.tolist())))
        self.assertTrue(np.all(~magnet_genome.flips))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))
