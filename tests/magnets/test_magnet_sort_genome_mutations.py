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
from optid.utils import Range, Grid, validate_tensor

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
        mtype = 'HH'

        # MagnetSet
        reference_size = np.array([10, 10, 2], dtype=np.float32)
        reference_field_vector = np.array([0, 0, 1], dtype=np.float32)
        flip_matrix = np.ones((3, 3), dtype=np.float32)
        names = [f'{index + 1:03d}' for index in range(count*2)]
        sizes = np.ones((count*2, 3)) * reference_size[np.newaxis, ...]
        field_vectors = np.random.uniform(size=(count*2, 3))

        magnet_set = MagnetSet(mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                               names=names, sizes=sizes, field_vectors=field_vectors)

        # MagnetSlots
        beams = [f'B{((index % 2) + 1):d}' for index in range(count)]
        slots = [f'S{(((index - (index % 2)) // 2) + 1):03d}' for index in range(count)]
        positions = np.zeros((count, 3), dtype=np.float32)
        gap_vectors = np.zeros((count, 3), dtype=np.float32)
        gap_vectors[:, 1] = 1.0
        direction_matrices = np.zeros((count, 3, 3), dtype=np.float32)
        direction_matrices[:, ...] = np.eye(3, dtype=np.float32)[np.newaxis, ...]

        magnet_slots = MagnetSlots(mtype=mtype, beams=beams, slots=slots, positions=positions,
                                   gap_vectors=gap_vectors, direction_matrices=direction_matrices)

        # MagnetSortLookup
        grid = Grid(x_range=Range(min=-1, max=1, steps=5),
                    z_range=Range(min=-1, max=1, steps=5),
                    s_range=Range(min=-1, max=1, steps=5))
        lookup = np.random.uniform(size=(count, *grid.steps, 3, 3))

        magnet_lookup = MagnetSortLookup(mtype=mtype, grid=grid, lookup=lookup)

        magnet_genome = MagnetSortGenome.from_random(seed=1234, magnet_set=magnet_set,
                                                     magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        return count, mtype, grid, magnet_set, magnet_slots, magnet_lookup, magnet_genome

    def test_flip(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, mtype, grid, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index = 1

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        flips_old = magnet_genome.flips.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.flip_mutation(index=index)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        flips_new = magnet_genome.flips.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
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

        count, mtype, grid, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        self.assertRaisesRegex(Exception, '.*', magnet_genome.flip_mutation, index=(count + 1))

    def test_random_flip(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, mtype, grid, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        flips_old = magnet_genome.flips.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.random_flip_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        flips_new = magnet_genome.flips.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
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

        count, mtype, grid, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index_a, index_b = 1, 3

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.exchange_mutation(index_a=index_a, index_b=index_b)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertEqual(2, np.sum(permutation_old != permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_bad_exchange(self):
        """
        Tests the MagnetSortGenome class throws exception when exchanging a bad pair of magnet slots.
        """

        count, mtype, grid, \
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

        count, mtype, grid, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.random_exchange_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertEqual(2, np.sum(permutation_old != permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_insertion(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, mtype, grid, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index_a, index_b = 0, 1

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.insertion_mutation(index_a=index_a, index_b=index_b)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertTrue(((index_b + 1) - index_a), np.sum(permutation_old != permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

        index_a, index_b = 1, (count + 1)

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.insertion_mutation(index_a=index_a, index_b=index_b)

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertTrue(((index_b + 1) - index_a), np.sum(permutation_old != permutation_new))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_bad_insertion(self):
        """
        Tests the MagnetSortGenome class throws exception when exchanging a bad pair of magnet slots.
        """

        count, mtype, grid, \
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

        count, mtype, grid, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutation
        magnet_genome.random_insertion_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

        self.assertTrue(np.sum(permutation_old != permutation_new) > 0)

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_random_mutations(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, mtype, grid, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        num_mutations = 16

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutations
        for _ in range(num_mutations):
            magnet_genome.random_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertEqual(sorted(set(permutation_old.tolist())), sorted(set(permutation_new.tolist())))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))

    def test_random_mutations_non_flippable(self):
        """
        Tests the MagnetSortGenome class calculate bfield deltas consistently after mutations.
        """

        count, mtype, grid, \
            _, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        # MagnetSet
        reference_size = np.array([10, 10, 2], dtype=np.float32)
        reference_field_vector = np.array([0, 0, 1], dtype=np.float32)
        flip_matrix = np.eye(3, dtype=np.float32)
        names = [f'{index + 1:03d}' for index in range(count * 2)]
        sizes = np.ones((count * 2, 3)) * reference_size[np.newaxis, ...]
        field_vectors = np.random.uniform(size=(count * 2, 3))

        magnet_set = MagnetSet(mtype=mtype, reference_size=reference_size,
                               reference_field_vector=reference_field_vector, flip_matrix=flip_matrix,
                               names=names, sizes=sizes, field_vectors=field_vectors)

        magnet_genome = MagnetSortGenome.from_random(seed=1234, magnet_set=magnet_set,
                                                     magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        self.assertTrue(np.all(~magnet_genome.flips))

        num_mutations = 16

        # Current genome state
        permutation_old = magnet_genome.permutation.copy()
        bfield_old = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_old, magnet_genome.calculate_bfield()))

        # Apply mutations
        for _ in range(num_mutations):
            magnet_genome.random_mutation()

        # New genome state
        permutation_new = magnet_genome.permutation.copy()
        bfield_new = magnet_genome.bfield.copy()

        validate_tensor(bfield_old, shape=(*grid.steps, 3), dtype=np.floating)
        self.assertTrue(np.allclose(bfield_new, magnet_genome.calculate_bfield()))
        self.assertEqual(sorted(set(permutation_old.tolist())), sorted(set(permutation_new.tolist())))
        self.assertTrue(np.all(~magnet_genome.flips))

        magnet_genome.recalculate_bfield()
        self.assertTrue(np.allclose(bfield_new, magnet_genome.bfield))
