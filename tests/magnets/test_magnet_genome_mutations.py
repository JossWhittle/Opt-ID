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


class MagnetGenomeTest(unittest.TestCase):
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

        # MagnetSlots
        size = np.array([10, 10, 2], dtype=np.float32)
        cutouts = np.array([
            [[0, 0, 0], [2, 2, 2]],
            [[8, 8, 0], [10, 10, 2]]
        ], dtype=np.float32)
        beams = [f'Q1' for index in range(count)]
        positions = []
        x_pos, z_pos, s_pos = (size[0] * -0.5), (-size[1]), 0
        for index in range(count):
            positions += [np.array([x_pos, z_pos, s_pos], dtype=np.float32)]
            s_pos += size[2]
        positions = np.stack(positions, axis=0)
        direction_matrices = np.repeat(np.eye(3, dtype=np.float32)[np.newaxis, ...], axis=0, repeats=count)
        flip_vectors = (np.arange(count * 3) % 2)
        flip_vectors[flip_vectors == 0] = -1
        flip_vectors = flip_vectors.reshape((count, 3)).astype(np.float32)

        # MagnetLookup
        x_range = Range(-1, 1, 5)
        z_range = Range(-1, 1, 5)
        s_range = Range(-1, 1, 5)
        lookup = np.random.uniform(size=(count, x_range[2], z_range[2], s_range[2], 3, 3))

        magnet_set = MagnetSet(magnet_type=magnet_type, names=names, field_vectors=field_vectors)

        magnet_slots = MagnetSlots(magnet_type=magnet_type, size=size, cutouts=cutouts,
                                   beams=beams, positions=positions,
                                   direction_matrices=direction_matrices,
                                   flip_vectors=flip_vectors)

        magnet_lookup = MagnetLookup(magnet_type=magnet_type, x_range=x_range,
                                     z_range=z_range, s_range=s_range, lookup=lookup)

        magnet_genome = MagnetGenome.from_magnet_set(magnet_set=magnet_set, seed=1234)

        return count, magnet_type, names, field_vectors, \
               size, cutouts, beams, positions, direction_matrices, flip_vectors, \
               x_range, z_range, s_range, lookup, \
               magnet_set, magnet_slots, magnet_lookup, magnet_genome

    def test_flip(self):
        """
        Tests the MagnetGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            size, cutouts, beams, positions, direction_matrices, flip_vectors, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index = 1
        flip_old = magnet_genome.flips[index]
        bfield_old = magnet_genome.calculate_bfield(magnet_set=magnet_set, magnet_slots=magnet_slots,
                                                    magnet_lookup=magnet_lookup)

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        bfield_delta = magnet_genome.flip(index, magnet_set=magnet_set, magnet_slots=magnet_slots,
                                          magnet_lookup=magnet_lookup)

        validate_tensor(bfield_delta, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        flip_new = magnet_genome.flips[index]
        bfield_new = magnet_genome.calculate_bfield(magnet_set=magnet_set, magnet_slots=magnet_slots,
                                                    magnet_lookup=magnet_lookup)

        validate_tensor(bfield_new, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        self.assertTrue((not flip_old) == flip_new)
        self.assertTrue(np.allclose((bfield_old + bfield_delta), bfield_new))
        self.assertFalse(np.allclose(bfield_old, bfield_new))

    def test_swap(self):
        """
        Tests the MagnetGenome class calculate bfield deltas consistently after mutations.
        """

        count, magnet_type, names, field_vectors, \
            size, cutouts, beams, positions, direction_matrices, flip_vectors, \
            x_range, z_range, s_range, lookup, \
            magnet_set, magnet_slots, magnet_lookup, magnet_genome = self.dummy_values()

        index_a, index_b = 1, 3
        permutation_old = magnet_genome.permutation[[index_a, index_b]]
        flips_old = magnet_genome.flips[[index_a, index_b]]
        bfield_old = magnet_genome.calculate_bfield(magnet_set=magnet_set, magnet_slots=magnet_slots,
                                                    magnet_lookup=magnet_lookup)

        validate_tensor(bfield_old, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        bfield_delta = magnet_genome.swap(index_a, index_b, magnet_set=magnet_set, magnet_slots=magnet_slots,
                                          magnet_lookup=magnet_lookup)

        validate_tensor(bfield_delta, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        permutation_new = magnet_genome.permutation[[index_a, index_b]]
        flips_new = magnet_genome.flips[[index_a, index_b]]
        bfield_new = magnet_genome.calculate_bfield(magnet_set=magnet_set, magnet_slots=magnet_slots,
                                                    magnet_lookup=magnet_lookup)

        validate_tensor(bfield_new, shape=(x_range.steps, z_range.steps, s_range.steps, 3), dtype=np.floating)

        self.assertTrue(np.allclose(permutation_old, permutation_new[::-1]))
        self.assertTrue(np.allclose(flips_old, flips_new[::-1]))
        self.assertTrue(np.allclose((bfield_old + bfield_delta), bfield_new))
        self.assertFalse(np.allclose(bfield_old, bfield_new))
