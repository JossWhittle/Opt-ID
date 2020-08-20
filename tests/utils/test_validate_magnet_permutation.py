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
import optid
from optid.utils import validate_magnet_permutation

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class ValidateMagnetPermutationTest(unittest.TestCase):
    """
    Tests the validate_magnet_permutation function can be imported and used correctly.
    """

    @staticmethod
    def dummy_magnet_permutation_values():
        """
        Creates a set of constant test values used for across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        count = 4
        magnet_permutation = np.stack([
            np.arange(count).astype(np.int32),
            (np.arange(count).astype(np.int32) % 2)
        ], axis=-1)
        return count, magnet_permutation

    def test_parameter_magnet_permutation(self):
        """
        Tests the validate_magnet_permutation function throws exceptions when called with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_permutation = self.dummy_magnet_permutation_values()

        validate_magnet_permutation(magnet_permutation=magnet_permutation)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_magnet_permutation,
                               magnet_permutation=magnet_permutation[..., np.newaxis])

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', validate_magnet_permutation,
                               magnet_permutation=None)

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', validate_magnet_permutation,
                               magnet_permutation=magnet_permutation.astype(np.float32))

    def test_duplicate_indices(self):
        """
        Tests the validate_magnet_permutation function throws exceptions when called with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_permutation = self.dummy_magnet_permutation_values()

        validate_magnet_permutation(magnet_permutation=magnet_permutation)

        self.assertRaisesRegex(optid.errors.ValidateMagnetPermutationDuplicateError, '.*', validate_magnet_permutation,
                               magnet_permutation=np.zeros((count, 2), dtype=np.int32))

    def test_boundary_indices(self):
        """
        Tests the validate_magnet_permutation function throws exceptions when called with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_permutation = self.dummy_magnet_permutation_values()

        validate_magnet_permutation(magnet_permutation=magnet_permutation)

        self.assertRaisesRegex(optid.errors.ValidateMagnetPermutationBoundaryError, '.*', validate_magnet_permutation,
                               magnet_permutation=np.stack([np.arange(count).astype(np.int32) - 1,
                                                            magnet_permutation[:, 1]], axis=-1))

        self.assertRaisesRegex(optid.errors.ValidateMagnetPermutationBoundaryError, '.*', validate_magnet_permutation,
                               magnet_permutation=np.stack([np.arange(count).astype(np.int32) + 1,
                                                            magnet_permutation[:, 1]], axis=-1))

    def test_flips(self):
        """
        Tests the validate_magnet_permutation function throws exceptions when called with incorrect parameters.
        """

        # Make dummy parameters
        count, magnet_permutation = self.dummy_magnet_permutation_values()

        validate_magnet_permutation(magnet_permutation=magnet_permutation)

        self.assertRaisesRegex(optid.errors.ValidateMagnetPermutationFlipError, '.*', validate_magnet_permutation,
                               magnet_permutation=np.stack([magnet_permutation[:, 0],
                                                            (np.arange(count).astype(np.int32) % 2) - 1], axis=-1))

        self.assertRaisesRegex(optid.errors.ValidateMagnetPermutationFlipError, '.*', validate_magnet_permutation,
                               magnet_permutation=np.stack([magnet_permutation[:, 0],
                                                            (np.arange(count).astype(np.int32) % 2) + 1], axis=-1))
