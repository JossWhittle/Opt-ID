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
from optid.utils import validate_magnet_cutouts

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class ValidateMagnetCutoutsTest(unittest.TestCase):
    """
    Tests the validate_magnet_cutouts function can be imported and used correctly.
    """

    @staticmethod
    def dummy_magnet_cutouts_values():
        """
        Creates a set of constant test values used for constructing and comparing MagnetSlots
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        magnet_size = np.array([10, 10, 2], dtype=np.float32)

        # Cutout the 2mm cubed region in the bottom left corner and top right corner
        magnet_cutouts = np.array([
            [[0, 0, 0], [2, 2, 2]],
            [[8, 8, 0], [2, 2, 2]]
        ], dtype=np.float32)

        return magnet_size, magnet_cutouts

    def test_parameter_magnet_cutouts(self):
        """
        Tests the validate_magnet_cutouts function throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        magnet_size, magnet_cutouts = self.dummy_magnet_cutouts_values()

        validate_magnet_cutouts(magnet_cutouts=magnet_cutouts, magnet_size=magnet_size)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=magnet_cutouts[..., 1:], magnet_size=magnet_size)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=magnet_cutouts[0], magnet_size=magnet_size)

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=magnet_cutouts.astype(np.int32), magnet_size=magnet_size)

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=None, magnet_size=magnet_size)

    def test_parameter_magnet_size(self):
        """
        Tests the validate_magnet_cutouts function throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        magnet_size, magnet_cutouts = self.dummy_magnet_cutouts_values()

        validate_magnet_cutouts(magnet_cutouts=magnet_cutouts, magnet_size=magnet_size)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=magnet_cutouts, magnet_size=np.array([10, 10, 2, 1], dtype=np.float32))

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=magnet_cutouts, magnet_size=np.array([[10, 10, 2]], dtype=np.float32))

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=magnet_cutouts, magnet_size=magnet_size.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=magnet_cutouts, magnet_size=None)

    def test_overlap(self):
        """
        Tests the validate_magnet_cutouts function throws exceptions when cutouts extend outside the magnet size.
        """

        # Make dummy parameters
        magnet_size, magnet_cutouts = self.dummy_magnet_cutouts_values()

        validate_magnet_cutouts(magnet_cutouts=magnet_cutouts, magnet_size=magnet_size)

        # Assert for cutout that goes past the bottom left near corner
        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsOverlapError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=np.array([[[-1, 0, 0], [2, 2, 2]]], dtype=np.float32),
                               magnet_size=magnet_size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsOverlapError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=np.array([[[0, -1, 0], [2, 2, 2]]], dtype=np.float32),
                               magnet_size=magnet_size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsOverlapError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=np.array([[[0, 0, -1], [2, 2, 2]]], dtype=np.float32),
                               magnet_size=magnet_size)

        # Assert for cutout that goes past the top right far corner
        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsOverlapError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=np.array([[[9, 8, 0], [2, 2, 2]]], dtype=np.float32),
                               magnet_size=magnet_size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsOverlapError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=np.array([[[8, 9, 0], [2, 2, 2]]], dtype=np.float32),
                               magnet_size=magnet_size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsOverlapError, '.*', validate_magnet_cutouts,
                               magnet_cutouts=np.array([[[8, 8, 1], [2, 2, 2]]], dtype=np.float32),
                               magnet_size=magnet_size)
