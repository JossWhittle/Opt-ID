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
    def dummy_cutouts_values():
        """
        Creates a set of constant test values used for constructing and comparing MagnetSlots
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        size = np.array([10, 10, 2], dtype=np.float32)

        # Cutout the 2mm cubed region in the bottom left corner and top right corner
        cutouts = np.array([
            [[0, 0, 0], [2, 2, 2]],
            [[8, 8, 0], [10, 10, 2]]
        ], dtype=np.float32)

        return size, cutouts

    def test_parameter_cutouts(self):
        """
        Tests the validate_magnet_cutouts function throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        size, cutouts = self.dummy_cutouts_values()

        validate_magnet_cutouts(cutouts=cutouts, size=size)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_magnet_cutouts,
                               cutouts=cutouts[..., 1:], size=size)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_magnet_cutouts,
                               cutouts=cutouts[0], size=size)

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', validate_magnet_cutouts,
                               cutouts=cutouts.astype(np.int32), size=size)

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', validate_magnet_cutouts,
                               cutouts=None, size=size)

    def test_parameter_size(self):
        """
        Tests the validate_magnet_cutouts function throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        size, cutouts = self.dummy_cutouts_values()

        validate_magnet_cutouts(cutouts=cutouts, size=size)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_magnet_cutouts,
                               cutouts=cutouts, size=np.array([10, 10, 2, 1], dtype=np.float32))

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_magnet_cutouts,
                               cutouts=cutouts, size=np.array([[10, 10, 2]], dtype=np.float32))

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', validate_magnet_cutouts,
                               cutouts=cutouts, size=size.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', validate_magnet_cutouts,
                               cutouts=cutouts, size=None)

    def test_size(self):
        """
        Tests the validate_magnet_cutouts function throws exceptions when cutouts extend outside the magnet size.
        """

        # Make dummy parameters
        size, cutouts = self.dummy_cutouts_values()

        validate_magnet_cutouts(cutouts=cutouts, size=size)

        # Assert for cutout that has zero size in an axis
        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsSizeError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[0, 0, 0], [0, 2, 2]]], dtype=np.float32),
                               size=size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsSizeError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[0, 0, 0], [2, 0, 2]]], dtype=np.float32),
                               size=size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsSizeError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[0, 0, 0], [2, 2, 0]]], dtype=np.float32),
                               size=size)

        # Assert for cutout that has negative size in an axis
        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsSizeError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[2, 0, 0], [0, 2, 2]]], dtype=np.float32),
                               size=size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsSizeError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[0, 2, 0], [2, 0, 2]]], dtype=np.float32),
                               size=size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsSizeError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[0, 0, 2], [2, 2, 0]]], dtype=np.float32),
                               size=size)

    def test_cutout_boundary(self):
        """
        Tests the validate_magnet_cutouts function throws exceptions when cutouts extend outside the magnet size.
        """

        # Make dummy parameters
        size, cutouts = self.dummy_cutouts_values()

        validate_magnet_cutouts(cutouts=cutouts, size=size)

        # Assert for cutout that goes past the bottom left near corner
        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsBoundaryError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[-1, 0, 0], [1, 2, 2]]], dtype=np.float32),
                               size=size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsBoundaryError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[0, -1, 0], [2, 1, 2]]], dtype=np.float32),
                               size=size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsBoundaryError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[0, 0, -1], [2, 2, 1]]], dtype=np.float32),
                               size=size)

        # Assert for cutout that goes past the top right far corner
        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsBoundaryError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[9, 8, 0], [11, 10, 2]]], dtype=np.float32),
                               size=size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsBoundaryError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[8, 9, 0], [10, 11, 2]]], dtype=np.float32),
                               size=size)

        self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsBoundaryError, '.*', validate_magnet_cutouts,
                               cutouts=np.array([[[8, 8, 1], [10, 10, 3]]], dtype=np.float32),
                               size=size)

    def test_cutout_overlap(self):
        """
        Tests the validate_magnet_cutouts function throws exceptions when cutouts extend outside the magnet size.
        """

        # Make dummy parameters
        size = np.array([10, 10, 10], dtype=np.float32)

        # 1x1x1 cutout in the centre
        cutout = np.array([[4.5, 4.5, 4.5], [5.5, 5.5, 5.5]], dtype=np.float32)

        valid_offsets = np.array([
            [[-1, 0, 0]], [[+1, 0, 0]], [[0, -1, 0]], [[0, +1, 0]], [[0, 0, -1]], [[0, 0, +1]],

            [[-1, -0.5, +0.5]], [[-1, +0.5, +0.5]], [[-1, -0.5, -0.5]], [[-1, +0.5, -0.5]],
            [[+1, -0.5, +0.5]], [[+1, +0.5, +0.5]], [[+1, -0.5, -0.5]], [[+1, +0.5, -0.5]],

            [[-0.5, -1, +0.5]], [[+0.5, -1, +0.5]], [[-0.5, -1, -0.5]], [[+0.5, -1, -0.5]],
            [[-0.5, +1, +0.5]], [[+0.5, +1, +0.5]], [[-0.5, +1, -0.5]], [[+0.5, +1, -0.5]],

            [[-0.5, +0.5, -1]], [[+0.5, +0.5, -1]], [[-0.5, -0.5, -1]], [[+0.5, -0.5, -1]],
            [[-0.5, +0.5, +1]], [[+0.5, +0.5, +1]], [[-0.5, -0.5, +1]], [[+0.5, -0.5, +1]],
        ], dtype=np.float32)

        for offset in valid_offsets:
            validate_magnet_cutouts(cutouts=np.stack([cutout, (cutout + offset)], axis=0),
                                    size=size)

        invalid_offsets = np.array([
            [[-0.99, 0, 0]], [[+0.99, 0, 0]], [[0, -0.99, 0]], [[0, +0.99, 0]], [[0, 0, -0.99]], [[0, 0, +0.99]],

            [[-0.99, -0.5, +0.5]], [[-0.99, +0.5, +0.5]], [[-0.99, -0.5, -0.5]], [[-0.99, +0.5, -0.5]],
            [[+0.99, -0.5, +0.5]], [[+0.99, +0.5, +0.5]], [[+0.99, -0.5, -0.5]], [[+0.99, +0.5, -0.5]],

            [[-0.5, -0.99, +0.5]], [[+0.5, -0.99, +0.5]], [[-0.5, -0.99, -0.5]], [[+0.5, -0.99, -0.5]],
            [[-0.5, +0.99, +0.5]], [[+0.5, +0.99, +0.5]], [[-0.5, +0.99, -0.5]], [[+0.5, +0.99, -0.5]],

            [[-0.5, +0.5, -0.99]], [[+0.5, +0.5, -0.99]], [[-0.5, -0.5, -0.99]], [[+0.5, -0.5, -0.99]],
            [[-0.5, +0.5, +0.99]], [[+0.5, +0.5, +0.99]], [[-0.5, -0.5, +0.99]], [[+0.5, -0.5, +0.99]],
        ], dtype=np.float32)

        for offset in invalid_offsets:
            self.assertRaisesRegex(optid.errors.ValidateMagnetCutoutsOverlapError, '.*', validate_magnet_cutouts,
                                   cutouts=np.stack([cutout, (cutout + offset)], axis=0),
                                   size=size)


