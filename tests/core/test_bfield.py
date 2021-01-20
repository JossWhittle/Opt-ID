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
import jax.numpy as jnp

# Test imports
import optid
from optid.core import bfield, affine

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class BfieldTest(unittest.TestCase):
    """
    Test bfield arithmetic functions.
    """

    def test_bfield_from_lookup(self):
        """
        Test that a lattice of 3x3 matrices can be applied to a common field vector.
        """

        lookup = jnp.array([
            [[affine.rotate_x(affine.radians(90))[:3, :3], affine.rotate_x(affine.radians(-90))[:3, :3]],
             [affine.rotate_z(affine.radians(90))[:3, :3], affine.rotate_z(affine.radians(-90))[:3, :3]]],
            [[affine.rotate_s(affine.radians(90))[:3, :3], affine.rotate_s(affine.radians(-90))[:3, :3]],
             [affine.rotate_z(affine.radians(90))[:3, :3], affine.rotate_z(affine.radians(-90))[:3, :3]]]
        ])

        self.assertTrue(np.allclose(bfield.bfield_from_lookup(lookup, jnp.array([1, 0, 0])),
                                    jnp.array([[[[ 1,  0,  0], [ 1,  0,  0]],
                                                [[ 0,  0,  1], [ 0,  0, -1]]],
                                               [[[ 0, -1,  0], [ 0,  1,  0]],
                                                [[ 0,  0,  1], [ 0,  0, -1]]]]), atol=1e-5))

    def test_bfield_from_interpolated_lookup(self):
        """
        Test that lookups can be interpolated.
        """

        lookup = jnp.array([
            [[[affine.rotate_x(affine.radians(90))[:3, :3], affine.rotate_x(affine.radians(-90))[:3, :3]],
              [affine.rotate_z(affine.radians(90))[:3, :3], affine.rotate_z(affine.radians(-90))[:3, :3]]],
             [[affine.rotate_s(affine.radians(90))[:3, :3], affine.rotate_s(affine.radians(-90))[:3, :3]],
              [affine.rotate_z(affine.radians(90))[:3, :3], affine.rotate_z(affine.radians(-90))[:3, :3]]]],
            [[[affine.scale(0, 0, 0)[:3, :3], affine.scale(0, 0, 0)[:3, :3]],
              [affine.scale(0, 0, 0)[:3, :3], affine.scale(0, 0, 0)[:3, :3]]],
             [[affine.scale(0, 0, 0)[:3, :3], affine.scale(0, 0, 0)[:3, :3]],
              [affine.scale(0, 0, 0)[:3, :3], affine.scale(0, 0, 0)[:3, :3]]]]
        ])

        self.assertTrue(np.allclose(
            bfield.bfield_from_interpolated_lookup(lookup, jnp.array([1, 0, 0]), jnp.array([0.0])),
            jnp.array([
                [[[ 1,  0,  0], [ 1,  0,  0]],
                 [[ 0,  0,  1], [ 0,  0, -1]]],
                [[[ 0, -1,  0], [ 0,  1,  0]],
                 [[ 0,  0,  1], [ 0,  0, -1]]]
            ]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            bfield.bfield_from_interpolated_lookup(lookup, jnp.array([1, 0, 0]), jnp.array([0.5])),
            jnp.array([
                [[[ 0.5,    0,    0], [ 0.5,    0,    0]],
                 [[   0,    0,  0.5], [   0,    0, -0.5]]],
                [[[   0, -0.5,    0], [   0,  0.5,    0]],
                 [[   0,    0,  0.5], [   0,    0, -0.5]]]
            ]),
            atol=1e-5))

        self.assertTrue(np.allclose(
            bfield.bfield_from_interpolated_lookup(lookup, jnp.array([1, 0, 0]), jnp.array([1.0])),
            jnp.array([
                [[[ 0,  0,  0], [ 0,  0,  0]],
                 [[ 0,  0,  0], [ 0,  0,  0]]],
                [[[ 0,  0,  0], [ 0,  0,  0]],
                 [[ 0,  0,  0], [ 0,  0,  0]]]
            ]),
            atol=1e-5))
