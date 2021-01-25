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
from beartype.roar import BeartypeException
import sys
import unittest
import numpy as np
import jax.numpy as jnp

# Test imports
import optid
from optid.geometry import Cuboid

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class CuboidTest(unittest.TestCase):
    """
    Test Cuboid class.
    """

    ####################################################################################################################

    def test_constructor_shape_array(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        geometry = Cuboid(shape=shape)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_shape_list(self):

        shape = [1, 1, 1]

        geometry = Cuboid(shape=shape)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, jnp.array(vertices, dtype=jnp.float32), atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_shape_tuple(self):

        shape = (1, 1, 1)

        geometry = Cuboid(shape=shape)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, jnp.array(vertices, dtype=jnp.float32), atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_shape_type_raises_exception(self):

        self.assertRaisesRegex(BeartypeException, '.*', Cuboid,
                               shape=None)

    def test_constructor_bad_shape_shape_raises_exception(self):

        shape = jnp.ones((2,), dtype=jnp.float32)

        self.assertRaisesRegex(ValueError, '.*', Cuboid,
                               shape=shape)

    def test_constructor_bad_polygon_array_type_raises_exception(self):

        shape = jnp.ones((3,), dtype=jnp.int32)

        self.assertRaisesRegex(TypeError, '.*', Cuboid,
                               shape=shape)

    def test_constructor_bad_shape_raises_exception(self):

        shape = jnp.array([-1, 1, 1], dtype=jnp.float32)

        self.assertRaisesRegex(ValueError, '.*', Cuboid,
                               shape=shape)
