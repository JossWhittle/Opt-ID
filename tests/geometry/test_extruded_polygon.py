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
from optid.geometry import ExtrudedPolygon

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class ExtrudedPolygonTest(unittest.TestCase):
    """
    Test ExtrudedPolygon class.
    """

    ####################################################################################################################

    def test_constructor_polygon_array(self):

        polygon = jnp.array([
            [-0.5, -0.5], [-0.5,  0.5], [0.5,  0.5], [0.5, -0.5]], dtype=jnp.float32)

        geometry = ExtrudedPolygon(polygon=polygon, thickness=1.0)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        polyhedra = [[
            [0, 1, 2, 3], [7, 6, 5, 4],
            [1, 5, 4, 0], [2, 6, 5, 1],
            [3, 7, 6, 2], [0, 4, 7, 3]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    def test_constructor_polygon_list(self):

        polygon = [
            [-0.5, -0.5], [-0.5,  0.5], [0.5,  0.5], [0.5, -0.5]]

        geometry = ExtrudedPolygon(polygon=polygon, thickness=1.0)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        polyhedra = [[
            [0, 1, 2, 3], [7, 6, 5, 4],
            [1, 5, 4, 0], [2, 6, 5, 1],
            [3, 7, 6, 2], [0, 4, 7, 3]]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.polyhedra, polyhedra)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_polygon_type_raises_exception(self):

        self.assertRaisesRegex(BeartypeException, '.*', ExtrudedPolygon,
                               polygon=None, thickness=1.0)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_thickness_type_raises_exception(self):

        polygon = [
            [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]

        self.assertRaisesRegex(BeartypeException, '.*', ExtrudedPolygon,
                               polygon=polygon, thickness=None)

    def test_constructor_bad_polygon_list_vertex_shape_raises_exception(self):

        polygon = [
            [-0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]

        self.assertRaisesRegex(ValueError, '.*', ExtrudedPolygon,
                               polygon=polygon, thickness=1.0)

    def test_constructor_bad_polygon_shape_raises_exception(self):

        polygon = jnp.ones((2, 2), dtype=jnp.float32)

        self.assertRaisesRegex(ValueError, '.*', ExtrudedPolygon,
                               polygon=polygon, thickness=1.0)

    def test_constructor_bad_polygon_array_vertices_shape_raises_exception(self):

        polygon = jnp.ones((4, 3), dtype=jnp.float32)

        self.assertRaisesRegex(ValueError, '.*', ExtrudedPolygon,
                               polygon=polygon, thickness=1.0)

    def test_constructor_bad_polygon_array_type_raises_exception(self):

        polygon = jnp.ones((4, 2), dtype=jnp.int32)

        self.assertRaisesRegex(TypeError, '.*', ExtrudedPolygon,
                               polygon=polygon, thickness=1.0)

    def test_constructor_bad_thickness_raises_exception(self):

        polygon = jnp.ones((4, 2), dtype=jnp.float32)

        self.assertRaisesRegex(TypeError, '.*', ExtrudedPolygon,
                               polygon=polygon, thickness=-1.0)
