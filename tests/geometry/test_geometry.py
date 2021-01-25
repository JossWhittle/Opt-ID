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
from optid.geometry import Geometry
from optid.core.affine import scale

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class GeometryTest(unittest.TestCase):
    """
    Test Geometry class.
    """

    ####################################################################################################################

    def test_constructor_vertices_array(self):

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        geometry = Geometry(vertices=vertices, faces=faces)

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_vertices_list(self):

        vertices = [
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]]

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        geometry = Geometry(vertices=vertices, faces=faces)

        self.assertTrue(np.allclose(geometry.vertices, jnp.array(vertices, dtype=jnp.float32), atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_vertices_raises_exception(self):

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertRaisesRegex(BeartypeException, '.*', Geometry,
                               vertices=None, faces=faces)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_faces_raises_exception(self):

        vertices = [
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]]

        self.assertRaisesRegex(BeartypeException, '.*', Geometry,
                               vertices=vertices, faces=None)

    def test_constructor_bad_vertices_list_vertex_shape_raises_exception(self):

        vertices = [
            [-0.5, -0.5      ], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]]

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_vertices_array_vertices_shape_raises_exception(self):

        vertices = jnp.ones((8, 2), dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_vertices_array_type_raises_exception(self):

        vertices = jnp.ones((8, 3), dtype=jnp.int32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertRaisesRegex(TypeError, '.*', Geometry,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_faces_vertex_out_of_bounds_raises_exception(self):

        vertices = jnp.ones((8, 3), dtype=jnp.float32)

        faces = [
            [8, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_faces_vertex_duplicated_raises_exception(self):

        vertices = jnp.ones((8, 3), dtype=jnp.float32)

        faces = [
            [0, 0, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, faces=faces)

    def test_constructor_bad_faces_face_not_polygon_raises_exception(self):

        vertices = jnp.ones((8, 3), dtype=jnp.float32)

        faces = [
            [0, 1], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertRaisesRegex(ValueError, '.*', Geometry,
                               vertices=vertices, faces=faces)

    ####################################################################################################################

    def test_transform(self):

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        geometry = Geometry(vertices=vertices, faces=faces)

        self.assertTrue(np.allclose(geometry.transform(scale(2, 2, 2)), (vertices * 2.0), atol=1e-5))

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_transform_bad_matrix_type_raises_exception(self):

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        geometry = Geometry(vertices=vertices, faces=faces)

        self.assertRaisesRegex(BeartypeException, '.*', geometry.transform,
                               matrix=None)

    def test_transform_bad_matrix_shape_raises_exception(self):

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        geometry = Geometry(vertices=vertices, faces=faces)

        self.assertRaisesRegex(ValueError, '.*', geometry.transform,
                               matrix=jnp.eye(3, dtype=jnp.float32))

    def test_transform_bad_matrix_array_type_raises_exception(self):

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        geometry = Geometry(vertices=vertices, faces=faces)

        self.assertRaisesRegex(TypeError, '.*', geometry.transform,
                               matrix=jnp.eye(4, dtype=jnp.int32))
