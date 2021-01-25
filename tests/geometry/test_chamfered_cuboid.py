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
from optid.geometry import ChamferedCuboid

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class ChamferedCuboidTest(unittest.TestCase):
    """
    Test ChamferedCuboid class.
    """

    ####################################################################################################################

    def test_constructor_shape_array(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        geometry = ChamferedCuboid(shape=shape)

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

        geometry = ChamferedCuboid(shape=shape)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_shape_tuple(self):

        shape = (1, 1, 1)

        geometry = ChamferedCuboid(shape=shape)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_shape_type_raises_exception(self):

        self.assertRaisesRegex(BeartypeException, '.*', ChamferedCuboid,
                               shape=None)

    def test_constructor_bad_shape_shape_raises_exception(self):

        shape = jnp.ones((2,), dtype=jnp.float32)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape)

    def test_constructor_bad_shape_array_type_raises_exception(self):

        shape = jnp.ones((3,), dtype=jnp.int32)

        self.assertRaisesRegex(TypeError, '.*', ChamferedCuboid,
                               shape=shape)

    def test_constructor_bad_shape_raises_exception(self):

        shape = jnp.array([-1, 1, 1], dtype=jnp.float32)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape)

    def test_constructor_chamfer_array_scalar(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = jnp.zeros((), dtype=jnp.float32)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_array_xz(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = jnp.zeros((2,), dtype=jnp.float32)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_array_scalar_corners(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = jnp.zeros((4,), dtype=jnp.float32)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_array_xz_corners(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = jnp.zeros((4, 2), dtype=jnp.float32)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_scalar(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = 0

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_tuple_xz(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = (0, 0)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_tuple_scalar_corners(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = (0, 0, 0, 0)

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_tuple_xz_corners(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = ((0, 0), (0, 0), (0, 0), (0, 0))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5]], dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5],
            [2, 3, 7, 6], [3, 0, 4, 7]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_bl(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = ((0.2, 0.1), (0, 0), (0, 0), (0, 0))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.3, -0.5, -0.5], [-0.5, -0.4, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
            [-0.3, -0.5,  0.5], [-0.5, -0.4,  0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5]],
            dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3, 4], [5, 6, 7, 8, 9],
            [0, 1, 6, 5], [1, 2, 7, 6],
            [2, 3, 8, 7], [3, 4, 9, 8],
            [4, 0, 5, 9]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_tl(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = ((0, 0), (0.2, 0.1), (0, 0), (0, 0))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.4, -0.5], [-0.3,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.4,  0.5], [-0.3,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5]],
            dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3, 4], [5, 6, 7, 8, 9],
            [0, 1, 6, 5], [1, 2, 7, 6],
            [2, 3, 8, 7], [3, 4, 9, 8],
            [4, 0, 5, 9]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_tr(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = ((0, 0), (0, 0), (0.2, 0.1), (0, 0))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.3,  0.5, -0.5], [ 0.5,  0.4, -0.5], [ 0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [ 0.3,  0.5,  0.5], [ 0.5,  0.4,  0.5], [ 0.5, -0.5,  0.5]],
            dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3, 4], [5, 6, 7, 8, 9],
            [0, 1, 6, 5], [1, 2, 7, 6],
            [2, 3, 8, 7], [3, 4, 9, 8],
            [4, 0, 5, 9]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    def test_constructor_chamfer_br(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = ((0, 0), (0, 0), (0, 0), (0.2, 0.1))

        geometry = ChamferedCuboid(shape=shape, chamfer=chamfer)

        vertices = jnp.array([
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.4, -0.5], [ 0.3, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.4,  0.5], [ 0.3, -0.5,  0.5]],
            dtype=jnp.float32)

        faces = [
            [0, 1, 2, 3, 4], [5, 6, 7, 8, 9],
            [0, 1, 6, 5], [1, 2, 7, 6],
            [2, 3, 8, 7], [3, 4, 9, 8],
            [4, 0, 5, 9]]

        self.assertTrue(np.allclose(geometry.vertices, vertices, atol=1e-5))
        self.assertEqual(geometry.faces, faces)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_chamfer_type_raises_exception(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        self.assertRaisesRegex(BeartypeException, '.*', ChamferedCuboid,
                               shape=shape, chamfer=None)

    def test_constructor_bad_chamfer_negative_raises_exception(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=-0.1)

    def test_constructor_bad_chamfer_shape_raises_exception(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = jnp.zeros((4, 3), dtype=jnp.float32)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_imbalanced_raises_exception(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = (0.1, 0)

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_left_collision_raises_exception(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = ((0.6, 0.6), (0.6, 0.6), (0, 0), (0, 0))

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_top_collision_raises_exception(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = ((0, 0), (0.6, 0.6), (0.6, 0.6), (0, 0))

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_right_collision_raises_exception(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = ((0, 0), (0, 0), (0.6, 0.6), (0.6, 0.6))

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)

    def test_constructor_bad_chamfer_bottom_collision_raises_exception(self):

        shape = jnp.array([1, 1, 1], dtype=jnp.float32)

        chamfer = ((0.6, 0.6), (0, 0), (0, 0), (0.6, 0.6))

        self.assertRaisesRegex(ValueError, '.*', ChamferedCuboid,
                               shape=shape, chamfer=chamfer)
