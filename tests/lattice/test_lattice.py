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
import sys
from beartype.roar import BeartypeException
import unittest
import numpy as np
import jax.numpy as jnp

# Test imports
import optid
from optid.core import affine

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class LatticeTest(unittest.TestCase):
    """
    Test optid.lattice.Lattice class.
    """

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_bad_shape_raises_exception(self):
        """
        Test lattice shape is validated.
        """

        optid.lattice.Lattice(unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(1, 1, 1))

        self.assertRaisesRegex(BeartypeException, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(1, 1, 1.5))

        self.assertRaisesRegex(BeartypeException, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(1,))

        self.assertRaisesRegex(BeartypeException, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(1, 1))

        self.assertRaisesRegex(BeartypeException, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(1, 1, 1, 1))

    def test_bad_shape_x_raises_exception(self):
        """
        Test lattice shape is validated.
        """

        self.assertRaisesRegex(ValueError, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(0, 2, 10))

        self.assertRaisesRegex(ValueError, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(-1, 2, 10))

    def test_bad_shape_z_raises_exception(self):
        """
        Test lattice shape is validated.
        """

        self.assertRaisesRegex(ValueError, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(2, 0, 10))

        self.assertRaisesRegex(ValueError, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(2, -1, 10))

    def test_bad_shape_s_raises_exception(self):
        """
        Test lattice shape is validated.
        """

        self.assertRaisesRegex(ValueError, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(2, 2, 0))

        self.assertRaisesRegex(ValueError, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.float32), shape=(2, 2, -1))

    def test_bad_matrix_type_raises_exception(self):
        """
        Test lattice raises exceptions from incorrectly typed matrices.
        """

        self.assertRaisesRegex(TypeError, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(4, dtype=jnp.int32), shape=(2, 2, 10))

    def test_bad_matrix_shape_raises_exception(self):
        """
        Test lattice raises exceptions from incorrectly sized matrices.
        """

        self.assertRaisesRegex(ValueError, '.*', optid.lattice.Lattice,
                               unit_to_world_matrix=jnp.eye(2, dtype=jnp.float32), shape=(2, 2, 10))

    def test_transform_points_unit_to_world(self):
        """
        Test lattice correctly transforms point lattice.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        self.assertTrue(np.allclose(lattice.transform_points_unit_to_world(
            jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)), jnp.array([1, 1, 5], dtype=jnp.float32), atol=1e-5))

    def test_transform_points_unit_to_world_bad_shape_raises_exception(self):
        """
        Test lattice raises exceptions from incorrectly sized point lattices.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_unit_to_world,
                               point_lattice=jnp.ones((2,), dtype=jnp.float32))

    def test_transform_points_unit_to_world_bad_type_raises_exception(self):
        """
        Test lattice raises exceptions from incorrectly typed point lattices.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        self.assertRaisesRegex(TypeError, '.*', lattice.transform_points_unit_to_world,
                               point_lattice=jnp.ones((3,), dtype=jnp.int32))

    def test_transform_points_unit_to_world_bad_point_raises_exception(self):
        """
        Test lattice raises exceptions from transforming points outside the lattice.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        lattice.transform_points_unit_to_world(jnp.array([0, 0, 0], dtype=jnp.float32),
                                               raise_out_of_bounds=True)

        lattice.transform_points_unit_to_world(jnp.array([0, 0, 0], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_unit_to_world(jnp.array([-0.6, -0.5, -0.5], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_unit_to_world(jnp.array([-0.5, -0.6, -0.5], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_unit_to_world(jnp.array([-0.5, -0.5, -0.6], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_unit_to_world(jnp.array([0.6, 0.5, 0.5], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_unit_to_world(jnp.array([0.5, 0.6, 0.5], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_unit_to_world(jnp.array([0.5, 0.5, 0.6], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_unit_to_world,
                               point_lattice=jnp.array([-0.6, -0.5, -0.5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_unit_to_world,
                               point_lattice=jnp.array([-0.5, -0.6, -0.5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_unit_to_world,
                               point_lattice=jnp.array([-0.5, -0.5, -0.6], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_unit_to_world,
                               point_lattice=jnp.array([0.6, 0.5, 0.5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_unit_to_world,
                               point_lattice=jnp.array([0.5, 0.6, 0.5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_unit_to_world,
                               point_lattice=jnp.array([0.5, 0.5, 0.6], dtype=jnp.float32),
                               raise_out_of_bounds=True)

    def test_transform_points_world_to_unit(self):
        """
        Test lattice correctly transforms point lattice.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        self.assertTrue(np.allclose(lattice.transform_points_world_to_unit(jnp.array([1, 1, 5], dtype=jnp.float32)),
                                    jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32), atol=1e-5))

    def test_transform_points_world_to_unit_bad_shape_raises_exception(self):
        """
        Test lattice raises exceptions from incorrectly sized point lattices.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_unit,
                               point_lattice=jnp.ones((2,), dtype=jnp.float32))

    def test_transform_points_world_to_unit_bad_type_raises_exception(self):
        """
        Test lattice raises exceptions from incorrectly typed point lattices.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        self.assertRaisesRegex(TypeError, '.*', lattice.transform_points_world_to_unit,
                               point_lattice=jnp.ones((3,), dtype=jnp.int32))

    def test_transform_points_world_to_unit_bad_point_raises_exception(self):
        """
        Test lattice raises exceptions from transforming points outside the lattice.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        lattice.transform_points_world_to_unit(jnp.array([0, 0, 0], dtype=jnp.float32),
                                               raise_out_of_bounds=True)

        lattice.transform_points_world_to_unit(jnp.array([0, 0, 0], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_world_to_unit(jnp.array([-1.1, -1, -5], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_world_to_unit(jnp.array([-1, -1.1, -5], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_world_to_unit(jnp.array([-1, -1, -5.1], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_world_to_unit(jnp.array([1.1, 1, 5], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_world_to_unit(jnp.array([1, 1.1, 5], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        lattice.transform_points_world_to_unit(jnp.array([1, 1, 5.1], dtype=jnp.float32),
                                               raise_out_of_bounds=False)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_unit,
                               point_lattice=jnp.array([-1.1, -1, -5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_unit,
                               point_lattice=jnp.array([-1, -1.1, -5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_unit,
                               point_lattice=jnp.array([-1, -1, -5.1], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_unit,
                               point_lattice=jnp.array([1.1, 1, 5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_unit,
                               point_lattice=jnp.array([1, 1.1, 5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_unit,
                               point_lattice=jnp.array([1, 1, 5.1], dtype=jnp.float32),
                               raise_out_of_bounds=True)

    def test_transform_points_world_to_orthonormal(self):
        """
        Test lattice correctly transforms point lattice.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        self.assertTrue(np.allclose(lattice.transform_points_world_to_orthonormal(jnp.array([1, 1, 5], dtype=jnp.float32)),
                                    jnp.array([2, 2, 10], dtype=jnp.float32), atol=1e-5))

    def test_transform_points_world_to_orthonormal_bad_shape_raises_exception(self):
        """
        Test lattice raises exceptions from incorrectly sized point lattices.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_orthonormal,
                               point_lattice=jnp.ones((2,), dtype=jnp.float32))

    def test_transform_points_world_to_orthonormal_bad_type_raises_exception(self):
        """
        Test lattice raises exceptions from incorrectly typed point lattices.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        self.assertRaisesRegex(TypeError, '.*', lattice.transform_points_world_to_orthonormal,
                               point_lattice=jnp.ones((3,), dtype=jnp.int32))

    def test_transform_points_world_to_orthonormal_bad_point_raises_exception(self):
        """
        Test lattice raises exceptions from transforming points outside the lattice.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 10), shape=(2, 2, 10))

        lattice.transform_points_world_to_orthonormal(jnp.array([0, 0, 0], dtype=jnp.float32),
                                                      raise_out_of_bounds=True)

        lattice.transform_points_world_to_orthonormal(jnp.array([0, 0, 0], dtype=jnp.float32),
                                                      raise_out_of_bounds=False)

        lattice.transform_points_world_to_orthonormal(jnp.array([-1.1, -1, -5], dtype=jnp.float32),
                                                      raise_out_of_bounds=False)

        lattice.transform_points_world_to_orthonormal(jnp.array([-1, -1.1, -5], dtype=jnp.float32),
                                                      raise_out_of_bounds=False)

        lattice.transform_points_world_to_orthonormal(jnp.array([-1, -1, -5.1], dtype=jnp.float32),
                                                      raise_out_of_bounds=False)

        lattice.transform_points_world_to_orthonormal(jnp.array([1.1, 1, 5], dtype=jnp.float32),
                                                      raise_out_of_bounds=False)

        lattice.transform_points_world_to_orthonormal(jnp.array([1, 1.1, 5], dtype=jnp.float32),
                                                      raise_out_of_bounds=False)

        lattice.transform_points_world_to_orthonormal(jnp.array([1, 1, 5.1], dtype=jnp.float32),
                                                      raise_out_of_bounds=False)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_orthonormal,
                               point_lattice=jnp.array([-1.1, -1, -5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_orthonormal,
                               point_lattice=jnp.array([-1, -1.1, -5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_orthonormal,
                               point_lattice=jnp.array([-1, -1, -5.1], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_orthonormal,
                               point_lattice=jnp.array([1.1, 1, 5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_orthonormal,
                               point_lattice=jnp.array([1, 1.1, 5], dtype=jnp.float32),
                               raise_out_of_bounds=True)

        self.assertRaisesRegex(ValueError, '.*', lattice.transform_points_world_to_orthonormal,
                               point_lattice=jnp.array([1, 1, 5.1], dtype=jnp.float32),
                               raise_out_of_bounds=True)

    def test_unit_lattice(self):
        """
        Test lattice correctly produces a lattice in unit space.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 2), shape=(2, 2, 2))

        self.assertTrue(np.allclose(lattice.unit_lattice,
                                    jnp.array([[[[-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5]],
                                                [[-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5]]],
                                               [[[ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5]],
                                                [[ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5]]]],
                                              dtype=jnp.float32), atol=1e-5))

    def test_world_lattice(self):
        """
        Test lattice correctly produces a lattice in world space.
        """

        lattice = optid.lattice.Lattice(unit_to_world_matrix=affine.scale(2, 2, 2), shape=(2, 2, 2))

        self.assertTrue(np.allclose(lattice.world_lattice,
                                    jnp.array([[[[-1, -1, -1], [-1, -1,  1]],
                                                [[-1,  1, -1], [-1,  1,  1]]],
                                               [[[ 1, -1, -1], [ 1, -1,  1]],
                                                [[ 1,  1, -1], [ 1,  1,  1]]]],
                                              dtype=jnp.float32), atol=1e-5))
