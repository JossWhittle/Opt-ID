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


from beartype import beartype
import typing as typ
import numpy as np
import jax.numpy as jnp
from .. import core


class Lattice:

    @beartype
    def __init__(self, unit_to_world_matrix: jnp.ndarray, x: int, z: int, s: int):
        self.unit_to_world_matrix = unit_to_world_matrix
        self.x = x
        self.z = z
        self.s = s

    @beartype
    def transform_points_world_to_unit(self, point_lattice: jnp.ndarray) -> jnp.ndarray:

        if point_lattice.shape[-1] != 3:
            raise ValueError(f'point_lattice must be shape (..., 3) but is : '
                             f'{point_lattice.shape}')

        if isinstance(point_lattice.dtype, jnp.float32):
            raise TypeError(f'point_lattice must have dtype (float32) but is : '
                            f'{point_lattice.dtype}')

        return core.affine.transform_points(point_lattice, self.world_to_unit_matrix)

    @beartype
    def transform_points_world_to_orthonormal(self, point_lattice: jnp.ndarray) -> jnp.ndarray:

        if point_lattice.shape[-1] != 3:
            raise ValueError(f'point_lattice must be shape (..., 3) but is : '
                             f'{point_lattice.shape}')

        if isinstance(point_lattice.dtype, jnp.float32):
            raise TypeError(f'point_lattice must have dtype (float32) but is : '
                            f'{point_lattice.dtype}')

        return core.affine.transform_points(point_lattice, self.world_to_orthonormal_matrix)

    @property
    @beartype
    def unit_lattice(self) -> jnp.ndarray:
        """
        Lattice tensor with the desired shape centred at origin spanning -0.5 to +0.5
        """
        return core.lattice.unit_lattice(self.x, self.z, self.s)

    @property
    @beartype
    def world_lattice(self) -> jnp.ndarray:
        """
        Lattice tensor with the desired shape in world coordinates.
        """
        return core.affine.transform_points(self.unit_lattice, self.unit_to_world_matrix)

    @property
    @beartype
    def world_to_unit_matrix(self) -> jnp.ndarray:
        """
        Matrix that maps world space coordinates to unit coordinates centred at origin -0.5 to +0.5.
        """
        return jnp.linalg.inv(self.unit_to_world_matrix)

    @property
    @beartype
    def world_to_orthonormal_matrix(self) -> jnp.ndarray:
        """
        Matrix that maps world space coordinates to orthonormal space.
        """
        return self.world_to_unit_matrix @ self.unit_to_orthonormal_matrix

    @property
    @beartype
    def unit_to_orthonormal_matrix(self) -> jnp.ndarray:
        """
        Matrix that maps from unit coordinates centred at origin -0.5 to +0.5 to orthonormal space.
        """
        return core.lattice.unit_to_orthonormal_matrix(self.x, self.z, self.s)

    @property
    @beartype
    def unit_to_world_matrix(self) -> jnp.ndarray:
        """
        Matrix that maps unit coordinates centred at origin -0.5 to +0.5 to world space.
        """
        return self._unit_to_world_matrix

    @unit_to_world_matrix.setter
    @beartype
    def unit_to_world_matrix(self, unit_to_world_matrix: jnp.ndarray):

        if unit_to_world_matrix.shape != (4, 4):
            raise ValueError(f'unit_to_world_matrix must be an affine matrix with shape (4, 4) but is : '
                             f'{unit_to_world_matrix.shape}')

        if isinstance(unit_to_world_matrix.dtype, jnp.float32):
            raise TypeError(f'unit_to_world_matrix must have dtype (float32) but is : '
                            f'{unit_to_world_matrix.dtype}')

        self._unit_to_world_matrix = unit_to_world_matrix

    @property
    @beartype
    def x(self) -> int:
        return self._x

    @x.setter
    @beartype
    def x(self, x: int):
        if x <= 0:
            raise ValueError(f'x shape must be a positive integer but is : {x}')
        self._x = x

    @property
    @beartype
    def z(self) -> int:
        return self._z

    @z.setter
    @beartype
    def z(self, z: int):
        if z <= 0:
            raise ValueError(f'z shape must be a positive integer but is : {z}')
        self._z = z

    @property
    @beartype
    def s(self) -> int:
        return self._s

    @s.setter
    @beartype
    def s(self, s: int):
        if s <= 0:
            raise ValueError(f's shape must be a positive integer but is : {s}')
        self._s = s
