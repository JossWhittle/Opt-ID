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


# External Imports
from beartype import beartype
import typing as typ
import numpy as np
import jax.numpy as jnp

# Opt-ID Imports
from ..core.affine import \
    transform_points

from ..core.lattice import \
    unit_lattice, unit_to_orthonormal_matrix, any_unit_point_out_of_bounds, any_orthonormal_point_out_of_bounds


class Lattice:

    @beartype
    def __init__(self,
            unit_to_world_matrix: jnp.ndarray,
            shape: typ.Tuple[int, int, int]):

        if unit_to_world_matrix.shape != (4, 4):
            raise ValueError(f'unit_to_world_matrix must be an affine matrix with shape (4, 4) but is : '
                             f'{unit_to_world_matrix.shape}')

        if unit_to_world_matrix.dtype != jnp.float32:
            raise TypeError(f'unit_to_world_matrix must have dtype (float32) but is : '
                            f'{unit_to_world_matrix.dtype}')

        self._unit_to_world_matrix = unit_to_world_matrix

        if np.any(np.array(shape) <= 0):
            raise ValueError(f'shape must be a 3-tuple of positive integers but is : '
                             f'{shape}')
        self._shape = shape

    @beartype
    def transform_points_unit_to_world(self,
            point_lattice: jnp.ndarray,
            raise_out_of_bounds: bool = True) -> jnp.ndarray:

        if point_lattice.shape[-1] != 3:
            raise ValueError(f'point_lattice must be shape (..., 3) but is : '
                             f'{point_lattice.shape}')

        if point_lattice.dtype != jnp.float32:
            raise TypeError(f'point_lattice must have dtype (float32) but is : '
                            f'{point_lattice.dtype}')

        if raise_out_of_bounds:
            if any_unit_point_out_of_bounds(point_lattice, 1e-5):
                raise ValueError(f'point_lattice contains world space coordinates outside the lattice')

        transformed_point_lattice = transform_points(point_lattice, self.unit_to_world_matrix)

        return transformed_point_lattice

    @beartype
    def transform_points_world_to_unit(self,
            point_lattice: jnp.ndarray,
            raise_out_of_bounds: bool = True) -> jnp.ndarray:

        if point_lattice.shape[-1] != 3:
            raise ValueError(f'point_lattice must be shape (..., 3) but is : '
                             f'{point_lattice.shape}')

        if point_lattice.dtype != jnp.float32:
            raise TypeError(f'point_lattice must have dtype (float32) but is : '
                            f'{point_lattice.dtype}')

        transformed_point_lattice = transform_points(point_lattice, self.world_to_unit_matrix)

        if raise_out_of_bounds:
            if any_unit_point_out_of_bounds(transformed_point_lattice, 1e-5):
                raise ValueError(f'point_lattice contains world space coordinates outside the lattice')

        return transformed_point_lattice

    @beartype
    def transform_points_world_to_orthonormal(self,
            point_lattice: jnp.ndarray,
            raise_out_of_bounds: bool = True) -> jnp.ndarray:

        if point_lattice.shape[-1] != 3:
            raise ValueError(f'point_lattice must be shape (..., 3) but is : '
                             f'{point_lattice.shape}')

        if point_lattice.dtype != jnp.float32:
            raise TypeError(f'point_lattice must have dtype (float32) but is : '
                            f'{point_lattice.dtype}')

        transformed_point_lattice = transform_points(point_lattice, self.world_to_orthonormal_matrix)

        if raise_out_of_bounds:
            if any_orthonormal_point_out_of_bounds(transformed_point_lattice, *self.shape, 1e-5):
                raise ValueError(f'point_lattice contains world space coordinates outside the lattice')

        return transformed_point_lattice

    @property
    @beartype
    def unit_lattice(self) -> jnp.ndarray:
        """
        Lattice tensor with the desired shape centred at origin spanning -0.5 to +0.5
        """
        return unit_lattice(*self.shape)

    @property
    @beartype
    def world_lattice(self) -> jnp.ndarray:
        """
        Lattice tensor with the desired shape in world coordinates.
        """
        return transform_points(self.unit_lattice, self.unit_to_world_matrix)

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
        return unit_to_orthonormal_matrix(*self.shape)

    @property
    @beartype
    def unit_to_world_matrix(self) -> jnp.ndarray:
        """
        Matrix that maps unit coordinates centred at origin -0.5 to +0.5 to world space.
        """
        return self._unit_to_world_matrix

    @property
    @beartype
    def shape(self) -> typ.Tuple[int, int, int]:
        return self._shape
