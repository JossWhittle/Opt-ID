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


class Geometry:

    @beartype
    def __init__(self,
            vertices: jnp.ndarray,
            faces: typ.List[typ.List[int]]):

        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError(f'vertices must be shape (N, 3) but is : '
                             f'{vertices.shape}')

        if vertices.dtype != jnp.float32:
            raise TypeError(f'vertices must have dtype (float32) but is : '
                            f'{vertices.dtype}')

        self._vertices = vertices

        def any_vertex_out_of_bounds(face):
            face = np.array(face)
            return np.any((face < 0) | (face >= vertices.shape[0]))

        if any(map(any_vertex_out_of_bounds, faces)):
            raise TypeError(f'faces must be list of lists of integers in range [0, {vertices.shape[0]}) but is : '
                            f'{faces}')

        self._faces = faces

    @beartype
    def transform(self, matrix: jnp.ndarray) -> jnp.ndarray:

        if matrix.shape != (4, 4):
            raise ValueError(f'matrix must be an affine matrix with shape (4, 4) but is : '
                             f'{matrix.shape}')

        if matrix.dtype != jnp.float32:
            raise TypeError(f'matrix must have dtype (float32) but is : '
                            f'{matrix.dtype}')

        return transform_points(self.vertices, matrix)

    @property
    @beartype
    def vertices(self) -> jnp.ndarray:
        return self._vertices

    @property
    @beartype
    def faces(self) -> typ.List[typ.List[int]]:
        return self._faces
