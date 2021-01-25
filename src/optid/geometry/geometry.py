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
            vertices: typ.Union[jnp.ndarray, typ.Sequence[typ.Sequence[typ.Union[float, int]]]],
            faces: typ.Sequence[typ.Sequence[int]]):
        """
        Construct a Geometry instance from a set of unique vertices in 3-space and a list of polygons.

        :param vertices:
            Tensor of vertices in 3-space of shape (N, 3).

        :param faces:
            List of lists of integer vertex IDs in the range [0, N). Each face must have at least 3 vertices.
        """

        if not isinstance(vertices, jnp.ndarray):

            def is_vertex_not_3d(vertex: typ.Sequence[typ.Union[float, int]]) -> bool:
                return len(vertex) != 3

            if any(map(is_vertex_not_3d, vertices)):
                raise ValueError(f'vertices must be a list of 3D XZS coordinates but is : '
                                 f'{vertices}')

            vertices = jnp.array(vertices, dtype=jnp.float32)

        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError(f'vertices must be shape (N, 3) but is : '
                             f'{vertices.shape}')

        if vertices.dtype != jnp.float32:
            raise TypeError(f'vertices must have dtype (float32) but is : '
                            f'{vertices.dtype}')

        self._vertices = vertices

        # Coerce sequence of sequences into list of lists
        faces = [[vertex for vertex in face] for face in faces]

        def any_vertex_out_of_bounds(face: typ.List[int]) -> bool:
            face = np.array(face)
            return np.any((face < 0) | (face >= vertices.shape[0]))

        if any(map(any_vertex_out_of_bounds, faces)):
            raise ValueError(f'faces must be list of lists of unique integers in range '
                             f'[0, {vertices.shape[0]}) but is : '
                             f'{faces}')

        def any_vertex_duplicated(face: typ.List[int]) -> bool:
            return len(set(face)) < len(face)

        if any(map(any_vertex_duplicated, faces)):
            raise ValueError(f'faces must be list of lists of unique integers but is : '
                             f'{faces}')

        def is_face_not_polygon(face: typ.List[int]) -> bool:
            return len(face) < 3

        if any(map(is_face_not_polygon, faces)):
            raise ValueError(f'faces must contain faces of at least 3 vertices but is : '
                             f'{faces}')

        self._faces = faces

    @beartype
    def transform(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Apply an affine matrix transformation to the vertices of this Geometry instance.

        :param matrix:
            Affine transformation matrix to apply.

        :return:
            Tensor the same shape as the vertices with the affine transformation applied.
        """

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
        """
        Tensor of vertices in 3-space.
        """
        return self._vertices

    @property
    @beartype
    def faces(self) -> typ.List[typ.List[int]]:
        """
        List of lists of integer vertex IDs.
        """
        return self._faces
