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
from sect.triangulation import constrained_delaunay_triangles

# Opt-ID Imports
from ..geometry import \
    Geometry


class ExtrudedPolygon(Geometry):

    @beartype
    def __init__(self,
            polygon: typ.Union[jnp.ndarray, typ.Sequence[typ.Sequence[typ.Union[float, int]]]],
            thickness: typ.Union[float, int]):
        """
        Construct an ExtrudedPolygon instance from a set of unique polygon vertices in 2-space and a thickness.

        :param polygon:
            Tensor of vertices in 2-space of shape (N, 2).

        :param thickness:
            Thickness of the geometry along the S-axis.
        """

        if not isinstance(polygon, jnp.ndarray):

            def is_vertex_not_2d(vertex: typ.Sequence[typ.Union[float, int]]) -> bool:
                return len(vertex) != 2

            if any(map(is_vertex_not_2d, polygon)):
                raise ValueError(f'polygon must be a list of 2D XZ coordinates but is : '
                                 f'{polygon}')

            polygon = jnp.array(polygon, dtype=jnp.float32)

        if polygon.shape[-1] != 2:
            raise ValueError(f'polygon must be a list of 2D XZ coordinates (N, 2) but is : '
                             f'{polygon.shape}')

        if polygon.shape[0] < 3:
            raise ValueError(f'polygon must be a list of at least 3 2D XZ coordinates (N >= 3, 2) but is : '
                             f'{polygon.shape}')

        if polygon.dtype != jnp.float32:
            raise TypeError(f'polygon must have dtype (float32) but is : '
                            f'{polygon.dtype}')

        thickness = float(thickness)

        if thickness <= 0:
            raise TypeError(f'thickness must a positive float but is : '
                            f'{thickness}')

        n = polygon.shape[0]
        s = thickness * 0.5

        vertices = jnp.concatenate([
            jnp.pad(polygon, ((0, 0), (0, 1)), constant_values=-s),
            jnp.pad(polygon, ((0, 0), (0, 1)), constant_values=+s)])

        def is_polygon_convex(polygon):
            polygon = np.concatenate([polygon, polygon[:2]], axis=0)
            return all((x0 - x1) * (z1 - z2) <= (z0 - z1) * (x1 - x2)
                       for (x0, z0), (x1, z1), (x2, z2) in zip(polygon[:2], polygon[1:-1], polygon[2:]))

        if is_polygon_convex(polygon):
            # Consider the entire end polygon at once if it is convex
            polygons = [list(range(len(polygon)))]

        else:
            # Triangulate non-convex end polygon into multiple convex polygons
            vertex_to_id = { (*vertex,): idx for idx, vertex in enumerate(polygon) }
            polygons = [[vertex_to_id[vertex] for vertex in face]
                        for face in constrained_delaunay_triangles(list(vertex_to_id.keys()))]

        # Produce a set of convex prisms representing the full extrusion
        polyhedra = [[

            [v for v in polygon],

            [(v + n) for v in reversed(polygon)],

            *[[v1, (v1 + n), (v0 + n), v0]
              for v0, v1 in zip(polygon, (polygon[1:] + polygon[:1]))]

            ] for polygon in polygons]

        super().__init__(vertices=vertices, polyhedra=polyhedra)

