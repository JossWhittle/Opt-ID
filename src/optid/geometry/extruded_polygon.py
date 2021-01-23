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
import jax.numpy as jnp

# Opt-ID Imports
from ..geometry import \
    Geometry


class ExtrudedPolygon(Geometry):

    @beartype
    def __init__(self,
            polygon: typ.Union[jnp.ndarray, typ.Sequence[typ.Sequence[typ.Union[float, int]]]],
            thickness: float):

        if not isinstance(polygon, jnp.ndarray):
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

        if thickness <= 0:
            raise TypeError(f'thickness must a positive float but is : '
                            f'{thickness}')

        n = polygon.shape[0]

        vertices = jnp.concatenate([
            jnp.concatenate([polygon, jnp.full((n, 1), -thickness / 2.0)], axis=-1),
            jnp.concatenate([polygon, jnp.full((n, 1), +thickness / 2.0)], axis=-1)], axis=0)

        faces = [
            # End polygons
            [v for v in range(n)], [(v + n) for v in range(n)],
            # Iterate over all the edge quads for the extrusion
            *[[v, ((v + 1) % n), (((v + 1) % n) + n), (v + n)] for v in range(n)]]

        super().__init__(vertices=vertices, faces=faces)

