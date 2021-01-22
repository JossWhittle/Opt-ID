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
from ..geometry import \
    Geometry


class Cuboid(Geometry):

    @beartype
    def __init__(self,
            shape: typ.Tuple[float, float, float]):

        if np.any(np.array(shape) <= 0):
            raise ValueError(f'shape must be greater than zero in every dimension but is : '
                             f'{shape}')

        self._shape = shape

        x, z, s = np.array(shape) / 2.0

        super().__init__(
            vertices=jnp.array([
                # 0             1             2             3
                [-x, -z, -s], [-x,  z, -s], [ x,  z, -s], [ x, -z, -s],
                # 4             5             6             7
                [-x, -z,  s], [-x,  z,  s], [ x,  z,  s], [ x, -z,  s],
            ], dtype=jnp.float32),

            faces=[
                # -X            +X
                [0, 1, 5, 4], [2, 3, 7, 6],
                # -Z            +Z
                [0, 3, 7, 4], [1, 2, 6, 5],
                # -S            +S
                [0, 1, 2, 3], [4, 5, 6, 7],
            ])

    @property
    @beartype
    def shape(self) -> typ.Tuple[float, float, float]:
        return self._shape
