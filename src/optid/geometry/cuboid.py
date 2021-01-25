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
    ExtrudedPolygon


class Cuboid(ExtrudedPolygon):

    @beartype
    def __init__(self,
            shape: typ.Union[jnp.ndarray, typ.Sequence[typ.Union[int, float]]]):
        """
        Construct a Cuboid instance.

        :param shape:
            Aligned size vector in 3-space.
        """

        if not isinstance(shape, jnp.ndarray):
            shape = jnp.array(shape, dtype=jnp.float32)

        if shape.shape != (3,):
            raise ValueError(f'shape must be a vector of shape (3,) but is : '
                             f'{shape.shape}')

        if shape.dtype != jnp.float32:
            raise TypeError(f'shape must have dtype (float32) but is : '
                            f'{shape.dtype}')

        if np.any(shape <= 0):
            raise ValueError(f'shape must be greater than zero in every dimension but is : '
                             f'{shape}')

        x, z, s = shape.tolist()
        x *= 0.5
        z *= 0.5

        polygon = jnp.array(
            [[-x, -z], [-x, z], [x, z], [x, -z]], dtype=jnp.float32)

        super().__init__(polygon=polygon, thickness=s)
