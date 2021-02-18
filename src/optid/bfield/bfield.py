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
from ..lattice import Lattice


class Bfield:

    @beartype
    def __init__(self,
            lattice: Lattice,
            field: jnp.ndarray):
        """
        Construct a Lookup instance to represent a Bfield lookup table of 3x3 rotation matrices sampled over a
        3-lattice of spatial coordinates.

        :param lattice:
            Lattice representing the spatial coordinates of the Bfield samples.

        :param field:
            Tensor representing the vector field.
        """

        self._lattice = lattice

        if field.ndim != 4:
            raise ValueError(f'field must be a lattice of vectors with shape (X, Z, S, 3) but is : '
                             f'{field.shape}')

        if field.shape[-1] != 3:
            raise ValueError(f'field must be a lattice of vectors with shape (..., 3) but is : '
                             f'{field.shape}')

        if field.shape[:-1] != self.lattice.shape:
            raise ValueError(f'field spatial dims must be equal to lattice shape '
                             f'{self.lattice.shape} but is : '
                             f'{field.shape[:-1]}')

        if field.dtype != jnp.float32:
            raise TypeError(f'field must have dtype (float32) but is : '
                            f'{field.dtype}')

        self._field = field

    @beartype
    def copy(self):
        return Bfield(lattice=self.lattice, field=self.field.copy())

    @property
    @beartype
    def lattice(self) -> Lattice:
        """
        Lattice representing the spatial locations of each vector.
        """
        return self._lattice

    @property
    @beartype
    def field(self) -> jnp.ndarray:
        """
        Tensor for the field vectors.
        """
        return self._field
