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
from ..lattice import \
    Lattice

from ..bfield import \
    Bfield

from ..core.bfield import \
    bfield_from_lookup


class Lookup:

    @beartype
    def __init__(self,
                 lattice: Lattice,
                 lookup: jnp.ndarray):
        """
        Construct a Lookup instance to represent a Bfield lookup table of 3x3 rotation matrices sampled over a
        3-lattice of spatial coordinates.

        :param lattice:
            Lattice representing the spatial coordinates of the Bfield samples.

        :param lookup:
            Tensor representing the 8D lookup table of 3x3 rotation matrices sampled over the two 3-lattices.
        """

        self._lattice = lattice

        if lookup.ndim != 5:
            raise ValueError(f'lookup must be a lattice of matrices with shape (X, Z, S, 3, 3) but is : '
                             f'{lookup.shape}')

        if lookup.shape[-2:] != (3, 3):
            raise ValueError(f'lookup must be a lattice of rotation matrices with shape (..., 3, 3) but is : '
                             f'{lookup.shape}')

        if lookup.shape[:-2] != self.lattice.shape:
            raise ValueError(f'lookup spatial dims must be equal to lattice shape '
                             f'{self.lattice.shape} but is : '
                             f'{lookup.shape[:-2]}')

        if lookup.dtype != jnp.float32:
            raise TypeError(f'lookup must have dtype (float32) but is : '
                            f'{lookup.dtype}')

        self._lookup = lookup

    @beartype
    def bfield(self, vector: jnp.ndarray) -> Bfield:
        """
        Extract a Bfield for a magnet with the desired magnetization vector from the lookup table.

        :param vector:
            Vector in 3-space for the major magnetization direction to map to all of the 3x3 rotation matrices.

        :return:
            Bfield as a 4D tensor of shape (X, Z, S, 3).
        """

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != jnp.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        return Bfield(lattice=self.lattice, field=bfield_from_lookup(self.lookup, vector))

    @property
    @beartype
    def lattice(self) -> Lattice:
        """
        Lattice representing the spatial locations of each 3x3 matrix in the lookup table.
        """
        return self._lattice

    @property
    @beartype
    def lookup(self) -> jnp.ndarray:
        """
        Tensor for the raw lookup table data.
        """
        return self._lookup
