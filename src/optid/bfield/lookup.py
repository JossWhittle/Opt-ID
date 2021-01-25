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
from ..core.bfield import bfield_from_lookup, bfield_from_interpolated_lookup


class Lookup:

    @beartype
    def __init__(self,
            shim_lattice: Lattice,
            lookup_lattice: Lattice,
            lookup: jnp.ndarray):
        """
        Construct a Lookup instance to represent a Bfield lookup table of 3x3 rotation matrices sampled over a
        3-lattice of spatial coordinates, and additionally sampled over a 3-lattice of shim offsets.

        :param shim_lattice:
            Lattice representing shim offsets the lookup table is duplicated over.

        :param lookup_lattice:
            Lattice representing the spatial coordinates of the Bfield samples.

        :param lookup:
            Tensor representing the 8D lookup table of 3x3 rotation matrices sampled over the two 3-lattices.
        """

        self._shim_lattice = shim_lattice
        self._lookup_lattice = lookup_lattice

        if lookup.ndim != 8:
            raise ValueError(f'lookup must be a lattice of matrices with shape (sX, sZ, sS, X, Z, S, 3, 3) but is : '
                             f'{lookup.shape}')

        if lookup.shape[-2:] != (3, 3):
            raise ValueError(f'lookup must be a lattice of rotation matrices with shape (..., 3, 3) but is : '
                             f'{lookup.shape}')

        if lookup.shape[:3] != self.shim_lattice.shape:
            raise ValueError(f'lookup shim dims must be equal to shim_lattice shape '
                             f'{self.shim_lattice.shape} but is : '
                             f'{lookup.shape[:3]}')

        if lookup.shape[3:-2] != self.lookup_lattice.shape:
            raise ValueError(f'lookup spatial dims must be equal to lookup_lattice shape '
                             f'{self.lookup_lattice.shape} but is : '
                             f'{lookup.shape[3:-2]}')

        if lookup.dtype != jnp.float32:
            raise TypeError(f'lookup must have dtype (float32) but is : '
                            f'{lookup.dtype}')

        self._lookup = lookup

    @beartype
    def bfield(self,
            vector: jnp.ndarray,
            shim: typ.Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Extract a Bfield for a magnet with the desired magnetization vector from the lookup table.

        :param vector:
            Vector in 3-space for the major magnetization direction to map to all of the 3x3 rotation matrices.

        :param shim:
            Coordinate in 3-space to use to interpolate between lookup tables at different shim offsets.

        :return:
            Bfield as a 4D tensor of shape (X, Z, S, 3).
        """

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != jnp.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        if shim is None:

            if self.lookup.shape[:3] != (1, 1, 1):
                raise ValueError(f'lookup contains shim dimensions but no shim coordinate is given : '
                                 f'{self.lookup.shape}')

            return bfield_from_lookup(self.lookup[0, 0, 0], vector)

        if shim.shape != (3,):
            raise ValueError(f'shim must be shape (3,) but is : '
                             f'{shim.shape}')

        if shim.dtype != jnp.float32:
            raise TypeError(f'shim must have dtype (float32) but is : '
                            f'{shim.dtype}')

        transformed_shim = self.shim_lattice.transform_points_world_to_orthonormal(
            shim, raise_out_of_bounds=True)

        return bfield_from_interpolated_lookup(self.lookup, transformed_shim, vector)

    @property
    @beartype
    def shim_lattice(self) -> Lattice:
        """
        Lattice representing the locations of lookup tables computed at different shim offsets.
        """
        return self._shim_lattice

    @property
    @beartype
    def lookup_lattice(self) -> Lattice:
        """
        Lattice representing the spatial locations of each 3x3 matrix in the lookup table.
        """
        return self._lookup_lattice

    @property
    @beartype
    def lookup(self) -> jnp.ndarray:
        """
        Tensor for the raw lookup table data.
        """
        return self._lookup
