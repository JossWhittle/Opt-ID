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
        return self._shim_lattice

    @property
    @beartype
    def lookup_lattice(self) -> Lattice:
        return self._lookup_lattice

    @property
    @beartype
    def lookup(self) -> jnp.ndarray:
        return self._lookup
