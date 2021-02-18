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
from ..bfield import \
    Lookup


class MagnetSlot:

    @beartype
    def __init__(self,
            slot: str,
            beam: str,
            world_matrix: jnp.ndarray,
            direction_matrix: jnp.ndarray,
            lookup: Lookup):
        """
        Construct a MagnetSlot instance.

        :param slot:
            String name for the slot.

        :param beam:
            String name for the beam.

        :param world_matrix:
            Affine matrix for where the magnet exists in world space.

        :param direction_matrix:
            Affine matrix for the major orientation of this slot applied at origin.

        :param lookup:
            Bfield lookup table.
        """

        if len(slot) == 0:
            raise ValueError(f'slot must be a non-empty string')

        self._slot = slot

        if len(beam) == 0:
            raise ValueError(f'beam must be a non-empty string')

        self._beam = beam

        self._name = f'{self._beam}::{self._slot}'

        if world_matrix.shape != (4, 4):
            raise ValueError(f'world_matrix must be an affine world_matrix with shape (4, 4) but is : '
                             f'{world_matrix.shape}')

        if world_matrix.dtype != jnp.float32:
            raise TypeError(f'world_matrix must have dtype (float32) but is : '
                            f'{world_matrix.dtype}')

        self._world_matrix = world_matrix

        if direction_matrix.shape != (4, 4):
            raise ValueError(f'direction_matrix must be an affine direction_matrix with shape (4, 4) but is : '
                             f'{direction_matrix.shape}')

        if direction_matrix.dtype != jnp.float32:
            raise TypeError(f'direction_matrix must have dtype (float32) but is : '
                            f'{direction_matrix.dtype}')

        self._direction_matrix = direction_matrix

        self._lookup = lookup

    @property
    @beartype
    def name(self) -> str:
        return self._name

    @property
    @beartype
    def slot(self) -> str:
        return self._slot

    @property
    @beartype
    def beam(self) -> str:
        return self._beam

    @property
    @beartype
    def world_matrix(self) -> jnp.ndarray:
        """
        Affine matrix with shape (4, 4) of the main positional transformation.
        """
        return self._world_matrix

    @property
    @beartype
    def direction_matrix(self) -> jnp.ndarray:
        """
        Affine matrix with shape (4, 4) of the major orientation transformation.
        """
        return self._direction_matrix

    @property
    @beartype
    def lookup(self) -> Lookup:
        return self._lookup
