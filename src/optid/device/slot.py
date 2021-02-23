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
import numbers
from beartype import beartype
import typing as typ
import jax.numpy as jnp

# Opt-ID Imports
from ..device import \
    SlotType, Candidate

from ..geometry import \
    Geometry


TCandidates = typ.Dict[str, Candidate]


class Slot:

    @beartype
    def __init__(self,
            beam,
            name: str,
            period: str,
            slot_type: SlotType,
            slot_matrix: jnp.ndarray):
        """
        Construct a Slot instance.

        :param beam:
            Parent Beam instance this slot is a member of.

        :param name:
            String name for the slot.

        :param period:
            String period name used for calculating device period length.

        :param slot_type:
            MagnetSlotType instance that this slot is one of.

        :param slot_matrix:
            Affine matrix for the placing this slot along its parent beam starting from 0 on the Z axis.
        """

        self._beam = beam

        self._slot_type = slot_type

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        if len(period) == 0:
            raise ValueError(f'period must be a non-empty string')

        self._period = period

        if slot_matrix.shape != (4, 4):
            raise ValueError(f'slot_matrix must be an affine world_matrix with shape (4, 4) but is : '
                             f'{slot_matrix.shape}')

        if slot_matrix.dtype != jnp.float32:
            raise TypeError(f'slot_matrix must have dtype (float32) but is : '
                            f'{slot_matrix.dtype}')

        self._slot_matrix = slot_matrix

    @beartype
    def world_matrix(self, gap: numbers.Real, phase: numbers.Real) -> jnp.ndarray:
        """
        Calculate the affine matrix that places this magnet slot into the world in the correct major
        orientation except flip state.

        :param gap:
            Device gap value to separate the beams on the Z axis.

        :param phase:
            Device phase value to shear the beams by on the S axis.

        :return:
            Affine matrix representing the major position of the slot in world space.
        """
        return self.slot_type.direction_matrix @ \
               self.slot_type.anchor_matrix @ \
               self.slot_matrix @ \
               self.beam.world_matrix(gap=gap, phase=phase)

    @property
    def beam(self):
        return self._beam

    @property
    @beartype
    def slot_type(self) -> SlotType:
        return self._slot_type

    @property
    @beartype
    def name(self) -> str:
        return self._name

    @property
    @beartype
    def qualified_name(self) -> str:
        return f'{self.beam.qualified_name}::{self.name}::{self.slot_type.name}::{self.slot_type.magnet.name}'

    @property
    @beartype
    def period(self) -> str:
        return self._period

    @property
    @beartype
    def slot_matrix(self) -> jnp.ndarray:
        return self._slot_matrix

    @property
    @beartype
    def candidates(self) -> TCandidates:
        return self.slot_type.magnet.candidates

    @property
    @beartype
    def geometry(self) -> Geometry:
        return self.slot_type.magnet.geometry

    @property
    @beartype
    def vector(self) -> jnp.ndarray:
        return self.slot_type.magnet.vector

