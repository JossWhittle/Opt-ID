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
from more_itertools import SequenceView
import numbers
from beartype import beartype
import typing as typ
import jax.numpy as jnp

# Opt-ID Imports
from ..core.affine import translate

from ..device import \
    Slot, SlotType


TVector = typ.Union[jnp.ndarray, typ.Sequence[numbers.Real]]


class Beam:

    @beartype
    def __init__(self,
            device,
            name: str,
            beam_matrix: jnp.ndarray,
            gap_vector: TVector,
            phase_vector: TVector):
        """
        Construct a Beam instance.

        :param device:
            Parent Device instance this Beam is a member of.

        :param name:
            String name for the beam.

        :param beam_matrix:
            Affine matrix for the placing this beam into the world.
            
        :param gap_vector:
            Vector representing the Beam's response to the gap value.

        :param phase_vector:
            Vector representing the Beam's response to the phase value.
        """

        self._device = device

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        if beam_matrix.shape != (4, 4):
            raise ValueError(f'beam_matrix must be an affine world_matrix with shape (4, 4) but is : '
                             f'{beam_matrix.shape}')

        if beam_matrix.dtype != jnp.float32:
            raise TypeError(f'beam_matrix must have dtype (float32) but is : '
                            f'{beam_matrix.dtype}')

        self._beam_matrix = beam_matrix
        
        if not isinstance(gap_vector, jnp.ndarray):
            gap_vector = jnp.array(gap_vector, dtype=jnp.float32)

        if gap_vector.shape != (3,):
            raise ValueError(f'gap_vector must be shape (3,) but is : '
                             f'{gap_vector.shape}')

        if gap_vector.dtype != jnp.float32:
            raise TypeError(f'gap_vector must have dtype (float32) but is : '
                            f'{gap_vector.dtype}')
        
        self._gap_vector = gap_vector

        if not isinstance(phase_vector, jnp.ndarray):
            phase_vector = jnp.array(phase_vector, dtype=jnp.float32)

        if phase_vector.shape != (3,):
            raise ValueError(f'phase_vector must be shape (3,) but is : '
                             f'{phase_vector.shape}')

        if phase_vector.dtype != jnp.float32:
            raise TypeError(f'phase_vector must have dtype (float32) but is : '
                            f'{phase_vector.dtype}')

        self._phase_vector = phase_vector

        self._centre_matrix = jnp.eye(4, dtype=jnp.float32)
        self._smin = 0
        self._smax = 0
        self._spacing = 0
        self._slots = list()

    @beartype
    def world_matrix(self, gap: numbers.Real, phase: numbers.Real) -> jnp.ndarray:
        """
        Calculate the affine matrix that places this beam into the world.

        :param gap:
            Device gap value to separate the beams on the Z axis.

        :param phase:
            Device phase value to shear the beams by on the S axis.

        :return:
            Affine matrix representing the major position of the beam in world space.
        """

        if gap < 0:
            raise ValueError(f'gap must be > 0 but is : '
                             f'{gap}')

        if phase < 0:
            raise ValueError(f'phase must be > 0 but is : '
                             f'{phase}')

        return self.centre_matrix @ \
               translate(*((self.gap_vector * gap) + (self.phase_vector * phase))) @ \
               self.beam_matrix @ \
               self.device.world_matrix

    @beartype
    def add_slot(self,
            period: str,
            slot_type: SlotType,
            after_spacing: numbers.Real = 0):

        bmin, bmax = slot_type.bounds

        if self.nslots == 0:
            self._smin = bmin[2]
            self._smax = bmax[2]
        else:
            self._smax += self._spacing + (bmax[2] - bmin[2])
            self._spacing = 0

        slot_matrix = translate(0, 0, (self._smax - bmax[2]))

        self._centre_matrix = translate(0, 0, (-self._smin) - ((self._smax - self._smin) / 2.0))

        slot = Slot(beam=self, name=f'{self.nslots:06d}', period=period,
                    slot_type=slot_type, slot_matrix=slot_matrix)

        self.add_spacing(spacing=after_spacing)

        self._slots.append(slot)
        return slot

    @beartype
    def add_spacing(self, spacing: numbers.Real):

        if spacing < 0:
            raise ValueError(f'spacing must be > 0 but is : '
                             f'{spacing}')

        self._spacing += spacing

    @property
    def device(self):
        return self._device

    @property
    @beartype
    def name(self) -> str:
        return self._name

    @property
    @beartype
    def qualified_name(self) -> str:
        return f'{self.device.name}::{self.name}'

    @property
    @beartype
    def beam_matrix(self) -> jnp.ndarray:
        return self._beam_matrix

    @property
    @beartype
    def gap_vector(self) -> jnp.ndarray:
        return self._gap_vector

    @property
    @beartype
    def phase_vector(self) -> jnp.ndarray:
        return self._phase_vector

    @property
    @beartype
    def length(self) -> float:
        return self._smax - self._smin

    @property
    @beartype
    def centre_matrix(self) -> jnp.ndarray:
        return self._centre_matrix

    @property
    @beartype
    def slots(self) -> typ.Sequence[Slot]:
        return SequenceView(self._slots)

    @property
    @beartype
    def nslots(self) -> int:
        return len(self._slots)
