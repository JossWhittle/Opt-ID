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
import numpy as np
import jax.numpy as jnp
import radia as rad


# Opt-ID Imports
from ..core.utils import \
    np_readonly

from ..core.affine import \
    transform_rescaled_vectors

from ..core.bfield import \
    radia_evaluate_bfield_on_lattice

from ..lattice import \
    Lattice

from .slot import \
    Slot

from .slot_type import \
    SlotType

from .magnet import \
    Magnet

from .candidate import \
    Candidate

from .slot_state import \
    SlotState


TCandidates = typ.Dict[str, Candidate]
TVector     = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]


class MagnetSlot(Slot):

    @beartype
    def __init__(self,
            beam,
            index: int,
            name: str,
            period: str,
            slot_type: SlotType,
            slot_matrix: np.ndarray):
        """
        Construct a Slot instance.

        :param beam:
            Parent Beam instance this slot is a member of.

        :param index:
            Integer index for the slot in the beam.

        :param name:
            String name for the slot.

        :param period:
            String period name used for calculating device period length.

        :param slot_type:
            SlotType instance that this slot is one of.

        :param slot_matrix:
            Affine matrix for the placing this slot along its parent beam starting from 0 on the Z axis.
        """
        if not isinstance(slot_type.element, Magnet):
            raise TypeError(f'slot_type.element must be type Magnet but is : '
                             f'{type(slot_type.element)}')

        super().__init__(beam=beam, index=index, name=name, period=period,
                         slot_type=slot_type, slot_matrix=slot_matrix)

        self._vector = transform_rescaled_vectors(self.magnet.vector, super().world_matrix(gap=0, phase=0))

    @beartype
    def world_matrix(self, gap: numbers.Real, phase: numbers.Real, flip: int = 0) -> np.ndarray:
        """
        Calculate the affine matrix that places this magnet slot into the world in the correct major
        orientation except flip state.

        :param gap:
            Device gap value to separate the beams on the Z axis.

        :param phase:
            Device phase value to shear the beams by on the S axis.

        :param flip:
            Flip matrix to use.

        :return:
            Affine matrix representing the major position of the slot in world space.
        """

        if (flip < 0) or (flip >= self.magnet.nflip):
            raise ValueError(f'flip must be in range [0, {self.magnet.nflip}) but is : '
                             f'{flip}')

        return self.magnet.flip_matrices[flip] @ \
               super().world_matrix(gap=gap, phase=phase)

    @beartype
    def bfield_from_vector(self,
            lattice: Lattice,
            vector: TVector,
            gap: numbers.Real,
            phase: numbers.Real,
            flip: int = 0,
            world_vector: bool = True) -> jnp.ndarray:

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != np.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        matrix   = self.world_matrix(gap=gap, phase=phase, flip=flip)
        geometry = self.geometry.transform(matrix)

        if not world_vector:
            vector = transform_rescaled_vectors(vector, matrix)

        rad.UtiDelAll()
        bfield = radia_evaluate_bfield_on_lattice(geometry.to_radia(vector), lattice.world_lattice)
        rad.UtiDelAll()
        return bfield

    @beartype
    def bfield(self,
            lattice: Lattice,
            gap: numbers.Real,
            phase: numbers.Real,
            flip: int = 0) -> jnp.ndarray:

        return self.bfield_from_vector(lattice=lattice, gap=gap, phase=phase, flip=flip,
                                       vector=self.magnet.vector, world_vector=False)

    @beartype
    def bfield_from_state(self,
            lattice: Lattice,
            gap: numbers.Real,
            phase: numbers.Real,
            slot_state: SlotState) -> jnp.ndarray:

        if slot_state.slot != self.qualified_name:
            raise ValueError(f'slot_state must be assigned to this slot but is : '
                             f'slot={self.qualified_name} state={slot_state.slot}')

        if slot_state.candidate not in self.candidates:
            raise ValueError(f'slot_state must be assigned to a valid candidate for this slot but is : '
                             f'{slot_state.candidate}')

        candidate = self.candidates[slot_state.candidate]

        return self.bfield_from_vector(lattice=lattice, gap=gap, phase=phase, flip=slot_state.flip,
                                       vector=candidate.vector, world_vector=False)

    @property
    @beartype
    def magnet(self) -> Magnet:
        return self.slot_type.element

    @property
    @beartype
    def candidates(self) -> TCandidates:
        return self.magnet.candidates

    @property
    @beartype
    def vector(self) -> np.ndarray:
        return np_readonly(self._vector)

