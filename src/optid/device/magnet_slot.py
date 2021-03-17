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
import numbers
import typing as typ
import numpy as np
import radia as rad


# Opt-ID Imports
from ..constants import VECTOR_ZERO
from ..core.affine import transform_rescaled_vectors
from ..core.bfield import radia_evaluate_bfield_on_lattice
from ..lattice import Lattice
from ..bfield import Bfield
from .pose import Pose
from .slot import Slot
from .slot_type import SlotType
from .magnet import Magnet
from .candidate import Candidate
from .state import State


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

    @beartype
    def world_matrix(self,
            pose: Pose,
            shim: TVector = VECTOR_ZERO,
            flip: int = 0) -> np.ndarray:
        """
        Calculate the affine matrix that places this magnet slot into the world in the correct major
        orientation except flip state.

        :param pose:
            Device Pose instance specifying gap and phase.

        :param shim:
            Shimming amount in XZS in the magnet aligned reference frame.

        :param flip:
            Flip matrix to use.

        :return:
            Affine matrix representing the major position of the slot in world space.
        """

        return self.magnet.flip_matrix(flip) @ \
               super().world_matrix(pose=pose, shim=shim)

    @beartype
    def bfield_from_vector(self,
            lattice: Lattice,
            vector: TVector,
            pose: Pose,
            shim: TVector = VECTOR_ZERO,
            flip: int = 0,
            world_vector: bool = True) -> Bfield:

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != np.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        matrix   = self.world_matrix(pose=pose, shim=shim, flip=flip)
        geometry = self.geometry.transform(matrix)

        if not world_vector:
            vector = transform_rescaled_vectors(vector, matrix)

        rad.UtiDelAll()
        bfield = radia_evaluate_bfield_on_lattice(geometry.to_radia(vector), lattice.world_lattice)
        rad.UtiDelAll()
        return Bfield(lattice=lattice, values=bfield)

    @beartype
    def bfield(self,
            lattice: Lattice,
            pose: Pose,
            shim: TVector = VECTOR_ZERO,
            flip: int = 0) -> Bfield:

        return self.bfield_from_vector(lattice=lattice, pose=pose, shim=shim, flip=flip,
                                       vector=self.magnet.vector, world_vector=False)

    @beartype
    def bfield_from_state(self,
            lattice: Lattice,
            pose: Pose,
            state: State) -> Bfield:

        if state.slot != self.qualified_name:
            raise ValueError(f'state must be assigned to this slot but is : '
                             f'slot={self.qualified_name} state={state.slot}')

        if state.candidate not in self.candidates:
            raise ValueError(f'state must be assigned to a valid candidate for this slot but is : '
                             f'{state.candidate}')

        candidate = self.candidates[state.candidate]

        return self.bfield_from_vector(lattice=lattice, pose=pose, shim=state.shim, flip=state.flip,
                                       vector=candidate.vector, world_vector=False)

    @property
    @beartype
    def magnet(self) -> Magnet:
        return self.slot_type.element

    @property
    @beartype
    def candidates(self) -> TCandidates:
        return self.magnet.candidates

