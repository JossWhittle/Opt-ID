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
from ..geometry import \
    Geometry

from ..device import \
    MagnetCandidate, MagnetSlot

from ..core.affine import \
    transform_rescaled_vectors

from ..bfield import \
    Bfield


TCandidates = typ.List[MagnetCandidate]
TSlots = typ.List[MagnetSlot]


class MagnetGroup:

    @beartype
    def __init__(self,
            name: str,
            geometry: Geometry,
            vector: jnp.ndarray,
            flip_matrices: jnp.ndarray,
            candidates: TCandidates,
            slots: TSlots):
        """
        Construct a Magnet instance pairing a magnet geometry with a field vector.

        :param name:
            String name for the magnet group.

        :param geometry:
            Geometry instance centred at origin.

        :param vector:
            Field vector for the magnet.

        :param flip_matrices:
            Stack of affine matrices for each flip state.

        :param candidates:
            List of MagnetCandidate objects.

        :param slots:
            List of MagnetSlot objects.
        """

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        self._geometry = geometry

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != jnp.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        self._vector = vector

        if (flip_matrices.ndim != 3) or (flip_matrices.shape[1:] != (4, 4)):
            raise ValueError(f'flip_matrices must be a list of affine matrices with shape (N >= 1, 4, 4) but is : '
                             f'{flip_matrices.shape}')

        if flip_matrices.dtype != jnp.float32:
            raise TypeError(f'flip_matrices must have dtype (float32) but is : '
                            f'{flip_matrices.dtype}')

        self._flip_matrices = flip_matrices

        if len(candidates) == 0:
            raise ValueError(f'candidates must be at least length 1 but is : '
                             f'{len(candidates)}')

        if len(candidates) != len(set(candidate.name for candidate in candidates)):
            raise ValueError(f'candidates must have unique names')

        norm_vector = vector / jnp.linalg.norm(vector)
        for candidate in candidates:
            norm_candidate_vector = candidate.vector / jnp.linalg.norm(candidate.vector)
            if jnp.dot(norm_vector, norm_candidate_vector) > 0.1:
                raise ValueError(f'candidates must point in the same direction as {norm_vector.tolist()} but is : '
                                 f'{norm_candidate_vector.tolist()}')

        self._candidates = candidates

        if len(slots) == 0:
            raise ValueError(f'slots must be at least length 1 but is : '
                             f'{len(slots)}')

        if len(slots) > len(candidates):
            raise ValueError(f'slots must be fewer than candidates ({len(candidates)}) but is : '
                             f'{len(slots)}')

        self._slots = slots

        self._bfield = self.calculate_expected_bfield()

    def calculate_slot_expected_bfield(self, index: int) -> jnp.ndarray:

        slot = self.slot(index)

        # Compute the world space transformation for the magnet slot
        matrix = slot.direction_matrix @ slot.world_matrix

        # Transform the candidates field vector into world space at the magnet slot orientation
        vector = transform_rescaled_vectors(self.vector, matrix)

        # Calculate the bfield contribution for the current candidate in the selected slot
        return slot.lookup.bfield(vector)

    def calculate_expected_bfield(self) -> jnp.ndarray:

        bfield = self.calculate_slot_expected_bfield(0)
        for index in range(1, self.nslot):
            bfield += self.calculate_slot_expected_bfield(index)

        return bfield

    @property
    @beartype
    def name(self) -> str:
        return self._name

    @property
    @beartype
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    @beartype
    def vector(self) -> jnp.ndarray:
        return self._vector

    @property
    @beartype
    def bfield(self) -> jnp.ndarray:
        return self._bfield

    @property
    @beartype
    def nflip(self) -> int:
        return len(self._flip_matrices)

    @beartype
    def flip_matrix(self, flip: int) -> jnp.ndarray:
        return self._flip_matrices[flip]

    @property
    @beartype
    def ncandidate(self) -> int:
        return len(self._candidates)

    @beartype
    def candidate(self, index: int) -> MagnetCandidate:
        return self._candidates[index]

    @property
    @beartype
    def nslot(self) -> int:
        return len(self._slots)

    @beartype
    def slot(self, index: int) -> MagnetSlot:
        return self._slots[index]
