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
from ..device import \
    Beam, Candidate


TVector     = typ.Union[jnp.ndarray, typ.Sequence[numbers.Real]]
TCandidates = typ.Dict[str, typ.Dict[str, Candidate]]
TSlots      = typ.Dict[str, typ.Dict[str, Candidate]]


class Device:

    @beartype
    def __init__(self,
            name: str,
            world_matrix: jnp.ndarray):
        """
        Construct a Device instance.

        :param name:
            String name for the device.

        :param world_matrix:
            Affine matrix for the placing this device into the world.
        """

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        if world_matrix.shape != (4, 4):
            raise ValueError(f'world_matrix must be an affine world_matrix with shape (4, 4) but is : '
                             f'{world_matrix.shape}')

        if world_matrix.dtype != jnp.float32:
            raise TypeError(f'world_matrix must have dtype (float32) but is : '
                            f'{world_matrix.dtype}')

        self._world_matrix = world_matrix

        self._beams = dict()

    @beartype
    def add_beam(self,
            name: str,
            beam_matrix: jnp.ndarray,
            gap_vector: TVector,
            phase_vector: TVector):

        if name in self._beams:
            raise ValueError(f'beams already contains a beam with name : '
                             f'{name}')

        beam = Beam(device=self, name=name, beam_matrix=beam_matrix, gap_vector=gap_vector, phase_vector=phase_vector)

        self._beams[name] = beam
        return beam

    @property
    @beartype
    def beams(self) -> dict:
        return dict(self._beams)

    @property
    @beartype
    def slots_by_type(self) -> TSlots:

        magnet_names = set(slot.slot_type.magnet_type.name for beam in self._beams.values() for slot in beam.slots)

        return { magnet_name: {
                    beam.name: [slot for slot in beam.slots if slot.slot_type.magnet_type.name == magnet_name]
                    for beam in self._beams.values() }
                 for magnet_name in magnet_names }

    @property
    @beartype
    def candidates_by_type(self) -> TCandidates:

        candidates = dict()
        for beam in self._beams.values():
            for slot in beam.slots:
                key = slot.slot_type.magnet_type.name
                if key not in candidates:
                    candidates[key] = dict(slot.candidates)

        return candidates

    @property
    @beartype
    def name(self) -> str:
        return self._name

    @property
    @beartype
    def world_matrix(self) -> jnp.ndarray:
        return self._world_matrix


