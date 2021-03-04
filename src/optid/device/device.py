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


# Opt-ID Imports
from ..core.utils import \
    np_readonly

from .beam import \
    Beam

from .candidate import \
    Candidate

from .slot import \
    Slot

from .slot_type import \
    SlotType

from .magnet_slot import \
    MagnetSlot

TVector     = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]
TCandidates = typ.Dict[str, typ.Dict[str, Candidate]]
TSlots      = typ.Dict[str, typ.Dict[str, Slot]]
TPeriodLengths = typ.Dict[str, numbers.Real]


class Device:

    @beartype
    def __init__(self,
            name: str,
            world_matrix: np.ndarray):
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

        if world_matrix.dtype != np.float32:
            raise TypeError(f'world_matrix must have dtype (float32) but is : '
                            f'{world_matrix.dtype}')

        self._world_matrix = world_matrix

        self._beams = dict()

    @beartype
    def add_beam(self,
            name: str,
            beam_matrix: np.ndarray,
            gap_vector: TVector,
            phase_vector: TVector):

        if name in self._beams:
            raise ValueError(f'beams already contains a beam with name : '
                             f'{name}')

        beam = Beam(device=self, name=name, beam_matrix=beam_matrix, gap_vector=gap_vector, phase_vector=phase_vector)

        self._beams[name] = beam
        return beam

    @beartype
    def add_slots(
            self,
            period: str,
            slot_types: typ.Dict[str, SlotType],
            after_spacing: numbers.Real = 0,
            name: typ.Optional[str] = None):

        if len(slot_types) != len(self.beams):
            raise ValueError(f'slot_types must contain keys for all beams in the device : '
                             f'{len(slot_types)} != {len(self.beams)}')

        for beam in self.beams.values():

            if beam.name not in slot_types:
                raise ValueError(f'slot_types must contain keys for all beams in the device : '
                                 f'{beam.name} not in {list(slot_types.keys())}')

            slot_type = slot_types[beam.name]

            beam.add_slot(period=period, slot_type=slot_type,
                          after_spacing=after_spacing, name=name)

    def validate(self):

        elements = dict()

        for beam in self.beams.values():
            for slot in beam.slots:

                key = slot.slot_type.element.name
                if key not in elements:
                    elements[key] = slot.slot_type.element
                elif elements[key] is not slot.slot_type.element:
                    raise ValueError(f'multiple magnets with same name refer to different objects : '
                                     f'{key}, {slot.qualified_name}')

        nslots_by_type = self.nslots_by_type
        for element_name, candidates in self.candidates_by_type.items():

            if nslots_by_type[element_name] > len(candidates):
                raise ValueError(f'device has more slots of type "{element_name} than candidates : '
                                 f'slots={nslots_by_type[element_name]} > candidates={len(candidates)}')

    @property
    @beartype
    def beams(self) -> dict:
        return dict(self._beams)

    @property
    @beartype
    def slots_by_type(self) -> TSlots:

        slots = dict()
        for beam in self.beams.values():
            for slot in beam.slots:

                element_name = slot.slot_type.element.name
                if element_name not in slots:
                    slots[element_name] = dict()

                if beam.name not in slots[element_name]:
                    slots[element_name][beam.name] = list()

                slots[element_name][beam.name].append(slot)

        return slots

    @property
    @beartype
    def candidates_by_type(self) -> TCandidates:

        candidates = dict()
        for beam in self.beams.values():
            for slot in beam.slots:

                if not isinstance(slot, MagnetSlot):
                    continue

                key = slot.slot_type.element.name
                if key not in candidates:
                    candidates[key] = dict(slot.candidates)

        return candidates

    @property
    @beartype
    def period_lengths(self) -> TPeriodLengths:

        period_lengths = dict()
        for beam in self.beams.values():

            for period, length in beam.period_lengths.items():

                if period not in period_lengths:
                    period_lengths[period] = (length, 1)
                else:
                    cur_length, cur_count = period_lengths[period]
                    period_lengths[period] = ((cur_length + length), (cur_count + 1))

        return { period: (length / count) for period, (length, count) in period_lengths.items() }

    @property
    @beartype
    def name(self) -> str:
        return str(self._name)

    @property
    @beartype
    def world_matrix(self) -> np.ndarray:
        return np_readonly(self._world_matrix)

    @property
    @beartype
    def nslots(self) -> int:
        return sum(beam.nslots for beam in self.beams.values())

    @property
    @beartype
    def nslots_by_beam(self) -> typ.Dict[str, int]:
        return { beam.name: beam.nslots for beam in self.beams.values() }

    @property
    @beartype
    def nslots_by_type(self) -> typ.Dict[str, int]:

        counts = dict()
        for beam in self.beams.values():
            for slot in beam.slots:

                magnet_name = slot.slot_type.element.name
                if magnet_name not in counts:
                    counts[magnet_name] = 0

                counts[magnet_name] += 1

        return counts

    @property
    @beartype
    def length(self) -> numbers.Real:
        return max(beam.length for beam in self.beams.values())

