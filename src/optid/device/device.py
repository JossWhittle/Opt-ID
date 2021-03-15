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
import pandas as pd
import pandera as pa


# Opt-ID Imports
from ..core.affine import \
    is_scale_preserving

from ..core.utils import \
    np_readonly

from .beam import \
    Beam

from optid.device import \
    Magnet

from .slot import \
    Slot

from .slot_type import \
    SlotType


TVector     = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]
TMagnets    = typ.Dict[str, Magnet]
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

        if not is_scale_preserving(world_matrix):
            raise ValueError(f'world_matrix must be an affine world_matrix that preserves scale')

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
        for element_name, magnet in self.magnets_by_type.items():

            if nslots_by_type[element_name] > len(magnet.candidates):
                raise ValueError(f'device has more slots of type "{element_name} than candidates : '
                                 f'slots={nslots_by_type[element_name]} > candidates={len(magnet.candidates)}')

    # @beartype
    # def genome_from_dataframe(self,
    #         df: pd.DataFrame,
    #         slot: str = 'slot',
    #         candidate: str = 'candidate',
    #         flip: str = 'flip'):
    #
    #     schema = pa.DataFrameSchema({
    #         slot:      pa.Column(pa.String, pa.Check((lambda col: (len(col.unique()) == len(col))),
    #                              error='slot names must be unique or null'), nullable=True, coerce=True),
    #         candidate: pa.Column(pa.String, pa.Check((lambda col: (len(col.unique()) == len(col))),
    #                              error='candidate names must be unique'), coerce=True),
    #         flip:      pa.Column(pa.Int, pa.Check((lambda col: col >= 0),
    #                              error='flip states must be >= 0'), coerce=True),
    #     })
    #
    #     df = schema.validate(df)
    #
    #     df_nslots      = len(df[~df['slot'].isnull()])
    #     df_ncandidates = len(df)
    #
    #     if df_nslots != self.nslots:
    #         raise ValueError(f'dataframe must have the same number of slots as this device : '
    #                          f'df={df_nslots} != device={self.nslots}')
    #
    #     if df_ncandidates != self.ncandidates:
    #         raise ValueError(f'dataframe must have the same number of candidates as this device : '
    #                          f'df={df_ncandidates} != device={self.ncandidates}')
    #
    #     states = dict()
    #     for beam in self.beams.values():
    #
    #         states[beam.name] = list()
    #         for slot in beam.slots:
    #
    #             if not isinstance(slot, MagnetSlot):
    #                 continue
    #
    #             df_slot = df[(df['slot'] == slot.qualified_name)]
    #
    #             if len(df_slot) != 1:
    #                 raise ValueError(f'dataframe is missing a slot named in this device : '
    #                                  f'{slot.qualified_name}')
    #
    #             df_slot = df_slot.iloc[0]
    #
    #             flip = df_slot['flip']
    #
    #             if flip < 0 or flip >= slot.magnet.nflip:
    #                 raise ValueError(f'flip state for slot is outside the value range in the device : '
    #                                  f'df={flip} device=[0, {slot.magnet.nflip})')
    #
    #             states[beam.name].append(State(slot=df_slot['slot'], candidate=df_slot['candidate'], flip=flip))
    #
    #     unused = list()
    #     for :
    #         unused.append(State(slot=None, candidate=df_slot['candidate'], flip=flip))
    #
    #     return Genome(states=states)

    @property
    @beartype
    def beams(self) -> dict:
        return dict(self._beams)

    @property
    @beartype
    def slots_by_type(self) -> TSlots:

        slots = dict()
        for beam in self.beams.values():
            for element_name, element_slots in beam.slots_by_type.items():

                if element_name not in slots:
                    slots[element_name] = dict()

                slots[element_name][beam.name] = element_slots

        return slots

    @property
    @beartype
    def magnet_slots_by_type(self) -> TSlots:

        slots = dict()
        for beam in self.beams.values():
            for element_name, element_slots in beam.magnet_slots_by_type.items():

                if element_name not in slots:
                    slots[element_name] = dict()

                slots[element_name][beam.name] = element_slots

        return slots

    @property
    @beartype
    def pole_slots_by_type(self) -> TSlots:

        slots = dict()
        for beam in self.beams.values():
            for element_name, element_slots in beam.pole_slots_by_type.items():

                if element_name not in slots:
                    slots[element_name] = dict()

                slots[element_name][beam.name] = element_slots

        return slots

    @property
    @beartype
    def magnets_by_type(self) -> TMagnets:

        magnets = dict()
        for beam in self.beams.values():
            for magnet in beam.magnets_by_type.values():

                if magnet.name not in magnets:
                    magnets[magnet.name] = magnet

        return magnets

    @property
    @beartype
    def period_lengths(self) -> TPeriodLengths:

        period_lengths = dict()
        for beam in self.beams.values():
            for period, length in beam.period_lengths.items():

                if period not in period_lengths:
                    period_lengths[period] = list()

                period_lengths[period].append(length)

        return { period: np.mean(period_lengths)
                 for period, period_lengths in period_lengths.items() }

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
    def ncandidates(self) -> int:
        return sum(self.ncandidates_by_type.values())

    @property
    @beartype
    def ncandidates_by_type(self) -> typ.Dict[str, int]:
        return { key: len(magnet.candidates) for key, magnet in self.magnets_by_type.items() }

    @property
    @beartype
    def length(self) -> numbers.Real:
        return max(beam.length for beam in self.beams.values())

