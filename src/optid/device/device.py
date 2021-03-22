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
from types import MappingProxyType
from more_itertools import SequenceView
from beartype import beartype
import numbers
import typing as typ
import numpy as np

# Opt-ID Imports
from ..utils.cached import Memoized, cached_property
from ..core.affine import is_scale_preserving
from ..core.utils import np_readonly
from ..bfield import Bfield
from ..lattice import Lattice
from .beam import Beam
from .magnet import Magnet
from .slot import Slot
from .slot_type import SlotType
from .pose import Pose
from .genome import Genome


TVector        = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]
TMagnets       = typ.Mapping[str, Magnet]
TSlots         = typ.Mapping[str, typ.Mapping[str, typ.Sequence[Slot]]]
TPeriodLengths = typ.Mapping[str, numbers.Real]


class Device(Memoized):

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

        # Handle memoized cached parameters
        self.invalidate_cache()

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

        # Handle memoized cached parameters
        self.invalidate_cache()

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

        nslots_by_type = self.nslot_by_type
        for element_name, magnet in self.magnets_by_type.items():

            if nslots_by_type[element_name] > len(magnet.candidates):
                raise ValueError(f'device has more slots of type "{element_name} than candidates : '
                                 f'slots={nslots_by_type[element_name]} > candidates={len(magnet.candidates)}')

    @beartype
    def bfield(self,
            lattice: Lattice,
            pose: Pose) -> Bfield:

        device_field = None
        for beam in self.beams.values():
            beam_field   = beam.bfield(lattice=lattice, pose=pose).field
            device_field = beam_field if (device_field is None) else (device_field + beam_field)

        return Bfield(lattice=lattice, field=device_field)

    @beartype
    def bfield_from_genome(self,
            lattice: Lattice,
            pose: Pose,
            genome: Genome) -> Bfield:

        device_field = None
        for beam in self.beams.values():
            beam_field   = beam.bfield_from_genome(lattice=lattice, pose=pose, genome=genome).field
            device_field = beam_field if (device_field is None) else (device_field + beam_field)

        return Bfield(lattice=lattice, field=device_field)

    @cached_property
    @beartype
    def beams(self) -> typ.Mapping[str, Beam]:
        return MappingProxyType(self._beams)

    @cached_property
    @beartype
    def slots_by_type(self) -> TSlots:

        slots = dict()
        for beam in self.beams.values():
            for element_name, element_slots in beam.slots_by_type.items():

                if element_name not in slots:
                    slots[element_name] = dict()

                slots[element_name][beam.name] = element_slots

        return MappingProxyType({ element_name: MappingProxyType({ beam_name: SequenceView(element_slots)
                                  for beam_name, element_slots in beams.items() })
                                  for element_name, beams in slots.items() })

    @cached_property
    @beartype
    def magnet_slots_by_type(self) -> TSlots:

        slots = dict()
        for beam in self.beams.values():
            for element_name, element_slots in beam.magnet_slots_by_type.items():

                if element_name not in slots:
                    slots[element_name] = dict()

                slots[element_name][beam.name] = element_slots

        return MappingProxyType({ element_name: MappingProxyType({ beam_name: SequenceView(element_slots)
                                  for beam_name, element_slots in beams.items() })
                                  for element_name, beams in slots.items() })

    @cached_property
    @beartype
    def pole_slots_by_type(self) -> TSlots:

        slots = dict()
        for beam in self.beams.values():
            for element_name, element_slots in beam.pole_slots_by_type.items():

                if element_name not in slots:
                    slots[element_name] = dict()

                slots[element_name][beam.name] = element_slots

        return MappingProxyType({ element_name: MappingProxyType({ beam_name: SequenceView(element_slots)
                                  for beam_name, element_slots in beams.items() })
                                  for element_name, beams in slots.items() })

    @cached_property
    @beartype
    def magnets_by_type(self) -> TMagnets:

        magnets = dict()
        for beam in self.beams.values():
            for magnet in beam.magnets_by_type.values():

                if magnet.name not in magnets:
                    magnets[magnet.name] = magnet

        return MappingProxyType(magnets)

    @cached_property
    @beartype
    def period_lengths(self) -> TPeriodLengths:

        period_lengths = dict()
        for beam in self.beams.values():
            for period, length in beam.period_lengths.items():

                if period not in period_lengths:
                    period_lengths[period] = list()

                period_lengths[period].append(length)

        return MappingProxyType({ period: np.mean(period_lengths)
                                  for period, period_lengths in period_lengths.items() })

    @property
    @beartype
    def name(self) -> str:
        return str(self._name)

    @cached_property
    @beartype
    def world_matrix(self) -> np.ndarray:
        return np_readonly(self._world_matrix)

    @cached_property
    @beartype
    def nslot(self) -> int:
        return sum(beam.nslot for beam in self.beams.values())

    @cached_property
    @beartype
    def nslot_by_beam(self) -> typ.Mapping[str, int]:
        return MappingProxyType({ beam.name: beam.nslot
                                  for beam in self.beams.values() })

    @cached_property
    @beartype
    def nslot_by_type(self) -> typ.Mapping[str, int]:

        counts = dict()
        for beam in self.beams.values():
            for slot in beam.slots:

                magnet_name = slot.slot_type.element.name
                if magnet_name not in counts:
                    counts[magnet_name] = 0

                counts[magnet_name] += 1

        return MappingProxyType(counts)

    @cached_property
    @beartype
    def ncandidate(self) -> int:
        return sum(self.ncandidate_by_type.values())

    @cached_property
    @beartype
    def ncandidate_by_type(self) -> typ.Mapping[str, int]:
        return MappingProxyType({ magnet.name: magnet.ncandidate
                                  for magnet in self.magnets_by_type.values() })

    @cached_property
    @beartype
    def length(self) -> numbers.Real:
        return max(beam.length for beam in self.beams.values())

