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


import typing

import optid
from optid.devices import MagnetSlotSpec, MagnetTypeSpec


logger = optid.utils.logging.get_logger('optid.devices.BeamSpec')


BeamSlotSpec = typing.NamedTuple('BeamSlotSpec', [
    ('name', str),
    ('direction_matrix', optid.types.TensorMatrix),
    ('spacing', float)
])


class BeamSpec:
    """
    Represents an insertion device composed of multiple magnet types in fixed arrangements.
    """

    def __init__(self, name : str, x_offset : float, z_offset : float):

        self._name : str = name
        self._elements : typing.List[BeamSlotSpec] = list()

        self._x_offset : float = x_offset
        self._z_offset : float = z_offset

    @property
    def name(self) -> str:
        return self._name

    @property
    def elements(self) -> typing.List[BeamSlotSpec]:
        return self._elements

    @property
    def x_offset(self) -> float:
        return self._x_offset

    @property
    def z_offset(self) -> float:
        return self._z_offset

    @property
    def count(self) -> int:
        return len(self.elements)

    def calculate_length(self, magnet_types : typing.Dict[str, MagnetTypeSpec]) -> float:
        # Determine the full length of the beam
        s_offset, spacing = 0, 0
        for magnet_spec in self.elements:
            # Apply the offset from the depth of the current magnet and spacing from previous iteration
            type_spec = magnet_types[magnet_spec.name]
            s_offset += (spacing + type_spec.s_size)
            # Note the spacing to apply before the next magnet is added. We defer adding this
            # spacing in case this is the last magnet.
            spacing = magnet_spec.spacing

        if spacing > 0.0:
            logger.warning('Beam [%s] with [%d] magnets has a final spacing of [%f] > 0.0 which has been ignored.',
                           self.name, len(self.elements), spacing)
        return s_offset

    def calculate_slots(self, x_size : float, z_size : float,
                        magnet_types : typing.Dict[str, MagnetTypeSpec]) -> typing.List[MagnetSlotSpec]:

        slots : typing.List[MagnetSlotSpec] = list()

        # Compute the length of this beam and compute the offset to centre it around 0 on the s-axis
        s_centre = (self.calculate_length(magnet_types=magnet_types) / 2)

        # Yield each device slot with the device fully centred around 0 on the s-axis
        s_offset, spacing = 0, 0
        for slot_index, magnet_spec in enumerate(self.elements):
            # Apply spacing from the previous magnet
            s_offset += spacing
            # Yield a fully specified device slot by merging per slot and slot global parameters
            type_spec = magnet_types[magnet_spec.name]
            slots.append(MagnetSlotSpec(beam=self.name, slot=f'S{slot_index:03d}', magnet_type=magnet_spec.name,
                                        x_size=x_size, z_size=z_size, s_size=type_spec.s_size,
                                        x_offset=self.x_offset, z_offset=self.z_offset, s_offset=(s_offset - s_centre),
                                        field_vector=type_spec.field_vector,
                                        direction_matrix=magnet_spec.direction_matrix,
                                        flip_matrix=type_spec.flip_matrix))

            # March forward by the depth of the magnet
            s_offset += type_spec.s_size
            # Note the after magnet spacing to be applied if there is a magnet after this one
            spacing = magnet_spec.spacing

        return slots

    def push_magnet(self, magnet_type : str, direction_matrix : optid.types.TensorMatrix, spacing : float = 0.0):
        assert spacing >= 0.0
        self.elements.append(BeamSlotSpec(name=magnet_type, direction_matrix=direction_matrix, spacing=spacing))

    def pop_magnet(self):
        assert len(self.elements) > 0
        self.elements.pop()
