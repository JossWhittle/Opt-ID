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
import numpy as np

import optid
from optid.devices import MagnetSlotSpec, MagnetTypeSpec
from optid.utils import validate_string, validate_tensor

logger = optid.utils.logging.get_logger('optid.devices.BeamSpec')


BeamSlotSpec = typing.NamedTuple('BeamSlotSpec', [
    ('mtype', str),
    ('direction_matrix', optid.types.TensorMatrix),
    ('spacing', float),
])


class BeamSpec:
    """
    Represents an insertion device composed of multiple magnet types in fixed arrangements.
    """

    def __init__(self,
                 beam : str,
                 offset : optid.types.TensorPoint,
                 gap_vector : optid.types.TensorVector,
                 phase_vector : optid.types.TensorVector):

        try:
            self._name = validate_string(beam, assert_non_empty=True)
        except Exception as ex:
            logger.exception('name must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._offset = validate_tensor(offset, shape=(3,))
        except Exception as ex:
            logger.exception('offset must be a float tensor of shape (3,)', exc_info=ex)
            raise ex

        try:
            self._gap_vector = validate_tensor(gap_vector, shape=(3,))
        except Exception as ex:
            logger.exception('gap_vector must be a float tensor of shape (3,)', exc_info=ex)
            raise ex

        try:
            self._phase_vector = validate_tensor(phase_vector, shape=(3,))
        except Exception as ex:
            logger.exception('phase_vector must be a float tensor of shape (3,)', exc_info=ex)
            raise ex

        self._elements = list()
        self._magnet_types = dict()

    @property
    def name(self) -> str:
        return self._name

    @property
    def elements(self) -> typing.List[BeamSlotSpec]:
        return self._elements

    @property
    def magnet_types(self) -> typing.Dict[str, MagnetTypeSpec]:
        return self._magnet_types

    @property
    def offset(self) -> optid.types.TensorPoint:
        return self._offset

    @property
    def gap_vector(self) -> optid.types.TensorVector:
        return self._gap_vector

    @property
    def phase_vector(self) -> optid.types.TensorVector:
        return self._phase_vector

    @property
    def count(self) -> int:
        return len(self.elements)

    def calculate_length(self) -> float:
        # Determine the full length of the beam
        s_offset, spacing = 0, 0
        for magnet_spec in self.elements:
            # Apply the offset from the depth of the current magnet and spacing from previous iteration
            type_spec = self.magnet_types[magnet_spec.mtype]
            s_offset += (spacing + type_spec.size[2])
            # Note the spacing to apply before the next magnet is added. We defer adding this
            # spacing in case this is the last magnet.
            spacing = magnet_spec.spacing

        if spacing > 0.0:
            logger.warning('Beam [%s] with [%d] magnets has a final spacing of [%f] > 0.0 which has been ignored.',
                           self.name, len(self.elements), spacing)
        return s_offset

    def calculate_slot_specs(self, gap : float, phase : float = 0.0,
                             offset : optid.types.TensorVector = optid.constants.VECTOR_ZERO) \
                             -> typing.List[MagnetSlotSpec]:

        # Compute dynamic offsets
        gap_offset   = self.gap_vector   * gap
        phase_offset = self.phase_vector * phase

        # List of all slot specs
        slots = list()

        # Compute the length of this beam and compute the offset to centre it around 0 on the s-axis
        s_centre = (self.calculate_length() / 2)

        # Yield each device slot with the device fully centred around 0 on the s-axis
        s_relative_offset, spacing = 0, 0
        for slot_index, slot_spec in enumerate(self.elements):
            # Apply spacing from the previous magnet
            s_relative_offset += spacing
            # Yield a fully specified device slot by merging per slot and slot global parameters
            mtype_spec = self.magnet_types[slot_spec.mtype]

            position = sum([self.offset, mtype_spec.offset, offset, gap_offset, phase_offset])
            position[2] += (s_relative_offset - s_centre)

            slots.append(MagnetSlotSpec(beam=self.name, slot=f'S{slot_index:06d}', mtype=slot_spec.mtype,
                                        size=mtype_spec.size, position=position, field_vector=mtype_spec.field_vector,
                                        direction_matrix=slot_spec.direction_matrix, flip_matrix=mtype_spec.flip_matrix,
                                        gap_vector=self.gap_vector, phase_vector=self.phase_vector))

            # March forward by the depth of the magnet
            s_relative_offset += mtype_spec.size[2]
            # Note the after magnet spacing to be applied if there is a magnet after this one
            spacing = slot_spec.spacing

        return slots

    def register_magnet_type(self, mtype : str,
                             size : optid.types.TensorVector,
                             offset : optid.types.TensorPoint,
                             field_vector : optid.types.TensorVector,
                             flip_matrix : optid.types.TensorMatrix):

        validate_string(mtype, assert_non_empty=True)
        assert mtype not in self.magnet_types.keys()

        self.magnet_types[mtype] = MagnetTypeSpec(mtype=mtype, size=size, offset=offset,
                                                  field_vector=field_vector, flip_matrix=flip_matrix)

    def push_magnet(self, mtype : str,
                    direction_matrix : optid.types.TensorMatrix,
                    spacing : float = 0.0):

        validate_string(mtype, assert_non_empty=True)
        assert mtype in self.magnet_types.keys()

        assert spacing >= 0.0
        self.elements.append(BeamSlotSpec(mtype=mtype, direction_matrix=direction_matrix, spacing=spacing))

    def pop_magnet(self):
        assert len(self.elements) > 0
        self.elements.pop()
