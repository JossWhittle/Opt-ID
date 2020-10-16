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
import itertools
import numpy as np

import optid
from optid.devices import BeamSpec
from optid.magnets import MagnetSet, MagnetSlots
from optid.utils import validate_string


logger = optid.utils.logging.get_logger('optid.devices.DeviceSpec')


class DeviceSpec:
    """
    Represents an insertion device composed of multiple magnet types in fixed arrangements.
    """

    def __init__(self, name : str):

        try:
            self._name = validate_string(name, assert_non_empty=True)
        except Exception as ex:
            logger.exception('beam must be a non-empty string', exc_info=ex)
            raise ex

        self._device_set = dict()
        self._beams = dict()

    @property
    def name(self) -> str:
        return self._name

    @property
    def beams(self) -> typing.Dict[str, BeamSpec]:
        return self._beams

    @property
    def device_set(self) -> typing.Dict[str, MagnetSet]:
        return self._device_set

    def register_magnet_sets(self, *args):
        for arg in args:
            self.register_magnet_set(arg)

    def register_magnet_set(self, magnet_set : MagnetSet):

        assert isinstance(magnet_set, MagnetSet)
        assert magnet_set.mtype not in self.device_set.keys()

        self.device_set[magnet_set.mtype] = magnet_set

    def register_beam(self, beam : str,
                      offset : optid.types.TensorPoint, gap_vector : optid.types.TensorVector,
                      phase_vector : optid.types.TensorVector = optid.constants.VECTOR_ZERO) -> BeamSpec:

        assert isinstance(beam, str)
        assert beam not in self.beams.keys()

        self.beams[beam] = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)
        return self.beams[beam]

    def register_magnet_type(self, beam : str, mtype : str,
                             rel_offset : optid.types.TensorPoint):

        assert isinstance(beam, str)
        assert beam in self.beams.keys()

        assert isinstance(mtype, str)
        assert mtype in self.device_set.keys()

        magnet_set = self.device_set[mtype]
        self.beams[beam].register_magnet_type(mtype=magnet_set.mtype, size=magnet_set.reference_size,
                                              offset=(magnet_set.reference_size * rel_offset),
                                              field_vector=magnet_set.reference_field_vector,
                                              flip_matrix=magnet_set.flip_matrix)

    def push_magnet(self, beam : str, mtype : str,
                    direction_matrix : optid.types.TensorMatrix,
                    spacing : float = 0.0):

        assert isinstance(beam, str)
        assert beam in self.beams.keys()

        assert isinstance(mtype, str)
        assert mtype in self.device_set.keys()

        self.beams[beam].push_magnet(mtype=mtype, direction_matrix=direction_matrix, spacing=spacing)

    def pop_magnet(self, beam : str):

        assert isinstance(beam, str)
        assert beam in self.beams.keys()

        self.beams[beam].pop_magnet()

    def calculate_slot_specs(self, gap : float, phase : float = 0.0,
                             offset : optid.types.TensorVector = optid.constants.VECTOR_ZERO):

        return { beam.name : beam.calculate_slot_specs(gap=gap, phase=phase, offset=offset)
                 for beam in self.beams.values() }

    def device_slots(self, gap : float, phase : float = 0.0,
                     offset : optid.types.TensorVector = optid.constants.VECTOR_ZERO):

        slots_specs = list(itertools.chain.from_iterable(
            self.calculate_slot_specs(gap=gap, phase=phase, offset=offset)))

        device_slots = dict()
        for mtype in self.device_set.keys():

            mtype_slots_specs  = list(filter((lambda s : s.mtype == mtype), slots_specs))
            beams              = [s.beam for s in mtype_slots_specs]
            slots              = [s.slot for s in mtype_slots_specs]
            positions          = np.stack([s.position         for s in mtype_slots_specs], axis=0)
            gap_vectors        = np.stack([s.gap_vector       for s in mtype_slots_specs], axis=0)
            direction_matrices = np.stack([s.direction_matrix for s in mtype_slots_specs], axis=0)

            device_slots[mtype] = MagnetSlots(mtype=mtype, beams=beams, slots=slots, positions=positions,
                                              gap_vectors=gap_vectors, direction_matrices=direction_matrices)

        # Assert that we extracted every magnet slot spec
        assert len(slots_specs) == sum(magnet_slots.count for magnet_slots in device_slots)

        return device_slots
