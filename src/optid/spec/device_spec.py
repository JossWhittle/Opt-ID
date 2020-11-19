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
from optid.spec import BeamSpec, MagnetSlotSpec
from optid.magnets import MagnetSet, MagnetSlots
from optid.utils import validate_string


logger = optid.utils.logging.get_logger('optid.spec.DeviceSpec')


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

        self._compiled     = False
        self._beam_specs   = dict()
        self._device_set   = dict()
        self._device_slots = None
        self._slot_specs   = None

    @property
    def compiled(self) -> bool:
        return self._compiled

    def assert_compiled(self):
        try:
            assert self.compiled
        except Exception as ex:
            logger.exception('device [%s] is not compiled', self.name, exc_info=ex)
            raise ex

    def invalidate_compilation(self):
        self._compiled     = False
        self._device_slots = None
        self._slot_specs   = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def beam_specs(self) -> typing.Dict[str, BeamSpec]:
        return self._beam_specs

    @property
    def device_set(self) -> typing.Dict[str, MagnetSet]:
        return self._device_set

    @property
    def device_slots(self) -> typing.Dict[str, MagnetSlots]:
        self.assert_compiled()
        return self._device_slots

    @property
    def slot_specs(self) -> typing.Dict[str, typing.List[MagnetSlotSpec]]:
        self.assert_compiled()
        return self._slot_specs

    @property
    def mtype_counts(self) -> typing.Dict[str, typing.List[MagnetSlotSpec]]:
        self.assert_compiled()
        return { magnet_slots.mtype : magnet_slots.count for magnet_slots in self.device_slots.values() }

    def register_magnet_sets(self, *args):
        self.invalidate_compilation()

        for arg in args:
            self.register_magnet_set(arg)

    def register_magnet_set(self, magnet_set : MagnetSet):
        self.invalidate_compilation()

        assert isinstance(magnet_set, MagnetSet)
        assert magnet_set.mtype not in self.device_set.keys()

        self.device_set[magnet_set.mtype] = magnet_set

    def register_beam(self, beam : str,
                      offset : optid.types.TensorPoint, gap_vector : optid.types.TensorVector,
                      phase_vector : optid.types.TensorVector = optid.constants.VECTOR_ZERO):
        self.invalidate_compilation()

        validate_string(beam, assert_non_empty=True)
        assert beam not in self.beam_specs.keys()

        self.beam_specs[beam] = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

    def register_magnet_type(self, beam : str, mtype : str,
                             rel_offset : optid.types.TensorPoint):
        self.invalidate_compilation()

        validate_string(beam, assert_non_empty=True)
        assert beam in self.beam_specs.keys()

        validate_string(mtype, assert_non_empty=True)
        assert mtype in self.device_set.keys()

        magnet_set = self.device_set[mtype]
        self.beam_specs[beam].register_magnet_type(mtype=magnet_set.mtype, size=magnet_set.reference_size,
                                                   offset=(magnet_set.reference_size * rel_offset),
                                                   field_vector=magnet_set.reference_field_vector,
                                                   flip_matrix=magnet_set.flip_matrix)

    def push_magnet(self, beam : str, mtype : str,
                    direction_matrix : optid.types.TensorMatrix,
                    spacing : float = 0.0):
        self.invalidate_compilation()

        validate_string(beam, assert_non_empty=True)
        assert beam in self.beam_specs.keys()

        validate_string(mtype, assert_non_empty=True)
        assert mtype in self.device_set.keys()

        self.beam_specs[beam].push_magnet(mtype=mtype, direction_matrix=direction_matrix, spacing=spacing)

    def pop_magnet(self, beam : str):
        self.invalidate_compilation()

        validate_string(beam, assert_non_empty=True)
        assert beam in self.beam_specs.keys()

        self.beam_specs[beam].pop_magnet()

    def calculate_slot_specs(self, gap : float, phase : float = 0.0,
                             offset : optid.types.TensorVector = optid.constants.VECTOR_ZERO):

        beam_keys = sorted(self.beam_specs.keys())
        return { beam : self.beam_specs[beam].calculate_slot_specs(gap=gap, phase=phase, offset=offset)
                 for beam in beam_keys }

    def calculate_device_slots(self, gap : float, phase : float = 0.0,
                               offset : optid.types.TensorVector = optid.constants.VECTOR_ZERO):

        slots_specs = list(itertools.chain.from_iterable(
            self.calculate_slot_specs(gap=gap, phase=phase, offset=offset).values()))

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
        assert len(slots_specs) == sum(magnet_slots.count for magnet_slots in device_slots.values())

        return device_slots

    def compile(self, gap : float, phase : float = 0.0,
                offset : optid.types.TensorVector = optid.constants.VECTOR_ZERO):
        self.invalidate_compilation()
        self._slot_specs   = self.calculate_slot_specs(gap=gap, phase=phase, offset=offset)
        self._device_slots = self.calculate_device_slots(gap=gap, phase=phase, offset=offset)
        self._compiled = True
        return self


class PeriodicDeviceSpec(DeviceSpec):

    def __init__(self, name : str, periods : int):
        super().__init__(name=name)

        try:
            self._periods = periods
            assert isinstance(self.periods, int)
            assert self.periods > 0

        except Exception as ex:
            logger.exception('periods must be a positive integer', exc_info=ex)
            raise ex

    @property
    def periods(self) -> int:
        return self._periods
