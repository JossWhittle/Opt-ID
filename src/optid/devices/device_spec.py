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

    def __init__(self, name : str,
                 beams : typing.Dict[str, BeamSpec],
                 magnet_set : typing.Dict[str, MagnetSet]):

        try:
            self._name = validate_string(name, assert_non_empty=True)
        except Exception as ex:
            logger.exception('name must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._beams : typing.Dict[str, BeamSpec] = beams

            for beam_name, beam in self.beams.items():
                validate_string(beam_name, assert_non_empty=True)
                assert beam_name == beam.name
                assert isinstance(beam, BeamSpec)

        except Exception as ex:
            logger.exception('beams must be BeamSpecs with non empty and consistent names', exc_info=ex)
            raise ex

        try:
            self._magnet_set : typing.Dict[str, MagnetSet] = magnet_set

            for mtype, mset in self.magnet_set.items():
                validate_string(mtype, assert_non_empty=True)
                assert mtype == mset.mtype
                assert isinstance(mset, MagnetSet)

        except Exception as ex:
            logger.exception('magnet sets must be MagnetSet with non empty and consistent mtype', exc_info=ex)
            raise ex

    @property
    def name(self) -> str:
        return self._name

    @property
    def beams(self) -> typing.Dict[str, BeamSpec]:
        return self._beams

    @property
    def magnet_set(self) -> typing.Dict[str, MagnetSet]:
        return self._magnet_set

    def calculate_slots(self, *args, **kargs):
        raise NotImplementedError()

    def magnet_slots(self, *args, **kargs):
        slots = list(itertools.chain.from_iterable(self.calculate_slots(*args, **kargs)))

        for mtype in self.magnet_set.keys():
            for slot in slots:
                if slot.mtype != mtype:
                    continue


class TwoBeamDeviceSpec(DeviceSpec):
    """
    Represents an insertion device composed of multiple magnet types in fixed arrangements.
    """

    def calculate_slots(self, z_gap : typing.Union[typing.Tuple[float, float], float]):
        assert 'TOP' in self.beams.keys()
        assert 'BTM' in self.beams.keys()

        if isinstance(z_gap, tuple):
            z_gap_top, z_gap_btm = z_gap
        elif isinstance(z_gap, float):
            z_gap_top, z_gap_btm = +(z_gap / 2.0), +(z_gap / 2.0)
        else:
            raise AssertionError('z_gap must be a float or tuple of two floats')

        assert z_gap_top >= 0
        assert z_gap_btm >= 0

        offset_top = np.array([0, +z_gap_top, 0], dtype=np.float32)
        offset_btm = np.array([0, -z_gap_btm, 0], dtype=np.float32)

        return [
            self.beams['TOP'].calculate_slots(offset=offset_top),
            self.beams['BTM'].calculate_slots(offset=offset_btm),
        ]
