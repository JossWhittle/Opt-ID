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
from optid.utils import validate_string
from optid.magnets import MagnetSet, MagnetSlots

logger = optid.utils.logging.get_logger('optid.devices.Device')


class Device:
    """
    Represents an insertion device composed of multiple magnet types in fixed arrangements.
    """

    def __init__(self,
                 name : str,
                 magnet_set : typing.Dict[str, MagnetSet],
                 magnet_slots : typing.Dict[str, MagnetSlots]):

        try:
            self._name = validate_string(name, assert_non_empty=True)
        except Exception as ex:
            logger.exception('mtype must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._magnet_set = magnet_set
            assert isinstance(self.magnet_set, dict)

            for mtype, mset in self.magnet_set.items():
                validate_string(mtype, assert_non_empty=True)
                assert isinstance(mset, MagnetSet)
                assert mtype == mset.mtype

        except Exception as ex:
            logger.exception('magnet_set must be a dictionary of MagnetSet instances', exc_info=ex)
            raise ex

        try:
            self._magnet_slots = magnet_slots
            assert isinstance(self.magnet_slots, dict)

            for mtype, mslots in self.magnet_slots.items():
                validate_string(mtype, assert_non_empty=True)
                assert isinstance(mslots, MagnetSlots)
                assert mtype == mslots.mtype

        except Exception as ex:
            logger.exception('magnet_slots must be a dictionary of MagnetSlots instances', exc_info=ex)
            raise ex

        try:
            for mtype, mslots in self.magnet_slots.items():
                assert mtype in self.magnet_set.keys()

        except Exception as ex:
            logger.exception('magnet_slots must have a matching keyed magnet_set', exc_info=ex)
            raise ex

        try:
            for mtype, mslots in self.magnet_slots.items():
                assert self.magnet_set[mtype].count >= mslots.count

        except Exception as ex:
            logger.exception('magnet_slots must have count less than or equal to its matching magnet_set', exc_info=ex)
            raise ex

    @property
    def name(self) -> str:
        return self._name

    @property
    def magnet_set(self) -> typing.Dict[str, MagnetSet]:
        return self._magnet_set

    @property
    def magnet_slots(self) -> typing.Dict[str, MagnetSlots]:
        return self._magnet_slots
