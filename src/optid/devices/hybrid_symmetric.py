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


import numpy as np

import optid
from optid.magnets import MagnetSet
from optid.devices import DeviceSpec
from optid.constants import VECTOR_Z, VECTOR_S, MATRIX_IDENTITY, MATRIX_ROTZ_180

logger = optid.utils.logging.get_logger('optid.devices.HybridSymmetricDeviceSpec')


class HybridSymmetricDeviceSpec(DeviceSpec):

    def __init__(self, name : str, periods : int, interstice : float, pole_size : float, terminal_size : float,
                 hh : MagnetSet, he : MagnetSet, ht : MagnetSet):
        super().__init__(name=name)

        try:
            self._periods = periods
            assert isinstance(self.periods, int)
            assert self.periods > 0

        except Exception as ex:
            logger.exception('periods must be a positive integer', exc_info=ex)
            raise ex

        try:
            self._interstice = interstice
            assert isinstance(self.interstice, float)
            assert self.interstice > 0

        except Exception as ex:
            logger.exception('interstice must be a positive float', exc_info=ex)
            raise ex

        try:
            self._pole_size = pole_size
            assert isinstance(self.pole_size, float)
            assert self.pole_size > 0

        except Exception as ex:
            logger.exception('pole_size must be a positive float', exc_info=ex)
            raise ex

        try:
            self._terminal_size = terminal_size
            assert isinstance(self.terminal_size, float)
            assert self.terminal_size > 0

        except Exception as ex:
            logger.exception('terminal_size must be a positive float', exc_info=ex)
            raise ex

        # Register all the magnet sets
        self.register_magnet_sets(hh, he, ht)

        # Register each beam to position them and define their directions of movement
        self.register_beam('TOP', offset=np.zeros((3,)), gap_vector=+(VECTOR_Z / 2))
        self.register_beam('BTM', offset=np.zeros((3,)), gap_vector=-(VECTOR_Z / 2))

        # Register each magnet type with each beam to set relative offsets
        rel_offset_top = (VECTOR_S + VECTOR_Z) * 0.5
        self.register_magnet_type(beam='TOP', mtype='HH', rel_offset=rel_offset_top)
        self.register_magnet_type(beam='TOP', mtype='HE', rel_offset=rel_offset_top)
        self.register_magnet_type(beam='TOP', mtype='HT', rel_offset=rel_offset_top)

        rel_offset_btm = (VECTOR_S - VECTOR_Z) * 0.5
        self.register_magnet_type(beam='BTM', mtype='HH', rel_offset=rel_offset_btm)
        self.register_magnet_type(beam='BTM', mtype='HE', rel_offset=rel_offset_btm)
        self.register_magnet_type(beam='BTM', mtype='HT', rel_offset=rel_offset_btm)

        # Determine magnet spacings
        spacing_term = ((pole_size / 2) + terminal_size)
        spacing_pole = (pole_size + (interstice * 2))

        # Top Beam Prefix
        self.push_magnet(beam='TOP', mtype='HT', direction_matrix=MATRIX_ROTZ_180, spacing=spacing_term)
        self.push_magnet(beam='TOP', mtype='HE', direction_matrix=MATRIX_IDENTITY, spacing=spacing_pole)

        # Bottom Beam Prefix
        self.push_magnet(beam='BTM', mtype='HT', direction_matrix=MATRIX_IDENTITY, spacing=spacing_term)
        self.push_magnet(beam='BTM', mtype='HE', direction_matrix=MATRIX_ROTZ_180, spacing=spacing_pole)

        for period in range(periods):
            # Top Beam Period
            self.push_magnet(beam='TOP', mtype='HH', direction_matrix=MATRIX_ROTZ_180, spacing=spacing_pole)
            self.push_magnet(beam='TOP', mtype='HH', direction_matrix=MATRIX_IDENTITY, spacing=spacing_pole)

            # Bottom Beam Period
            self.push_magnet(beam='BTM', mtype='HH', direction_matrix=MATRIX_IDENTITY, spacing=spacing_pole)
            self.push_magnet(beam='BTM', mtype='HH', direction_matrix=MATRIX_ROTZ_180, spacing=spacing_pole)

        # Top Beam Suffix
        self.push_magnet(beam='TOP', mtype='HE', direction_matrix=MATRIX_ROTZ_180, spacing=spacing_term)
        self.push_magnet(beam='TOP', mtype='HT', direction_matrix=MATRIX_IDENTITY)

        # Bottom Beam Suffix
        self.push_magnet(beam='BTM', mtype='HE', direction_matrix=MATRIX_IDENTITY, spacing=spacing_term)
        self.push_magnet(beam='BTM', mtype='HT', direction_matrix=MATRIX_ROTZ_180)

    @property
    def periods(self) -> int:
        return self._periods

    @property
    def interstice(self) -> float:
        return self._interstice

    @property
    def pole_size(self) -> float:
        return self._pole_size

    @property
    def terminal_size(self) -> float:
        return self._terminal_size
