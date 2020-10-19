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
from optid.constants import VECTOR_Z, VECTOR_S, MATRIX_IDENTITY, MATRIX_ROTS_180, MATRIX_ROTZ_180

logger = optid.utils.logging.get_logger('optid.devices.PPMAntisymmetricDeviceSpec')


class PPMAntisymmetricDeviceSpec(DeviceSpec):

    def __init__(self, name : str, periods : int, interstice : float,
                 hh : MagnetSet, he : MagnetSet, vv : MagnetSet, ve : MagnetSet):
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

        # Register all the magnet sets
        self.register_magnet_sets(hh, he, vv, ve)
        assert hh.mtype == 'HH'
        assert he.mtype == 'HE'
        assert vv.mtype == 'VV'
        assert ve.mtype == 'VE'

        # Register each beam to position them and define their directions of movement
        self.register_beam('TOP', offset=np.zeros((3,)), gap_vector=+(VECTOR_Z / 2))
        self.register_beam('BTM', offset=np.zeros((3,)), gap_vector=-(VECTOR_Z / 2))

        # Register each magnet type with each beam to set relative offsets
        rel_offset_top = (VECTOR_S + VECTOR_Z) * 0.5
        self.register_magnet_type(beam='TOP', mtype='HH', rel_offset=rel_offset_top)
        self.register_magnet_type(beam='TOP', mtype='HE', rel_offset=rel_offset_top)
        self.register_magnet_type(beam='TOP', mtype='VV', rel_offset=rel_offset_top)
        self.register_magnet_type(beam='TOP', mtype='VE', rel_offset=rel_offset_top)

        rel_offset_btm = (VECTOR_S - VECTOR_Z) * 0.5
        self.register_magnet_type(beam='BTM', mtype='HH', rel_offset=rel_offset_btm)
        self.register_magnet_type(beam='BTM', mtype='HE', rel_offset=rel_offset_btm)
        self.register_magnet_type(beam='BTM', mtype='VV', rel_offset=rel_offset_btm)
        self.register_magnet_type(beam='BTM', mtype='VE', rel_offset=rel_offset_btm)

        # Top Beam Prefix
        self.push_magnet(beam='TOP', mtype='HE', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
        self.push_magnet(beam='TOP', mtype='VE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

        # Bottom Beam Prefix
        self.push_magnet(beam='BTM', mtype='HE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
        self.push_magnet(beam='BTM', mtype='VE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

        for period in range(periods):
            # Top Beam Period
            self.push_magnet(beam='TOP', mtype='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
            self.push_magnet(beam='TOP', mtype='VV', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
            self.push_magnet(beam='TOP', mtype='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
            self.push_magnet(beam='TOP', mtype='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

            # Bottom Beam Period
            self.push_magnet(beam='BTM', mtype='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
            self.push_magnet(beam='BTM', mtype='VV', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
            self.push_magnet(beam='BTM', mtype='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
            self.push_magnet(beam='BTM', mtype='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

        # Top Beam Suffix
        self.push_magnet(beam='TOP', mtype='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
        self.push_magnet(beam='TOP', mtype='VE', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
        self.push_magnet(beam='TOP', mtype='HE', direction_matrix=MATRIX_ROTZ_180)

        # Bottom Beam Suffix
        self.push_magnet(beam='BTM', mtype='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
        self.push_magnet(beam='BTM', mtype='VE', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
        self.push_magnet(beam='BTM', mtype='HE', direction_matrix=MATRIX_IDENTITY)

    @property
    def periods(self) -> int:
        return self._periods

    @property
    def interstice(self) -> float:
        return self._interstice
