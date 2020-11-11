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


from functools import partial
import numpy as np

import optid
from optid.magnets import MagnetSet
from optid.spec import PeriodicDeviceSpec
from optid.constants import VECTOR_ZERO, VECTOR_Z, VECTOR_S, MATRIX_IDENTITY, MATRIX_ROTS_180, MATRIX_ROTZ_180

logger = optid.utils.logging.get_logger('optid.devices.PPMAntisymmetricDeviceSpec')


class PPMAntisymmetricDeviceSpec(PeriodicDeviceSpec):

    def __init__(self, name : str, periods : int, interstice : float,
                 hh : MagnetSet, he : MagnetSet, vv : MagnetSet, ve : MagnetSet):
        super().__init__(name=name, periods=periods)

        # Register all the magnet sets
        self.register_magnet_sets(hh, he, vv, ve)
        assert hh.mtype == 'HH'
        assert he.mtype == 'HE'
        assert vv.mtype == 'VV'
        assert ve.mtype == 'VE'

        # Register each beam to position them and define their directions of movement
        self.register_beam('TOP', offset=VECTOR_ZERO, gap_vector=+(VECTOR_Z / 2))
        self.register_beam('BTM', offset=VECTOR_ZERO, gap_vector=-(VECTOR_Z / 2))

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


MagnetSetHorizontal = partial(optid.magnets.MagnetSet,
                              reference_field_vector=VECTOR_S,
                              flip_matrix=MATRIX_ROTS_180)

MagnetSetHH = partial(MagnetSetHorizontal, mtype='HH')

MagnetSetHE = partial(MagnetSetHorizontal, mtype='HE')

MagnetSetVertical = partial(optid.magnets.MagnetSet,
                            reference_field_vector=VECTOR_Z,
                            flip_matrix=MATRIX_ROTZ_180)

MagnetSetVV = partial(MagnetSetVertical, mtype='VV')

MagnetSetVE = partial(MagnetSetVertical, mtype='VE')
