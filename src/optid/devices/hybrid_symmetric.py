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

import optid
from optid.magnets import MagnetSet
from optid.spec import DeviceSpec
from optid.constants import VECTOR_ZERO, VECTOR_Z, VECTOR_S, MATRIX_IDENTITY, MATRIX_ROTZ_180, MATRIX_ROTS_180

logger = optid.utils.logging.get_logger('optid.devices.HybridSymmetricDeviceSpec')


class HybridSymmetricDeviceSpec(DeviceSpec):

    def __init__(self, name : str, periods : int, interstice : float, pole_size : float, terminal_size : float,
                 hh : MagnetSet, he : MagnetSet, ht : MagnetSet):
        super().__init__(name=name, periods=periods)

        # Register all the magnet sets
        assert hh.mtype == 'HH'
        assert he.mtype == 'HE'
        assert ht.mtype == 'HT'
        self.register_magnet_sets(hh, he, ht)

        # Register each beam to position them and define their directions of movement
        self.register_beam('TOP', offset=VECTOR_ZERO, gap_vector=+(VECTOR_Z / 2))
        self.register_beam('BTM', offset=VECTOR_ZERO, gap_vector=-(VECTOR_Z / 2))

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
        self.push_magnet(beam='TOP', mtype='HT', period=None,
                         direction_matrix=MATRIX_ROTZ_180, spacing=spacing_term)
        self.push_magnet(beam='TOP', mtype='HE', period=None,
                         direction_matrix=MATRIX_IDENTITY, spacing=spacing_pole)

        # Bottom Beam Prefix
        self.push_magnet(beam='BTM', mtype='HT', period=None,
                         direction_matrix=MATRIX_IDENTITY, spacing=spacing_term)
        self.push_magnet(beam='BTM', mtype='HE', period=None,
                         direction_matrix=MATRIX_ROTZ_180, spacing=spacing_pole)

        for period in range(periods):
            # Top Beam Period
            self.push_magnet(beam='TOP', mtype='HH', period=period,
                             direction_matrix=MATRIX_ROTZ_180, spacing=spacing_pole)
            self.push_magnet(beam='TOP', mtype='HH', period=period,
                             direction_matrix=MATRIX_IDENTITY, spacing=spacing_pole)

            # Bottom Beam Period
            self.push_magnet(beam='BTM', mtype='HH', period=period,
                             direction_matrix=MATRIX_IDENTITY, spacing=spacing_pole)
            self.push_magnet(beam='BTM', mtype='HH', period=period,
                             direction_matrix=MATRIX_ROTZ_180, spacing=spacing_pole)

        # Top Beam Suffix
        self.push_magnet(beam='TOP', mtype='HE', period=None,
                         direction_matrix=MATRIX_ROTZ_180, spacing=spacing_term)
        self.push_magnet(beam='TOP', mtype='HT', period=None,
                         direction_matrix=MATRIX_IDENTITY)

        # Bottom Beam Suffix
        self.push_magnet(beam='BTM', mtype='HE', period=None,
                         direction_matrix=MATRIX_IDENTITY, spacing=spacing_term)
        self.push_magnet(beam='BTM', mtype='HT', period=None,
                         direction_matrix=MATRIX_ROTZ_180)


# Helpers for horizontal magnets
MagnetSetH  = partial(optid.magnets.MagnetSet, field_vector=VECTOR_S, flip_matrix=MATRIX_ROTS_180)
MagnetSetHH = partial(MagnetSetH, mtype='HH')
MagnetSetHE = partial(MagnetSetH, mtype='HE')
MagnetSetHT = partial(MagnetSetH, mtype='HT')

MagnetSetH_from_sim_file  = partial(optid.magnets.MagnetSet.from_sim_file,
                                    field_vector=VECTOR_S, flip_matrix=MATRIX_ROTS_180)
MagnetSetHH_from_sim_file = partial(MagnetSetH_from_sim_file, mtype='HH')
MagnetSetHE_from_sim_file = partial(MagnetSetH_from_sim_file, mtype='HE')
MagnetSetHT_from_sim_file = partial(MagnetSetH_from_sim_file, mtype='HT')
