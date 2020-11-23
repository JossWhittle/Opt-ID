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
from optid.constants import VECTOR_X, VECTOR_Z, VECTOR_S, \
                            MATRIX_IDENTITY, MATRIX_ROTS_180, MATRIX_ROTZ_180, MATRIX_ROTS_90, \
                            MATRIX_ROTS_270, MATRIX_ROTX_180, MATRIX_ROTS_270_ROTZ_180, MATRIX_ROTS_270_ROTX_180

logger = optid.utils.logging.get_logger('optid.devices.APPLESymmetricDeviceSpec')


class APPLESymmetricDeviceSpec(DeviceSpec):

    def __init__(self, name : str, periods : int, interstice : float, phasing_interstice : float, terminal_size : float,
                 hh : MagnetSet, he : MagnetSet, vv : MagnetSet, ve : MagnetSet):
        super().__init__(name=name, periods=periods)

        # Register all the magnet sets
        assert hh.mtype == 'HH'
        assert he.mtype == 'HE'
        assert vv.mtype == 'VV'
        assert ve.mtype == 'VE'
        self.register_magnet_sets(hh, he, vv, ve)

        # Register each beam to position them and define their directions of movement
        abs_offset_left  = VECTOR_X *  (phasing_interstice / 2)
        abs_offset_right = VECTOR_X * -(phasing_interstice / 2)
        self.register_beam('Q1', offset=abs_offset_left,  gap_vector=+(VECTOR_Z / 2), phase_vector=(VECTOR_S * -0.5))
        self.register_beam('Q2', offset=abs_offset_right, gap_vector=+(VECTOR_Z / 2), phase_vector=(VECTOR_S * +0.5))
        self.register_beam('Q3', offset=abs_offset_right, gap_vector=-(VECTOR_Z / 2), phase_vector=(VECTOR_S * -0.5))
        self.register_beam('Q4', offset=abs_offset_left,  gap_vector=-(VECTOR_Z / 2), phase_vector=(VECTOR_S * +0.5))

        # Register each magnet type with each beam to set relative offsets
        rel_offset_q1 = (VECTOR_S + VECTOR_Z + VECTOR_X) * 0.5
        self.register_magnet_type(beam='Q1', mtype='HH', rel_offset=rel_offset_q1)
        self.register_magnet_type(beam='Q1', mtype='HE', rel_offset=rel_offset_q1)
        self.register_magnet_type(beam='Q1', mtype='VV', rel_offset=rel_offset_q1)
        self.register_magnet_type(beam='Q1', mtype='VE', rel_offset=rel_offset_q1)

        rel_offset_q2 = (VECTOR_S + VECTOR_Z - VECTOR_X) * 0.5
        self.register_magnet_type(beam='Q2', mtype='HH', rel_offset=rel_offset_q2)
        self.register_magnet_type(beam='Q2', mtype='HE', rel_offset=rel_offset_q2)
        self.register_magnet_type(beam='Q2', mtype='VV', rel_offset=rel_offset_q2)
        self.register_magnet_type(beam='Q2', mtype='VE', rel_offset=rel_offset_q2)

        rel_offset_q3 = (VECTOR_S - VECTOR_Z - VECTOR_X) * 0.5
        self.register_magnet_type(beam='Q3', mtype='HH', rel_offset=rel_offset_q3)
        self.register_magnet_type(beam='Q3', mtype='HE', rel_offset=rel_offset_q3)
        self.register_magnet_type(beam='Q3', mtype='VV', rel_offset=rel_offset_q3)
        self.register_magnet_type(beam='Q3', mtype='VE', rel_offset=rel_offset_q3)

        rel_offset_q4 = (VECTOR_S - VECTOR_Z + VECTOR_X) * 0.5
        self.register_magnet_type(beam='Q4', mtype='HH', rel_offset=rel_offset_q4)
        self.register_magnet_type(beam='Q4', mtype='HE', rel_offset=rel_offset_q4)
        self.register_magnet_type(beam='Q4', mtype='VV', rel_offset=rel_offset_q4)
        self.register_magnet_type(beam='Q4', mtype='VE', rel_offset=rel_offset_q4)

        # Q1 Beam Prefix
        self.push_magnet(beam='Q1', mtype='HE', direction_matrix=MATRIX_IDENTITY, spacing=terminal_size)
        self.push_magnet(beam='Q1', mtype='VE', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
        self.push_magnet(beam='Q1', mtype='HE', direction_matrix=MATRIX_ROTS_270_ROTZ_180, spacing=interstice)

        # Q2 Beam Prefix
        self.push_magnet(beam='Q2', mtype='HE', direction_matrix=MATRIX_ROTS_270, spacing=terminal_size)
        self.push_magnet(beam='Q2', mtype='VE', direction_matrix=MATRIX_ROTX_180, spacing=interstice)
        self.push_magnet(beam='Q2', mtype='HE', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)

        # Q3 Beam Prefix
        self.push_magnet(beam='Q3', mtype='HE', direction_matrix=MATRIX_ROTS_270_ROTX_180, spacing=terminal_size)
        self.push_magnet(beam='Q3', mtype='VE', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
        self.push_magnet(beam='Q3', mtype='HE', direction_matrix=MATRIX_ROTS_180, spacing=interstice)

        # Q4 Beam Prefix
        self.push_magnet(beam='Q4', mtype='HE', direction_matrix=MATRIX_ROTX_180, spacing=terminal_size)
        self.push_magnet(beam='Q4', mtype='VE', direction_matrix=MATRIX_ROTX_180, spacing=interstice)
        self.push_magnet(beam='Q4', mtype='HE', direction_matrix=MATRIX_ROTS_90, spacing=interstice)

        for period in range(periods):
            # Q1 Beam Period
            self.push_magnet(beam='Q1', mtype='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
            self.push_magnet(beam='Q1', mtype='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
            self.push_magnet(beam='Q1', mtype='VV', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
            self.push_magnet(beam='Q1', mtype='HH', direction_matrix=MATRIX_ROTS_270_ROTZ_180, spacing=interstice)

            # Q2 Beam Period
            self.push_magnet(beam='Q2', mtype='VV', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
            self.push_magnet(beam='Q2', mtype='HH', direction_matrix=MATRIX_ROTS_270, spacing=interstice)
            self.push_magnet(beam='Q2', mtype='VV', direction_matrix=MATRIX_ROTX_180, spacing=interstice)
            self.push_magnet(beam='Q2', mtype='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)

            # Q3 Beam Period
            self.push_magnet(beam='Q3', mtype='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
            self.push_magnet(beam='Q3', mtype='HH', direction_matrix=MATRIX_ROTS_270_ROTX_180, spacing=interstice)
            self.push_magnet(beam='Q3', mtype='VV', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
            self.push_magnet(beam='Q3', mtype='HH', direction_matrix=MATRIX_ROTS_180, spacing=interstice)

            # Q4 Beam Period
            self.push_magnet(beam='Q4', mtype='VV', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
            self.push_magnet(beam='Q4', mtype='HH', direction_matrix=MATRIX_ROTX_180, spacing=interstice)
            self.push_magnet(beam='Q4', mtype='VV', direction_matrix=MATRIX_ROTX_180, spacing=interstice)
            self.push_magnet(beam='Q4', mtype='HH', direction_matrix=MATRIX_ROTS_90, spacing=interstice)

        # Q1 Beam Suffix
        self.push_magnet(beam='Q1', mtype='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
        self.push_magnet(beam='Q1', mtype='HE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
        self.push_magnet(beam='Q1', mtype='VE', direction_matrix=MATRIX_ROTS_180, spacing=terminal_size)
        self.push_magnet(beam='Q1', mtype='HE', direction_matrix=MATRIX_ROTS_270_ROTZ_180)

        # Q2 Beam Suffix
        self.push_magnet(beam='Q2', mtype='VV', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
        self.push_magnet(beam='Q2', mtype='HE', direction_matrix=MATRIX_ROTS_270, spacing=interstice)
        self.push_magnet(beam='Q2', mtype='VE', direction_matrix=MATRIX_ROTX_180, spacing=terminal_size)
        self.push_magnet(beam='Q2', mtype='HE', direction_matrix=MATRIX_ROTZ_180)

        # Q3 Beam Suffix
        self.push_magnet(beam='Q3', mtype='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
        self.push_magnet(beam='Q3', mtype='HE', direction_matrix=MATRIX_ROTS_270_ROTX_180, spacing=interstice)
        self.push_magnet(beam='Q3', mtype='VE', direction_matrix=MATRIX_ROTS_180, spacing=terminal_size)
        self.push_magnet(beam='Q3', mtype='HE', direction_matrix=MATRIX_ROTS_180)

        # Q4 Beam Suffix
        self.push_magnet(beam='Q4', mtype='VV', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
        self.push_magnet(beam='Q4', mtype='HE', direction_matrix=MATRIX_ROTX_180, spacing=interstice)
        self.push_magnet(beam='Q4', mtype='VE', direction_matrix=MATRIX_ROTX_180, spacing=terminal_size)
        self.push_magnet(beam='Q4', mtype='HE', direction_matrix=MATRIX_ROTS_90)


# Helpers for horizontal magnets
MagnetSetH  = partial(optid.magnets.MagnetSet, field_vector=VECTOR_S, flip_matrix=MATRIX_ROTS_180)
MagnetSetHH = partial(MagnetSetH, mtype='HH')
MagnetSetHE = partial(MagnetSetH, mtype='HE')

MagnetSetH_from_sim_file  = partial(optid.magnets.MagnetSet.from_sim_file,
                                    field_vector=VECTOR_S, flip_matrix=MATRIX_ROTS_180)
MagnetSetHH_from_sim_file = partial(MagnetSetH_from_sim_file, mtype='HH')
MagnetSetHE_from_sim_file = partial(MagnetSetH_from_sim_file, mtype='HE')

# Helpers for vertical magnets
MagnetSetV  = partial(optid.magnets.MagnetSet, field_vector=VECTOR_Z, flip_matrix=MATRIX_IDENTITY)
MagnetSetVV = partial(MagnetSetV, mtype='VV')
MagnetSetVE = partial(MagnetSetV, mtype='VE')

MagnetSetV_from_sim_file  = partial(optid.magnets.MagnetSet.from_sim_file,
                                    field_vector=VECTOR_Z, flip_matrix=MATRIX_IDENTITY)
MagnetSetVV_from_sim_file = partial(MagnetSetH_from_sim_file, mtype='VV')
MagnetSetVE_from_sim_file = partial(MagnetSetH_from_sim_file, mtype='VE')
