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

from optid.magnets import MagnetSet
from optid.devices import TwoBeamDeviceSpec, BeamSpec
from optid.constants import VECTOR_Z, VECTOR_S, MATRIX_IDENTITY, MATRIX_ROTS_180, MATRIX_ROTZ_180


class PPMAntisymmetricDeviceSpec(TwoBeamDeviceSpec):

    def __init__(self, name : str, periods : int,
                 hh : MagnetSet, he : MagnetSet, vv : MagnetSet, ve : MagnetSet,
                 interstice : float = 0.03):

        assert isinstance(hh, MagnetSet)
        assert isinstance(he, MagnetSet)
        assert isinstance(vv, MagnetSet)
        assert isinstance(ve, MagnetSet)

        # Register Top Beam Magnets
        beam_top = BeamSpec('TOP', offset=np.array([0, 0, 0], dtype=np.float32))
        offset_top = ((+VECTOR_Z) + (+VECTOR_S)) * 0.5

        beam_top.register_magnet_type('HH', size=hh.reference_size, offset=(hh.reference_size * offset_top),
                                      field_vector=hh.reference_field_vector, flip_matrix=hh.flip_matrix)

        beam_top.register_magnet_type('HE', size=he.reference_size, offset=(he.reference_size * offset_top),
                                      field_vector=he.reference_field_vector, flip_matrix=he.flip_matrix)

        beam_top.register_magnet_type('VV', size=vv.reference_size, offset=(vv.reference_size * offset_top),
                                      field_vector=vv.reference_field_vector, flip_matrix=vv.flip_matrix)

        beam_top.register_magnet_type('VE', size=ve.reference_size, offset=(ve.reference_size * offset_top),
                                      field_vector=ve.reference_field_vector, flip_matrix=ve.flip_matrix)

        # Register Bottom Beam Magnets
        beam_btm = BeamSpec('BTM', offset=np.array([0, 0, 0], dtype=np.float32))
        offset_btm = ((-VECTOR_Z) + (+VECTOR_S)) * 0.5

        beam_btm.register_magnet_type('HH', size=hh.reference_size, offset=(hh.reference_size * offset_btm),
                                      field_vector=hh.reference_field_vector, flip_matrix=hh.flip_matrix)

        beam_btm.register_magnet_type('HE', size=he.reference_size, offset=(he.reference_size * offset_btm),
                                      field_vector=he.reference_field_vector, flip_matrix=he.flip_matrix)

        beam_btm.register_magnet_type('VV', size=vv.reference_size, offset=(vv.reference_size * offset_btm),
                                      field_vector=vv.reference_field_vector, flip_matrix=vv.flip_matrix)

        beam_btm.register_magnet_type('VE', size=ve.reference_size, offset=(ve.reference_size * offset_btm),
                                      field_vector=ve.reference_field_vector, flip_matrix=ve.flip_matrix)

        # Top Beam Prefix
        beam_top.push_magnet(mtype='HE', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
        beam_top.push_magnet(mtype='VE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

        # Bottom Beam Prefix
        beam_btm.push_magnet(mtype='HE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
        beam_btm.push_magnet(mtype='VE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

        for period in range(periods):
            # Top Beam Period
            beam_top.push_magnet(mtype='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
            beam_top.push_magnet(mtype='VV', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
            beam_top.push_magnet(mtype='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
            beam_top.push_magnet(mtype='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

            # Bottom Beam Period
            beam_btm.push_magnet(mtype='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
            beam_btm.push_magnet(mtype='VV', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
            beam_btm.push_magnet(mtype='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
            beam_btm.push_magnet(mtype='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

        # Top Beam Suffix
        beam_top.push_magnet(mtype='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
        beam_top.push_magnet(mtype='VE', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
        beam_top.push_magnet(mtype='HE', direction_matrix=MATRIX_ROTZ_180)

        # Bottom Beam Suffix
        beam_btm.push_magnet(mtype='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
        beam_btm.push_magnet(mtype='VE', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
        beam_btm.push_magnet(mtype='HE', direction_matrix=MATRIX_IDENTITY)

        super().__init__(
            name=name,
            beams={ beam_top.name : beam_top, beam_btm.name : beam_btm },
            magnet_set={ hh.mtype : hh, he.mtype : he, vv.mtype : vv, ve.mtype : ve })


