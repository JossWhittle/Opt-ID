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


from optid.devices import DeviceSpec
from optid.constants import VECTOR_Z, VECTOR_S, MATRIX_IDENTITY, MATRIX_ROTS_180, MATRIX_ROTZ_180


def ppm_antisymmetric(device_name : str,
                      periods : int, interstice : float, minimum_gap : float,
                      x_size : float, z_size : float,
                      hh_s_size : float, vv_s_size : float,
                      he_s_size : float, ve_s_size : float):

    # Define the Device
    device = DeviceSpec(device_name, x_size=x_size, z_size=z_size)

    # Register Beams
    device.register_beam('TOP', x_offset=(x_size / 2), z_offset=(minimum_gap / 2))
    device.register_beam('BTM', x_offset=(x_size / 2), z_offset=((-z_size) - (minimum_gap / 2)))

    # Register Magnet Types
    device.register_magnet_type('HH', s_size=hh_s_size, field_vector=VECTOR_S, flip_matrix=MATRIX_ROTS_180)
    device.register_magnet_type('HE', s_size=he_s_size, field_vector=VECTOR_S, flip_matrix=MATRIX_ROTS_180)
    device.register_magnet_type('VV', s_size=vv_s_size, field_vector=VECTOR_Z, flip_matrix=MATRIX_ROTZ_180)
    device.register_magnet_type('VE', s_size=ve_s_size, field_vector=VECTOR_Z, flip_matrix=MATRIX_ROTZ_180)

    # Top Beam Prefix
    device.push_magnet(beam='TOP', magnet_type='HE', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
    device.push_magnet(beam='TOP', magnet_type='VE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

    # Bottom Beam Prefix
    device.push_magnet(beam='BTM', magnet_type='HE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
    device.push_magnet(beam='BTM', magnet_type='VE', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

    for period in range(periods):
        # Top Beam Period
        device.push_magnet(beam='TOP', magnet_type='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
        device.push_magnet(beam='TOP', magnet_type='VV', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
        device.push_magnet(beam='TOP', magnet_type='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
        device.push_magnet(beam='TOP', magnet_type='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

        # Bottom Beam Period
        device.push_magnet(beam='BTM', magnet_type='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
        device.push_magnet(beam='BTM', magnet_type='VV', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
        device.push_magnet(beam='BTM', magnet_type='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
        device.push_magnet(beam='BTM', magnet_type='VV', direction_matrix=MATRIX_IDENTITY, spacing=interstice)

    # Top Beam Suffix
    device.push_magnet(beam='TOP', magnet_type='HH', direction_matrix=MATRIX_IDENTITY, spacing=interstice)
    device.push_magnet(beam='TOP', magnet_type='VE', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
    device.push_magnet(beam='TOP', magnet_type='HE', direction_matrix=MATRIX_ROTZ_180)

    # Bottom Beam Suffix
    device.push_magnet(beam='BTM', magnet_type='HH', direction_matrix=MATRIX_ROTZ_180, spacing=interstice)
    device.push_magnet(beam='BTM', magnet_type='VE', direction_matrix=MATRIX_ROTS_180, spacing=interstice)
    device.push_magnet(beam='BTM', magnet_type='HE', direction_matrix=MATRIX_IDENTITY)

    # Assert all Beams contain the same number of magnets
    assert len(set(device.beam_counts.values())) == 1
    # Assert all Beams are the same length along the s-axis
    assert len(set(device.beam_lengths.values())) == 1

    return device
