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


# External Imports
import numbers
from beartype import beartype
import numpy as np


# Opt-ID Imports
from .magnet import \
    Magnet

from .pole import \
    Pole

from .slot_type import \
    SlotType

from .device import \
    Device

from ..constants import \
    MATRIX_IDENTITY, MATRIX_ROTX_180


class HybridDevice(Device):

    @beartype
    def __init__(self,
            name: str,
            nperiods: int,
            hh_magnet: Magnet,
            he_magnet: Magnet,
            ht_magnet: Magnet,
            pp_pole: Pole,
            pt_pole: Pole,
            interstice: numbers.Real = 0.0625,
            symmetric: bool = True,
            world_matrix: np.ndarray = MATRIX_IDENTITY):

        super().__init__(name=name, world_matrix=world_matrix)

        if hh_magnet.name != 'HH':
            raise ValueError(f'hh_magnet.name must be HH but is : '
                             f'{hh_magnet.name}')

        if he_magnet.name != 'HE':
            raise ValueError(f'he_magnet.name must be HE but is : '
                             f'{he_magnet.name}')

        if ht_magnet.name != 'HT':
            raise ValueError(f'ht_magnet.name must be HT but is : '
                             f'{ht_magnet.name}')

        if pp_pole.name != 'PP':
            raise ValueError(f'pp_pole.name must be PP but is : '
                             f'{pp_pole.name}')

        if pt_pole.name != 'PT':
            raise ValueError(f'pt_pole.name must be PT but is : '
                             f'{pt_pole.name}')

        hh_top_fwd = SlotType(name='+S', element=hh_magnet, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        hh_top_bwd = SlotType(name='-S', element=hh_magnet, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_ROTX_180)
        he_top_fwd = SlotType(name='+S', element=he_magnet, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        he_top_bwd = SlotType(name='-S', element=he_magnet, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_ROTX_180)
        ht_top_fwd = SlotType(name='+S', element=ht_magnet, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        ht_top_bwd = SlotType(name='-S', element=ht_magnet, anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_ROTX_180)
        pp_top     = SlotType(name='PP', element=pp_pole,   anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)
        pt_top     = SlotType(name='PT', element=pt_pole,   anchor=(0.5, 0, 0.5), direction_matrix=MATRIX_IDENTITY)

        hh_btm_fwd = SlotType(name='+S', element=hh_magnet, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        hh_btm_bwd = SlotType(name='-S', element=hh_magnet, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
        he_btm_fwd = SlotType(name='+S', element=he_magnet, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        he_btm_bwd = SlotType(name='-S', element=he_magnet, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
        ht_btm_fwd = SlotType(name='+S', element=ht_magnet, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        ht_btm_bwd = SlotType(name='-S', element=ht_magnet, anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_ROTX_180)
        pp_btm     = SlotType(name='PP', element=pp_pole,   anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)
        pt_btm     = SlotType(name='PT', element=pt_pole,   anchor=(0.5, 1, 0.5), direction_matrix=MATRIX_IDENTITY)

        self.add_beam(name='TOP', beam_matrix=MATRIX_IDENTITY, gap_vector=(0,  0.5, 0), phase_vector=(0, 0, 0))
        self.add_beam(name='BTM', beam_matrix=MATRIX_IDENTITY, gap_vector=(0, -0.5, 0), phase_vector=(0, 0, 0))

        period = 'START'
        self.add_slots(slot_types={ 'TOP': ht_top_fwd, 'BTM': ht_btm_bwd }, period=period, after_spacing=interstice)
        self.add_slots(slot_types={ 'TOP': pt_top,     'BTM': pt_btm     }, period=period, after_spacing=interstice)
        self.add_slots(slot_types={ 'TOP': he_top_bwd, 'BTM': he_btm_fwd }, period=period, after_spacing=interstice)

        for index in range(nperiods):
            period = f'{index:04d}'
            self.add_slots(slot_types={ 'TOP': pp_top,     'BTM': pp_btm     }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': hh_top_fwd, 'BTM': hh_btm_bwd }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': pp_top,     'BTM': pp_btm     }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': hh_top_bwd, 'BTM': hh_btm_fwd }, period=period, after_spacing=interstice)

        if symmetric:
            period = 'SYM'
            self.add_slots(slot_types={ 'TOP': pp_top, 'BTM': pp_btm }, period=period, after_spacing=interstice)

            period = 'END'
            self.add_slots(slot_types={ 'TOP': he_top_fwd, 'BTM': he_btm_bwd }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': pt_top,     'BTM': pt_btm     }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': ht_top_bwd, 'BTM': ht_btm_fwd }, period=period)

        else:
            period = 'ANTISYM'
            self.add_slots(slot_types={ 'TOP': pp_top,     'BTM': pp_btm     }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': hh_top_fwd, 'BTM': hh_btm_bwd }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': pp_top,     'BTM': pp_btm     }, period=period, after_spacing=interstice)

            period = 'END'
            self.add_slots(slot_types={ 'TOP': he_top_bwd, 'BTM': he_btm_fwd }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': pt_top,     'BTM': pt_btm     }, period=period, after_spacing=interstice)
            self.add_slots(slot_types={ 'TOP': ht_top_fwd, 'BTM': ht_btm_bwd }, period=period)
