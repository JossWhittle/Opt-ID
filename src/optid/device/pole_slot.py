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
from .slot import \
    Slot

from .slot_type import \
    SlotType

from .pole import \
    Pole


class PoleSlot(Slot):

    @beartype
    def __init__(self,
                 beam,
                 index: int,
                 name: str,
                 period: str,
                 slot_type: SlotType,
                 slot_matrix: np.ndarray):
        """
        Construct a Slot instance.

        :param beam:
            Parent Beam instance this slot is a member of.

        :param index:
            Integer index for the slot in the beam.

        :param name:
            String name for the slot.

        :param period:
            String period name used for calculating device period length.

        :param slot_type:
            SlotType instance that this slot is one of.

        :param slot_matrix:
            Affine matrix for the placing this slot along its parent beam starting from 0 on the Z axis.
        """
        if not isinstance(slot_type.element, Pole):
            raise TypeError(f'slot_type.element must be type Pole but is : '
                            f'{type(slot_type.element)}')

        super().__init__(beam=beam, index=index, name=name, period=period,
                         slot_type=slot_type, slot_matrix=slot_matrix)
