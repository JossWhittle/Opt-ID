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
from beartype import beartype
import numbers
import typing as typ
import numpy as np


# Opt-ID Imports
from ..constants import VECTOR_ZERO
from ..core.utils import np_readonly


TVector = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]


class State:

    @beartype
    def __init__(self,
            slot: typ.Optional[str],
            candidate: str,
            flip: int):
        """
        Construct a State instance.

        :param slot:
            String name for the slot if this state is assigned to one.

        :param candidate:
            String name for the candidate.

        :param flip:
            Integer flip state for the selection.
        """

        if (slot is not None) and (len(slot) == 0):
            raise ValueError(f'slot must be a non-empty string')

        self._slot = slot

        if len(candidate) == 0:
            raise ValueError(f'candidate must be a non-empty string')

        self._candidate = candidate

        if flip < 0:
            raise ValueError(f'flip must be >= 0 but is : '
                             f'{flip}')

        self._flip = flip

    @property
    @beartype
    def slot(self) -> typ.Optional[str]:
        return None if (self._slot is None) else str(self._slot)

    @property
    @beartype
    def candidate(self) -> str:
        return str(self._candidate)

    @property
    @beartype
    def flip(self) -> int:
        return self._flip
