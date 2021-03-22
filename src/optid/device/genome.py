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
import typing as typ

# Opt-ID Imports
from .state import State


TStates = typ.Dict[str, State]
TUnused = typ.List[State]


class Genome:

    @beartype
    def __init__(self,
                 slots: TStates,
                 pool: TUnused):

        self._slots = slots
        self._pool = pool

    @property
    @beartype
    def slots(self) -> TStates:
        return self._slots

    @property
    @beartype
    def nslot(self) -> int:
        return len(self._slots)

    @property
    @beartype
    def pool(self) -> TUnused:
        return self._pool

    @property
    @beartype
    def npool(self) -> int:
        return len(self._pool)
