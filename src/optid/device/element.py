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
import typing as typ
import numpy as np

# Opt-ID Imports
from ..geometry import Geometry


TVector   = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]
TMaterial = typ.Callable[[int], int]


class Element:

    @beartype
    def __init__(self,
            name: str,
            geometry: Geometry,
            material: typ.Optional[TMaterial] = None):

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        self._geometry = geometry

        self._material = material if (material is not None) else (lambda obj: obj)

    @property
    @beartype
    def name(self) -> str:
        return str(self._name)

    @property
    @beartype
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    @beartype
    def material(self) -> TMaterial:
        return self._material
