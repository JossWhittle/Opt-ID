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
import jax.numpy as jnp

# Opt-ID Imports
from ..core.utils import \
    np_readonly

from ..core.affine import \
    translate

from .magnet import \
    Magnet


TAnchor = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]
TBounds = typ.Tuple[np.ndarray, np.ndarray]


class SlotType:

    @beartype
    def __init__(self,
                 name: str,
                 magnet: Magnet,
                 anchor: TAnchor,
                 direction_matrix: np.ndarray):

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        self._magnet = magnet

        if not isinstance(anchor, np.ndarray):
            anchor = np.array(anchor, dtype=np.float32)

        if anchor.shape != (3,):
            raise ValueError(f'anchor must be shape (3,) but is : '
                             f'{anchor.shape}')

        if anchor.dtype != np.float32:
            raise TypeError(f'anchor must have dtype (float32) but is : '
                            f'{anchor.dtype}')

        self._anchor = anchor

        if direction_matrix.shape != (4, 4):
            raise ValueError(f'direction_matrix must be an affine matrix with shape (4, 4) but is : '
                             f'{direction_matrix.shape}')

        if direction_matrix.dtype != np.float32:
            raise TypeError(f'direction_matrix must have dtype (float32) but is : '
                            f'{direction_matrix.dtype}')

        self._direction_matrix = direction_matrix

        bmin, bmax = magnet.geometry.transform(direction_matrix).bounds
        anchor_matrix = translate(*(-((bmin * (1.0 - anchor)) + (bmax * anchor))))

        self._anchor_matrix = anchor_matrix

        self._bounds = magnet.geometry.transform(direction_matrix @ anchor_matrix).bounds

    @property
    @beartype
    def name(self) -> str:
        return str(self._name)

    @property
    @beartype
    def qualified_name(self) -> str:
        return f'{self.name}::{self.magnet.name}'

    @property
    @beartype
    def magnet(self) -> Magnet:
        return self._magnet

    @property
    @beartype
    def anchor(self) -> np.ndarray:
        return np_readonly(self._anchor)

    @property
    @beartype
    def anchor_matrix(self) -> np.ndarray:
        return np_readonly(self._anchor_matrix)

    @property
    @beartype
    def direction_matrix(self) -> np.ndarray:
        return np_readonly(self._direction_matrix)

    @property
    @beartype
    def bounds(self) -> TBounds:
        return self._bounds
