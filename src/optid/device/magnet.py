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
import pandas as pd


# Opt-ID Imports
from ..constants import \
    MATRIX_IDENTITY

from ..core.affine import \
    is_scale_preserving

from ..core.utils import \
    np_readonly

from ..geometry import \
    Geometry

from .candidate import \
    Candidate

from .element import \
    Element


TVector          = typ.Union[np.ndarray, typ.Sequence[numbers.Real]]
TCandidatesParam = typ.Union[str, pd.DataFrame, typ.Sequence[Candidate]]
TCandidates      = typ.Dict[str, Candidate]
TFlipMatrices    = typ.Optional[typ.Union[np.ndarray, typ.Sequence[np.ndarray]]]
TMaterial        = typ.Callable[[int], int]


class Magnet(Element):

    @beartype
    def __init__(self,
            name: str,
            geometry: Geometry,
            vector: TVector,
            candidates: TCandidatesParam,
            flip_matrices: TFlipMatrices = None,
            material: typ.Optional[TMaterial] = None,
            rescale_vector: bool = True):

        super().__init__(name=name, geometry=geometry, material=material)

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != np.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        if np.allclose(np.linalg.norm(vector), 0, atol=1e-5):
            raise ValueError(f'vector must have positive length')

        self._vector = vector

        flip_matrices = [] if (flip_matrices is None) else [matrix for matrix in flip_matrices]

        if not all(map(is_scale_preserving, flip_matrices)):
            raise ValueError(f'flip_matrices must all be scale preserving affine matrices')

        flip_matrices = np.array([MATRIX_IDENTITY] + flip_matrices, dtype=np.float32)

        if (flip_matrices.ndim != 3) or (flip_matrices.shape[1:] != (4, 4)):
            raise ValueError(f'flip_matrices must be a list of affine matrices with shape (N >= 1, 4, 4) but is : '
                             f'{flip_matrices.shape}')

        if flip_matrices.dtype != np.float32:
            raise TypeError(f'flip_matrices must have dtype (float32) but is : '
                            f'{flip_matrices.dtype}')

        self._flip_matrices = flip_matrices

        if isinstance(candidates, str):
            candidates = pd.read_csv(candidates)

        if isinstance(candidates, pd.DataFrame):
            candidates = Candidate.from_dataframe(candidates)

        if not isinstance(candidates, list):
            candidates = list(candidates)

        if len(candidates) == 0:
            raise ValueError(f'candidates must be at least length 1 but is : '
                             f'{len(candidates)}')

        if len(candidates) != len(set(candidate.name for candidate in candidates)):
            raise ValueError(f'candidates must have unique names')

        self._candidates = { candidate.name: candidate for candidate in candidates }

        if rescale_vector:
            mean_candidate = np.mean([np.linalg.norm(candidate.vector)
                                      for candidate in self.candidates.values()])

            self._vector = (self._vector / np.linalg.norm(self._vector)) * mean_candidate

    @property
    @beartype
    def vector(self) -> np.ndarray:
        return np_readonly(self._vector)

    @beartype
    def flip_matrix(self, flip: int) -> np.ndarray:

        if (flip < 0) or (flip >= self.nflip):
            raise ValueError(f'flip must be in range [0, {self.nflip}) but is : '
                             f'{flip}')

        return np_readonly(self._flip_matrices[flip])

    @property
    @beartype
    def nflip(self) -> int:
        return len(self._flip_matrices)

    @property
    @beartype
    def candidates(self) -> TCandidates:
        return dict(self._candidates)


