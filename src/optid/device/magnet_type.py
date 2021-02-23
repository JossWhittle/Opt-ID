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
import jax.numpy as jnp

# Opt-ID Imports
from ..geometry import \
    Geometry

from ..device import \
    MagnetCandidate


TVector       = typ.Union[jnp.ndarray, typ.Sequence[numbers.Real]]
TCandidates   = typ.Sequence[MagnetCandidate]
TFlipMatrices = typ.Union[jnp.ndarray, typ.Sequence[jnp.ndarray]]
TMaterial     = typ.Callable[[int], int]

class MagnetType:

    @beartype
    def __init__(self,
            name: str,
            geometry: Geometry,
            vector: TVector,
            flip_matrices: TFlipMatrices,
            candidates: TCandidates,
            material: typ.Optional[TMaterial] = None):

        if len(name) == 0:
            raise ValueError(f'name must be a non-empty string')

        self._name = name

        self._geometry = geometry

        if not isinstance(vector, jnp.ndarray):
            vector = jnp.array(vector, dtype=jnp.float32)

        if vector.shape != (3,):
            raise ValueError(f'vector must be shape (3,) but is : '
                             f'{vector.shape}')

        if vector.dtype != jnp.float32:
            raise TypeError(f'vector must have dtype (float32) but is : '
                            f'{vector.dtype}')

        self._vector = vector

        if not isinstance(flip_matrices, jnp.ndarray):
            flip_matrices = jnp.array(flip_matrices, dtype=jnp.float32)

        if (flip_matrices.ndim != 3) or (flip_matrices.shape[1:] != (4, 4)):
            raise ValueError(f'flip_matrices must be a list of affine matrices with shape (N >= 1, 4, 4) but is : '
                             f'{flip_matrices.shape}')

        if flip_matrices.dtype != jnp.float32:
            raise TypeError(f'flip_matrices must have dtype (float32) but is : '
                            f'{flip_matrices.dtype}')

        self._flip_matrices = flip_matrices

        if not isinstance(candidates, list):
            candidates = list(candidates)

        if len(candidates) == 0:
            raise ValueError(f'candidates must be at least length 1 but is : '
                             f'{len(candidates)}')

        if len(candidates) != len(set(candidate.name for candidate in candidates)):
            raise ValueError(f'candidates must have unique names')

        self._candidates = { candidate.name: candidate for candidate in candidates }

        self._material = material if (material is not None) else (lambda obj: obj)

    @property
    @beartype
    def name(self) -> str:
        return self._name

    @property
    @beartype
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    @beartype
    def vector(self) -> jnp.ndarray:
        return self._vector

    @property
    @beartype
    def material(self) -> TMaterial:
        return self._material

    @property
    @beartype
    def flip_matrices(self) -> jnp.ndarray:
        return self._flip_matrices

    @property
    @beartype
    def candidates(self) -> typ.Dict[str, MagnetCandidate]:
        return dict(self._candidates)


