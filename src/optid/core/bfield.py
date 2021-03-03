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
import jax
import numpy as np
import radia as rad

# Opt-ID Imports


@beartype
def radia_evaluate_bfield_on_lattice(
        radia_object: int,
        lattice: np.ndarray) -> np.ndarray:
    """
    Wraps rad.Fld to take JAX tensors as inputs and return them as outputs.

    :param radia_object:
        Handle to the radia object to simulate the field of.

    :param lattice:
        Tensor representing 3-space world coordinates to evaluate the field at.

    :return:
        Tensor of 3-space field vectors at each location in the lattice.
    """

    if lattice.shape[-1] != 3:
        raise ValueError(f'lattice must be a mesh of vectors in 3-space with shape (..., 3) but is : '
                         f'{lattice.shape}')

    if lattice.dtype != np.float32:
        raise TypeError(f'lattice must have dtype (float32) but is : '
                        f'{lattice.dtype}')

    return np.array(rad.Fld(radia_object, 'b', lattice.reshape((-1, 3)).tolist()),
                    dtype=np.float32).reshape(lattice.shape)


@jax.jit
def bfield_from_lookup(lookup, vector):
    """
    Compute the bfield from a magnet with the given field vector using a lookup table of field rotation matrices.

    :param lookup:
        Lattice of 3x3 rotation matrices representing field curvature and scale over a spatial lattice.

        Magnet shape and geometry is baked into the lookup table, but the actual magnetization direction can be applied
        by a simple matmul of the desired field vector against the matrix at each location on the lattice, yielding
        a lattice of field 3-vectors.

    :param vector:
        Field vector for the magnet whose field we want to solve for.

    :return:
        Lattice of 3-vectors representing the field direction and magnitude at each spatial location represented on
        the lattice of the lookup table.
    """
    return lookup @ vector
