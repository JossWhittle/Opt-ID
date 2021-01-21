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
import jax

# Opt-ID Imports
from .lattice import orthonormal_interpolate


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


@jax.jit
def bfield_from_interpolated_lookup(lookup, vector, shim):
    """
    Compute the bfield from a magnet with the given field vector using a lookup table of field rotation matrices.

    :param lookup:
        Lattice of 3x3 rotation matrices representing field curvature and scale over a spatial lattice, duplicated
        over a secondary (leading) lattice representing the lookup table at multiple shim offsets.

        Magnet shape and geometry is baked into the lookup table, but the actual magnetization direction can be applied
        by a simple matmul of the desired field vector against the matrix at each location on the lattice, yielding
        a lattice of field 3-vectors.

    :param vector:
        Field vector for the magnet whose field we want to solve for.

    :param shim:
        Shim lattice in unit orthonormal coordinates representing the leading dimensions of the lookup table to
        interpolate over before the field vector is applied.

        (lookup : (10, 5, 5, 5, 3, 3),     shim : (1,)) -> interp lookup : (5, 5, 5, 3, 3) -> bfield : (5, 5, 5, 3)
        (lookup : (10, 10, 5, 5, 5, 3, 3), shim : (2,)) -> interp lookup : (5, 5, 5, 3, 3) -> bfield : (5, 5, 5, 3)

    :return:
        Lattice of 3-vectors representing the field direction and magnitude at each spatial location represented on
        the lattice of the lookup table.
    """
    return bfield_from_lookup(orthonormal_interpolate(lookup, shim), vector)
