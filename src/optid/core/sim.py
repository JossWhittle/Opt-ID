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
import jax.numpy as jnp
import radia as rad

# Opt-ID Imports


@beartype
def bfield_from_lattice(radia_object: int, lattice: jnp.ndarray) -> jnp.ndarray:
    """
    Wraps rad.Fld to take JAX tensors as inputs and return them as outputs.

    :param radia_object:
        Handle to the radia object to simulate the field of.

    :param lattice:
        Tensor representing 3-space world coordinates to evaluate the field at.

    :return:
        Tensor of 3-space field vectors at each location in the lattice.
    """
    return jnp.array(rad.Fld(radia_object, 'b', lattice.reshape((-1, 3)).tolist()),
                     dtype=jnp.float32).reshape(lattice.shape)
