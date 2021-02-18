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


# Utility imports
from beartype.roar import BeartypeException
import sys
import unittest
import numpy as np
import jax.numpy as jnp

# Test imports
import optid
from optid.bfield import Lookup
from optid.lattice import Lattice
from optid.core.affine import scale

# Configure debug logging
optid.utils.logging.attach_console_logger(remove_existing=True)


class LookupTest(unittest.TestCase):
    """
    Test Lookup class.
    """

    ####################################################################################################################

    def test_constructor(self):

        lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup  = jnp.ones((2, 2, 2, 3, 3), dtype=jnp.float32)

        Lookup(lattice=lattice, lookup=lookup)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_lookup_lattice_raises_exception(self):

        lookup = jnp.ones((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.float32)

        self.assertRaisesRegex(BeartypeException, '.*', Lookup,
                               lattice=None, lookup=lookup)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_lookup_raises_exception(self):

        lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))

        self.assertRaisesRegex(BeartypeException, '.*', Lookup,
                               lattice=lattice, lookup=None)

    def test_constructor_bad_lookup_type_raises_exception(self):

        lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup  = jnp.ones((2, 2, 2, 3, 3), dtype=jnp.int32)

        self.assertRaisesRegex(TypeError, '.*', Lookup,
                               lattice=lattice, lookup=lookup)

    def test_constructor_bad_lookup_shape_raises_exception(self):

        lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))

        self.assertRaisesRegex(ValueError, '.*', Lookup,
                               lattice=lattice,
                               lookup=jnp.ones((2, 2, 2, 3, 3, 1), dtype=jnp.float32))

        self.assertRaisesRegex(ValueError, '.*', Lookup,
                               lattice=lattice,
                               lookup=jnp.ones((2, 2, 2, 3), dtype=jnp.float32))

        self.assertRaisesRegex(ValueError, '.*', Lookup,
                               lattice=lattice,
                               lookup=jnp.ones((2, 2, 2, 1, 1), dtype=jnp.float32))

        self.assertRaisesRegex(ValueError, '.*', Lookup,
                               lattice=lattice,
                               lookup=jnp.ones((2, 1, 2, 3, 3), dtype=jnp.float32))

    ####################################################################################################################

    def test_bfield(self):

        lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup  = jnp.zeros((2, 2, 2, 3, 3), dtype=jnp.float32)
        lookup  = lookup.at[...].set(jnp.eye(3, dtype=jnp.float32).reshape((1, 1, 1, 3, 3)))

        bfield_lookup = Lookup(lattice=lattice, lookup=lookup)

        bfield = bfield_lookup.bfield((jnp.ones((3,), dtype=jnp.float32) * 0.5))

        self.assertIs(bfield.lattice, lattice)
        self.assertEqual(bfield.field.shape, (2, 2, 2, 3))
        self.assertTrue(np.allclose(bfield.field, 0.5, atol=1e-5))

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_bfield_bad_vector_raises_exception(self):

        lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup  = jnp.zeros((2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(lattice=lattice, lookup=lookup)

        self.assertRaisesRegex(BeartypeException, '.*', bfield_lookup.bfield,
                               vector=None)

    def test_bfield_bad_vector_type_raises_exception(self):

        lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup  = jnp.zeros((2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(lattice=lattice, lookup=lookup)

        self.assertRaisesRegex(TypeError, '.*', bfield_lookup.bfield,
                               vector=jnp.ones((3,), dtype=jnp.int32))

    def test_bfield_bad_vector_shape_raises_exception(self):

        lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup  = jnp.zeros((2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(lattice=lattice, lookup=lookup)

        self.assertRaisesRegex(ValueError, '.*', bfield_lookup.bfield,
                               vector=jnp.ones((2,), dtype=jnp.float32))
