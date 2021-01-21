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

        shim_lattice   = Lattice(unit_to_world_matrix=scale(0.1, 0.1, 0.1), shape=(3, 3, 3))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.ones((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.float32)

        Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_shim_lattice_raises_exception(self):

        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.ones((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.float32)

        self.assertRaisesRegex(BeartypeException, '.*', Lookup,
                               shim_lattice=None, lookup_lattice=lookup_lattice, lookup=lookup)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_lookup_lattice_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(0.1, 0.1, 0.1), shape=(3, 3, 3))
        lookup         = jnp.ones((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.float32)

        self.assertRaisesRegex(BeartypeException, '.*', Lookup,
                               shim_lattice=shim_lattice, lookup_lattice=None, lookup=lookup)

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_constructor_bad_lookup_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(0.1, 0.1, 0.1), shape=(3, 3, 3))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))

        self.assertRaisesRegex(BeartypeException, '.*', Lookup,
                               shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=None)

    def test_constructor_bad_lookup_type_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(0.1, 0.1, 0.1), shape=(3, 3, 3))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.ones((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.int32)

        self.assertRaisesRegex(TypeError, '.*', Lookup,
                               shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

    def test_constructor_bad_lookup_shape_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(0.1, 0.1, 0.1), shape=(3, 3, 3))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))

        self.assertRaisesRegex(ValueError, '.*', Lookup,
                               shim_lattice=shim_lattice, lookup_lattice=lookup_lattice,
                               lookup=jnp.ones((3, 3, 3, 2, 2, 2, 3, 3, 1), dtype=jnp.float32))

        self.assertRaisesRegex(ValueError, '.*', Lookup,
                               shim_lattice=shim_lattice, lookup_lattice=lookup_lattice,
                               lookup=jnp.ones((3, 3, 3, 2, 2, 2, 3), dtype=jnp.float32))

        self.assertRaisesRegex(ValueError, '.*', Lookup,
                               shim_lattice=shim_lattice, lookup_lattice=lookup_lattice,
                               lookup=jnp.ones((3, 3, 3, 2, 2, 2, 1, 1), dtype=jnp.float32))

        self.assertRaisesRegex(ValueError, '.*', Lookup,
                               shim_lattice=shim_lattice, lookup_lattice=lookup_lattice,
                               lookup=jnp.ones((1, 1, 1, 2, 2, 2, 3, 3), dtype=jnp.float32))

        self.assertRaisesRegex(ValueError, '.*', Lookup,
                               shim_lattice=shim_lattice, lookup_lattice=lookup_lattice,
                               lookup=jnp.ones((3, 3, 3, 1, 1, 1, 3, 3), dtype=jnp.float32))

    ####################################################################################################################

    def test_bfield(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(1, 1, 1), shape=(1, 1, 1))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.zeros((1, 1, 1, 2, 2, 2, 3, 3), dtype=jnp.float32)
        lookup         = lookup.at[...].set(jnp.eye(3, dtype=jnp.float32).reshape((1, 1, 1, 1, 1, 3, 3)))

        bfield_lookup = Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

        bfield = bfield_lookup.bfield((jnp.ones((3,), dtype=jnp.float32) * 0.5))

        self.assertEqual(bfield.shape, (2, 2, 2, 3))
        self.assertTrue(np.allclose(bfield, 0.5, atol=1e-5))

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_bfield_bad_vector_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(1, 1, 1), shape=(1, 1, 1))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.zeros((1, 1, 1, 2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

        self.assertRaisesRegex(BeartypeException, '.*', bfield_lookup.bfield,
                               vector=None)

    def test_bfield_bad_vector_type_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(1, 1, 1), shape=(1, 1, 1))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.zeros((1, 1, 1, 2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

        self.assertRaisesRegex(TypeError, '.*', bfield_lookup.bfield,
                               vector=jnp.ones((3,), dtype=jnp.int32))

    def test_bfield_bad_vector_shape_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(1, 1, 1), shape=(1, 1, 1))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.zeros((1, 1, 1, 2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

        self.assertRaisesRegex(ValueError, '.*', bfield_lookup.bfield,
                               vector=jnp.ones((2,), dtype=jnp.float32))

    def test_bfield_shim(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(1, 1, 1), shape=(3, 3, 3))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.zeros((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.float32)
        lookup         = lookup.at[...].set(jnp.eye(3, dtype=jnp.float32).reshape((1, 1, 1, 1, 1, 3, 3)))

        bfield_lookup = Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

        bfield = bfield_lookup.bfield((jnp.ones((3,), dtype=jnp.float32) * 0.5),
                                      shim=jnp.zeros((3,), dtype=jnp.float32))

        self.assertEqual(bfield.shape, (2, 2, 2, 3))
        self.assertTrue(np.allclose(bfield, 0.5, atol=1e-5))

    @unittest.skipIf(sys.flags.optimize > 0, 'BearType optimized away.')
    def test_bfield_bad_shim_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(1, 1, 1), shape=(3, 3, 3))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.zeros((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

        self.assertRaisesRegex(BeartypeException, '.*', bfield_lookup.bfield,
                               vector=jnp.ones((3,), dtype=jnp.float32),
                               shim='None')

    def test_bfield_bad_shim_missing_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(1, 1, 1), shape=(3, 3, 3))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.zeros((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

        self.assertRaisesRegex(ValueError, '.*', bfield_lookup.bfield,
                               vector=jnp.ones((3,), dtype=jnp.float32),
                               shim=None)

    def test_bfield_bad_shim_type_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(1, 1, 1), shape=(3, 3, 3))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.zeros((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

        self.assertRaisesRegex(TypeError, '.*', bfield_lookup.bfield,
                               vector=jnp.ones((3,), dtype=jnp.float32),
                               shim=jnp.zeros((3,), dtype=jnp.int32))

    def test_bfield_bad_shim_shape_raises_exception(self):

        shim_lattice   = Lattice(unit_to_world_matrix=scale(1, 1, 1), shape=(3, 3, 3))
        lookup_lattice = Lattice(unit_to_world_matrix=scale(1, 1, 2), shape=(2, 2, 2))
        lookup         = jnp.zeros((3, 3, 3, 2, 2, 2, 3, 3), dtype=jnp.float32)

        bfield_lookup = Lookup(shim_lattice=shim_lattice, lookup_lattice=lookup_lattice, lookup=lookup)

        self.assertRaisesRegex(ValueError, '.*', bfield_lookup.bfield,
                               vector=jnp.ones((3,), dtype=jnp.float32),
                               shim=jnp.zeros((2,), dtype=jnp.float32))
