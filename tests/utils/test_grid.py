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
import unittest
import numpy as np

# Test imports
import optid
from optid.utils import Range, Grid, validate_tensor

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class ValidateRangeTest(unittest.TestCase):
    """
    Tests the validate_range function can be imported and used correctly.
    """

    def test_element_type(self):
        """
        Tests the validate_range function checks ranges have three entries.
        """

        Grid(x_range=Range(min=-1, max=1, steps=10),
             z_range=Range(min=-1, max=1, steps=10),
             s_range=Range(min=-1, max=1, steps=10))

        self.assertRaisesRegex(Exception, '.*', Grid,
                               x_range=None,
                               z_range=Range(min=-1, max=1, steps=10),
                               s_range=Range(min=-1, max=1, steps=10))

        self.assertRaisesRegex(Exception, '.*', Grid,
                               x_range=Range(min=-1, max=1, steps=10),
                               z_range=None,
                               s_range=Range(min=-1, max=1, steps=10))

        self.assertRaisesRegex(Exception, '.*', Grid,
                               x_range=Range(min=-1, max=1, steps=10),
                               z_range=Range(min=-1, max=1, steps=10),
                               s_range=None)

    def test_min(self):

        grid = Grid(x_range=Range(min=-1, max=1, steps=10),
                    z_range=Range(min=-2, max=2, steps=20),
                    s_range=Range(min=-3, max=3, steps=30))

        self.assertTrue(np.allclose(grid.min, [-1, -2, -3]))

    def test_max(self):

        grid = Grid(x_range=Range(min=-1, max=1, steps=10),
                    z_range=Range(min=-2, max=2, steps=20),
                    s_range=Range(min=-3, max=3, steps=30))

        self.assertTrue(np.allclose(grid.max, [1, 2, 3]))

    def test_steps(self):

        grid = Grid(x_range=Range(min=-1, max=1, steps=10),
                    z_range=Range(min=-2, max=2, steps=20),
                    s_range=Range(min=-3, max=3, steps=30))

        self.assertTrue(np.allclose(grid.steps, [10, 20, 30]))

    def test_eq(self):

        grid_a = Grid(x_range=Range(min=-1, max=1, steps=10),
                      z_range=Range(min=-2, max=2, steps=20),
                      s_range=Range(min=-3, max=3, steps=30))

        grid_b = Grid(x_range=Range(min=-1, max=1, steps=10),
                      z_range=Range(min=-2, max=2, steps=20),
                      s_range=Range(min=-3, max=3, steps=30))

        grid_c = Grid(x_range=Range(min=-1, max=1, steps=20),
                      z_range=Range(min=-2, max=2, steps=20),
                      s_range=Range(min=-3, max=3, steps=20))

        grid_d = Grid(x_range=Range(min=-2, max=1, steps=10),
                      z_range=Range(min=-2, max=2, steps=20),
                      s_range=Range(min=-2, max=3, steps=30))

        grid_e = Grid(x_range=Range(min=-1, max=2, steps=10),
                      z_range=Range(min=-2, max=2, steps=20),
                      s_range=Range(min=-3, max=2, steps=30))

        self.assertEqual(grid_a, grid_b)
        self.assertNotEqual(grid_a, grid_c)
        self.assertNotEqual(grid_a, grid_d)
        self.assertNotEqual(grid_a, grid_e)

    def test_meshgrid(self):

        grid = Grid(x_range=Range(min=-1, max=1, steps=10),
                    z_range=Range(min=-1, max=1, steps=10),
                    s_range=Range(min=-1, max=1, steps=10))

        arr = grid.meshgrid
        validate_tensor(arr, shape=(10, 10, 10, 3))

        self.assertTrue(np.allclose(arr[0, 0, 0], [-1, -1, -1]))
        self.assertTrue(np.allclose(arr[-1, -1, -1], [1, 1, 1]))

    def test_iter(self):

        grid = Grid(x_range=Range(min=-1, max=1, steps=10),
                    z_range=Range(min=-1, max=1, steps=10),
                    s_range=Range(min=-1, max=1, steps=10))

        arr = list(grid.iter())

        self.assertTrue(np.allclose(arr[0][0], [0, 0, 0]))
        self.assertTrue(np.allclose(arr[0][1], [-1, -1, -1]))

        self.assertTrue(np.allclose(arr[-1][0], [9, 9, 9]))
        self.assertTrue(np.allclose(arr[-1][1], [1, 1, 1]))
