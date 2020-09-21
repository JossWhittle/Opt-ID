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

# Test imports
import optid
from optid.utils import Range

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

        Range(-1, 1, 10)

        self.assertRaisesRegex(Exception, '.*', Range, min=None, max=1, steps=10)

        self.assertRaisesRegex(Exception, '.*', Range, min=-1, max=None, steps=10)

        self.assertRaisesRegex(Exception, '.*', Range, min=-1, max=1, steps=None)

    def test_steps(self):
        """
        Tests the validate_range function checks ranges have a minimum <= maximum.
        """

        Range(-1, 1, 10)
        Range(-1, 1, 2)
        Range(1, 1, 1)

        self.assertRaisesRegex(Exception, '.*', Range, min=-1, max=1, steps=1)
        self.assertRaisesRegex(Exception, '.*', Range, min=-1, max=1, steps=0)
        self.assertRaisesRegex(Exception, '.*', Range, min=-1, max=1, steps=-1)

    def test_boundary(self):
        """
        Tests the validate_range function checks ranges have a minimum <= maximum.
        """

        Range(-1, 1, 10)

        self.assertRaisesRegex(Exception, '.*', Range, min=1, max=-1, steps=2)

    def test_singularity(self):
        """
        Tests the validate_range function checks ranges have a minimum == maximum and steps > 1.
        """

        Range(1, 1, 1)

        self.assertRaisesRegex(Exception, '.*', Range, min=1, max=1, steps=2)
        self.assertRaisesRegex(Exception, '.*', Range, min=1, max=1, steps=0)
        self.assertRaisesRegex(Exception, '.*', Range, min=1, max=1, steps=-1)

    def test_linspace(self):

        rng = Range(-1, 1, 10)

        arr = rng.linspace
        self.assertEqual(arr[0], rng.min)
        self.assertEqual(arr[-1], rng.max)
        self.assertEqual(len(arr), rng.steps)

    def test_iter(self):

        rng = Range(-1, 1, 10)

        arr = list(rng.iter())
        self.assertEqual(arr[0], (0, rng.min))
        self.assertEqual(arr[-1], ((rng.steps - 1), rng.max))
        self.assertEqual(len(arr), rng.steps)