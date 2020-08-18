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
from optid.utils import validate_range

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class ValidateRangeTest(unittest.TestCase):
    """
    Tests the validate_range function can be imported and used correctly.
    """

    def test_type(self):
        """
        Tests the validate_range function checks for values type.
        """

        validate_range(axis_range=(-1, 1, 10))

        self.assertRaisesRegex(optid.errors.ValidateRangeTypeError, '.*', validate_range,
                               axis_range=None)

    def test_shape(self):
        """
        Tests the validate_range function checks ranges have three entries.
        """

        validate_range(axis_range=(-1, 1, 10))

        self.assertRaisesRegex(optid.errors.ValidateRangeShapeError, '.*', validate_range,
                               axis_range=(-1, 1, 10, 1234))

        self.assertRaisesRegex(optid.errors.ValidateRangeShapeError, '.*', validate_range,
                               axis_range=(-1, 1))

    def test_element_type(self):
        """
        Tests the validate_range function checks ranges have three entries.
        """

        validate_range(axis_range=(-1, 1, 10))

        self.assertRaisesRegex(optid.errors.ValidateRangeElementTypeError, '.*', validate_range,
                               axis_range=(None, 1, 10))

        self.assertRaisesRegex(optid.errors.ValidateRangeElementTypeError, '.*', validate_range,
                               axis_range=(-1, None, 10))

        self.assertRaisesRegex(optid.errors.ValidateRangeElementTypeError, '.*', validate_range,
                               axis_range=(-1, 1, None))

    def test_steps(self):
        """
        Tests the validate_range function checks ranges have a minimum <= maximum.
        """

        validate_range(axis_range=(-1, 1, 10))

        validate_range(axis_range=(-1, 1, 1))

        self.assertRaisesRegex(optid.errors.ValidateRangeStepsError, '.*', validate_range,
                               axis_range=(-1, 1, 0))

        self.assertRaisesRegex(optid.errors.ValidateRangeStepsError, '.*', validate_range,
                               axis_range=(-1, 1, -1))

    def test_boundary(self):
        """
        Tests the validate_range function checks ranges have a minimum <= maximum.
        """

        validate_range(axis_range=(-1, 1, 10))

        self.assertRaisesRegex(optid.errors.ValidateRangeBoundaryError, '.*', validate_range,
                               axis_range=(1, -1, 10))

    def test_singularity(self):
        """
        Tests the validate_range function checks ranges have a minimum == maximum and steps > 1.
        """

        validate_range(axis_range=(1, 1, 1))

        self.assertRaisesRegex(optid.errors.ValidateRangeSingularityError, '.*', validate_range,
                               axis_range=(1, 1, 10))
