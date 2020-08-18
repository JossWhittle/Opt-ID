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
from optid.utils import validate_string_list

import optid
optid.utils.logging.attach_console_logger(remove_existing=True)


class ValidateStringListTest(unittest.TestCase):
    """
    Tests the validate_string_list class can be imported and used correctly.
    """

    def test_type(self):
        """
        Tests the validate_string_list function checks for values type.
        """

        validate_string_list(values=['hello', 'world'])

        self.assertRaisesRegex(optid.errors.ValidateStringListTypeError, '.*', validate_string_list,
                               values=None)

    def test_element_type(self):
        """
        Tests the validate_string_list function checks for element type.
        """

        self.assertRaisesRegex(optid.errors.ValidateStringListElementTypeError, '.*', validate_string_list,
                               values=['hello', None, 'world'])

    def test_shape(self):
        """
        Tests the validate_string_list function checks if the list shape if requested.
        """

        validate_string_list(values=['hello', 'world'], shape=2)

        self.assertRaisesRegex(optid.errors.ValidateStringListShapeError, '.*', validate_string_list,
                               values=['hello', 'world'], shape=1234)

    def test_empty(self):
        """
        Tests the validate_string_list function checks if the list is empty if requested.
        """

        validate_string_list(values=[], assert_non_empty_list=False)

        self.assertRaisesRegex(optid.errors.ValidateStringListEmptyError, '.*', validate_string_list,
                               values=[], assert_non_empty_list=True)

    def test_element_empty(self):
        """
        Tests the validate_string_list function checks for empty string elements if requested.
        """

        validate_string_list(values=['hello', '', 'world'], assert_non_empty_strings=False)

        self.assertRaisesRegex(optid.errors.ValidateStringListElementEmptyError, '.*', validate_string_list,
                               values=['hello', '', 'world'], assert_non_empty_strings=True)

    def test_element_uniqueness(self):
        """
        Tests the validate_string_list function checks for unique string elements if requested.
        """

        validate_string_list(values=['a', 'b', 'c'], assert_unique_strings=True)

        validate_string_list(values=['a', 'a', 'c'], assert_unique_strings=False)

        self.assertRaisesRegex(optid.errors.ValidateStringListElementUniquenessError, '.*', validate_string_list,
                               values=['a', 'a', 'c'], assert_unique_strings=True)

