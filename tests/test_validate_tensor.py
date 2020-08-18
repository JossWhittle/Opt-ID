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
from optid.utils import validate_tensor

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class ValidateTensorTest(unittest.TestCase):
    """
    Tests the validate_tensor function can be imported and used correctly.
    """

    def test_type(self):
        """
        Tests the validate_tensor function checks for values type.
        """

        validate_tensor(tensor=np.ones((2, 2), dtype=np.float32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', validate_tensor,
                               tensor=None)

    def test_element_type(self):
        """
        Tests the validate_tensor function checks for element type.
        """

        validate_tensor(tensor=np.ones((2, 2), dtype=np.float32), dtype=np.float32)

        validate_tensor(tensor=np.ones((2, 2), dtype=np.int32), dtype=np.int32)

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', validate_tensor,
                               tensor=np.ones((2, 2), dtype=np.int32), dtype=np.float32)

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', validate_tensor,
                               tensor=np.ones((2, 2), dtype=np.float32), dtype=np.int32)

    def test_shape(self):
        """
        Tests the validate_string_list function checks if the list shape if requested.
        """

        validate_tensor(tensor=np.ones((2, 2), dtype=np.float32), shape=(2, 2))

        validate_tensor(tensor=np.ones((2, 2), dtype=np.float32), shape=(None, 2))

        validate_tensor(tensor=np.ones((2, 2), dtype=np.float32), shape=(2, None))

        validate_tensor(tensor=np.ones((2, 2), dtype=np.float32), shape=(None, None))

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_tensor,
                               tensor=np.ones((2, 2), dtype=np.float32), shape=(2, 2, 1))

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_tensor,
                               tensor=np.ones((2, 2), dtype=np.float32), shape=(3, 2))

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_tensor,
                               tensor=np.ones((2, 2), dtype=np.float32), shape=(2, 3))

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_tensor,
                               tensor=np.ones((2, 2), dtype=np.float32), shape=(3, 3))

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_tensor,
                               tensor=np.ones((2, 2), dtype=np.float32), shape=(None, 3))

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', validate_tensor,
                               tensor=np.ones((2, 2), dtype=np.float32), shape=(3, None))
