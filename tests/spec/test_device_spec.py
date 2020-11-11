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
import os
import unittest
import tempfile
import shutil
import inspect
import numpy as np

# Test imports
import optid
from optid.spec import DeviceSpec

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class DeviceSpecTest(unittest.TestCase):
    """
    Tests the DeviceSpec class can be imported and used correctly.
    """

    def test_constructor(self):
        """
        Tests the DeviceSpec class can be constructed with correct parameters.
        """

        name = 'TEST'

        device_spec = DeviceSpec(name)

        self.assertEqual(device_spec.name, name)

