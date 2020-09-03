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


import io
import typing
import nptyping as npt
import pickle
import numpy as np

import optid
from optid.errors import FileHandleError

logger = optid.utils.logging.get_logger('optid.devices.Device')


class DeviceSpec:
    """
    Represents an insertion device composed of multiple magnet types in fixed arrangements.
    """

    def __init__(self):
        pass
