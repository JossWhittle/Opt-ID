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


import typing
import nptyping as npt
import numpy as np

import optid
from optid.utils import Range
from optid.utils.logging import get_logger

logger = get_logger('optid.utils.Grid')


class Grid:

    def __init__(self, x_range : Range, z_range : Range, s_range : Range):
        """
        Represents a 3D sampling grid as the cartesian product of 3 Range instances.

        Parameters
        ----------
        x_range : Range
            Sampling across the x-axis.

        z_range : Range
            Sampling across the z-axis.

        s_range : Range
            Sampling across the s-axis.
        """

        try:
            self._x_range = x_range
            assert isinstance(self.x_range, Range)
        except Exception as ex:
            logger.exception('x_range must be a Range', exc_info=ex)
            raise ex

        try:
            self._z_range = z_range
            assert isinstance(self.z_range, Range)
        except Exception as ex:
            logger.exception('z_range must be a Range', exc_info=ex)
            raise ex

        try:
            self._s_range = s_range
            assert isinstance(self.s_range, Range)
        except Exception as ex:
            logger.exception('s_range must be a Range', exc_info=ex)
            raise ex

    @property
    def x_range(self) -> optid.types.TensorRange:
        return self._x_range

    @property
    def z_range(self) -> optid.types.TensorRange:
        return self._z_range

    @property
    def s_range(self) -> optid.types.TensorRange:
        return self._s_range

    @property
    def min(self) -> typing.Tuple[float, float, float]:
        return self.x_range.min, self.z_range.min, self.s_range.min

    @property
    def max(self) -> typing.Tuple[float, float, float]:
        return self.x_range.max, self.z_range.max, self.s_range.max

    @property
    def steps(self) -> typing.Tuple[int, int, int]:
        return self.x_range.steps, self.z_range.steps, self.s_range.steps

    def __eq__(self, other : 'Grid') -> bool:
        return (self.x_range == other.x_range) and \
               (self.z_range == other.z_range) and \
               (self.s_range == other.s_range)

    @property
    def meshgrid(self) -> optid.types.TensorGrid:
        return np.stack(np.meshgrid(self.x_range.linspace,
                                    self.z_range.linspace,
                                    self.s_range.linspace), axis=-1)

    def iter(self):
        for s_step, s_value in self.s_range.iter():
            for z_step, z_value in self.z_range.iter():
                for x_step, x_value in self.x_range.iter():
                    yield (x_step, z_step, s_step), (x_value, z_value, s_value)
