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

from optid.utils.logging import get_logger

logger = get_logger('optid.magnets.MagnetSortLookup')


class Range:

    def __init__(self, min : float, max : float, steps : int):
        """
        Represents a sample range in a single axis.

        Parameters
        ----------
        min : float
            The minimum value on the left edge of the range.

        max : float
            The maximum value on the right edge of the range.

        steps : int
            The number of steps between the two edges of the range. Must be positive and must be equal to 1 if the min
            is the same as the max.
        """
        try:
            self._min = float(min)
        except Exception as ex:
            logger.exception('min must be a float', exc_info=ex)
            raise ex

        try:
            self._max = float(max)
        except Exception as ex:
            logger.exception('max must be a float', exc_info=ex)
            raise ex

        try:
            self._steps = int(steps)
            assert self.steps > 0
        except Exception as ex:
            logger.exception('steps must be a positive int', exc_info=ex)
            raise ex

        try:
            assert self.min <= self.max
        except Exception as ex:
            logger.exception('min must be less than or equal to max', exc_info=ex)
            raise ex

        try:
            assert (self.min < self.max) or ((self.min == self.max) and (self.steps == 1))
        except Exception as ex:
            logger.exception('if range is singular steps must be 1', exc_info=ex)
            raise ex

        try:
            assert (self.min == self.max) or ((self.min < self.max) and (self.steps > 1))
        except Exception as ex:
            logger.exception('if min is less than max then steps must be greater than 1', exc_info=ex)
            raise ex

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    @property
    def steps(self) -> int:
        return self._steps

    def __eq__(self, other : 'Range') -> bool:
        return (self.min == other.min) and \
               (self.max == other.max) and \
               (self.steps == other.steps)

    @property
    def linspace(self) -> npt.NDArray[(typing.Any,), npt.Float]:
        return np.linspace(self.min, self.max, self.steps)

    def iter(self):
        for step, value in enumerate(self.linspace):
            yield step, value
