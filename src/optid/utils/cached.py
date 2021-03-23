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


# External Imports
from beartype import beartype
import typing as typ
from functools import cached_property


def invalidates_cached_properties(func):
    @beartype
    def wrap(self: Memoized, *args, **kargs) -> typ.Any:
        result = func(self, *args, **kargs)
        self.invalidate_cached_properties()
        return result
    return wrap


class Memoized:

    def invalidate_cached_properties(self):

        for key, value in self.__class__.__dict__.items():
            if isinstance(value, cached_property):
                self.__dict__.pop(key, None)
