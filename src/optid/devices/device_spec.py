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
import itertools

import optid
from optid.devices import BeamSpec, MagnetTypeSpec


logger = optid.utils.logging.get_logger('optid.devices.DeviceSpec')


class DeviceSpec:
    """
    Represents an insertion device composed of multiple magnet types in fixed arrangements.
    """

    def __init__(self, name : str, x_size : float, z_size : float):

        self._name : str = name
        self._beams : typing.Dict[str, BeamSpec] = dict()
        self._magnet_types : typing.Dict[str, MagnetTypeSpec] = dict()

        self._x_size : float = x_size
        self._z_size : float = z_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def beams(self) -> typing.Dict[str, BeamSpec]:
        return self._beams

    @property
    def x_size(self) -> float:
        return self._x_size

    @property
    def z_size(self) -> float:
        return self._z_size

    @property
    def magnet_types(self) -> typing.Dict[str, MagnetTypeSpec]:
        return self._magnet_types

    @property
    def length(self) -> float:
        return max(self.beam_lengths.values())

    @property
    def beam_lengths(self) -> typing.Dict[str, float]:
        return { beam.name: beam.calculate_length(magnet_types=self.magnet_types)
                 for beam in self.beams.values() }

    @property
    def count(self) -> int:
        return sum(self.beam_counts)

    @property
    def beam_counts(self) -> typing.Dict[str, int]:
        return { beam.name : beam.count for beam in self.beams.values() }

    @property
    def slots(self):
        return itertools.chain.from_iterable(self.beam_slots.values())

    @property
    def beam_slots(self):
        return { beam.name : beam.calculate_slots(x_size=self.x_size, z_size=self.z_size,
                                                  magnet_types=self.magnet_types)
                 for beam in self.beams.values() }

    def register_beam(self, name : str, x_offset : float, z_offset : float):

        assert isinstance(name, str)
        assert name not in self.beams.keys()
        self.beams[name] = BeamSpec(name=name, x_offset=x_offset, z_offset=z_offset)

    def register_magnet_type(self, name : str, s_size : float,
                             field_vector : optid.types.TensorVector,
                             flip_matrix : optid.types.TensorMatrix):

        assert isinstance(name, str)
        assert name not in self.magnet_types.keys()
        self.magnet_types[name] = MagnetTypeSpec(name=name, s_size=s_size, field_vector=field_vector,
                                                 flip_matrix=flip_matrix)

    def push_magnet(self, beam : str, magnet_type : str, direction_matrix : optid.types.TensorMatrix,
                    spacing : float = 0.0):

        assert isinstance(beam, str)
        assert beam in self.beams.keys()
        assert isinstance(magnet_type, str)
        assert magnet_type in self.magnet_types.keys()
        self.beams[beam].push_magnet(magnet_type=magnet_type, direction_matrix=direction_matrix, spacing=spacing)

    def pop_magnet(self, beam : str):

        assert isinstance(beam, str)
        assert beam in self.beams.keys()
        self.beams[beam].pop_magnet()
