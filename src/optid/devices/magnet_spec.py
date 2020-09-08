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


import optid


MagnetTypeSpec = typing.NamedTuple('MagnetTypeSpec', [
    ('name', str),
    ('s_size', float),
    ('field_vector', optid.types.TensorVector),
    ('flip_matrix', optid.types.TensorMatrix)
])

MagnetSlotSpec = typing.NamedTuple('MagnetSlotSpec', [
    ('beam', str), ('slot', str), ('magnet_type', str),
    ('x_size',   float), ('z_size',   float), ('s_size',   float),
    ('x_offset', float), ('z_offset', float), ('s_offset', float),
    ('field_vector', optid.types.TensorVector),
    ('direction_matrix', optid.types.TensorMatrix),
    ('flip_matrix', optid.types.TensorMatrix)
])
