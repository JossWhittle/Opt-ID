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

BinaryFileHandle  = typing.Union[str, typing.BinaryIO]
ASCIIFileHandle   = typing.Union[str, typing.TextIO]

ListStrings       = typing.List[str]

TensorVector      = npt.NDArray[(3,), npt.Float]
TensorVectors     = npt.NDArray[(typing.Any, 3), npt.Float]

TensorPoint       = npt.NDArray[(3,), npt.Float]
TensorPoints      = npt.NDArray[(typing.Any, 3), npt.Float]

TensorMatrix      = npt.NDArray[(3, 3), npt.Float]
TensorMatrices    = npt.NDArray[(typing.Any, 3, 3), npt.Float]

TensorPermutation = npt.NDArray[(typing.Any,), npt.Int]
TensorFlips       = npt.NDArray[(typing.Any,), npt.Bool]

TensorBfield      = npt.NDArray[(typing.Any, typing.Any, typing.Any, 3), npt.Float]
TensorSortLookup  = npt.NDArray[(typing.Any, typing.Any, typing.Any, typing.Any, 3, 3), npt.Float]

TensorRange       = npt.NDArray[(typing.Any,), npt.Float]
TensorGrid        = npt.NDArray[(typing.Any, typing.Any, typing.Any, 3), npt.Float]