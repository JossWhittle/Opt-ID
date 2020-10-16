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


import numpy as np


MAX_RAND_INT = (np.iinfo(np.int32).max - 1)

VECTOR_ZERO = np.array([0, 0, 0], dtype=np.float32)

VECTOR_X = np.array([1, 0, 0], dtype=np.float32)
VECTOR_Z = np.array([0, 1, 0], dtype=np.float32)
VECTOR_S = np.array([0, 0, 1], dtype=np.float32)

MATRIX_IDENTITY          = np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], dtype=np.float32)
MATRIX_ROTX_180          = np.array([[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]], dtype=np.float32)
MATRIX_ROTZ_180          = np.array([[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]], dtype=np.float32)
MATRIX_ROTS_90           = np.array([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=np.float32)
MATRIX_ROTS_180          = np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]], dtype=np.float32)
MATRIX_ROTS_270          = np.array([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]], dtype=np.float32)
MATRIX_ROTS_270_ROTX_180 = np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]], dtype=np.float32)
MATRIX_ROTS_270_ROTZ_180 = np.array([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]], dtype=np.float32)
