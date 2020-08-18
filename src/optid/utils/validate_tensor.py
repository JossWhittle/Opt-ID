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
import numpy as np


class ValidateTensorErrorBase(Exception):
    """
    Base Exception to inherit from for tensor errors.
    """


class ValidateTensorTypeError(ValidateTensorErrorBase):
    """
    Exception to throw when a tensor does not have the expected dtype.
    """

    def __init__(self, expected_dtype : typing.Any, observed_dtype : typing.Any):
        super().__init__()
        self._expected_dtype = expected_dtype
        self._observed_dtype = observed_dtype

    @property
    def expected_dtype(self):
        return self._expected_dtype

    @property
    def observed_dtype(self):
        return self._observed_dtype

    def __str__(self):
        return f'tensor does not have the expected dtype: ' \
               f'expected {self.expected_dtype}, observed {self.observed_dtype}'


class ValidateTensorShapeError(ValidateTensorErrorBase):
    """
    Exception to throw when a tensor does not have the expected shape.
    """

    def __init__(self,
                 expected_shape: typing.Union[typing.Tuple[typing.Optional[int]], typing.List[typing.Optional[int]]],
                 observed_shape: typing.List[int]):
        super().__init__()
        self._expected_shape = expected_shape
        self._observed_shape = observed_shape

    @property
    def expected_shape(self):
        return self._expected_shape

    @property
    def observed_shape(self):
        return self._observed_shape

    def __str__(self):
        return f'tensor does not have the expected shape: ' \
               f'expected {self.expected_shape}, observed {self.observed_shape}'


class ValidateTensorElementTypeError(ValidateTensorErrorBase):
    """
    Exception to throw when a tensor does not have the expected dtype.
    """

    def __init__(self, expected_dtype : typing.Any, observed_dtype : typing.Any):
        super().__init__()
        self._expected_dtype = expected_dtype
        self._observed_dtype = observed_dtype

    @property
    def expected_dtype(self):
        return self._expected_dtype

    @property
    def observed_dtype(self):
        return self._observed_dtype

    def __str__(self):
        return f'tensor does not have the expected dtype: ' \
               f'expected {self.expected_dtype}, observed {self.observed_dtype}'


def validate_tensor(tensor : np.ndarray,
                    shape : typing.Optional[typing.Tuple[typing.Optional[int], ...]] = None,
                    dtype : typing.Any = np.floating):
    """
    Tests whether a given numpy tensor has the number of dimensions and shape matching a shape pattern, and that
    the dtype matches an expected dtype. Raises an exception on invalid tensor inputs.

    Parameters
    ----------
    tensor : np.ndarray
        A tensor whose size and dtype we want to validate match expected values.

    shape : None or tuple(None or int, ...)
        A python tuple of None or int values to match the shape of the tensor against. If shape is None then
        do not validate tensor shape.

    dtype : np.dtype
        A numpy dtype that the tensor is expected to match against.

    Returns
    -------
    If the tensor is valid and matches the expected shape and type then return the tensor to allow
    streamlined assignment.
    """

    if not isinstance(tensor, np.ndarray):
        raise ValidateTensorTypeError(expected_dtype=np.ndarray, observed_dtype=type(tensor))

    if shape is not None:
        if tensor.ndim != len(shape):
            raise ValidateTensorShapeError(expected_shape=shape, observed_shape=tensor.shape)

        for tensor_dim, shape_dim in zip(tensor.shape, shape):
            if (shape_dim is not None) and (tensor_dim != shape_dim):
                raise ValidateTensorShapeError(expected_shape=shape, observed_shape=tensor.shape)

    if not np.issubdtype(tensor.dtype, dtype):
        raise ValidateTensorElementTypeError(expected_dtype=dtype, observed_dtype=tensor.dtype)

    # Return the tensor if it is valid
    return tensor
