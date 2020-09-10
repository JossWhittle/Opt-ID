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

import optid
from optid.utils import validate_tensor, validate_string, validate_string_list
from optid.utils.logging import get_logger

logger = get_logger('optid.magnets.MagnetSet')


class MagnetSet:
    """
    Represents a set of named candidate magnets of the same type and size with their measured field strength vectors.
    """

    def __init__(self,
                 mtype : str,
                 reference_size : optid.types.TensorVector,
                 reference_field_vector : optid.types.TensorVector,
                 flip_matrix : optid.types.TensorMatrix,
                 names : optid.types.ListStrings,
                 sizes : optid.types.TensorVectors,
                 field_vectors : optid.types.TensorVectors):
        """
        Constructs a MagnetSet instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        mtype : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        reference_size : float tensor (3,)
            A float tensor of a single 3-dim size for the AABB surrounding the reference magnet geometry.

        reference_field_vector : float tensor (3,)
            A float tensor of a single 3-dim field vector for the magnetization of the reference magnet.

        flip_matrix : float tensor (3, 3)
            A float matrix of shape (3, 3) representing the flips that would be applied to a magnet in order to swap
            the direction of its minor axis vectors while keeping its major (easy) axis vector.

        names : list(str)
            A list of unique non-empty strings representing the named identifier for each physical magnet as
            specified by the manufacturer or build team.

        sizes : float tensor (M, 3)
            A tensor of M 3-dim float vectors of shape (M, 3) representing the measured sizes of the AABB of each
            individual magnets geometry.

        field_vectors : float tensor (M, 3)
            A tensor of M 3-dim float vectors of shape (M, 3) representing the average magnetic field strength
            measurements for each magnet in this set.
        """

        try:
            self._mtype = validate_string(mtype, assert_non_empty=True)
        except Exception as ex:
            logger.exception('mtype must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._reference_size = validate_tensor(reference_size, shape=(3,))
        except Exception as ex:
            logger.exception('reference_size must be a float tensor of shape (3,)', exc_info=ex)
            raise ex

        try:
            self._reference_field_vector = validate_tensor(reference_field_vector, shape=(3,))
        except Exception as ex:
            logger.exception('reference_field_vector must be a float tensor of shape (3,)', exc_info=ex)
            raise ex

        try:
            self._flip_matrix = validate_tensor(flip_matrix, shape=(3, 3))
            self._flippable = not np.allclose(self.flip_matrix, np.eye(3, dtype=np.float32))
        except Exception as ex:
            logger.exception('flip_matrix must be a float tensor of shape (3, 3)', exc_info=ex)
            raise ex

        try:
            self._names = validate_string_list(names, assert_non_empty_list=True, assert_non_empty_strings=True,
                                                      assert_unique_strings=True)

            # Number of magnets derived from number of names provided. All other inputs must be consistent.
            self._count = len(self.names)
        except Exception as ex:
            logger.exception('names must be a non-empty list of non-empty and unique strings', exc_info=ex)
            raise ex

        try:
            self._sizes = validate_tensor(sizes, shape=(self.count, 3))
        except Exception as ex:
            logger.exception('sizes must be a float tensor of shape (M, 3)', exc_info=ex)
            raise ex

        try:
            self._field_vectors = validate_tensor(field_vectors, shape=(self.count, 3))
        except Exception as ex:
            logger.exception('field_vectors must be a float tensor of shape (M, 3)', exc_info=ex)
            raise ex

    @property
    def mtype(self) -> str:
        return self._mtype

    @property
    def reference_size(self) -> optid.types.TensorVector:
        return self._reference_size

    @property
    def reference_field_vector(self) -> optid.types.TensorVector:
        return self._reference_field_vector

    @property
    def flip_matrix(self) -> optid.types.TensorMatrix:
        return self._flip_matrix

    @property
    def flippable(self) -> bool:
        return self._flippable

    @property
    def names(self) -> optid.types.ListStrings:
        return self._names

    @property
    def sizes(self) -> optid.types.TensorVectors:
        return self._sizes

    @property
    def field_vectors(self) -> optid.types.TensorVectors:
        return self._field_vectors

    @property
    def count(self) -> int:
        return self._count

    def save(self, file : optid.types.BinaryFileHandle):
        """
        Saves a MagnetSet instance to a .magset file.

        Parameters
        ----------
        file : str or open writable file handle
            A path to where a .magset file should be created or overwritten, or an open writable file handle to
            a .magset file.
        """

        logger.info('Saving magnet set...')
        optid.utils.io.save(file, dict(
            mtype=self.mtype,
            reference_size=self.reference_size,
            reference_field_vector=self.reference_field_vector,
            flip_matrix=self.flip_matrix,
            names=self.names,
            sizes=self.sizes,
            field_vectors=self.field_vectors
        ))

    @staticmethod
    def from_file(file : optid.types.BinaryFileHandle) -> 'MagnetSet':
        """
        Constructs a MagnetSet instance from a .magset file.

        Parameters
        ----------
        file : str or open file handle
            A path to a .magset file or an open file handle to a .magset file.

        Returns
        -------
        A MagnetSet instance with the desired values loaded from the .magset file.
        """

        logger.info('Loading magnet set...')
        return MagnetSet(**optid.utils.io.from_file(file))
