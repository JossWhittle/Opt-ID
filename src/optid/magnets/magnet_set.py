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
                 size : optid.types.TensorVector,
                 field_vector : optid.types.TensorVector,
                 flip_matrix : optid.types.TensorMatrix,
                 names : optid.types.ListStrings,
                 sizes : optid.types.TensorVectors,
                 field_vectors : optid.types.TensorVectors,
                 rescale_field_vector : bool = True):
        """
        Constructs a MagnetSet instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        mtype : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        size : float tensor (3,)
            A float tensor of a single 3-dim size for the AABB surrounding the reference magnet geometry.

        field_vector : float tensor (3,)
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

        rescale_field_vector : bool
            If true then normalize the magnitude of the reference field vector by the average magnitude of the
            per magnet field vectors.
        """

        try:
            self._mtype = validate_string(mtype, assert_non_empty=True)
        except Exception as ex:
            logger.exception('mtype must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._size = validate_tensor(size, shape=(3,))
        except Exception as ex:
            logger.exception('size must be a float tensor of shape (3,)', exc_info=ex)
            raise ex

        try:
            self._field_vector = validate_tensor(field_vector, shape=(3,))
        except Exception as ex:
            logger.exception('field_vector must be a float tensor of shape (3,)', exc_info=ex)
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

        if rescale_field_vector:
            reference_norm = np.linalg.norm(self.field_vector, axis=-1)
            average_norm = np.mean(np.linalg.norm(self.field_vectors, axis=-1))
            self._field_vector = (self.field_vector / reference_norm) * average_norm

    @property
    def mtype(self) -> str:
        return self._mtype

    @property
    def size(self) -> optid.types.TensorVector:
        return self._size

    @property
    def field_vector(self) -> optid.types.TensorVector:
        return self._field_vector

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
            size=self.size,
            field_vector=self.field_vector,
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
        return MagnetSet(**optid.utils.io.from_file(file), rescale_field_vector=False)

    @staticmethod
    def from_sim_file(mtype : str,
                      size : optid.types.TensorVector,
                      field_vector : optid.types.TensorVector,
                      flip_matrix : optid.types.TensorMatrix,
                      file : optid.types.ASCIIFileHandle,
                      rescale_field_vector : bool = True) -> 'MagnetSet':
        """
        Constructs a MagnetSet instance using per magnet names and field vectors from a .sim file provided by
        the magnet manufacturer.
        Parameters
        ----------
        mtype : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        size : float tensor (3,)
            A float tensor of a single 3-dim size for the AABB surrounding the reference magnet geometry.

        field_vector : float tensor (3,)
            A float tensor of a single 3-dim field vector for the magnetization of the reference magnet.

        flip_matrix : float tensor (3, 3)
            A float matrix of shape (3, 3) representing the flips that would be applied to a magnet in order to swap
            the direction of its minor axis vectors while keeping its major (easy) axis vector.

        file : str or open file handle
            A path to a .sim file or an open file handle to a .sim file containing per magnet names and field
            vectors as provided by the magnet manufacturer.

        rescale_field_vector : bool
            If true then normalize the magnitude of the reference field vector by the average magnitude of the
            per magnet field vectors.

        Returns
        -------
        A MagnetSet instance with the desired values loaded from the .sim file.
        """

        def read_file(mtype : str,
                      size : optid.types.TensorVector,
                      field_vector : optid.types.TensorVector,
                      flip_matrix : optid.types.TensorMatrix,
                      file_handle : typing.TextIO,
                      rescale_field_vector : bool) -> 'MagnetSet':

            try:
                # Load the data into python lists
                names, sizes, field_vectors = [], [], []

                for line_index, line in enumerate(file_handle):
                    # Skip this line if it is blank
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    logger.debug('Line [%d] : [%s]', line_index, line)

                    # Unpack and parse values for the current magnet
                    name, field_x, field_z, field_s = line.split()
                    names += [name]
                    sizes += [size]
                    field_vectors += [(float(field_x), float(field_z), float(field_s))]

                sizes = np.stack(sizes, axis=0)
                field_vectors = np.array(field_vectors, dtype=np.float32)

            except Exception as ex:
                logger.exception('Failed to load magnet set from .sim file', exc_info=ex)
                raise ex

            return MagnetSet(mtype=mtype, size=size, field_vector=field_vector, flip_matrix=flip_matrix,
                             names=names, sizes=sizes, field_vectors=field_vectors,
                             rescale_field_vector=rescale_field_vector)

        if isinstance(file, io.TextIOWrapper):
            # Load directly from the already open file handle
            logger.info('Loading magnet set [%s] from open .sim file handle', mtype)
            return read_file(mtype=mtype, size=size, field_vector=field_vector,
                             flip_matrix=flip_matrix, file_handle=file,
                             rescale_field_vector=rescale_field_vector)

        elif isinstance(file, str):
            # Open the .sim file in a closure to ensure it gets closed on error
            with open(file, 'r') as file_handle:
                logger.info('Loading magnet set [%s] from .sim file [%s]', mtype, file)
                return read_file(mtype=mtype, size=size, field_vector=field_vector,
                                 flip_matrix=flip_matrix, file_handle=file_handle,
                                 rescale_field_vector=rescale_field_vector)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise optid.utils.io.FileHandleError()
