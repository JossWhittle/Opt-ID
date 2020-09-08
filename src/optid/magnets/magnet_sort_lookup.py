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
import nptyping as npt
import pickle

import optid
from optid.utils import Range, validate_tensor, validate_string, validate_range
from optid.errors import FileHandleError

logger = optid.utils.logging.get_logger('optid.magnets.MagnetSortLookup')


class MagnetSortLookup:
    """
    Represents a B-field contribution lookup table for a set of magnet slots.
    """

    def __init__(self,
                 magnet_type : str,
                 x_range : Range, z_range : Range, s_range : Range,
                 lookup : optid.types.TensorSortLookup):
        """
        Constructs a MagnetSortLookup instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        magnet_type : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        x_range : Range(min : float, max : float, steps : int)
            A tuple representing the sample range across the x-axis for the lookup table. This is used for validating
            that multiple MagnetSortLookup objects represent lookup tables constructed over the same sampling grid.
            The grid is used to construct samples w.r.t an np.linspace(*x_range) == np.linspace(min, max, steps)

        z_range : Range(min : float, max : float, steps : int)
            A tuple representing the sample range across the z-axis for the lookup table.
            See documentation for x_range parameter.

        s_range : Range(min : float, max : float, steps : int)
            A tuple representing the sample range across the s-axis for the lookup table.
            See documentation for x_range parameter.

        lookup : float tensor (S, x_steps, z_steps, s_steps, 3, 3)
            A float tensor of shape (slots, x_steps, z_steps, s_steps, 3, 3) representing the unscaled magnetic field
            strengths in the x, z, and s axes (last two dimensions), sampled at each spatial location represented by
            the lattice over x_range, z_range, and s_range using np.meshgrid (middle three dimensions), and duplicated
            for each of the magnet slots (first dimension).

            At each sample location of each slot entry we store B-field contribution in each axis assuming each axis
            is the major field axis:
                [[Bxx, Bzx, Bsx]
                 [Bxz, Bzz, Bsz]
                 [Bxs, Bzs, Bss]]

            Given a measured magnet strength vector (Mx, Mz, Ms) from a MagnetSet element that is currently placed
            within a magnet slot modelled by a MagnetSlots and a MagnetSortLookup instance, we compute the scaled
            magnetic field strength intensities in each axis at each sample location specified by the lookup table
            by matmul'ing the matrix at that location against the measured magnet strength.
                Fx = (Bxx * Mx) + (Bxz * Mz) + (Bxs * Ms)
                Fz = (Bzx * Mx) + (Bzz * Mz) + (Bzs * Ms)
                Fs = (Bsx * Mx) + (Bsz * Mz) + (Bss * Ms)
        """

        try:
            self._magnet_type = validate_string(magnet_type, assert_non_empty=True)
        except Exception as ex:
            logger.exception('name must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._x_range = validate_range(x_range)
        except Exception as ex:
            logger.exception('x_range must be a valid range', exc_info=ex)
            raise ex

        try:
            self._z_range = validate_range(z_range)
        except Exception as ex:
            logger.exception('z_range must be a valid range', exc_info=ex)
            raise ex

        try:
            self._s_range = validate_range(s_range)
        except Exception as ex:
            logger.exception('s_range must be a valid range', exc_info=ex)
            raise ex

        try:
            self._lookup = validate_tensor(lookup, shape=(None,
                                                          self.x_range.steps,
                                                          self.z_range.steps,
                                                          self.s_range.steps, 3, 3))

            # Number of magnet slots derived from shape of lookup tensor.
            self._count = self.lookup.shape[0]
        except Exception as ex:
            logger.exception('lookup must be float tensor of shape (N, x_steps, z_steps, s_steps, 3, 3)', exc_info=ex)
            raise ex

    @property
    def magnet_type(self) -> str:
        return self._magnet_type

    @property
    def x_range(self) -> Range:
        return self._x_range

    @property
    def z_range(self) -> Range:
        return self._z_range

    @property
    def s_range(self) -> Range:
        return self._s_range

    @property
    def lookup(self) -> optid.types.TensorSortLookup:
        return self._lookup

    @property
    def count(self) -> int:
        return self._count

    def save(self, file : typing.Union[str, typing.BinaryIO]):
        """
        Saves a MagnetSortLookup instance to a .magsortlookup file.

        Parameters
        ----------
        file : str or open writable file handle
            A path to where a .magsortlookup file should be created or overwritten, or an open writable file handle to
            a .magsortlookup file.
        """

        def write_file(file_handle : typing.BinaryIO):
            """
            Private helper function for writing data to a .magsortlookup file given an already open file handle.

            Parameters
            ----------
            file_handle : open writable file handle
                An open writable file handle to a .magsortlookup file.
            """

            # Pack members into .magsortlookup file as a single tuple
            pickle.dump((self.magnet_type, self.x_range, self.z_range, self.s_range, self.lookup), file_handle)

            logger.info('Saved magnet lookup to .magsortlookup file handle')

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logger.info('Saving magnet slots to .magsortlookup file handle')
            write_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .magsortlookup file in a closure to ensure it gets closed on error
            with open(file, 'wb') as file_handle:
                logger.info('Saving magnet slots to .magsortlookup file [%s]', file)
                write_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()

    @staticmethod
    def from_file(file : typing.Union[str, typing.BinaryIO]) -> 'MagnetSortLookup':
        """
        Constructs a MagnetSlots instance from a .magsortlookup file.

        Parameters
        ----------
        file : str or open file handle
            A path to a .magsortlookup file or an open file handle to a .magsortlookup file.

        Returns
        -------
        A MagnetSet instance with the desired values loaded from the .magsortlookup file.
        """

        def read_file(file_handle : typing.BinaryIO) -> 'MagnetSortLookup':
            """
            Private helper function for reading data from a .magsortlookup file given an already open file handle.

            Parameters
            ----------
            file_handle : open file handle
                An open file handle to a .magsortlookup file.

            Returns
            -------
            A MagnetSet instance with the desired values loaded from the .magsortlookup file.
            """

            # Unpack members from .magsortlookup file as a single tuple
            (magnet_type, x_range, z_range, s_range, lookup) = pickle.load(file_handle)

            # Offload object construction and validation to the MagnetSlots constructor
            magnet_lookup = MagnetSortLookup(magnet_type=magnet_type, x_range=x_range,
                                             z_range=z_range, s_range=s_range, lookup=lookup)

            logger.info('Loaded magnet lookup [%s] with [%d] slots', magnet_type, magnet_lookup.count)

            return magnet_lookup

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logger.info('Loading magnet set from .magsortlookup file handle')
            return read_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .magsortlookup file in a closure to ensure it gets closed on error
            with open(file, 'rb') as file_handle:
                logger.info('Loading magnet set from .magsortlookup file [%s]', file)
                return read_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()
