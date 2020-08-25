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
import numpy as np

import optid
from optid.magnets import MagnetSet
from optid.utils import validate_tensor, validate_string
from optid.errors import FileHandleError, \
    ValidateMagnetGenomePermutationDuplicateError, \
    ValidateMagnetGenomePermutationBoundaryError

logger = optid.utils.logging.get_logger('optid.magnets.MagnetGenome')


class MagnetGenome:
    """
    Represents a permutation of magnets of the same type w.r.t a MagnetSet, MagnetSlots, and MagnetSlotsLookup
    object that are shared between multiple genomes. Instances of MagnetGenome hold an integer permutation tensor
    for magnet ordering and flip state, and hold an np.random.SeedSequence object used to split the RNG key for
    child genomes.

    The permutation of magnet orderings and flips holds values for all N magnets in the MagnetSet with N >= S
    for S MagnetSlots in the device configuration. We need to store the current permutation order of magnets that
    are not currently used in the device so that we can reload the genome states and have consistent RNG usage for
    future mutations.
    """

    def __init__(self,
                 magnet_type : str,
                 magnet_permutation : npt.NDArray[(typing.Any,), npt.Int],
                 magnet_flips : npt.NDArray[(typing.Any,), npt.Int],
                 rng_states : typing.Tuple[np.random.RandomState, np.random.RandomState]):
        """
        Constructs a MagnetGenome instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        magnet_type : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        magnet_permutation : int tensor (M,)
            A tensor of integer value pairs representing MagnetSet index values of [0, M-1].

        magnet_flips : bool tensor (M,)
            A tensor of integer value pairs representing flip state values of [False, True].

        rng_states : tuple of numpy random states
            A pair of rng states used to initialize two independent numpy random generators. One rng is used for
            spawning child genomes, the other is used exclusively for mutating this genome.
            This makes rng seeding safe for parallel execution.
        """

        try:
            self._magnet_type = validate_string(magnet_type, assert_non_empty=True)
        except Exception as ex:
            logger.exception('magnet_type must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._magnet_permutation = validate_tensor(magnet_permutation, shape=(None,), dtype=np.integer)
            self._count = self.magnet_permutation.shape[0]

            if len(set(self.magnet_permutation.tolist())) != self.count:
                raise ValidateMagnetGenomePermutationDuplicateError()

            if np.any((self.magnet_permutation < 0) | (self.magnet_permutation >= self.count)):
                raise ValidateMagnetGenomePermutationBoundaryError(permutation_size=self.count,
                                                                   min_index=self.magnet_permutation.min(),
                                                                   max_index=self.magnet_permutation.max())
        except Exception as ex:
            logger.exception('magnet_permutation must be an int tensor of shape (M,)', exc_info=ex)
            raise ex

        try:
            # Use of np.bool_ with trailing underscore is required for correct type checking!
            self._magnet_flips = validate_tensor(magnet_flips, shape=(self.count,), dtype=np.bool_)
        except Exception as ex:
            logger.exception('magnet_flips must be an bool tensor of shape (M,)', exc_info=ex)
            raise ex

        try:
            self._rng_children, self._rng_mutations = rng_states
            assert isinstance(self.rng_children, np.random.RandomState)
            assert isinstance(self.rng_mutations, np.random.RandomState)
        except Exception as ex:
            logger.exception('rng_states must be a tuple of valid numpy random states', exc_info=ex)
            raise ex

    @property
    def magnet_type(self):
        return self._magnet_type

    @property
    def magnet_permutation(self):
        return self._magnet_permutation

    @property
    def magnet_flips(self):
        return self._magnet_flips

    @property
    def count(self):
        return self._count

    @property
    def rng_children(self):
        return self._rng_children

    @property
    def rng_mutations(self):
        return self._rng_mutations

    @staticmethod
    def from_magnet_set(magnet_set : MagnetSet, seed : int) -> 'MagnetGenome':
        """
        Creates a randomly initialized genome using a MagnetSet for size information, and an integer seed to initialize
        RNGs for this genome instance.

        Parameters
        ----------
        magnet_set : MagnetSet
            A MagnetSet class instance to use to determine the size of the genome and the magnet type string.

        seed : int
            An integer seed to produce initialize the RNGs for this genome and sample the initial permutation and
            flip states.

        Returns
        -------
        A randomly initialized MagnetGenome for the given MagnetSet instance.
        """

        # Seed one RNG and use it to seed a second independent RNG
        rng_children  = np.random.RandomState(seed=seed)
        rng_mutations = np.random.RandomState(seed=rng_children.randint(np.iinfo(np.int32).max - 1))

        # Use rng_mutations to sample the initial genome permutation and flip states
        magnet_permutation = rng_mutations.permutation(magnet_set.count)
        magnet_flips = rng_mutations.randint(low=0, high=2, size=(magnet_set.count,)).astype(np.bool)

        # Construct and return the randomly initialized genome
        return MagnetGenome(magnet_type=magnet_set.magnet_type, magnet_permutation=magnet_permutation,
                            magnet_flips=magnet_flips, rng_states=(rng_children, rng_mutations))

    @staticmethod
    def from_magnet_genome(magnet_genome : 'MagnetGenome') -> 'MagnetGenome':
        """
        Creates an independent genome instance from an existing one. The child genome will have the same initial
        permutation and flip state as the parent but will have independent RNG states.

        Parameters
        ----------
        magnet_genome : MagnetGenome
            The existing genome instance to initialize from. The parents rng_children will be used to seed the RNGs
            in the new genome, modifying its state.

        Returns
        -------
        The independent child genome.
        """

        # Seed one RNG and use it to seed a second independent RNG
        rng_children = np.random.RandomState(seed=magnet_genome.rng_children.randint(np.iinfo(np.int32).max - 1))
        rng_mutations = np.random.RandomState(seed=rng_children.randint(np.iinfo(np.int32).max - 1))

        # Construct and return the independent genome
        return MagnetGenome(magnet_type=magnet_genome.magnet_type, magnet_permutation=magnet_genome.magnet_permutation,
                            magnet_flips=magnet_genome.magnet_flips, rng_states=(rng_children, rng_mutations))

    def save(self, file : typing.Union[str, typing.BinaryIO]):
        """
        Saves a MagnetGenome instance to a .maggenome file.

        Parameters
        ----------
        file : str or open writable file handle
            A path to where a .maggenome file should be created or overwritten, or an open writable file handle to
            a .maggenome file.
        """

        def write_file(file_handle : typing.BinaryIO):
            """
            Private helper function for writing data to a .maggenome file given an already open file handle.

            Parameters
            ----------
            file_handle : open writable file handle
                An open writable file handle to a .maggenome file.
            """

            # Pack members into .maggenome file as a single tuple
            pickle.dump((self.magnet_type, self.magnet_permutation, self.magnet_flips,
                         (self.rng_children, self.rng_mutations)), file_handle)

            logger.info('Saved magnet set to .maggenome file handle')

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logger.info('Saving magnet set to .maggenome file handle')
            write_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .maggenome file in a closure to ensure it gets closed on error
            with open(file, 'wb') as file_handle:
                logger.info('Saving magnet set to .maggenome file [%s]', file)
                write_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()

    @staticmethod
    def from_file(file : typing.Union[str, typing.BinaryIO]) -> 'MagnetGenome':
        """
        Constructs a MagnetGenome instance from a .maggenome file.

        Parameters
        ----------
        file : str or open file handle
            A path to a .maggenome file or an open file handle to a .maggenome file.

        Returns
        -------
        A MagnetGenome instance with the desired values loaded from the .maggenome file.
        """

        def read_file(file_handle : typing.BinaryIO) -> 'MagnetGenome':
            """
            Private helper function for reading data from a .maggenome file given an already open file handle.

            Parameters
            ----------
            file_handle : open file handle
                An open file handle to a .maggenome file.

            Returns
            -------
            A MagnetGenome instance with the desired values loaded from the .maggenome file.
            """

            # Unpack members from .maggenome file as a single tuple
            (magnet_type, magnet_permutation, magnet_flips, rng_states) = pickle.load(file_handle)

            # Offload object construction and validation to the MagnetGenome constructor
            magnet_genome = MagnetGenome(magnet_type=magnet_type, magnet_permutation=magnet_permutation,
                                         magnet_flips=magnet_flips, rng_states=rng_states)

            logger.info('Loaded magnet genome for type [%s] with [%d] magnets', magnet_type, magnet_genome.count)

            return magnet_genome

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logger.info('Loading magnet set from .maggenome file handle')
            return read_file(file_handle=file)

        elif isinstance(file, str):
            # Open the .maggenome file in a closure to ensure it gets closed on error
            with open(file, 'rb') as file_handle:
                logger.info('Loading magnet set from .maggenome file [%s]', file)
                return read_file(file_handle=file_handle)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()
