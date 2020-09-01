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
from optid.magnets import MagnetSet, MagnetSlots, MagnetLookup
from optid.utils import validate_tensor, validate_string
from optid.errors import FileHandleError, \
    ValidateMagnetGenomePermutationDuplicateError, \
    ValidateMagnetGenomePermutationBoundaryError

import logging
logger = optid.utils.logging.get_logger('optid.magnets.MagnetGenome')


Permutation_Type = npt.NDArray[(typing.Any,), npt.Int]
Flips_Type       = npt.NDArray[(typing.Any,), npt.Bool]
Bfield_Type      = npt.NDArray[(typing.Any, typing.Any, typing.Any, 3), npt.Float]


class MagnetGenome:
    """
    Represents a permutation of magnets of the same type w.r.t a MagnetSet, MagnetSlots, and MagnetLookup
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
                 permutation : Permutation_Type,
                 flips : Flips_Type,
                 rng_states : typing.Tuple[np.random.RandomState, np.random.RandomState],
                 magnet_set : MagnetSet,
                 magnet_slots : MagnetSlots,
                 magnet_lookup : MagnetLookup):
        """
        Constructs a MagnetGenome instance and validates the values are the correct types and consistent sizes.

        Parameters
        ----------
        magnet_type : str
            A non-empty string name for this magnet type that should be unique in the context of the full insertion
            device. Names such as 'HH', 'VV', 'HE', 'VE', 'HT' are common.

        permutation : int tensor (M,)
            A tensor of integer value pairs representing MagnetSet index values of [0, M-1].

        flips : bool tensor (M,)
            A tensor of integer value pairs representing flip state values of [False, True].

        rng_states : tuple of numpy random states
            A pair of rng states used to initialize two independent numpy random generators. One rng is used for
            spawning child genomes, the other is used exclusively for mutating this genome.
            This makes rng seeding safe for parallel execution.

        magnet_set : MagnetSet
            The set of real magnet values used to gather field vectors.

        magnet_slots : MagnetSlots
            The configuration of magnet slots used to gather the flip matrix.

        magnet_lookup : MagnetLookup
            The lookup table used for the magnet slots to compute the bfields from.
        """

        try:
            self._magnet_type = validate_string(magnet_type, assert_non_empty=True)
        except Exception as ex:
            logger.exception('magnet_type must be a non-empty string', exc_info=ex)
            raise ex

        try:
            self._magnet_set = magnet_set
            assert isinstance(self.magnet_set, MagnetSet)
            assert self.magnet_type == self.magnet_set.magnet_type

            self._count = self.magnet_set.count
        except Exception as ex:
            logger.exception('magnet_set must be a MagnetSet instance', exc_info=ex)
            raise ex

        try:
            self._magnet_slots = magnet_slots
            assert isinstance(self.magnet_slots, MagnetSlots)
            assert self.magnet_type == self.magnet_slots.magnet_type
            assert self.count >= self.magnet_slots.count

        except Exception as ex:
            logger.exception('magnet_slots must be a MagnetSlots instance', exc_info=ex)
            raise ex

        try:
            self._magnet_lookup = magnet_lookup
            assert isinstance(self.magnet_lookup, MagnetLookup)
            assert self.magnet_type == self.magnet_lookup.magnet_type
            assert self.count >= self.magnet_lookup.count
            assert self.magnet_slots.count == self.magnet_lookup.count

        except Exception as ex:
            logger.exception('magnet_lookup must be a MagnetLookup instance', exc_info=ex)
            raise ex

        try:
            self._permutation = validate_tensor(permutation, shape=(self.count,), dtype=np.integer)

            if len(set(self.permutation.tolist())) != self.count:
                raise ValidateMagnetGenomePermutationDuplicateError()

            if np.any((self.permutation < 0) | (self.permutation >= self.count)):
                raise ValidateMagnetGenomePermutationBoundaryError(permutation_size=self.count,
                                                                   min_index=self.permutation.min(),
                                                                   max_index=self.permutation.max())
        except Exception as ex:
            logger.exception('permutation must be an int tensor of shape (M,)', exc_info=ex)
            raise ex

        try:
            # Use of np.bool_ with trailing underscore is required for correct type checking!
            self._flips = validate_tensor(flips, shape=(self.count,), dtype=np.bool_)

        except Exception as ex:
            logger.exception('flips must be an bool tensor of shape (M,)', exc_info=ex)
            raise ex

        try:
            self._rng_children, self._rng_mutations = rng_states
            assert isinstance(self.rng_children, np.random.RandomState)
            assert isinstance(self.rng_mutations, np.random.RandomState)
        except Exception as ex:
            logger.exception('rng_states must be a tuple of valid numpy random states', exc_info=ex)
            raise ex

    @property
    def magnet_type(self) -> str:
        return self._magnet_type

    @property
    def permutation(self) -> Permutation_Type:
        return self._permutation

    @property
    def flips(self) -> Flips_Type:
        return self._flips

    @property
    def count(self) -> int:
        return self._count

    @property
    def rng_children(self) -> np.random.RandomState:
        return self._rng_children

    @property
    def rng_mutations(self) -> np.random.RandomState:
        return self._rng_mutations

    @property
    def magnet_set(self) -> MagnetSet:
        return self._magnet_set

    @property
    def magnet_slots(self) -> MagnetSlots:
        return self._magnet_slots

    @property
    def magnet_lookup(self) -> MagnetLookup:
        return self._magnet_lookup

    def calculate_slot_bfield(self, index : int) -> Bfield_Type:
        """
        Computes the bfield for the magnet from the magnet set used by the genome to fill the selected magnet slot.

        Parameters
        ----------
        index : int
            The magnet slot to compute the bfield for.

        Returns
        -------
        The bfield computed from the given magnet slot.
        """

        # Get the set index and flip state for the magnet in the Nth slot of the genome
        set_index  = self.permutation[index]

        # Get the field strength vector (before potential flipping) for the real magnet currently in the desired slot
        field_vector = self.magnet_set.field_vectors[set_index]

        if self.magnet_slots.flippable and self.flips[index]:
            # Flip the field vector by the flip matrix if the genome says this slot is currently flipped
            field_vector = np.dot(self.magnet_slots.flip_matrix, field_vector)

        # Scale the lookup w.r.t the (potentially flipped) field vector for the magnet under consideration
        bfield = np.dot(self.magnet_lookup.lookup[index], field_vector)

        return bfield

    def calculate_bfield(self) -> Bfield_Type:
        """
        Computes the full bfield for the genome.

        Returns
        -------
        The full bfield computed from this genome.
        """

        # Compute the sum of the individual bfield contributions for each magnet slot
        return sum(self.calculate_slot_bfield(index=index)
                   for index in range(self.magnet_slots.count))

    def flip_mutation(self, index : int) -> Bfield_Type:
        """
        Performs a flip mutation at the selected magnet slot and returns the bfield delta from performing the mutation.

        Parameters
        ----------
        index : int
            The magnet slot to flip.

        Returns
        -------
        The bfield delta computed from the mutation.
        """

        # Can only apply flip mutations if this genome is flippable (non-identity flip matrix on magnet slots)
        assert self.magnet_slots.flippable

        # Index needs to be valid within the magnet_slots which may be fewer than the full magnet_genome and magnet_set
        # Flipping a magnet within the genome but not currently in an active slot would be a waste of computation
        assert 0 <= index < self.magnet_slots.count

        # Compute the bfield contribution of this magnet before the mutation
        bfield_old = self.calculate_slot_bfield(index=index)

        # Apply the mutation
        self.flips[index] = ~self.flips[index]

        # Compute the bfield contribution of this magnet after the mutation
        bfield_new = self.calculate_slot_bfield(index=index)

        # Compute the additive delta to the bfield that this full mutation (removal+flip+insertion) would produce
        bfield_delta = (bfield_new - bfield_old)

        return bfield_delta

    def random_flip_mutation(self) -> Bfield_Type:
        """
        Performs a flip mutation at a random magnet slot and returns the bfield delta from performing the mutation.

        Returns
        -------
        The bfield delta computed from the mutation.
        """

        # Sample index in the range of used slots
        index = self.rng_mutations.randint(low=0, high=self.magnet_slots.count)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Applying random flip mutation to slot [%d] of [%s] genome...',
                         index, self.magnet_type)

        # Perform the selected flip mutation
        return self.flip_mutation(index=index)

    def exchange_mutation(self, index_a : int, index_b : int) -> Bfield_Type:
        """
        Performs a swap mutation between the selected magnet slots and returns the bfield delta from performing the
        mutation.

        Parameters
        ----------
        index_a : int
            The first magnet slot to swap.

        index_b : int
            The second magnet slot to swap.

        Returns
        -------
        The bfield delta computed from the mutation.
        """

        # At least one of the indices being swapped must be for a magnet currently being used in a magnet slot
        # Swapping two magnets within the genome but not currently in active slots would be a waste of computation
        assert (0 <= index_a < self.magnet_slots.count) or \
               (0 <= index_b < self.magnet_slots.count)
        assert (0 <= index_a < self.magnet_set.count) and \
               (0 <= index_b < self.magnet_set.count) and \
               (index_a != index_b)

        # Compute the bfield contribution before the mutation
        bfield_old = []
        if index_a < self.magnet_slots.count:
            bfield_old += [self.calculate_slot_bfield(index=index_a)]
        if index_b < self.magnet_slots.count:
            bfield_old += [self.calculate_slot_bfield(index=index_b)]

        # Apply the mutation
        self.permutation[[index_a, index_b]] = self.permutation[[index_b, index_a]]
        self.flips[[index_a, index_b]] = self.flips[[index_b, index_a]]

        # Compute the bfield contribution after the mutation
        bfield_new = []
        if index_a < self.magnet_slots.count:
            bfield_new += [self.calculate_slot_bfield(index=index_a)]
        if index_b < self.magnet_slots.count:
            bfield_new += [self.calculate_slot_bfield(index=index_b)]

        # Compute the additive delta to the bfield that this full mutation (removal+swap+insertion) would produce
        bfield_delta = (sum(bfield_new) - sum(bfield_old))

        return bfield_delta

    def random_exchange_mutation(self) -> Bfield_Type:
        """
        Performs a exchange mutation at a random magnet slot and returns the bfield delta from performing the mutation.

        Returns
        -------
        The bfield delta computed from the mutation.
        """

        # Sample first index in the range of used slots
        index_a = self.rng_mutations.randint(low=0, high=self.magnet_slots.count)

        # Sample second index in the full range including currently unused magnets
        # Specifically the range is one fewer than the full set to avoid collisions between the two indices
        index_b = self.rng_mutations.randint(low=0, high=(self.magnet_set.count - 1))

        # If the second index is equal to the first or is larger than it, then shift it by +1 to uniformly sample
        # over the other indices
        if index_b >= index_a:
            index_b += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Applying random exchange mutation to slots [%d] and [%d] of [%s] genome...',
                         index_a, index_b, self.magnet_type)

        # Perform the selected exchange mutation
        return self.exchange_mutation(index_a=index_a, index_b=index_b)

    def random_mutation(self) -> Bfield_Type:
        """
        Performs a random mutation at a random magnet slot and returns the bfield delta from performing the mutation.

        Returns
        -------
        The bfield delta computed from the mutation.
        """

        # The set of mutation functions we can choose between
        mutation_fns = [self.random_exchange_mutation, self.random_flip_mutation]

        # Normalize the probabilities for each type of mutation so they sum to 1
        probabilities  = np.array([True, self.magnet_slots.flippable]).astype(np.float32)
        probabilities /= np.sum(probabilities)

        # Sample a single mutation index using the above normalized probabilities
        mutation = self.rng_mutations.choice(probabilities.shape[0], p=probabilities)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Applying random mutation [%s] to [%s] genome...',
                         ['exchange', 'flip'][mutation], self.magnet_type)

        # Apply the selected mutation, return the bfield delta
        return mutation_fns[mutation]()

    @staticmethod
    def from_random(seed : int,
                    magnet_set : MagnetSet,
                    magnet_slots : MagnetSlots,
                    magnet_lookup : MagnetLookup) -> 'MagnetGenome':
        """
        Creates a randomly initialized genome using a MagnetSet for size information, a MagnetSlots for flip masking,
        and an integer seed to initialize RNGs for this genome instance.

        Parameters
        ----------
        seed : int
            An integer seed to produce initialize the RNGs for this genome and sample the initial permutation and
            flip states.

        magnet_set : MagnetSet
            A MagnetSet class instance to use to determine the size of the genome and the magnet type string.

        magnet_slots : MagnetSlots
            A MagnetSlots class instance to use to determine the whether elements are flippable. Magnet type must be
            consistent.

        magnet_lookup : MagnetLookup
            The lookup table used for the magnet slots to compute the bfields from.

        Returns
        -------
        A randomly initialized MagnetGenome for the given MagnetSet instance.
        """

        assert magnet_set.magnet_type == magnet_slots.magnet_type

        # Seed one RNG and use it to seed a second independent RNG
        max_rand_int  = (np.iinfo(np.int32).max - 1)
        rng_children  = np.random.RandomState(seed=seed)
        rng_mutations = np.random.RandomState(seed=rng_children.randint(low=0, high=max_rand_int))

        # Use rng_mutations to sample the initial genome permutation and flip states
        permutation = rng_mutations.permutation(magnet_set.count)

        if magnet_slots.flippable:
            flips = rng_mutations.randint(low=0, high=2, size=(magnet_set.count,)).astype(np.bool)
        else:
            flips = np.zeros((magnet_set.count,), dtype=np.bool)

        # Construct and return the randomly initialized genome
        return MagnetGenome(magnet_type=str(magnet_set.magnet_type),
                            permutation=permutation, flips=flips, rng_states=(rng_children, rng_mutations),
                            magnet_set=magnet_set, magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

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
        max_rand_int  = (np.iinfo(np.int32).max - 1)
        rng_children  = np.random.RandomState(seed=magnet_genome.rng_children.randint(low=0, high=max_rand_int))
        rng_mutations = np.random.RandomState(seed=rng_children.randint(low=0, high=max_rand_int))

        # Construct and return the independent genome
        return MagnetGenome(magnet_type=str(magnet_genome.magnet_type), permutation=magnet_genome.permutation.copy(),
                            flips=magnet_genome.flips.copy(), rng_states=(rng_children, rng_mutations),
                            magnet_set=magnet_genome.magnet_set,
                            magnet_slots=magnet_genome.magnet_slots,
                            magnet_lookup=magnet_genome.magnet_lookup)

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
            pickle.dump((self.magnet_type, self.permutation, self.flips,
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
    def from_file(file : typing.Union[str, typing.BinaryIO],
                  magnet_set : MagnetSet,
                  magnet_slots : MagnetSlots,
                  magnet_lookup : MagnetLookup) -> 'MagnetGenome':
        """
        Constructs a MagnetGenome instance from a .maggenome file.

        Parameters
        ----------
        file : str or open file handle
            A path to a .maggenome file or an open file handle to a .maggenome file.

        magnet_set : MagnetSet
            A MagnetSet class instance to use to determine the size of the genome and the magnet type string.

        magnet_slots : MagnetSlots
            A MagnetSlots class instance to use to determine the whether elements are flippable. Magnet type must be
            consistent.

        magnet_lookup : MagnetLookup
            The lookup table used for the magnet slots to compute the bfields from.

        Returns
        -------
        A MagnetGenome instance with the desired values loaded from the .maggenome file.
        """

        def read_file(file_handle : typing.BinaryIO,
                      magnet_set : MagnetSet,
                      magnet_slots : MagnetSlots,
                      magnet_lookup : MagnetLookup) -> 'MagnetGenome':
            """
            Private helper function for reading data from a .maggenome file given an already open file handle.

            Parameters
            ----------
            file_handle : open file handle
                An open file handle to a .maggenome file.

            magnet_set : MagnetSet
            A MagnetSet class instance to use to determine the size of the genome and the magnet type string.

            magnet_slots : MagnetSlots
                A MagnetSlots class instance to use to determine the whether elements are flippable. Magnet type must be
                consistent.

            magnet_lookup : MagnetLookup
                The lookup table used for the magnet slots to compute the bfields from.

            Returns
            -------
            A MagnetGenome instance with the desired values loaded from the .maggenome file.
            """

            # Unpack members from .maggenome file as a single tuple
            (magnet_type, permutation, flips, rng_states) = pickle.load(file_handle)

            # Offload object construction and validation to the MagnetGenome constructor
            magnet_genome = MagnetGenome(magnet_type=magnet_type, permutation=permutation, flips=flips,
                                         rng_states=rng_states, magnet_set=magnet_set,
                                         magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

            logger.info('Loaded magnet genome for type [%s] with [%d] magnets', magnet_type, magnet_genome.count)

            return magnet_genome

        if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
            # Load directly from the already open file handle
            logger.info('Loading magnet set from .maggenome file handle')
            return read_file(file_handle=file, magnet_set=magnet_set,
                             magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        elif isinstance(file, str):
            # Open the .maggenome file in a closure to ensure it gets closed on error
            with open(file, 'rb') as file_handle:
                logger.info('Loading magnet set from .maggenome file [%s]', file)
                return read_file(file_handle=file_handle, magnet_set=magnet_set,
                                 magnet_slots=magnet_slots, magnet_lookup=magnet_lookup)

        else:
            # Assert that the file object provided is an open file handle or can be used to open one
            raise FileHandleError()
