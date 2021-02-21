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


# External Imports
from beartype import beartype
import typing as typ
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

# Opt-ID Imports
from ..core.affine import \
    transform_rescaled_vectors

from ..core.bfield import \
    bfield_from_lookup

from ..device import \
    MagnetGroup


class MagnetGenome:

    @beartype
    def __init__(self,
            magnet_group: MagnetGroup,
            prngkey: jnp.ndarray,
            order: np.ndarray,
            flips: np.ndarray,
            mask: typ.Optional[np.ndarray] = None,
            static_bfield: typ.Optional[jnp.ndarray] = None,
            dynamic_bfield: typ.Optional[jnp.ndarray] = None):
        """
        Construct a MagnetGenome instance.

        :param magnet_group:
            Immutable reference for the optimisation domain.

        :param prngkey:
            JAX random state.

        :param order:
            Permutation of integers with shape (C,).

        :param flips:
            Integer flip state of each candidate with shape (C,).

        :param mask:
            Optional boolean masking of which candidates / slots can be optimised (C,).
            Assumed all active if not given.

        :param static_bfield:
            Optional pre-computed Bfield of just the inactive slots that are masked off.
            Computed if not given. Remains None if all slots are active.

        :param dynamic_bfield:
            Optional pre-computed Bfield of the current genome.
            Computed if not given.
        """

        self._magnet_group = magnet_group
        self._mutation_prngkey, self._children_prngkey = jax.random.split(prngkey)

        if order.shape != (magnet_group.ncandidate,):
            raise ValueError(f'order must be shape ({magnet_group.ncandidate},) but is : '
                             f'{order.shape}')

        if order.dtype != np.int32:
            raise TypeError(f'order must have dtype (int32) but is : '
                            f'{order.dtype}')

        self._order = order

        if flips.shape != (magnet_group.ncandidate,):
            raise ValueError(f'flips must be shape ({magnet_group.ncandidate},) but is : '
                             f'{flips.shape}')

        if flips.dtype != np.int32:
            raise TypeError(f'flips must have dtype (int32) but is : '
                            f'{flips.dtype}')

        if np.any((flips < 0) | (flips >= magnet_group.nflip)):
            raise ValueError(f'flips must all be in range [0, {magnet_group.nflip}) but is : '
                            f'{flips}')

        self._flips = flips

        if mask is not None:
            if mask.shape != (magnet_group.ncandidate,):
                raise ValueError(f'mask must be shape ({magnet_group.ncandidate},) but is : '
                                 f'{mask.shape}')

            if mask.dtype != np.bool_:
                raise TypeError(f'mask must have dtype (float32) but is : '
                                f'{mask.dtype}')

            nactive_slots = np.count_nonzero(mask[:self.magnet_group.nslot])
            if nactive_slots < 1:
                raise ValueError(f'mask must have at least one active slots')

            nactive_candidates = np.count_nonzero(mask)
            if nactive_candidates <= nactive_slots:
                raise ValueError(f'mask must have more active candidates than active slots')
        else:
            mask = np.ones((magnet_group.ncandidate,), dtype=np.bool_)

        # Unpack the mask to determine the ranges of slots and candidates that can be optimized
        self._mask = mask
        self._mask.setflags(write=False)
        self._active_candidates = np.argwhere(mask)
        self._active_slots = np.argwhere(mask[:self.magnet_group.nslot])
        self._inactive_slots = np.argwhere(~mask[:self.magnet_group.nslot])

        # Use the given initial masked bfield for inactive slots or compute it using the mask
        if static_bfield is None:
            static_bfield = self.calculate_static_bfield()
        else:
            if static_bfield.ndim != 4:
                raise ValueError(f'static_bfield must be a lattice of vectors with shape (X, Z, S, 3) but is : '
                                 f'{static_bfield.shape}')

            if static_bfield.shape[-1] != 3:
                raise ValueError(f'static_bfield must be a lattice of vectors with shape (..., 3) but is : '
                                 f'{static_bfield.shape}')

            if static_bfield.dtype != jnp.float32:
                raise TypeError(f'static_bfield must have dtype (float32) but is : '
                                f'{static_bfield.dtype}')

        self._static_bfield  = static_bfield

        # Use the given initial full bfield for the magnet group or compute it
        if dynamic_bfield is None:
            dynamic_bfield = self.calculate_dynamic_bfield()
        else:
            if dynamic_bfield.ndim != 4:
                raise ValueError(f'dynamic_bfield must be a lattice of vectors with shape (X, Z, S, 3) but is : '
                                 f'{dynamic_bfield.shape}')

            if dynamic_bfield.shape[-1] != 3:
                raise ValueError(f'dynamic_bfield must be a lattice of vectors with shape (..., 3) but is : '
                                 f'{dynamic_bfield.shape}')

            if dynamic_bfield.dtype != jnp.float32:
                raise TypeError(f'dynamic_bfield must have dtype (float32) but is : '
                                f'{dynamic_bfield.dtype}')

        self._dynamic_bfield = dynamic_bfield

    @beartype
    def calculate_slot_bfield(self, index: int) -> jnp.ndarray:

        if index < 0 or index >= self.magnet_group.nslot:
            raise ValueError(f'index must be in range [0, {self.magnet_group.nslot}) but is : '
                             f'{index}')

        # Extract data for this slot
        candidate   = self.magnet_group.candidate(self.order[index])
        slot        = self.magnet_group.slot(index)
        flip_matrix = self.magnet_group.flip_matrix(self.flips[index])

        # Compute the world space transformation for the magnet slot
        matrix = slot.direction_matrix @ flip_matrix @ slot.world_matrix

        # Transform the candidates field vector into world space at the magnet slot orientation
        vector = transform_rescaled_vectors(candidate.vector, matrix)

        # Calculate the bfield contribution for the current candidate in the selected slot
        return bfield_from_lookup(slot.lookup, vector)

    @beartype
    def calculate_bfield(self) -> jnp.ndarray:
        return self.calculate_dynamic_bfield() + self.static_bfield

    @beartype
    def calculate_dynamic_bfield(self) -> jnp.ndarray:

        # Compute the contribution over the active slots
        bfield = self.calculate_slot_bfield(self._active_slots[0])
        for index in self._active_slots[1:]:
            bfield += self.calculate_slot_bfield(index)

        return bfield

    @beartype
    def calculate_static_bfield(self) -> typ.Optional[jnp.ndarray]:

        # If no slots are inactive then the masked bfield is None
        if len(self._inactive_slots) == 0:
            return None

        # Compute the contribution over the inactive slots
        bfield = self.calculate_slot_bfield(self._inactive_slots[0])
        for index in self._inactive_slots[1:]:
            bfield += self.calculate_slot_bfield(index)

        return bfield

    @beartype
    def clone(self) -> 'MagnetGenome':

        self._children_prngkey, clone_prngkey = jax.random.split(self.children_prngkey)

        return MagnetGenome(magnet_group=self.magnet_group, prngkey=clone_prngkey,
                            order=self.order.copy(), flips=self.flips.copy(), mask=self.mask,
                            dynamic_bfield=self.dynamic_bfield.copy(), static_bfield=self.static_bfield)

    @beartype
    def swap_mutation(self, index_a: int, index_b: int):

        if index_a < 0 or index_a >= self.magnet_group.nslot:
            raise ValueError(f'index_a must be in range [0, {self.magnet_group.nslot}) but is : '
                             f'{index_a}')

        if index_b <= index_a or index_b >= self.magnet_group.ncandidate:
            raise ValueError(f'index_b must be in range ({index_a}, {self.magnet_group.ncandidate}) but is : '
                             f'{index_b}')

        if not self.mask[index_a]:
            raise ValueError(f'index_a slot must not be masked but is : '
                             f'{self.mask[index_a]}')

        if not self.mask[index_b]:
            raise ValueError(f'index_b slot must not be masked but is : '
                             f'{self.mask[index_b]}')

        # Compute the bfield contribution before the mutation
        bfield_old = self.calculate_slot_bfield(index_a)
        if index_b < self.magnet_group.nslot:
            bfield_old += self.calculate_slot_bfield(index_b)

        # Apply the mutation
        self.order[[index_a, index_b]] = self.order[[index_b, index_a]]
        self.flips[[index_a, index_b]] = self.flips[[index_b, index_a]]

        # Compute the bfield contribution after the mutation
        bfield_new = self.calculate_slot_bfield(index_a)
        if index_b < self.magnet_group.nslot:
            bfield_new += self.calculate_slot_bfield(index_b)

        # Compute the additive delta to the bfield that this full mutation (removal+swap+insertion) would produce
        self._dynamic_bfield += (bfield_new - bfield_old)

    def random_swap_mutation(self):

        # Split the prngkey for the mutation
        self._mutation_prngkey, prngkey_a, prngkey_b = jax.random.split(self.mutation_prngkey, num=3)

        # Sample a pair of uniform random magnet candidates where one candidate is currently in use
        index_a = jax.random.choice(prngkey_a, a=self._active_slots, shape=())
        index_b = jax.random.choice(prngkey_b, a=self._active_candidates[self._active_candidates > index_a], shape=())

        # Apply the mutation
        self.swap_mutation(index_a, index_b)

    @beartype
    def flip_mutation(self, index: int, flip: int):

        if index < 0 or index >= self.magnet_group.nslot:
            raise ValueError(f'index must be in range [0, {self.magnet_group.nslot}) but is : '
                             f'{index}')

        if flip < 0 or flip >= self.magnet_group.nflip:
            raise ValueError(f'flip must be in range [0, {self.magnet_group.nflip}) but is : '
                             f'{flip}')

        if flip == self.flips[index]:
            raise ValueError(f'flip mutation must change the flip state')

        if not self.mask[index]:
            raise ValueError(f'index slot must not be masked but is : '
                             f'{self.mask[index]}')

        # Compute the bfield contribution before the mutation
        bfield_old = self.calculate_slot_bfield(index)

        # Apply the mutation
        self.flips[index] = flip

        # Compute the bfield contribution after the mutation
        bfield_new = self.calculate_slot_bfield(index)

        # Compute the additive delta to the bfield that this full mutation (removal+flip+insertion) would produce
        self._dynamic_bfield += (bfield_new - bfield_old)

    def random_flip_mutation(self):

        # Split the prngkey for the mutation
        self._mutation_prngkey, prngkey_index, prngkey_flip = jax.random.split(self.mutation_prngkey, num=3)

        # Sample a uniform random magnet slot for candidate in use
        index = jax.random.choice(prngkey_index, a=self._active_slots, shape=())

        # Sample a uniform random flip state excluding the current state
        flip = jax.random.randint(prngkey_flip, shape=(), minval=0, maxval=(self.magnet_group.nflip - 1))
        flip = (flip + 1) if (flip >= self.flips[index]) else flip

        # Apply the mutation
        self.flip_mutation(index, flip)

    @beartype
    def shift_mutation(self, index_a: int, index_b: int, shift: int):

        if index_a < 0 or index_a >= self.magnet_group.nslot:
            raise ValueError(f'index_a must be in range [0, {self.magnet_group.nslot}) but is : '
                             f'{index_a}')

        if index_b <= index_a or index_b >= self.magnet_group.ncandidate:
            raise ValueError(f'index_b must be in range ({index_a}, {self.magnet_group.ncandidate}) but is : '
                             f'{index_b}')

        if shift == 0:
            raise ValueError(f'shift must be non-zero')

        # Determine range of candidates that will be effected by this mutation
        candidate_indices = self._active_candidates[(self._active_candidates >= index_a) &
                                                    (self._active_candidates <= index_b)]

        nindices = len(candidate_indices)
        if np.abs(shift) >= nindices:
            raise ValueError(f'shift must be in range ({-nindices}, {nindices}) but is : '
                             f'{shift}')

        # Determine the range of slots that will be effected
        slot_indices = candidate_indices[candidate_indices < self.magnet_group.nslot]

        # Determine if it is cheaper to recompute the bfield of update it in place
        update_bfield = (len(slot_indices) * 2) > self.magnet_group.nslot

        if update_bfield:
            # Compute the bfield contribution before the mutation
            bfield_old = self.calculate_slot_bfield(slot_indices[0])
            for index in slot_indices[1:]:
                bfield_old += self.calculate_slot_bfield(index)

        # Apply the mutation
        new_indices = np.roll(candidate_indices, axis=0, shift=shift)
        self.order[new_indices] = self.order[candidate_indices]
        self.flips[new_indices] = self.flips[candidate_indices]

        if update_bfield:
            # Compute the bfield contribution after the mutation
            bfield_new = self.calculate_slot_bfield(slot_indices[0])
            for index in slot_indices[1:]:
                bfield_new += self.calculate_slot_bfield(index)

            # Compute the additive delta to the bfield that this full mutation (removal+shift+insertion) would produce
            self._dynamic_bfield += (bfield_new - bfield_old)

        else:
            self._dynamic_bfield = self.calculate_dynamic_bfield()

    def random_shift_mutation(self):

        # Split the prngkey for the mutation
        self._mutation_prngkey, prngkey_a, prngkey_b, prngkey_shift = jax.random.split(self.mutation_prngkey, num=4)

        # Sample a pair of uniform random magnet candidates where one candidate is currently in use
        index_a = jax.random.choice(prngkey_a, a=self._active_slots, shape=())
        index_b = jax.random.choice(prngkey_b, a=self._active_candidates[self._active_candidates > index_a], shape=())

        # Sample an offset for the shift
        shift = jax.random.choice(prngkey_shift, a=[-1, 1], shape=())

        # Apply the mutation
        self.shift_mutation(index_a, index_b, shift)

    @beartype
    def to_dataframe(self, include_unused: bool = False):

        rows = []
        for index in range(self.magnet_group.nslot):

            # Extract data for this slot
            slot            = self.magnet_group.slot(index)
            candidate_index = self.order[index]
            candidate       = self.magnet_group.candidate(candidate_index)
            flip_index      = self.flips[index]
            flip_matrix     = self.magnet_group.flip_matrix(flip_index)
            world_matrix    = slot.direction_matrix @ flip_matrix @ slot.world_matrix
            mask            = self.mask[index]

            rows += [{
                'magnet_group': self.magnet_group.name,
                'slot_index': index,
                'slot_name': slot.name,
                'slot_beam': slot.beam,

                'candidate_index': candidate_index,
                'candidate_name': candidate.name,

                'mask': mask,

                'direction_matrix': slot.direction_matrix.tolist(),

                'flip_index': flip_index,
                'flip_matrix': flip_matrix.tolist(),

                'world_matrix': world_matrix.tolist(),
            }]

        if include_unused:
            for index in range(self.magnet_group.nslot, self.magnet_group.ncandidate):

                # Extract data for this unused candidate
                candidate_index = self.order[index]
                candidate       = self.magnet_group.candidate(candidate_index)
                mask            = self.mask[index]
                flip_index      = self.flips[index]
                flip_matrix     = self.magnet_group.flip_matrix(flip_index)

                rows += [{
                    'magnet_group': self.magnet_group.name,

                    'candidate_index': candidate_index,
                    'candidate_name': candidate.name,

                    'mask': mask,

                    'flip_index': flip_index,
                    'flip_matrix': flip_matrix.tolist(),
                }]

        return pd.DataFrame(rows)

    @property
    @beartype
    def magnet_group(self) -> MagnetGroup:
        return self._magnet_group

    @property
    @beartype
    def mutation_prngkey(self) -> jnp.ndarray:
        return self._mutation_prngkey

    @property
    @beartype
    def children_prngkey(self) -> jnp.ndarray:
        return self._children_prngkey

    @property
    @beartype
    def dynamic_bfield(self) -> jnp.ndarray:
        return self._dynamic_bfield

    @property
    @beartype
    def static_bfield(self) -> jnp.ndarray:
        return self._static_bfield

    @property
    @beartype
    def bfield(self) -> jnp.ndarray:
        return self.dynamic_bfield + self.static_bfield

    @property
    @beartype
    def order(self) -> np.ndarray:
        return self._order

    @property
    @beartype
    def flips(self) -> np.ndarray:
        return self._flips

    @property
    @beartype
    def mask(self) -> np.ndarray:
        return self._mask
