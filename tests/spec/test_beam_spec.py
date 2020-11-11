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


# Utility imports
import unittest
import numpy as np

# Test imports
import optid
from optid.spec import BeamSpec, MagnetSlotSpec
from optid.utils import validate_string, validate_tensor

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class BeamSpecTest(unittest.TestCase):
    """
    Tests the BeamSpec class can be imported and used correctly.
    """

    @staticmethod
    def dummy_beam_spec_values():
        """
        Creates a set of constant test values used for constructing and comparing BeamSpec
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        beam         = 'TEST'
        offset       = np.zeros((3,), dtype=np.float32)
        gap_vector   = np.array([0, 1, 0], dtype=np.float32)
        phase_vector = np.array([0, 0, 1], dtype=np.float32)

        return beam, offset, gap_vector, phase_vector

    def test_constructor(self):
        """
        Tests the BeamSpec class can be constructed with correct parameters.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        beam_spec = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

        self.assertEqual(beam_spec.name, beam)
        self.assertTrue(np.allclose(beam_spec.offset, offset))
        self.assertTrue(np.allclose(beam_spec.gap_vector, gap_vector))
        self.assertTrue(np.allclose(beam_spec.phase_vector, phase_vector))
        self.assertEqual(len(beam_spec.elements), 0)
        self.assertEqual(len(beam_spec.magnet_types), 0)
        self.assertEqual(beam_spec.count, 0)
        self.assertEqual(beam_spec.calculate_length(), 0)

    def test_constructor_raises_on_bad_parameters_name(self):
        """
        Tests the BeamSpec class throws exceptions when constructed with incorrect parameters.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        fixed_params = dict(offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

        self.assertRaisesRegex(optid.errors.ValidateStringEmptyError, '.*', BeamSpec, **fixed_params,
                               beam='')

    def test_constructor_raises_on_bad_parameters_offset(self):
        """
        Tests the BeamSpec class throws exceptions when constructed with incorrect parameters.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        fixed_params = dict(beam=beam, gap_vector=gap_vector, phase_vector=phase_vector)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', BeamSpec, **fixed_params,
                               offset=np.random.uniform(size=(4,)))

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', BeamSpec, **fixed_params,
                               offset=offset.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', BeamSpec, **fixed_params,
                               offset=None)

    def test_constructor_raises_on_bad_parameters_gap_vector(self):
        """
        Tests the BeamSpec class throws exceptions when constructed with incorrect parameters.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        fixed_params = dict(beam=beam, offset=offset, phase_vector=phase_vector)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', BeamSpec, **fixed_params,
                               gap_vector=np.random.uniform(size=(4,)))

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', BeamSpec, **fixed_params,
                               gap_vector=gap_vector.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', BeamSpec, **fixed_params,
                               gap_vector=None)

    def test_constructor_raises_on_bad_parameters_phase_vector(self):
        """
        Tests the BeamSpec class throws exceptions when constructed with incorrect parameters.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        fixed_params = dict(beam=beam, offset=offset, gap_vector=gap_vector)

        self.assertRaisesRegex(optid.errors.ValidateTensorShapeError, '.*', BeamSpec, **fixed_params,
                               phase_vector=np.random.uniform(size=(4,)))

        self.assertRaisesRegex(optid.errors.ValidateTensorElementTypeError, '.*', BeamSpec, **fixed_params,
                               phase_vector=phase_vector.astype(np.int32))

        self.assertRaisesRegex(optid.errors.ValidateTensorTypeError, '.*', BeamSpec, **fixed_params,
                               phase_vector=None)

    def test_register_magnet_type(self):
        """
        Tests the BeamSpec class can register magnet types with offsets.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        beam_spec = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

        mtype        = 'TEST'
        size         = np.ones((3,), dtype=np.float32)
        offset       = np.zeros((3,), dtype=np.float32)
        field_vector = np.array([0, 1, 0], dtype=np.float32)
        flip_matrix  = np.eye(3, dtype=np.float32)

        beam_spec.register_magnet_type(mtype=mtype, size=size, offset=offset,
                                       field_vector=field_vector, flip_matrix=flip_matrix)

        self.assertEqual(len(beam_spec.magnet_types), 1)
        self.assertTrue(mtype in beam_spec.magnet_types.keys())

    def test_register_magnet_type_raises_on_duplicate_mtype(self):
        """
        Tests the BeamSpec class throws exceptions when registering duplicate magnet type.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        beam_spec = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

        mtype        = 'TEST'
        size         = np.ones((3,), dtype=np.float32)
        offset       = np.zeros((3,), dtype=np.float32)
        field_vector = np.array([0, 1, 0], dtype=np.float32)
        flip_matrix  = np.eye(3, dtype=np.float32)

        beam_spec.register_magnet_type(mtype=mtype, size=size, offset=offset,
                                       field_vector=field_vector, flip_matrix=flip_matrix)

        self.assertRaisesRegex(Exception, '.*', beam_spec.register_magnet_type,
                               mtype=mtype, size=size, offset=offset,
                               field_vector=field_vector, flip_matrix=flip_matrix)

        self.assertEqual(len(beam_spec.magnet_types), 1)
        self.assertTrue(mtype in beam_spec.magnet_types.keys())

    def test_push_magnet(self):
        """
        Tests the BeamSpec class can push magnets.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        beam_spec = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

        mtype            = 'TEST'
        size             = np.ones((3,), dtype=np.float32)
        offset           = np.zeros((3,), dtype=np.float32)
        field_vector     = np.array([0, 1, 0], dtype=np.float32)
        flip_matrix      = np.eye(3, dtype=np.float32)
        direction_matrix = np.eye(3, dtype=np.float32)

        beam_spec.register_magnet_type(mtype=mtype, size=size, offset=offset,
                                       field_vector=field_vector, flip_matrix=flip_matrix)

        beam_spec.push_magnet(mtype=mtype, direction_matrix=direction_matrix, spacing=1)

        self.assertEqual(len(beam_spec.elements), 1)
        self.assertEqual(beam_spec.count, 1)
        self.assertEqual(beam_spec.calculate_length(), 1)

    def test_push_magnet_raises_on_bad_mtype(self):
        """
        Tests the BeamSpec class throws exceptions when mtype is not registered.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        beam_spec = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

        mtype            = 'TEST'
        direction_matrix = np.eye(3, dtype=np.float32)

        self.assertRaisesRegex(Exception, '.*', beam_spec.push_magnet,
                               mtype=mtype, direction_matrix=direction_matrix, spacing=1)

    def test_pop_magnet(self):
        """
        Tests the BeamSpec class can pop magnets.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        beam_spec = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

        mtype            = 'TEST'
        size             = np.ones((3,), dtype=np.float32)
        offset           = np.zeros((3,), dtype=np.float32)
        field_vector     = np.array([0, 1, 0], dtype=np.float32)
        flip_matrix      = np.eye(3, dtype=np.float32)
        direction_matrix = np.eye(3, dtype=np.float32)

        beam_spec.register_magnet_type(mtype=mtype, size=size, offset=offset,
                                       field_vector=field_vector, flip_matrix=flip_matrix)

        beam_spec.push_magnet(mtype=mtype, direction_matrix=direction_matrix, spacing=1)

        self.assertEqual(len(beam_spec.elements), 1)
        self.assertEqual(beam_spec.count, 1)
        self.assertEqual(beam_spec.calculate_length(), 1)

        beam_spec.push_magnet(mtype=mtype, direction_matrix=direction_matrix, spacing=1)

        self.assertEqual(len(beam_spec.elements), 2)
        self.assertEqual(beam_spec.count, 2)
        self.assertEqual(beam_spec.calculate_length(), 3)

        beam_spec.pop_magnet()

        self.assertEqual(len(beam_spec.elements), 1)
        self.assertEqual(beam_spec.count, 1)
        self.assertEqual(beam_spec.calculate_length(), 1)

    def test_pop_magnet_raises_on_empty_beam(self):
        """
        Tests the BeamSpec class throws exceptions when pop is called while beam is empty.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        beam_spec = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

        mtype            = 'TEST'
        size             = np.ones((3,), dtype=np.float32)
        offset           = np.zeros((3,), dtype=np.float32)
        field_vector     = np.array([0, 1, 0], dtype=np.float32)
        flip_matrix      = np.eye(3, dtype=np.float32)

        beam_spec.register_magnet_type(mtype=mtype, size=size, offset=offset,
                                       field_vector=field_vector, flip_matrix=flip_matrix)

        self.assertRaisesRegex(Exception, '.*', beam_spec.pop_magnet)

    def test_calculate_magnet_slots(self):
        """
        Tests the BeamSpec class can calculate magnet slots.
        """

        beam, offset, gap_vector, phase_vector = self.dummy_beam_spec_values()

        beam_spec = BeamSpec(beam=beam, offset=offset, gap_vector=gap_vector, phase_vector=phase_vector)

        mtype            = 'TEST'
        size             = np.ones((3,), dtype=np.float32)
        offset           = np.zeros((3,), dtype=np.float32)
        field_vector     = np.array([0, 1, 0], dtype=np.float32)
        flip_matrix      = np.eye(3, dtype=np.float32)
        direction_matrix = np.eye(3, dtype=np.float32)

        beam_spec.register_magnet_type(mtype=mtype, size=size, offset=offset,
                                       field_vector=field_vector, flip_matrix=flip_matrix)

        beam_spec.push_magnet(mtype=mtype, direction_matrix=direction_matrix, spacing=1)
        beam_spec.push_magnet(mtype=mtype, direction_matrix=direction_matrix, spacing=1)

        self.assertEqual(len(beam_spec.elements), 2)
        self.assertEqual(beam_spec.count, 2)
        self.assertEqual(beam_spec.calculate_length(), 3)

        slots = beam_spec.calculate_slot_specs(gap=0, phase=0, offset=np.zeros((3,), dtype=np.float32))

        self.assertEqual(len(slots), 2)

        for slot in slots:
            self.assertTrue(isinstance(slot, MagnetSlotSpec))
            self.assertEqual(slot.beam, beam)
            validate_string(slot.slot, assert_non_empty=True)
            self.assertEqual(slot.mtype, mtype)
            self.assertTrue(np.allclose(slot.size, size))
            validate_tensor(slot.position, shape=(3,))
            self.assertTrue(np.allclose(slot.field_vector, field_vector))
            self.assertTrue(np.allclose(slot.direction_matrix, direction_matrix))
            self.assertTrue(np.allclose(slot.flip_matrix, flip_matrix))
            self.assertTrue(np.allclose(slot.gap_vector, gap_vector))
            self.assertTrue(np.allclose(slot.phase_vector, phase_vector))

        self.assertTrue(slots[0].position[2] < slots[1].position[2])
