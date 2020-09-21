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
import os
import unittest
import tempfile
import shutil
import inspect
import numpy as np

# Test imports
import optid
from optid.magnets import MagnetSortLookup
from optid.utils import Range, Grid

# Configure debug logging
from optid.utils.logging import attach_console_logger
attach_console_logger(remove_existing=True)


class MagnetSortLookupTest(unittest.TestCase):
    """
    Tests the MagnetSortLookup class can be imported and used correctly.
    """

    @staticmethod
    def dummy_magnet_lookup_values():
        """
        Creates a set of constant test values used for constructing and comparing MagnetSortLookup
        instances across test cases.

        Returns
        -------
        A tuple of the necessary fields.
        """

        count = 4
        mtype  = 'HH'
        grid = Grid(x_range=Range(min=-1, max=1, steps=5),
                    z_range=Range(min=-1, max=1, steps=5),
                    s_range=Range(min=-1, max=1, steps=5))
        lookup = np.empty((count, *grid.steps, 3, 3), dtype=np.float32)
        lookup[..., :, :] = np.eye(3, dtype=np.float32)

        return count, mtype, grid, lookup

    def test_constructor(self):
        """
        Tests the MagnetSortLookup class can be constructed with correct parameters.
        """

        # Make dummy parameters
        count, mtype, grid, lookup = self.dummy_magnet_lookup_values()

        # Construct MagnetSortLookup instance
        magnet_lookup = MagnetSortLookup(mtype=mtype, grid=grid, lookup=lookup)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_lookup.count, count)
        self.assertEqual(magnet_lookup.grid, grid)
        self.assertEqual(magnet_lookup.mtype, mtype)
        self.assertTrue(np.allclose(magnet_lookup.lookup, lookup))

    def test_constructor_raises_on_bad_parameters_mtype(self):
        """
        Tests the MagnetSortLookup class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, grid, lookup = self.dummy_magnet_lookup_values()

        fixed_params = dict(grid=grid, lookup=lookup)

        self.assertRaises(optid.errors.ValidateStringEmptyError, MagnetSortLookup, **fixed_params,
                          mtype='')

        self.assertRaises(optid.errors.ValidateStringTypeError, MagnetSortLookup, **fixed_params,
                          mtype=None)

    def test_constructor_raises_on_bad_parameters_grid(self):
        """
        Tests the MagnetSortLookup class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, grid, lookup = self.dummy_magnet_lookup_values()

        self.assertRaises(Exception, MagnetSortLookup,
                          mtype=mtype, grid=None, lookup=lookup)

    def test_constructor_raises_on_bad_parameters_lookup(self):
        """
        Tests the MagnetSortLookup class throws exceptions when constructed with incorrect parameters.
        """

        # Make dummy parameters
        count, mtype, grid, lookup = self.dummy_magnet_lookup_values()

        fixed_params = dict(mtype=mtype, grid=grid)

        # Assert constructor throws error from incorrectly shaped lookup
        self.assertRaises(optid.errors.ValidateTensorShapeError, MagnetSortLookup, **fixed_params,
                          lookup=lookup[0])

        # Assert constructor throws error from incorrectly typed lookup
        self.assertRaises(optid.errors.ValidateTensorElementTypeError, MagnetSortLookup, **fixed_params,
                          lookup=lookup.astype(np.int32))

        # Assert constructor throws error from incorrectly typed lookup
        self.assertRaises(optid.errors.ValidateTensorTypeError, MagnetSortLookup, **fixed_params,
                          lookup=None)

    def test_save(self):
        """
        Tests the MagnetSortLookup class can be saved to a .magsortlookup file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, mtype, grid, lookup = self.dummy_magnet_lookup_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magsortlookup')

            # Construct MagnetSortLookup instance
            magnet_lookup = MagnetSortLookup(mtype=mtype, grid=grid, lookup=lookup)

            # Save the MagnetSlots to the temporary directory
            magnet_lookup.save(file=tmp_file_path)

            # Throw away the local object and reload it from the temporary file
            magnet_lookup = MagnetSortLookup.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_lookup.count, count)
        self.assertEqual(magnet_lookup.grid, grid)
        self.assertEqual(magnet_lookup.mtype, mtype)
        self.assertTrue(np.allclose(magnet_lookup.lookup, lookup))

    def test_save_open_file_handle(self):
        """
        Tests the MagnetSortLookup class can be saved to a .magsortlookup file using the member function
        and reloaded using the static factory function while retaining the data.
        """

        # Make dummy parameters
        count, mtype, grid, lookup = self.dummy_magnet_lookup_values()

        # Run the round trip file save + load in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_file_path = os.path.join(tmp_path, 'example.magsortlookup')

            # Construct MagnetSortLookup instance
            magnet_lookup = MagnetSortLookup(mtype=mtype, grid=grid, lookup=lookup)

            with open(tmp_file_path, 'wb') as tmp_file_handle:
                # Save the MagnetSortLookup to the temporary directory
                magnet_lookup.save(file=tmp_file_handle)

            # Throw away the local object and reload it from the temporary file
            magnet_lookup = MagnetSortLookup.from_file(file=tmp_file_path)

            # Clean up the temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_lookup.count, count)
        self.assertEqual(magnet_lookup.grid, grid)
        self.assertEqual(magnet_lookup.mtype, mtype)
        self.assertTrue(np.allclose(magnet_lookup.lookup, lookup))

    def test_save_raises_on_bad_parameters(self):
        """
        Tests the MagnetSortLookup class save member function raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Make dummy parameters
        count, mtype, grid, lookup = self.dummy_magnet_lookup_values()

        # Construct MagnetSortLookup instance
        magnet_lookup = MagnetSortLookup(mtype=mtype, grid=grid, lookup=lookup)

        # Attempt to save to a bad file parameter
        self.assertRaises(optid.errors.FileHandleError, magnet_lookup.save, file=None)

    def test_static_from_file(self):
        """
        Tests the MagnetSortLookup class can be constructed from a .magsortlookup file using the static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, mtype, grid, lookup = self.dummy_magnet_lookup_values()

        # Construct MagnetSortLookup instance
        magnet_lookup = MagnetSortLookup.from_file(file=os.path.join(inputs_path, 'example.magsortlookup'))

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_lookup.count, count)
        self.assertEqual(magnet_lookup.grid, grid)
        self.assertEqual(magnet_lookup.mtype, mtype)
        self.assertTrue(np.allclose(magnet_lookup.lookup, lookup))

    def test_static_from_file_open_file_handle(self):
        """
        Tests the MagnetSortLookup class can be constructed from an open handle to a .magsortlookup file using the
        static factory function.
        """

        # Construct absolute path to the data for this test function
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                 os.path.splitext(os.path.basename(__file__))[0],
                                 inspect.stack()[0][3])

        # Inputs sub directory to load this tests input data from
        inputs_path = os.path.join(data_path, 'inputs')

        # Make dummy parameters
        count, mtype, grid, lookup = self.dummy_magnet_lookup_values()

        with open(os.path.join(inputs_path, 'example.magsortlookup'), 'rb') as file_handle:
            # Construct MagnetSortLookup instance
            magnet_lookup = MagnetSortLookup.from_file(file=file_handle)

        # Assert object members have been correctly assigned
        self.assertEqual(magnet_lookup.count, count)
        self.assertEqual(magnet_lookup.grid, grid)
        self.assertEqual(magnet_lookup.mtype, mtype)
        self.assertTrue(np.allclose(magnet_lookup.lookup, lookup))

    def test_static_from_file_raises_on_bad_parameters(self):
        """
        Tests the MagnetSortLookup class raises an error when the file parameter is neither
        as string file path or an open file handle.
        """

        # Attempt to load from to a bad file parameter
        self.assertRaises(optid.errors.FileHandleError, MagnetSortLookup.from_file, file=None)
