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
import pickle

from optid.types import BinaryFileHandle
from optid.utils.logging import get_logger

logger = get_logger('optid.utils.io')


class FileHandleError(Exception):
    """
    Exception to throw when a variable is not a valid file path or open file handle.
    """

    def __str__(self):
        return 'file must be a string file path or a file handle to an already open file'


def save(file : BinaryFileHandle, params : dict):
    """
    Saves a file using a set of parameters.

    Parameters
    ----------
    file : str or open writable file handle
        A path to where a file should be created or overwritten, or an open writable file handle.

    params : dict
        A dictionary of named parameters to be saved.
    """

    if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
        # Load directly from the already open file handle
        logger.info('Saving to file handle')
        pickle.dump(params, file)

    elif isinstance(file, str):
        # Open the file in a closure to ensure it gets closed on error
        with open(file, 'wb') as file_handle:
            logger.info('Saving to file [%s]', file)
            pickle.dump(params, file_handle)

    else:
        # Assert that the file object provided is an open file handle or can be used to open one
        raise FileHandleError()


def from_file(file : BinaryFileHandle) -> dict:
    """
    Loads a dictionary of parameters from a file.

    Parameters
    ----------
    file : str or open file handle
        A path to a file or an open readable file handle.

    Returns
    -------
    A dictionary of named parameters.
    """

    if isinstance(file, (io.RawIOBase, io.BufferedIOBase, typing.BinaryIO)):
        # Load directly from the already open file handle
        logger.info('Loading from file handle')
        return pickle.load(file)

    elif isinstance(file, str):
        # Open the file in a closure to ensure it gets closed on error
        with open(file, 'rb') as file_handle:
            logger.info('Loading from file [%s]', file)
            return pickle.load(file_handle)

    else:
        # Assert that the file object provided is an open file handle or can be used to open one
        raise FileHandleError()
