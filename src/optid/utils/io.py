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
import typing as typ
from beartype import beartype
import pickle
import numpy as np
import jax
import jax.numpy as jnp


class JAXPickler(pickle.Pickler):

    @beartype
    def __init__(self,
                 file: typ.Any,
                 device_map: typ.Optional[typ.Dict[str, str]] = None,
                 raise_on_missing_device: bool = True):
        """
        Construct a JAX aware Pickler instance.

        :param file:
            File-like handle with read mode.

        :param device_map:
            Optional dictionary mapping strings to strings where the keys are device names as found of this host,
            and the values are device names as written to the file.

        :param raise_on_missing_device:
            Bool representing whether to throw exceptions when unknown device names are encountered. If false then
            the default device (None) is used.
        """
        super().__init__(file)

        self._device_map = device_map
        self._raise_on_missing_device = raise_on_missing_device

        # Check that all the keys in the device map refer to valid device names on this host
        if self.raise_on_missing_device and (self.device_map is not None):
            for device_name in self.device_map.keys():
                if device_name not in self.devices:
                    raise pickle.UnpicklingError(f'device name {device_name} is not recognized in '
                                                 f'{list(self.devices.keys())}')

    @property
    @beartype
    def devices(self) -> typ.Dict[str, typ.Any]:
        return { str(device): device for device in jax.devices() }

    @property
    @beartype
    def device_map(self) -> typ.Optional[typ.Dict[str, str]]:
        return self._device_map

    @property
    @beartype
    def raise_on_missing_device(self) -> bool:
        return self._raise_on_missing_device

    def persistent_id(self, obj):

        # Only trigger for jax tensors, ignore native numpy tensors
        if isinstance(obj, jnp.ndarray) and not isinstance(obj, np.ndarray):

            # Get the device name for the tensor
            device_name = str(obj.device_buffer.device())

            # Remap the device name if it has an entry in the device name map
            if self.device_map is not None:
                device_name = self.device_map.get(device_name, device_name)

            # Convert the data in the jax tensor to a numpy tensor for pickling
            values = np.array(obj)

            # Return a tuple to be pickled as normal
            return 'jax.numpy.ndarray', device_name, values
        else:

            # Return None for all other types to indicate they should be pickled as normal
            return None


class JAXUnpickler(pickle.Unpickler):

    @beartype
    def __init__(self,
                 file: typ.Any,
                 device_map: typ.Optional[typ.Dict[str, str]] = None,
                 raise_on_missing_device: bool = True):
        """
        Construct a JAX aware Unpickler instance.

        :param file:
            File-like handle with read mode.

        :param device_map:
            Optional dictionary mapping strings to strings where the keys are device names as read from the file,
            and the values are device names as found of this host.

        :param raise_on_missing_device:
            Bool representing whether to throw exceptions when unknown device names are encountered. If false then
            the default device (None) is used.
        """
        super().__init__(file)

        self._device_map = device_map
        self._raise_on_missing_device = raise_on_missing_device

        # Check that all the mapped values in the device map refer to valid device names on this host
        if self.raise_on_missing_device and (self.device_map is not None):
            for device_name in self.device_map.values():
                if device_name not in self.devices:
                    raise pickle.UnpicklingError(f'device name {device_name} is not recognized in '
                                                 f'{list(self.devices.keys())}')

    @property
    @beartype
    def devices(self) -> typ.Dict[str, typ.Any]:
        return { str(device): device for device in jax.devices() }

    @property
    @beartype
    def device_map(self) -> typ.Optional[typ.Dict[str, str]]:
        return self._device_map

    @property
    @beartype
    def raise_on_missing_device(self) -> bool:
        return self._raise_on_missing_device

    def persistent_load(self, obj):

        # Attempt to unpack the object as a tuple representing the jax tensor
        # If unpacking the tuple fails then this object will be unpickled as normal
        tag, device_name, values = obj

        if tag == 'jax.numpy.ndarray':

            # Remap the encoded name of the device this tensor should be placed on
            # If the name is not in the device map then leave it as is
            if self.device_map is not None:
                device_name = self.device_map.get(device_name, device_name)

            if self.raise_on_missing_device and (device_name not in self.devices):
                raise pickle.UnpicklingError(f'device name {device_name} is not recognized in '
                                             f'{list(self.devices.keys())}')

            # Use the encoded device name to get jax device object
            device = self.devices.get(device_name, None)

            # Recover the JAX tensor
            return jax.device_put(values, device=device)
        else:

            # This load function only supports triggering for jax.numpy.ndarray instances
            raise pickle.UnpicklingError('unsupported persistent object')
