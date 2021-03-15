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
from jax.interpreters.pxla import ShardedDeviceArray
from jax.interpreters.xla import DeviceArray


class JAXPickler(pickle.Pickler):

    @beartype
    def __init__(self,
                 file: typ.Any,
                 *args,
                 device_map: typ.Optional[typ.Dict[str, str]] = None,
                 raise_on_missing_device: bool = True,
                 **kargs):
        """
        Construct a JAX aware Pickler instance.

        :param file:
            File-like handle with read mode.

        :param args:
            All other positional parameters are forwarded to pickle.Pickler.__init__.

        :param device_map:
            Optional dictionary mapping strings to strings where the keys are device names as found of this host,
            and the values are device names as written to the file.

        :param raise_on_missing_device:
            Bool representing whether to throw exceptions when unknown device names are encountered. If false then
            the default device (None) is used.

        :param kargs:
            All other keyword parameters are forwarded to pickle.Pickler.__init__.
        """
        super().__init__(file, *args, **kargs)

        self._device_map = device_map
        self._raise_on_missing_device = raise_on_missing_device

        # Check that all the keys in the device map refer to valid device names on this host
        if self.raise_on_missing_device and (self.device_map is not None):
            for device_name in self.device_map.keys():
                if device_name not in self.devices:
                    raise pickle.PicklingError(f'device name {device_name} is not recognized in '
                                               f'{list(self.devices.keys())}')

    @property
    @beartype
    def devices(self) -> typ.Dict[str, typ.Any]:
        """
        Property that extracts a dictionary of device names mapping to JAX device objects.

        :return:
            Dict of str -> Device representing the local runtime configuration.
        """
        return { str(device): device for device in jax.devices() }

    @property
    @beartype
    def device_map(self) -> typ.Optional[typ.Dict[str, str]]:
        """
        Property that extracts a dictionary of local device names mapping to names used for pickling.

        :return:
            Dict of str -> str.
        """
        return self._device_map

    @property
    @beartype
    def raise_on_missing_device(self) -> bool:
        """
        Property for whether to allow unknown device names to be replaced by the default device or raise an exception.

        :return:
            If true exceptions are thrown on unknown device, otherwise default device (None) is used.
        """
        return self._raise_on_missing_device

    def persistent_id(self, obj):
        """
        Attempt to encode a JAX device tensor to an encoded tuple.

        :param obj:
            JAX DeviceArray.

        :return:
            Tuple of (str, data) where data is a tuple (device_name, numpy_values) for JAX DeviceArray's.
        """

        if isinstance(obj, ShardedDeviceArray):

            # TODO implement sharded device array pickling
            raise pickle.PicklingError(f'JAXPickler currently does not support ShardedDeviceArray')

        if isinstance(obj, DeviceArray):

            # Get the device name for the tensor
            device_name = str(obj.device_buffer.device())

            # Remap the device name if it has an entry in the device name map
            if self.device_map is not None:
                device_name = self.device_map.get(device_name, device_name)

            # Convert the data in the jax tensor to a numpy tensor for pickling
            values = np.array(obj)

            # Return a tuple to be pickled as normal
            return 'jax.interpreters.xla.DeviceArray', (device_name, values)

        else:

            # Return None for all other types to indicate they should be pickled as normal
            return None


class JAXUnpickler(pickle.Unpickler):

    @beartype
    def __init__(self,
                 file: typ.Any,
                 *args,
                 device_map: typ.Optional[typ.Dict[str, str]] = None,
                 raise_on_missing_device: bool = True,
                 **kargs):
        """
        Construct a JAX aware Unpickler instance.

        :param file:
            File-like handle with read mode.

        :param args:
            All other positional parameters are forwarded to pickle.Unpickler.__init__.

        :param device_map:
            Optional dictionary mapping strings to strings where the keys are device names as read from the file,
            and the values are device names as found of this host.

        :param raise_on_missing_device:
            Bool representing whether to throw exceptions when unknown device names are encountered. If false then
            the default device (None) is used.

        :param kargs:
            All other keyword parameters are forwarded to pickle.Unpickler.__init__.
        """
        super().__init__(file, *args, **kargs)

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
        """
        Property that extracts a dictionary of device names mapping to JAX device objects.

        :return:
            Dict of str -> Device representing the local runtime configuration.
        """
        return { str(device): device for device in jax.devices() }

    @property
    @beartype
    def device_map(self) -> typ.Optional[typ.Dict[str, str]]:
        """
        Property that extracts a dictionary of device names used for pickling to local device names.

        :return:
            Dict of str -> str.
        """
        return self._device_map

    @property
    @beartype
    def raise_on_missing_device(self) -> bool:
        """
        Property for whether to allow unknown device names to be replaced by the default device or raise an exception.

        :return:
            If true exceptions are thrown on unknown device, otherwise default device (None) is used.
        """
        return self._raise_on_missing_device

    def persistent_load(self, obj):
        """
        Attempt to recover a JAX device tensor from an encoded tuple.

        :param obj:
            Tuple of (str, data) where data is a tuple (device_name, numpy_values) for JAX DeviceArray's.

        :return:
            The recovered JAX DeviceArray.
        """

        # Attempt to unpack the object as a tuple representing the jax tensor
        # If unpacking the tuple fails then this object will be unpickled as normal
        tag, data = obj

        if tag == 'jax.interpreters.pxla.ShardedDeviceArray':

            # TODO implement sharded device array unpickling
            raise pickle.UnpicklingError(f'JAXUnpickler currently does not support ShardedDeviceArray')

        elif tag == 'jax.interpreters.xla.DeviceArray':

            # Attempt to unpack the DeviceArray data
            device_name, values = data

            # Remap the encoded name of the device this tensor should be placed on
            # If the name is not in the device map then leave it as is
            if self.device_map is not None:
                device_name = self.device_map.get(device_name, device_name)

            if self.raise_on_missing_device and (device_name not in self.devices):
                raise pickle.UnpicklingError(f'device name {device_name} is not recognized in '
                                             f'{list(self.devices.keys())}')

            # Use the encoded device name to get jax device object or default to None if not defined
            device = self.devices.get(device_name, None)

            # Recover the JAX tensor
            return jax.device_put(values, device=device)

        else:

            # This load function only supports JAX DeviceArray instances
            raise pickle.UnpicklingError('unsupported persistent object')
