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


from optid.utils.validate_tensor import TensorShapeError, TensorTypeError

from optid.utils.validate_string import StringEmptyError, StringTypeError

from optid.utils.validate_string_list import StringListTypeError, StringListEmptyError, StringListShapeError, \
                                             StringListElementTypeError, StringListElementEmptyError, \
                                             StringListElementUniquenessError

from optid.errors.file_handle_error import FileHandleError
