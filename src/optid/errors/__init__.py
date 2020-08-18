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


from optid.utils.validate_range import \
    ValidateRangeTypeError, ValidateRangeShapeError, ValidateRangeElementTypeError, \
    ValidateRangeBoundaryError, ValidateRangeStepsError, ValidateRangeSingularityError

from optid.utils.validate_tensor import \
    ValidateTensorTypeError, ValidateTensorShapeError, ValidateTensorElementTypeError

from optid.utils.validate_string import \
    ValidateStringEmptyError, ValidateStringTypeError

from optid.utils.validate_string_list import \
    ValidateStringListTypeError, ValidateStringListEmptyError, ValidateStringListShapeError, \
    ValidateStringListElementTypeError, ValidateStringListElementEmptyError, \
    ValidateStringListElementUniquenessError

from optid.errors.file_handle_error import \
    FileHandleError
