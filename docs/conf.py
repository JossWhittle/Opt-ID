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


import os
import sys


# Build docs form the root project directory
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('..'))

# Enable Sphinx extensions
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'http://docs.python.org/' : None
}
language = 'en'

templates_path = ['_templates']

source_suffix = '.rst'
master_doc    = 'index'

project   = u'Opt-ID'
copyright = u'2017, Diamond Light Source'

pygments_style = 'sphinx'

import sphinx_rtd_theme

html_static_path   = ['_static']
html_theme         = 'sphinx_rtd_theme'
html_theme_path    = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'collapse_navigation': False,
}
