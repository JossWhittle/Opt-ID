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


from setuptools import setup, find_packages

# Attempt to import packages that we force the installation of...
# Throws exception at package install time if not found!
#   - JAX can be JAX CPU or GPU and requiring one can force the overwrite of an existing version
#   - Radia requires manual build steps
import jax
import radia

setup(
    version='3.0a',
    name='opt-id',
    description='Optimisation of IDs using Python and Opt-AI',
    url='https://github.com/DiamondLightSource/Opt-ID',
    author='Mark Basham, Joss Whittle',
    author_email='mark.basham@rfi.ac.uk, joss.whittle@rfi.ac.uk',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='tests',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Operating System :: POSIX :: Linux',
    ],
    license='Apache License, Version 2.0',
    zip_safe=False,
    install_requires=[
        'numpy>=1.18.0',
        'nptyping',
        'beartype',
        'h5py',
        'pandas',
        'sect',
        'vtk',
        'pyvista',
        'tetgen',
    ]
)
