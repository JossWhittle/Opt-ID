from setuptools import setup, find_packages

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
)
