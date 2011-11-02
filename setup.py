#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError as e:
    from distutils.core import setup

long_description = '''
    Peach is a pure-python module, based on SciPy and NumPy to implement
    algorithms for computational intelligence and machine learning. Methods
    implemented include, but are not limited to, artificial neural networks,
    fuzzy logic, genetic algorithms, swarm intelligence and much more.

    The aim of this library is primarily educational. Nonetheless, care was
    taken to make the methods implemented also very efficient.
'''

setup(
    name='Peach',
    version='0.3.1',
    url='http://code.google.com/p/peach/',
    download_url='http://code.google.com/p/peach/downloads/list',
    license='GNU Lesser General Public License',
    author='Jose Alexandre Nalon',
    author_email='jnalon@gmail.com',
    description='Python library for computational intelligence and machine learning',
    long_description=long_description,
    classifiers=[
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='artificial intelligence neural network genetic algorithm fuzzy logic optimization artificial life',
    packages=[
        'peach',
        'peach.fuzzy',
        'peach.ga',
        'peach.nn',
        'peach.optm',
        'peach.pso',
        'peach.sa'
    ],
    install_requires=[
        'bitarray',
    ],
)

