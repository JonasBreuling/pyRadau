#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup
from numpy.distutils.core import Extension, setup
from numpy.distutils.command import build_src

import numpy as np

LONG_DESCRIPTION = """
RADAU13 is an implicit 13th order adaptive Runge-Kutta implementation of the
RADAU IIA method by Ernst Hairer and Gerhard Wanner. The Fortran code is
available from Ernst Hairer's website

 http://www.unige.ch/~hairer/software.html

and a detailled description is available in their book

Hairer, Norsett and Wanner (1993): Solving Ordinary Differential Equations.
Nonstiff Problems.  2nd edition. Springer Series in Comput. Math., vol. 8.

This package does not use f2py, but instead calls the fortran function from a C
extension. The solver features dense/continous output, support for implicit
problems with a mass matrix, DAEs up to index 3 and multiple error control
algorithms. The python interface includes all features except band-width
control (for now), but some of the advanced/complex settings are not
documented (yet).

""".strip() + "\n\n"

class no_f2py_build_scr(build_src.build_src):
    def f2py_sources(self, sources, extension):
        return sources

ext_modules = [ Extension('pyradau13', sources = [ 'pyradau13.c', 'lib/radau.f', 'lib/decsol.f', 'lib/dc_decsol.f' ]) ]

setup(
    name = 'pyradau13',
    version = '0.4.1',
    author = 'Phillip Berndt',
    author_email = 'phillip.berndt@googlemail.com',
    license = 'GPL',
    url = 'https://git.imp.fu-berlin.de/pberndt/pyradau13',
    description = 'Python wrapper around the Hairer/Wanner implementation of the RADAU13 ODE solver',
    long_description = LONG_DESCRIPTION,
    include_dirs = [np.get_include()],
    ext_modules = ext_modules,
    setup_requires=["numpy"],
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Operating System :: POSIX",
        "Operating System :: Unix",
    ],
    cmdclass = {
        "build_src": no_f2py_build_scr,
    },
    test_suite = "test.TestIntegration"
)
