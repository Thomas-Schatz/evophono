# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:05:44 2018

@author: Thomas Schatz

Script to compile our cython implementation of edit distance.

Usage example: python ./compiled_ED.py build_ext --build-lib ./
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os
path = os.path.dirname(os.path.realpath(__file__))

extension = Extension("edit_distance",
                      [os.path.join(path, "edit_distance.pyx")],
                       extra_compile_args=["-O3"],
                       include_dirs=[numpy.get_include()])

setup(name="Edit distance cython implementation",
      ext_modules=cythonize(extension))