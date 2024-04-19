#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# inplace extension module
module = Extension("_relokate",
                   ["relokate.i","relokate.c"],
                   include_dirs = [numpy_include],
                   )

# NumyTypemapTests setup
setup(  name        = "relokate",
        description = "Used to reset the tongue contour points so that they are aligned with vectors from the center",
        author      = "Kwang Seob Kim",
        version     = "1.0",
        ext_modules = [module]
        )