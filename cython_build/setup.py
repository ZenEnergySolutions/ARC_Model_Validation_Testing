from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("ess_requirements.pyx"),
    include_dirs=[numpy.get_include()]
)