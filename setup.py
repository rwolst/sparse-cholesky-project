"""Setup file for the package."""

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = []

# simulate_fast
ext_modules += [
        Extension("py_eigen.py_eigen_cpp",
            sources=["py_eigen/py_eigen_cpp.pyx",
                     "py_eigen/my_program.cpp"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=[#"-Ofast", "-ffast-math", "-march=native",
                                "-O3",
                                #"-ffast-math",
                                "-march=native",
                                "-fopenmp"],
            extra_link_args=["-fopenmp"],
            language='c++',
            libraries=["openblas"],
            define_macros=[('NDEBUG', 1)]
            )
        ]

# setup
setup(
  name="py_eigen",
  packages=["py_eigen"],
  cmdclass={'build_ext': build_ext},
  ext_modules=ext_modules
)
