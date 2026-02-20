"""Build script for Cython-accelerated sparse DP.

Usage:
    cd PaST/solvers
    python setup_cython.py build_ext --inplace

This compiles _sparse_dp_cython.pyx into a shared library that can be
imported directly by optimal_benchmark_dp.py.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "_sparse_dp_cython",
        sources=["_sparse_dp_cython.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="sparse_dp_cython",
    ext_modules=cythonize(
        extensions,
        force=True,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": 3,
        },
    ),
)
