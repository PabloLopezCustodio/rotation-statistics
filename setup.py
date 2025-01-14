import os
import numpy
import sys

try:
    from setuptools import setup, Extension, find_packages
except ImportError:
    print("Error: setuptools is not installed. Install it using 'pip install setuptools'.", file=sys.stderr)
    sys.exit(1)

try:
    from Cython.Build import cythonize
except ImportError:
    raise RuntimeError("Cython is required to build this package. Install it with 'pip install Cython'.")

FUNCS_PATH = os.path.join("src", "rotstats")

setup(
    name="rotstats",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"src.rotstats.data": ["*.json"]},
    ext_modules=cythonize(Extension(name="rotstats.func",
                                    sources=[os.path.join(FUNCS_PATH, "func.pyx")],
                                    include_dirs=['./', numpy.get_include()]),
                          annotate=True),
    python_requires='>=3',
    install_requires=['matplotlib<=3.9','numpy', 'scipy'],
    zip_safe=False,
    author="Pablo Lopez-Custodio",
    author_email="custodio825@gmail.com"
)
