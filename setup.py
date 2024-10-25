from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "bouchard_sornett",  # Module name
        ["BouchardSornettOptionPricing_pybind.cpp", "BouchardSornettOptionPricing.cpp"],  # Source files
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="bouchard_sornett",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},  # Use build_ext from pybind11.setup_helpers
)
