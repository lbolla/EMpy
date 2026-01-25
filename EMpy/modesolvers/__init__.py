"""Collection of different algorithms to find modes of optical waveguides.

Modesolvers:

    1. FD.
    2. FMM.


"""

__all__ = ["FD", "FMM", "geometries"]
__author__ = "Lorenzo Bolla"

from . import FD, FMM, geometries
