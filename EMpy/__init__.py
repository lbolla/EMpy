"""EMpy: Electromagnetic Python.

The package contains some useful routines to study electromagnetic problems with Python.

    1. An implementation of the L{Transfer Matrix<EMpy.transfer_matrix>} algorithm, both isotropic and anisotropic.
    2. An implementation of the L{Rigorous Coupled Wave Analysis<EMpy.RCWA>}, both isotropic and anisotropic.
    3. A collection of L{Modesolvers<EMpy.modesolvers>} to find modes of optical waveguides.
    4. A library of L{materials<EMpy.materials>}.

It is based on U{numpy<http://numpy.scipy.org>} and U{scipy<http://www.scipy.org>}.

"""

__all__ = [
    "constants",
    "devices",
    "materials",
    "modesolvers",
    "RCWA",
    "scattering",
    "transfer_matrix",
    "utils",
]
__author__ = "Lorenzo Bolla"

from . import constants
from . import devices
from . import materials
from . import modesolvers
from . import RCWA
from . import scattering
from . import transfer_matrix
from . import utils

try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development environments or edge cases
    try:
        from importlib.metadata import version, PackageNotFoundError
        __version__ = version("electromagneticpython")
    except PackageNotFoundError:
        __version__ = "0.0.0.dev0"