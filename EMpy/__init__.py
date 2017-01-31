"""EMpy: Electromagnetic Python.

The package contains some useful routines to study electromagnetic problems with Python.

    1. An implementation of the L{Transfer Matrix<EMpy.transfer_matrix>} algorithm, both isotropic and anisotropic.
    2. An implementation of the L{Rigorous Coupled Wave Analysis<EMpy.RCWA>}, both isotropic and anisotropic.
    3. A collection of L{Modesolvers<EMpy.modesolvers>} to find modes of optical waveguides.
    4. A library of L{materials<EMpy.materials>}.

It is based on U{numpy<http://numpy.scipy.org>} and U{scipy<http://www.scipy.org>}.

"""
from __future__ import absolute_import

__all__ = ['constants', 'devices', 'materials', 'modesolvers', 'RCWA', 'scattering', 'transfer_matrix',
           'utils']
__author__ = 'Lorenzo Bolla'

from numpy.testing import Tester

from . import constants
from . import devices
from . import materials
from . import modesolvers
from . import RCWA
from . import scattering
from . import transfer_matrix
from . import utils
from .version import version as __version__

test = Tester().test
