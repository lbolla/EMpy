# pylint: disable=C0301

"""Useful constants.

Constants used in mathematics and electromagnetism.

@var c: U{Speed of light<http://en.wikipedia.org/wiki/Speed_of_light>} [m/s].
@var mu0: U{Magnetic Permeability<http://en.wikipedia.org/wiki/Permeability_(electromagnetism)>} [N/A^2].
@var eps0: U{Electric Permettivity<http://en.wikipedia.org/wiki/Permittivity>} [F/m].
@var h: U{Plank's constant<http://en.wikipedia.org/wiki/Plank%27s_constant>} [W s^2].
@var k: U{Boltzmann's constant<http://en.wikipedia.org/wiki/Boltzmann_constant>} [J/K].

"""

__author__ = 'Lorenzo Bolla'

from numpy import pi

c = 299792458.
mu0 = 4 * pi * 1e-7
eps0 = 1. / (c**2 * mu0)

h = 6.62606896e-34
h_bar = h / (2 * pi)
k = 1.3806504e-23
