# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC

from astropy import units
from astropy.units import imperial
import numpy as np

__all__ = ['AtranModel']


class AtranModel(ABC):

    kft = 1000.0 * imperial.ft
    reference_airmass = np.sqrt(2)
    reference_altitude = 41 * kft

    def __init__(self, options):
        self.am_coeffs = None
        self.alt_coeffs = None
        self.reference_transmission = 1.0
        self.poly_am = None
        self.poly_alt = None
        self.configure_options(options)

    def configure_options(self, options):
        if not isinstance(options, dict):
            raise ValueError("Options is not a dictionary.")

        am_coeffs = options.get('amcoeffs')
        if am_coeffs is None:
            raise ValueError("Undefined option: 'atran.amcoeffs'")
        self.am_coeffs = np.array([float(x) for x in am_coeffs])
        self.poly_am = np.poly1d(self.am_coeffs[::-1])

        alt_coeffs = options.get('altcoeffs')
        if alt_coeffs is None:
            raise ValueError("Undefined option: 'atran.altcoeffs'")
        self.alt_coeffs = np.array([float(x) for x in alt_coeffs])
        self.poly_alt = np.poly1d(self.alt_coeffs[::-1])

        reference = options.get('reference')
        if reference is None:
            raise ValueError("Undefined option: 'atran.reference'")
        self.reference_transmission = float(reference)

    def get_relative_transmission(self, altitude, elevation):

        if not isinstance(elevation, units.Quantity):
            elevation = elevation * units.deg

        if isinstance(altitude, units.Quantity):
            altitude = altitude.to(imperial.ft)
        else:
            altitude = altitude * self.kft

        d_am = (1.0 / np.sin(elevation)) - self.reference_airmass
        result = self.poly_am(d_am)

        d_alt = ((altitude - self.reference_altitude)
                 / self.kft).decompose().value
        result *= self.poly_alt(d_alt)

        return result.value
