# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC

from astropy import units
from astropy.units import imperial
import numpy as np

from sofia_redux.scan.utilities.utils import get_float_list


__all__ = ['AtranModel']

assert imperial


class AtranModel(ABC):

    units.imperial.enable()
    kft = 1000.0 * units.Unit('ft')
    reference_airmass = np.sqrt(2)  # 45 degrees elevation
    reference_altitude = 41 * kft

    def __init__(self, options):
        """
        Initialize the ATRAN model for SOFIA.

        The ATRAN model is used to derive the relative transmission correction
        factor for a given altitude above the Earth's surface, observing a
        source at a given elevation.  This can be used to determine the
        atmospheric opacity.  Please see
        :func:`AtranModel.get_relative_transmission` and
        :func:`AtranModel.get_zenith_tau` for further details.

        The options should contain the following keywords::

          - amcoeffs: The polynomial coefficients defining the air mass
          factor as a function of elevation from the reference air mass.
          - altcoeffs : The polynomial coefficients defining the altitude
          factor as a function of altitude from the reference altitude.
          - reference : The reference transmission factor.

        Parameters
        ----------
        options : dict
        """
        self.am_coeffs = None
        self.alt_coeffs = None
        self.reference_transmission = 1.0
        self.poly_am = None
        self.poly_alt = None
        self.configure_options(options)

    def configure_options(self, options):
        """
        Parse the options into the model.

        Parameters
        ----------
        options : dict
            The options must contain the following keys: amcoeffs, altcoeffs,
            and reference.

        Returns
        -------
        None
        """
        if not isinstance(options, dict):
            raise ValueError(f"Options must be a {dict}.  Received {options}.")

        am_coeffs = get_float_list(options.get('amcoeffs'))
        if am_coeffs is None:
            raise ValueError("Undefined option: 'atran.amcoeffs'")
        alt_coeffs = get_float_list(options.get('altcoeffs'))
        if alt_coeffs is None:
            raise ValueError("Undefined option: 'atran.altcoeffs'")
        reference = options.get('reference')
        if reference is None:
            raise ValueError("Undefined option: 'atran.reference'")

        self.am_coeffs = np.array([float(x) for x in am_coeffs])
        self.poly_am = np.poly1d(self.am_coeffs[::-1])
        self.alt_coeffs = np.array([float(x) for x in alt_coeffs])
        self.poly_alt = np.poly1d(self.alt_coeffs[::-1])
        self.reference_transmission = float(reference)

    def get_relative_transmission(self, altitude, elevation):
        """
        Model the relative transmission at the given altitude and elevation.

        The relative transmission (c) is given as::

            c = altitude_factor * air_mass_factor

        where::

            delta_alt = altitude - reference_altitude
            air_mass = 1 / sin(elevation)
            delta_atm = air_mass - reference_air_mass

            altitude_factor = sum_i(alt_coeff[i] * delta_alt^i)
            air_mass_factor = sum_i(atm_coeff[i] * delta_atm^i)

        The reference air mass is generally taken to be sqrt(2), equivalent to
        an elevation of 45 degrees, and the reference altitude for SOFIA is
        41,000 ft.

        Parameters
        ----------
        altitude : float or units.Quantity
            The altitude above the Earth's surface.  If a float value is
            provided, it is assumed to be in kilo-feet (1000 feet).
        elevation : float or units.Quantity
            The elevation of the observed source.  If a float value is
            provided, it is assumed to be in degrees.

        Returns
        -------
        relative_transmission : float
        """
        if not isinstance(elevation, units.Quantity):
            elevation = elevation * units.Unit('degree')

        if isinstance(altitude, units.Quantity):
            # kft is not a thing, but we want this in imperial units for later
            altitude = altitude.to(units.Unit('ft'))
        else:
            altitude = altitude * self.kft

        air_mass = 1.0 / np.sin(elevation)
        if isinstance(air_mass, units.Quantity):
            air_mass = air_mass.value

        delta_air_mass = air_mass - self.reference_airmass
        air_mass_factor = self.poly_am(delta_air_mass)

        delta_altitude = (altitude - self.reference_altitude) / self.kft
        delta_altitude = delta_altitude.decompose().value
        altitude_factor = self.poly_alt(delta_altitude)

        return air_mass_factor * altitude_factor

    def get_zenith_tau(self, altitude, elevation):
        """
        Model the atmospheric opacity for a given altitude and elevation.

        The opacity (tau) can be estimated as::

            zenith_tau = -log(ref * c) * sin(elevation)

        This may be rewritten as::

            transmission = exp(-zenith_tau * air_mass)

        where ref is the reference transmission of this model, and c is the
        relative transmission returned by
        :func:`AtranModel.get_relative_transmission`.

        Parameters
        ----------
        altitude : float or units.Quantity
            The altitude above the Earth's surface.  If a float value is
            provided, it is assumed to be in kilo-feet (1000 feet).
        elevation : float or units.Quantity
            The elevation of the observed source.  If a float value is
            provided, it is assumed to be in degrees.

        Returns
        -------
        tau : float
        """
        if not isinstance(elevation, units.Quantity):
            elevation = elevation * units.Unit('degree')

        sin_elevation = np.sin(elevation).value
        c = self.get_relative_transmission(altitude, elevation)
        tau = -np.log(self.reference_transmission * c) * sin_elevation
        return tau
