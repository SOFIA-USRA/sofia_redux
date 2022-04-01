# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units, constants
import numpy as np

from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.utils import (
    to_header_float, insert_info_in_header)

__all__ = ['SofiaSpectroscopyInfo']


class SofiaSpectroscopyInfo(InfoBase):

    velocity_unit = units.Unit('km') / units.Unit('s')

    def __init__(self):
        """
        Initialize the SOFIA spectroscopy information.

        Contains information on SOFIA spectroscopic parameters such as the
        bandwidth, resolution, frequencies, and velocities.
        """
        super().__init__()
        self.front_end = None
        self.back_end = None
        self.bandwidth = np.nan * units.Unit('MHz')
        self.frequency_resolution = np.nan * units.Unit('MHz')
        self.tsys = np.nan * units.Unit('Kelvin')
        self.observing_frequency = np.nan * units.Unit('MHz')
        self.image_frequency = np.nan * units.Unit('MHz')
        self.rest_frequency = np.nan * units.Unit('MHz')
        self.velocity_type = None
        self.frame_velocity = np.nan * self.velocity_unit
        self.source_velocity = np.nan * self.velocity_unit

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'spec'

    def apply_configuration(self):
        """
        Update spectroscopic information with FITS header information.

        Updates the information by taking the following keywords from the
        FITS header::

          FRONTEND - The frontend device name (str)
          BACKEND - The backend device name (str)
          BANDWID - The total spectral bandwidth (MHz)
          FREQRES - The spectral frequency resolution (MHz)
          TSYS - The system temperature (K)
          OBSFREQ - The observing frequency at the reference channel (MHz)
          IMAGFREQ - The image frequency at the reference channel (MHz)
          RESTFREQ - The rest frequency at the reference channel (MHz)
          VELDEF - The velocity system definition (str)
          VFRAME - Radial velocity of the reference frame wrt observer (km/s)
          RVSYS - The source velocity wrt the observer (km/s)

        Returns
        -------
        None
        """
        options = self.options
        if options is None:
            return
        mhz = units.Unit('MHz')
        self.front_end = options.get_string('FRONTEND')
        self.back_end = options.get_string('BACKEND')
        self.bandwidth = options.get_float('BANDWID') * mhz
        self.frequency_resolution = options.get_float('FREQRES') * mhz
        self.tsys = options.get_float('TSYS') * units.Unit('Kelvin')
        self.observing_frequency = options.get_float('OBSFREQ') * mhz
        self.image_frequency = options.get_float('IMAGFREQ') * mhz
        self.rest_frequency = options.get_float('RESTFREQ') * mhz
        self.velocity_type = options.get_string('VELDEF')
        self.frame_velocity = options.get_float('VFRAME') * self.velocity_unit
        self.source_velocity = options.get_float('RVSYS') * self.velocity_unit

    def get_redshift(self):
        """
        Return the redshift of the source determined from it's velocity.

        The redshift is calculated as::

            z = sqrt( (1 + v/c) / (1 - v/c) ) - 1

        where v is the source velocity and c is the speed of light.  I.e., the
        relativistic doppler shift along the line of sight.

        Returns
        -------
        redshift : float
        """
        v_over_c = (self.source_velocity / constants.c).decompose().value
        return np.sqrt((1.0 + v_over_c) / (1.0 - v_over_c)) - 1.0

    def edit_header(self, header):
        """
        Edit an image header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.

        Returns
        -------
        None
        """
        info = [
            ('COMMENT', "<------ SOFIA Spectroscopy Data ------>"),
            ('FRONTEND', self.front_end, 'Frontend device name.'),
            ('BACKEND', self.back_end, 'Backend device name.'),
            ('BANDWID', to_header_float(self.bandwidth, 'MHz'),
             '(MHz) Total spectral bandwidth.'),
            ('FREQRES', to_header_float(self.frequency_resolution, 'MHz'),
             '(MHz) Spectral frequency resolution.'),
            ('TSYS', to_header_float(self.tsys, 'K'),
             '(K) System temperature.'),
            ('OBSFREQ', to_header_float(self.observing_frequency, 'MHz'),
             '(MHz) Observing frequency at reference channel.'),
            ('IMAGFREQ', to_header_float(self.image_frequency, 'MHz'),
             '(MHz) Image frequency at reference channel.'),
            ('RESTFREQ', to_header_float(self.rest_frequency, 'MHz'),
             '(MHz) Rest frequency at reference channel.'),
            ('VELDEF', self.velocity_type, 'Velocity system definition.'),
            ('VFRAME', to_header_float(self.frame_velocity, 'km/s'),
             '(km/s) Radial velocity of reference frame wrt observer.'),
            ('RVSYS', to_header_float(self.source_velocity, 'km/s'),
             '(km/s) Source radial velocity wrt observer.')
        ]
        insert_info_in_header(header, info, delete_special=True)

    def get_table_entry(self, name):
        """
        Return a parameter value for the given name.

        Parameters
        ----------
        name : str
            The name of the parameter to retrieve.

        Returns
        -------
        value
        """
        if name == 'bw':
            return self.bandwidth.to('GHz')
        if name == 'df':
            return self.frequency_resolution.to('MHz')
        elif name == 'tsys':
            return self.tsys.to('Kelvin')
        elif name == 'fobs':
            return self.observing_frequency.to('GHz')
        elif name == 'frest':
            return self.rest_frequency.to('GHz')
        elif name == 'vsys':
            return self.velocity_type
        elif name == 'vframe':
            return self.frame_velocity.to('km/s')
        elif name == 'vrad':
            return self.source_velocity.to('km/s')
        elif name == 'z':
            return self.get_redshift()
        else:
            return super().get_table_entry(name)
