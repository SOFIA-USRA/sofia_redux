# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units

from sofia_redux.scan.custom.sofia.info.scanning import SofiaScanningInfo
from sofia_redux.scan.utilities.bracketed_values import BracketedValues
from sofia_redux.scan.utilities.utils import (
    to_header_float, UNKNOWN_INT_VALUE, UNKNOWN_STRING_VALUE)
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.utilities.utils import insert_info_in_header

degree = units.Unit('degree')
arcsec = units.Unit('arcsec')
second = units.Unit('second')

__all__ = ['SofiaExtendedScanningInfo']


class SofiaExtendedScanningInfo(SofiaScanningInfo):

    def __init__(self):
        super().__init__()
        self.pattern = ''
        self.coordinate_system = ''
        self.amplitude = None  # Coordinate2D (arcsec)
        self.current_position_angle = np.nan * degree
        self.duration = np.nan * second
        self.rel_frequency = np.nan
        self.rel_phase = np.nan * degree
        self.t0 = np.nan * second
        self.gyro_time_window = np.nan * second
        self.iterations = UNKNOWN_INT_VALUE
        self.subscans = UNKNOWN_INT_VALUE
        self.raster_length = np.nan * arcsec
        self.raster_step = np.nan * arcsec
        self.is_cross_scanning = False
        self.n_steps = UNKNOWN_INT_VALUE
        self.tracking_enabled = UNKNOWN_INT_VALUE
        self.position_angle = BracketedValues(np.nan, np.nan, unit='degree')

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'scan'

    def apply_configuration(self):
        super().apply_configuration()
        options = self.options
        if options is None:
            return

        self.pattern = options.get_string(
            'SCNPATT', default=UNKNOWN_STRING_VALUE)
        self.coordinate_system = options.get_string(
            'SCNCRSYS', default=UNKNOWN_STRING_VALUE)
        self.amplitude = Coordinate2D([options.get_float('SCNAMPXL'),
                                       options.get_float('SCNAMPEL')],
                                      unit='arcsec')
        self.current_position_angle = options.get_float('SCNANGLC') * degree
        self.position_angle = BracketedValues(options.get_float('SCNANGLS'),
                                              options.get_float('SCNANGLF'),
                                              unit='degree')
        self.duration = options.get_float('SCNDUR') * second
        self.iterations = options.get_int('SCNITERS',
                                          default=UNKNOWN_INT_VALUE)
        self.subscans = options.get_int('SCNNSUBS', default=UNKNOWN_INT_VALUE)
        self.raster_length = options.get_float('SCNLEN') * arcsec
        self.raster_step = options.get_float('SCNSTEP') * arcsec
        self.n_steps = options.get_int('SCNSTEPS', default=UNKNOWN_INT_VALUE)
        self.is_cross_scanning = options.get_bool('SCNCROSS')
        self.rel_frequency = options.get_float('SCNFQRAT')
        self.rel_phase = options.get_float('SCNPHASE') * degree
        self.t0 = options.get_float('SCNTOFF') * second
        self.gyro_time_window = options.get_float('SCNTWAIT') * second
        self.tracking_enabled = options.get_int('SCNTRKON',
                                                default=UNKNOWN_INT_VALUE)

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
        super().edit_header(header)

        if self.amplitude is None:
            self.amplitude = Coordinate2D([np.nan, np.nan], unit='arcsec')

        info = [
            ('COMMENT', "<------ SOFIA Extra Scanning Data ------>"),
            ('SCNPATT', self.pattern, "Scan pattern."),
            ('SCNCRSYS', self.coordinate_system, "Scan coordinate system."),
            ('SCNANGLC',
             to_header_float(self.current_position_angle, 'degree'),
             "(deg) current scan angle."),
            ('SCNANGLS', to_header_float(self.position_angle.start, 'degree'),
             "(deg) initial scan angle."),
            ('SCNANGLF', to_header_float(self.position_angle.end, 'degree'),
             "(deg) final scan angle."),
            ('SCNDUR', to_header_float(self.duration, 'second'),
             "(s) scan duration."),
            ('SCNITERS', self.iterations, "scan iterations."),
            ('SCNNSUBS', self.subscans, "number of subscans."),
            ('SCNTRKON', self.tracking_enabled, "[0,1] Is tracking enabled?"),
            ('SCNTWAIT', to_header_float(self.gyro_time_window, 'second'),
             "(s) Track relock time window."),
            ('SCNLEN', to_header_float(self.raster_length, 'arcsec'),
             "(arcsec) Raster scan length."),
            ('SCNSTEP', to_header_float(self.raster_step, 'arcsec'),
             "(arcsec) Raster scan step size."),
            ('SCNSTEPS', self.n_steps, "Raster scan steps."),
            ('SCNCROSS', self.is_cross_scanning, "cross scanning?"),
            ('SCNAMPXL', to_header_float(self.amplitude.x, 'arcsec'),
             "(arcsec) cross-elevation amplitude."),
            ('SCNAMPEL', to_header_float(self.amplitude.y, 'arcsec'),
             "(arcsec) elevation amplitude."),
            ('SCNFQRAT', to_header_float(self.rel_frequency),
             "Lissajous y/x frequency ratio."),
            ('SCNPHASE', to_header_float(self.rel_phase, 'degree'),
             "(deg) Lissajous y/x relative phase."),
            ('SCNTOFF', to_header_float(self.t0, 'second'),
             "(s) Lissajous time offset.")
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
        if name == 'pattern':
            return self.pattern
        elif name == 'sys':
            return self.coordinate_system
        elif name == 'PA':
            return self.current_position_angle.to('degree')
        elif name == 'T':
            return self.duration.to('second')
        elif name == 'iters':
            return self.iterations
        elif name == 'nsub':
            return self.subscans
        elif name == 'trk':
            if self.tracking_enabled == UNKNOWN_INT_VALUE:
                return '?'
            return self.tracking_enabled != 0
        elif name == 'X':
            return self.raster_length.to('arcsec')
        elif name == 'dY':
            return self.raster_step.to('arcsec')
        elif name == 'strips':
            return self.n_steps
        elif name == 'cross?':
            return self.is_cross_scanning
        elif name == 'Ax':
            return self.amplitude.x.to('arcsec')
        elif name == 'Ay':
            return self.amplitude.y.to('arcsec')
        elif name == 'frel':
            return self.rel_frequency
        elif name == 'phi0':
            return self.rel_phase.to('degree')
        elif name == 't0':
            return self.t0.to('second')
        else:
            return super().get_table_entry(name)
