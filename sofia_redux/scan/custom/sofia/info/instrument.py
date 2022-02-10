# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy import units, constants

from sofia_redux.scan.info.camera.instrument import CameraInstrumentInfo
from sofia_redux.scan.utilities.utils import (
    to_header_float, insert_info_in_header)

__all__ = ['SofiaInstrumentInfo']


class SofiaInstrumentInfo(CameraInstrumentInfo):

    telescope_diameter = 2.5 * units.Unit('m')

    def __init__(self):
        super().__init__()
        self.set_mount("NASMYTH_COROTATING")
        self.instrument_name = None
        self.data_type = None
        self.instrument_config = None
        self.instrument_mode = None
        self.mccs_mode = None
        self.spectral_element_1 = None
        self.spectral_element_2 = None
        self.slit_id = None
        self.detector_channel = None
        self.spectral_resolution = np.nan
        self.exposure_time = np.nan * units.Unit('s')
        self.total_integration_time = np.nan * units.Unit('s')
        self.wavelength = np.nan * units.Unit('um')

    def apply_configuration(self):
        options = self.options
        if options is None:
            return
        self.instrument_name = options.get_string("INSTRUME")
        self.data_type = options.get_string("DATATYPE")
        self.instrument_config = options.get_string("INSTCFG")
        self.instrument_mode = options.get_string("INSTMODE")
        self.mccs_mode = options.get_string("MCCSMODE")
        self.spectral_element_1 = options.get_string("SPECTEL1")
        self.spectral_element_2 = options.get_string("SPECTEL2")
        self.slit_id = options.get_string("SLIT")
        self.detector_channel = options.get_string("DETCHAN")
        self.spectral_resolution = options.get_float("RESOLUN")
        self.exposure_time = options.get_float("EXPTIME") * units.Unit('s')
        self.total_integration_time = options.get_float("TOTINT"
                                                        ) * units.Unit('s')
        self.wavelength = options.get_float("WAVECENT") * units.Unit('um')

        if 'aperture' in self.configuration:
            d = self.configuration['aperture'] * units.Unit('m')
        else:
            d = self.telescope_diameter

        self.angular_resolution = (1.22 * self.wavelength
                                   / d).decompose().value
        self.angular_resolution = (self.angular_resolution
                                   * units.Unit('radian'))
        self.frequency = (constants.c / self.wavelength).to(units.Unit('Hz'))

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
            ('COMMENT', "<------ SOFIA Instrument Data ------>"),
            ('INSTRUME', self.instrument_name, 'Name of SOFIA instrument.'),
            ('DATATYPE', self.data_type, 'Data type.'),
            ('INSTCFG', self.instrument_config, 'Instrument configuration.'),
            ('INSTMODE', self.instrument_mode, 'Instrument observing mode.'),
            ('MCCSMODE', self.mccs_mode, 'MCCS mode.'),
            ('EXPTIME', to_header_float(self.exposure_time, 'second'),
             '(s) total effective on-source time.'),
            ('SPECTEL1', self.spectral_element_1, 'First spectral element.'),
            ('SPECTEL2', self.spectral_element_2, 'Second spectral element.')
        ]

        if not np.isnan(self.wavelength):
            info.append(('WAVECENT', to_header_float(self.wavelength, 'um'),
                         '(um) wavelength at passband center.'))

        if self.slit_id is not None:
            info.append(('SLIT', self.slit_id, 'Slit identifier.'))

        if not np.isnan(self.spectral_resolution):
            info.append(('RESOLUN', self.spectral_resolution,
                         'Spectral resolution.'))

        if self.detector_channel is not None:
            info.append(('DETCHAN', self.detector_channel,
                         'Detector channel ID.'))

        if not np.isnan(self.total_integration_time):
            info.append(('TOTINT',
                         to_header_float(self.total_integration_time,
                                         'second'),
                         '(s) Total integration time.'))

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
        if name == 'wave':
            return self.wavelength.to('um')
        elif name == 'exp':
            return self.exposure_time.to('second')
        elif name == 'inttime':
            return self.total_integration_time.to('second')
        elif name == 'datatype':
            return self.data_type
        elif name == 'mode':
            return self.instrument_mode
        elif name == 'cfg':
            return self.instrument_config
        elif name == 'slit':
            return self.slit_id
        elif name == 'spec1':
            return self.spectral_element_1
        elif name == 'spec2':
            return self.spectral_element_2
        else:
            return super().get_table_entry(name)
