# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units, log
import numpy as np
import pandas as pd

from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.custom.sofia.info.instrument import SofiaInstrumentInfo
from sofia_redux.scan.utilities.utils import to_header_float

__all__ = ['FifiLsInstrumentInfo']


um = units.Unit('um')
arcsec = units.Unit('arcsec')
second = units.Unit('s')
hz = units.Unit('Hz')


class FifiLsInstrumentInfo(SofiaInstrumentInfo):

    def __init__(self):
        """
        Initialize the FIFI-LS instrument information.

        Contains information on the FIFI-LS instrument parameters.
        """
        super().__init__()
        self.name = 'fifi_ls'
        self.channel = None
        self.alpha = np.nan
        self.ramps = -1
        self.resolution = Coordinate2D1(xy=[5, 5] * arcsec, z=0 * um)
        self.spectral_resolution = 1000.0  # Default

    @property
    def xy_resolution(self):
        """
        Return the average spatial resolution.

        Returns
        -------
        units.Quantity
        """
        return np.sqrt(self.resolution.x * self.resolution.y)

    @xy_resolution.setter
    def xy_resolution(self, resolution):
        """
        Set the average spatial resolution.

        Parameters
        ----------
        resolution : units.Quantity

        Returns
        -------
        None
        """
        self.resolution.xy_coordinates.set([resolution, resolution])

    @property
    def z_resolution(self):
        """
        Return the average spectral resolution.

        Returns
        -------
        units.Quantity
        """
        return self.resolution.z

    @z_resolution.setter
    def z_resolution(self, resolution):
        """
        Set the average spectral resolution.

        Parameters
        ----------
        resolution : units.Quantity

        Returns
        -------
        None
        """
        self.resolution.z_coordinates.set(resolution)

    @staticmethod
    def get_spectral_unit():
        """
        Return the size unit of the instrument.

        Returns
        -------
        units.Unit
        """
        return units.Unit('um')

    def get_spectral_size(self):
        """
        Return the instrument spectral point size.

        Returns
        -------
        units.Quantity
        """
        return self.z_resolution

    def apply_configuration(self):
        """
        Update HAWC+ instrument information with FITS header information.

        Updates the chopping information by taking the following keywords from
        the FITS header::

          SMPLFREQ - The detector readout rate (Hz)

        Returns
        -------
        None
        """
        super().apply_configuration()
        if self.options is None:  # pragma: no cover
            return

        self.channel = self.options.get_string('CHANNEL')
        if self.channel is None:
            raise ValueError("Unable to determine the primary array: "
                             "No CHANNEL key in header.")
        self.channel = self.channel.upper().strip()
        ch = self.channel[0]

        self.ramps = self.options.get_float(f'RAMPLN_{ch}',
                                            default=np.nan)
        if np.isnan(self.ramps):
            raise ValueError(f"Unable to determine the number of ramps: "
                             f"No RAMPLN_{ch} key in header.")

        self.alpha = self.options.get_float('ALPHA', default=np.nan) * second
        if np.isnan(self.alpha):
            raise ValueError("Unable to determine ramp sampling interval: "
                             "No ALPHA key in header.")

        self.sampling_interval = self.alpha * self.ramps
        self.integration_time = self.sampling_interval
        self.read_resolution()

    def read_resolution(self):
        """
        Determine the spatial and spectral resolution.

        Returns
        -------
        None
        """
        self.resolution = Coordinate2D1(xy_unit=self.get_size_unit(),
                                        z_unit=self.get_spectral_unit())
        self.xy_resolution = 5 * arcsec
        self.spectral_resolution = 1000.0
        filename = self.configuration.priority_file('spectral_resolution.txt')
        if filename is None:
            log.warning('Could not locate spectral_resolution.txt '
                        'configuration file: Will set default resolutions.')
            return

        names = ['ch', 'wavelength', 'res', 'fwhm']
        df = pd.read_csv(filename, comment='#', names=names,
                         delim_whitespace=True)

        if self.channel == 'BLUE':
            order = self.options.get_string('G_ORD_B', default='unknown')
            if order == 'unknown':
                ch = 'unknown'
            else:
                ch = f'{self.channel[0].lower()}{order.strip()}'
        elif self.channel == 'RED':
            ch = 'r'
        else:
            ch = 'unknown'

        rows = df.loc[df['ch'] == ch]
        if len(rows) == 0:
            log.warning(f'Could not locate channel {ch} in resolution file: '
                        f'Will set default resolutions.')
            return

        wavelength = self.options.get_float(
            f'G_WAVE_{self.channel[0].upper()}', default=np.nan)
        if np.isnan(wavelength):
            log.warning(f'Could not determine wavelength mean: '
                        f'Will set default resolution')
            return

        offset = (rows.wavelength - wavelength).abs()
        row = rows[offset == offset.min()].iloc[0]
        self.xy_resolution = float(row.fwhm) * arcsec
        self.spectral_resolution = float(row.res)
        self.z_resolution = self.wavelength / self.spectral_resolution

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
        header['SMPLFREQ'] = (
            to_header_float(1.0 / self.sampling_interval, 'Hz'),
            '(Hz) Detector readout rate.')
        header['CHANNEL'] = self.channel, 'Detector channel'
        header['ALPHA'] = (to_header_float(self.alpha, 'second'),
                           'Alpha value in fifiTime correlation')
        header['RAMPLN'] = (int(to_header_float(self.ramps)),
                            'Number of readouts per ramp')

    def get_point_size(self):
        """
        Return the instrument point size (instrument resolution).

        Returns
        -------
        point_size : Coordinate2D1
            The point size in (x,y) spatial coordinates and (z) spectral
            coordinates.
        """
        return self.resolution.copy()

    def get_source_size(self):
        """
        Return the size of the source for the instrument.

        Returns
        -------
        units.Quantity
        """
        if self.configuration is None:
            xy_source_size = 0.0
            z_source_size = 0.0
        else:
            source_size = self.configuration.get_float_list('sourcesize',
                                                            default=None)
            if source_size is None or len(source_size) == 0:
                xy_source_size = 0.0
                z_source_size = 0.0
            elif len(source_size) == 1:
                xy_source_size = source_size[0]
                z_source_size = 0.0
            else:
                xy_source_size, z_source_size = source_size[:2]

        xy_source_size *= self.get_size_unit()
        z_source_size *= self.get_spectral_unit()

        xy_beam_size = self.xy_resolution
        z_beam_size = self.z_resolution
        xy_size = np.hypot(xy_source_size, xy_beam_size)
        z_size = np.hypot(z_source_size, z_beam_size)
        return Coordinate2D1([xy_size, xy_size, z_size])

    def edit_image_header(self, header, scans=None):
        """
        Edit an image header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.
        scans : list (Scan), optional
            A list of scans to use during editing.

        Returns
        -------
        None
        """
        super().edit_image_header(header, scans=scans)
        header['BEAM'] = (to_header_float(self.resolution.x, 'arcsec'),
                          'The instrument FWHM (arcsec) of the beam.')
        header['BEAMZ'] = (to_header_float(self.resolution.z, 'um'),
                           'The instrument spectral FWHM (um).')
