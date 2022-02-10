# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.info.instrument import InstrumentInfo
from sofia_redux.scan.utilities.utils import to_header_float

__all__ = ['CameraInstrumentInfo']


class CameraInstrumentInfo(InstrumentInfo):

    def __init__(self):
        self.rotation = 0.0 * units.Unit('deg')
        super().__init__()

    def parse_image_header(self, header):
        """
        Apply settings from a FITS image header.

        Parameters
        ----------
        header : astropy.fits.Header

        Returns
        -------
        None
        """
        if 'BEAM' in header:
            self.resolution = float(header['BEAM']) * units.Unit('arcsec')

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
        header['BEAM'] = (to_header_float(self.resolution, 'arcsec'),
                          'The instrument FWHM (arcsec) of the beam.')
