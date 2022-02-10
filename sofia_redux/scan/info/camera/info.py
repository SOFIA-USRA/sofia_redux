# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import log, units
import numpy as np

from sofia_redux.scan.info.info import Info
from sofia_redux.scan.info.camera.instrument import CameraInstrumentInfo
from sofia_redux.scan.source_models.pixel_map import PixelMap
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.flags.mounts import Mount

__all__ = ['CameraInfo']


class CameraInfo(Info):

    def __init__(self, configuration_path=None):
        """
        Initialize a CameraInfo object.

        Parameters
        ----------
        configuration_path : str, optional
            An alternate directory path to the configuration tree to be
            used during the reduction.  The default is
            <package>/data/configurations.
        """
        super().__init__(configuration_path=configuration_path)
        self.name = 'camera'
        self.instrument = CameraInstrumentInfo()

    def get_source_model_instance(self, scans, reduction=None):
        """
        Return the source model applicable to the channel type.

        Parameters
        ----------
        scans : list (Scan)
            A list of scans for which to create the source model.
        reduction : Reduction, optional
            The reduction to which the model will belong.

        Returns
        -------
        Map
        """
        source_type = self.configuration.get_string('source.type')
        if source_type == 'pixelmap':
            return PixelMap(info=self, reduction=reduction)
        else:
            return super().get_source_model_instance(scans,
                                                     reduction=reduction)

    def get_rotation_angle(self):
        """
        Return the rotation angle of the camera.

        Returns
        -------
        angle : units.Quantity
        """
        return self.instrument.rotation

    def parse_image_header(self, header):
        """
        Parse a FITS image header.

        Parameters
        ----------
        header : astropy.io.fits.Header
            The FITS image header to parse.

        Returns
        -------
        None
        """
        super().parse_image_header(header)
        resolution = header.get('BEAM')
        if resolution is None:
            resolution = self.resolution.to('arcsec')
        else:
            resolution = resolution * units.Unit('arcsec')
        self.resolution = resolution

    def edit_image_header(self, header, scans=None):
        """
        Edit a FITS image header with the channel information.

        Parameters
        ----------
        header : astropy.io.fits.Header
            The FITS image header to edit.
        scans : Scan or list (Scan), optional
            A list of scans or single scan to also edit with.

        Returns
        -------
        None
        """
        super().edit_image_header(header, scans=scans)
        header['BEAM'] = (self.resolution.to('arcsec').value,
                          'The instrument FWHM (arcsec) of the beam.')

    def set_pointing(self, scan=None):
        """
        Set the pointing for the channels.

        Parameters
        ----------
        scan : Scan, optional
            A scan to set pointing.

        Returns
        -------
        None
        """
        if self.configuration.get_bool('point'):
            return
        log.debug("Setting 'point' option to obtain "
                  "pointing/calibration data.")
        self.configuration.set_option('point', True)
        if scan is not None:
            scan.configuration.set_option('point', True)

    def get_pointing_center_offset(self):
        """
        Return the pointing center offset.

        Returns
        -------
        offset : Coordinate2D
        """
        return Coordinate2D(np.zeros(2), unit='arcsec')

    def get_pointing_offset(self, rotation_angle=None):
        """
        Return the pointing offset.

        Parameters
        ----------
        rotation_angle : units.Quantity, optional
            The rotation angle.

        Returns
        -------
        offset : Coordinate2D
        """
        if rotation_angle is None:
            rotation_angle = self.get_rotation_angle()

        offset = Coordinate2D(np.zeros(2), unit='arcsec')

        if self.instrument.mount == Mount.CASSEGRAIN:
            sin_a = np.sin(rotation_angle)
            cos_a = np.cos(rotation_angle)
            dp = self.get_pointing_center_offset()
            offset.set_x((dp.x + (1.0 - cos_a)) + (dp.y * sin_a))
            offset.set_y((dp.x * sin_a) + (dp.y * (1.0 - cos_a)))

        return offset

    @abstractmethod
    def max_pixels(self):
        """
        Return the maximum number of pixels.

        Returns
        -------
        count : int
        """
        pass
