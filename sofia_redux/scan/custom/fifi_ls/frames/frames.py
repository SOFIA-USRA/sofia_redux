# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.time import Time
import numpy as np

from sofia_redux.scan.coordinate_systems.index_3d import Index3D
from sofia_redux.scan.custom.fifi_ls.flags.frame_flags import (
    FifiLsFrameFlags)
from sofia_redux.scan.custom.fifi_ls.frames import \
    fifi_ls_frame_numba_functions as fnf
from sofia_redux.scan.custom.sofia.frames.frames import SofiaFrames

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.utilities.utils import safe_sidereal_time

__all__ = ['FifiLsFrames']


arcsec = units.Unit('arcsec')


class FifiLsFrames(SofiaFrames):

    flagspace = FifiLsFrameFlags

    def __init__(self):
        """
        Initialize FIFI-LS frames.

        FIFI-LS frames contain the timestream data for FIFI-LS integrations.
        """
        super().__init__()

    def copy(self):
        """
        Return a copy of the FIFI-LS frames.

        Returns
        -------
        FifiLsFrames
        """
        return super().copy()

    @property
    def default_channel_fields(self):
        """
        Returns default frame/channel type default values.

        This framework is similar to `default_field_types`, but is used to
        populate frame/channel data of shape (n_frames, n_channels).

        Returns
        -------
        fields : dict
        """
        fields = super().default_channel_fields
        fields['map_index'] = (Index3D, -1)
        return fields

    @property
    def info(self):
        """
        Return the scan info object.

        Returns
        -------
        sofia_redux.scan.custom.fifi_ls.info.info.FifiLsInfo
        """
        return super().info

    @property
    def site(self):
        """
        Return the LON/LAT SOFIA coordinates.

        Returns
        -------
        GeodeticCoordinates
        """
        return self.sofia_location

    def detector_coordinates_to_equatorial_offsets(self, coordinates):
        """
        Convert detector offsets to equatorial offsets.

        Parameters
        ----------
        coordinates : Coordinate2D

        Returns
        -------
        equatorial_offsets : Coordinate2D
        """
        detector = self.info.detector_array
        return detector.detector_coordinates_to_equatorial_offsets(coordinates)

    def detector_coordinates_to_equatorial(self, coordinates):
        """
        Convert detector coordinates to equatorial coordinates

        Parameters
        ----------
        coordinates : Coordinate2D

        Returns
        -------
        EquatorialCoordinates
        """
        detector = self.info.detector_array
        return detector.detector_coordinates_to_equatorial(coordinates)

    def read_hdul(self, hdul):
        """
        Populate the data from a list of HDUs.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            An HDU list containing FIFI-LS data from a WSH redux step.

        Returns
        -------
        None
        """
        do_uncorrected = self.configuration.get_bool('fifi_ls.uncorrected')

        if do_uncorrected and 'UNCORRECTED_FLUX' in hdul:
            flux_hdu = hdul['UNCORRECTED_FLUX']
        else:
            flux_hdu = hdul['FLUX']
            do_uncorrected = False

        spexels = self.channels.data.spexel
        spaxels = self.channels.data.spaxel
        self.data = np.asarray(flux_hdu.data[:, spexels, spaxels],
                               dtype=float)

        if do_uncorrected:
            variance = np.asarray(
                hdul['UNCORRECTED_STDDEV'].data, dtype=float) ** 2
        else:
            variance = np.asarray(hdul['STDDEV'].data, dtype=float) ** 2
        variance = variance[:, spexels, spaxels]
        self.relative_weight = fnf.get_relative_frame_weights(variance)

        is_lab = self.configuration.get_bool('lab')
        self.has_telescope_info[...] = not is_lab
        if is_lab:
            self.valid[...] = True
            return

        detector = self.info.detector_array
        self.instrument_vpa[...] = detector.detector_angle
        self.telescope_vpa[...] = detector.sky_angle

        # Rotation from pixel coordinates to telescope coordinates
        self.set_rotation(self.instrument_vpa - self.telescope_vpa)

        # Rotation from telescope coordinates to equatorial
        self.set_parallactic_angle(self.telescope_vpa)

        start_time = Time(self.astrometry.time_stamp)
        n_frames = self.size
        dt = self.info.instrument.sampling_interval
        times = start_time + np.arange(n_frames) * dt
        self.utc = times.unix
        self.mjd = times.mjd

        lon = np.linspace(self.info.aircraft.longitude.start,
                          self.info.aircraft.longitude.end, n_frames)
        lat = np.linspace(self.info.aircraft.latitude.start,
                          self.info.aircraft.latitude.end, n_frames)
        site = GeodeticCoordinates(np.stack((lon, lat)), copy=False)
        self.site[...] = site  # linked to sofia_location attribute

        self.lst = safe_sidereal_time(times, 'mean', longitude=lon)

        xs = hdul['XS'].data[:, spexels, spaxels] * arcsec
        ys = hdul['YS'].data[:, spexels, spaxels] * arcsec
        boresight_xy = detector.get_boresight_trajectory(xs, ys)
        self.equatorial[...] = self.detector_coordinates_to_equatorial(
            boresight_xy)

        reference = self.info.astrometry.equatorial
        horizontal_offset = self.equatorial.get_native_offset_from(reference)

        self.object_equatorial.coordinates[0] = reference.coordinates[0]
        self.object_equatorial.coordinates[1] = reference.coordinates[1]

        self.equatorial_native_to_horizontal_offset(
            horizontal_offset, in_place=True)
        self.horizontal_offset[...] = horizontal_offset

        apparent = self.equatorial.copy()
        self.info.astrometry.to_apparent.precess(apparent)
        horizontal = apparent.to_horizontal(self.site, self.lst)
        self.horizontal[...] = horizontal
        self.chopper_position.zero()

        self.valid[...] = True

    def validate(self):
        """
        Validate frame data after read.

        Should set the `validated` (checked) attribute if necessary.

        Returns
        -------
        None
        """
        fnf.validate(
            valid=self.valid,
            validated=self.validated,
            weight=self.relative_weight,
            check_coordinates=self.has_telescope_info,
            equatorial_null=self.equatorial.is_null(),
            equatorial_nan=self.equatorial.is_nan(),
            horizontal_nan=self.horizontal.is_nan(),
            chopper_nan=self.chopper_position.is_nan(),
            lst=self.lst,
            site_lon=self.site.longitude.value,
            site_lat=self.site.latitude.value,
            telescope_vpa=self.telescope_vpa.value,
            instrument_vpa=self.instrument_vpa.value)

        sample_skip = self.flagspace.convert_flag('SAMPLE_SKIP').value
        self.sample_flag[np.isnan(self.data)] |= sample_skip
        super().validate()
