# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units, log
from astropy.time import Time
import numpy as np
import os
import pandas as pd

from sofia_redux.scan.custom.sofia.info.detector_array import (
    SofiaDetectorArrayInfo)
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.utilities.utils import to_header_float

__all__ = ['FifiLsDetectorArrayInfo']

arcsec = units.Unit('arcsec')
degree = units.Unit('degree')
mm = units.Unit('mm')
um = units.Unit('um')


class FifiLsDetectorArrayInfo(SofiaDetectorArrayInfo):

    tel_sim_to_detector = 0.842
    n_spexel = 16
    n_spaxel = 25
    pixels = n_spexel * n_spaxel
    spaxel_rows = 5
    spaxel_cols = 5
    default_boresight_offset = Coordinate2D([0.0, 0.0], unit='arcsec')
    center_spaxel = 12
    pixel_indices = np.arange(pixels)
    spexel_indices = pixel_indices // n_spaxel
    spaxel_indices = pixel_indices % n_spaxel

    def __init__(self):
        """
        Initialize the FIFI-LS detector array information.

        Contains information specific to the FIFI-LS detector array.
        """
        super().__init__()
        self.plate_scale = np.nan * arcsec / mm
        self.pixel_sizes = Coordinate2D([np.nan, np.nan], unit='arcsec')
        self.equatorial = EquatorialCoordinates([np.nan, np.nan],
                                                unit='degree')
        self.boresight_equatorial = self.equatorial.copy()
        self.obs_equatorial = self.equatorial.copy()
        self.delta_map = Coordinate2D([0, 0], unit='arcsec')
        self.delta_map_rotated = Coordinate2D([0, 0], unit='arcsec')
        self.sky_angle = np.nan * degree
        self.detector_angle = np.nan * degree
        self.spatial_file = None
        self.pixel_positions = None
        self.pixel_offsets = None
        self.pixel_width = None
        self.date_obs = None
        self.channel = 'UNKNOWN'
        self.prime_array = 'UNKNOWN'
        self.dichroic = np.nan * um
        self.coefficients_1 = None
        self.coefficients_2 = None
        self.flip_sign = False
        self.rotation = 0 * degree
        self.boresight_equatorial_offset = self.default_boresight_offset.copy()
        self.native_boresight_offset = self.default_boresight_offset.copy()
        self.boresight_index = Coordinate2D([0.0, 0.0])

    @property
    def ch(self):
        """
        Return the single letter representation of the channel.

        Returns
        -------
        str
        """
        return self.channel[0].lower()

    @property
    def int_date(self):
        """
        Return the observation date as an integer of the form YYYYMMDD

        Returns
        -------
        int
        """
        if not isinstance(self.date_obs, Time):
            return -1
        return int(self.date_obs.isot.split('T')[0].replace('-', ''))

    def apply_configuration(self):
        """
        Update detector array information with FITS header information.

        Updates the detector information by taking the following keywords from
        the FITS header::

           stuff

        Returns
        -------
        None
        """
        if self.options is None:
            return

        self.plate_scale = self.options.get_float('PLATSCAL') * arcsec / mm
        self.date_obs = Time(self.options.get_string('DATE-OBS'))

        ra = self.options.get_hms_time('OBSRA', angle=True)
        dec = self.options.get_dms_angle('OBSDEC')
        self.equatorial = EquatorialCoordinates([ra, dec], unit='degree')
        self.boresight_equatorial = self.equatorial.copy()

        obs_lambda = self.options.get_float('OBSLAM', default=0) * degree
        obs_beta = self.options.get_float('OBSBET', default=0) * degree
        self.obs_equatorial = EquatorialCoordinates(
            [obs_lambda, obs_beta])

        map_d_lambda = self.options.get_float('DLAM_MAP', default=0) * arcsec
        map_d_beta = self.options.get_float('DBET_MAP', default=0) * arcsec
        self.delta_map = Coordinate2D([map_d_lambda, map_d_beta])
        self.boresight_equatorial.subtract_offset(self.delta_map)

        self.sky_angle = self.options.get_float('SKY_ANGL', default=0) * degree
        self.detector_angle = self.options.get_float(
            'DET_ANGL', default=0) * degree

        if self.sky_angle == 0:
            self.rotation = self.detector_angle
        else:
            self.rotation = 0 * degree

        self.channel = self.options.get_string(
            'CHANNEL', default='BLUE').upper().strip()
        if self.channel == 'RED':
            self.pixel_width = 3 * mm
        else:
            self.pixel_width = 1.5 * mm

        self.flip_sign = self.int_date < 20150501  # May, 2015
        self.pixel_size = self.plate_scale * self.pixel_width
        self.pixel_sizes = Coordinate2D([self.pixel_size, self.pixel_size])
        self.dichroic = self.options.get_float('DICHROIC') * um
        self.prime_array = self.options.get_string('PRIMARAY').strip().upper()
        self.calculate_delta_coefficients()
        self.set_boresight()
        self.calculate_pixel_offsets()

    def calculate_pixel_offsets(self):
        """
        Calculate the pixel offsets on the detector in arcseconds.

        Returns
        -------
        None
        """
        default_file = self.configuration.priority_file(
            os.path.join('spatial_cal', 'poscal_default.txt'))
        if default_file is None:
            raise ValueError(
                "Could not locate default date spatial calibration file.")
        df = pd.read_csv(
            default_file, comment='#', names=['date', 'ch', 'file'],
            delim_whitespace=True, dtype={'date': int})

        ch = self.ch
        date_int = self.int_date
        spatial_file = self.configuration.priority_file(
            df[(df['ch'] == ch) & (df['date'] >= date_int)].iloc[0].file)
        if spatial_file is None:  # pragma: no cover
            raise ValueError(
                "Could not locate spatial calibration file.")

        self.spatial_file = spatial_file
        df = pd.read_csv(spatial_file, names=['x', 'y'], dtype=float,
                         delim_whitespace=True, comment='#')
        self.pixel_positions = Coordinate2D([df['x'].values, df['y'].values],
                                            unit='mm')
        self.pixel_offsets = Coordinate2D(
            self.pixel_positions.coordinates * self.plate_scale)

    def calculate_delta_coefficients(self):
        """
        Used to derive offsets of the telescope boresight from the instrument.

        Returns
        -------
        None
        """
        self.coefficients_1 = None
        self.coefficients_2 = None
        coefficient_file = self.configuration.priority_file(
            os.path.join('spatial_cal', 'FIFI_LS_DeltaVector_Coeffs.txt'))
        if coefficient_file is None:
            raise ValueError(f"Could not find file: {coefficient_file}")

        df = pd.read_csv(
            coefficient_file, comment='#', delim_whitespace=True,
            names=['dt', 'ch', 'dch', 'bx', 'ax', 'rx', 'by', 'ay', 'ry'])

        dichroic = int(self.dichroic.value)
        try:
            rows = df[(df['dt'] >= self.int_date)
                      & (df['ch'] == self.prime_array[0].lower())
                      & (df['dch'] == dichroic)].sort_values('dt')
        except TypeError:  # pragma: no cover
            rows = []
        if len(rows) == 0:
            raise ValueError(f"No boresight offsets for {self.date_obs}")

        c = rows.iloc[0]
        self.coefficients_1 = np.zeros((2, 3))
        scale = self.tel_sim_to_detector
        self.coefficients_1[0] = c['ax'] * scale, c['bx'] * scale, c['rx']
        self.coefficients_1[1] = c['ay'] * scale, c['by'] * scale, c['ry']

        if self.prime_array[0].lower() == self.ch:
            return

        # Otherwise the prime array is not the channel and we need two sets
        # of coefficients

        rows = df[(df['dt'] >= self.int_date) &
                  (df['ch'] == self.ch) &
                  (df['dch'] == dichroic)].sort_values('dt').reset_index()
        c = rows.iloc[0]
        self.coefficients_2 = np.zeros((2, 3))
        self.coefficients_2[0] = c['ax'] * scale, c['bx'] * scale, c['rx']
        self.coefficients_2[1] = c['ay'] * scale, c['by'] * scale, c['ry']

    def set_boresight(self):
        """
        Set the boresight index of the detector array.

        Returns
        -------
        None
        """
        prime_ch = self.prime_array[0].upper()
        i0 = self.options.get_float(f'G_STRT_{prime_ch}')  # prime inductosyn
        cx0, cy0 = self.coefficients_1
        dx = cx0[1] + (cx0[0] * i0) - cx0[2]
        dy = cy0[1] - (cy0[0] * i0) - cy0[2]

        if self.coefficients_2 is not None:
            # grating inductosyn
            i1 = self.options.get_float(f'G_STRT_{self.ch.upper()}')
            cx1, cy1 = self.coefficients_2
            dx += (cx1[1] - cx0[1]) + (cx1[0] * i1 - cx0[0] * i0)
            dy -= (cy1[1] - cy0[1]) + (cy1[0] * i1 - cy0[0] * i0)

        cos_a, sin_a = np.cos(self.detector_angle), np.sin(self.detector_angle)
        map_dx = (cos_a * self.delta_map.x) - (sin_a * self.delta_map.y)
        map_dy = (sin_a * self.delta_map.x) + (cos_a * self.delta_map.y)
        self.delta_map_rotated = Coordinate2D([map_dx, map_dy], unit=arcsec)
        bx = self.plate_scale * dx * mm
        by = -self.plate_scale * dy * mm
        self.native_boresight_offset = Coordinate2D([bx, by], unit=arcsec)
        self.boresight_equatorial_offset = Coordinate2D(
            [bx + map_dx, by + map_dy])

        if not self.flip_sign:
            self.boresight_equatorial_offset.invert()

        if self.rotation != 0:
            self.boresight_equatorial_offset.rotate(self.rotation)

        log.info(f'Derived Boresight --> {self.native_boresight_offset}')
        log.info(f'Derived equatorial boresight offset --> '
                 f'{self.boresight_equatorial_offset}')

    def get_boresight_equatorial(self, xs, ys):
        """
        Get the boresight equatorial trajectory given input detector
        coordinates.

        Parameters
        ----------
        xs : units.Quantity
            The x-coordinates of shape (n, n_pixels)
        ys : units.Quantity
            The y-coordinates of shape (n, n_pixels)

        Returns
        -------
        boresight_equatorial : EquatorialCoordinates
        """
        return self.detector_coordinates_to_equatorial(
            self.get_boresight_trajectory(xs, ys))

    def get_boresight_trajectory(self, xs, ys):
        """
        Given x and y FIFI-LS coordinates, return the boresight trajectory.

        Parameters
        ----------
        xs : units.Quantity
            The x-coordinates of shape (n, n_pixels), (n_pixels,),
            (n_spexels, n_spaxels) or (n, n_spexels, n_spaxels)
        ys : units.Quantity
            The y-coordinates of shape (n, n_pixels)

        Returns
        -------
        detector_offsets : Coordinate2D
        """
        # This is for the center spaxel and zeroth spexel
        if xs.ndim == 3:
            x = xs[:, 0, self.center_spaxel]
            y = ys[:, 0, self.center_spaxel]
        elif xs.ndim == 2:
            if xs.shape[1] == self.pixels:
                x = xs[:, self.center_spaxel]
                y = ys[:, self.center_spaxel]
            else:
                x = xs[0, self.center_spaxel]
                y = ys[0, self.center_spaxel]
        elif xs.ndim == 1:
            x = xs[self.center_spaxel]
            y = ys[self.center_spaxel]
        else:
            raise ValueError(f"Incorrect xs and ys input shape: {xs.shape}")

        center_offset = self.pixel_offsets[self.center_spaxel]
        pixel_coordinates = Coordinate2D([x, y])
        rotate = isinstance(self.rotation, units.Quantity
                            ) and self.rotation != 0

        if not center_offset.is_null():
            if rotate:
                pixel_coordinates.rotate(-self.rotation)

            if not self.flip_sign:
                pixel_coordinates.invert()

            pixel_coordinates.subtract(self.delta_map_rotated)
            pixel_coordinates.invert_y()

            bx, by = pixel_coordinates.coordinates.copy()
            bx -= center_offset.x
            by += center_offset.y

            boresight = Coordinate2D([bx, by])  # Native coordinates
            boresight.invert_y()
            boresight.add(self.delta_map_rotated)

            if not self.flip_sign:
                boresight.invert()
            if rotate:
                boresight.rotate(self.rotation)

        else:
            boresight = pixel_coordinates.copy()

        return boresight

    def initialize_channel_data(self, data):
        """
        Apply this information to create and populate the channel data.

        The following attributes are determined from the detector::

          - col: The column on the array
          - row: The row on the array
          - spexel: The spexel index
          - spaxel: The spaxel index
          - position: The spaxel offset on the array (arcseconds)
          - wavelength : The spexel wavelength (um). Set to NaN.

        Additionally, the channel string ID is set to::

          <CHANNEL>[<spexel>,<spaxel>]

        where spexel and spaxel are described above and CHANNEL may be one of
        {B, R} for the RED and BLUE channels respectively.

        Parameters
        ----------
        data : FifiLsChannelData

        Returns
        -------
        None
        """
        index = np.arange(self.pixels)
        data.fixed_index = index
        data.set_default_values()
        data.spexel = self.spexel_indices.copy()
        data.spaxel = self.spaxel_indices.copy()
        data.flag = np.zeros(self.pixels, dtype=int)
        data.col = data.spaxel // self.spaxel_cols
        data.row = data.spaxel % self.spaxel_cols
        data.channel_id = np.array(
            [f'{self.ch.upper()}[{x[0]},{x[1]}]'
             for x in zip(data.spexel, data.spaxel)])

    def detector_coordinates_to_equatorial_offsets(self, coordinates):
        """
        Convert detector coordinates to equatorial offsets.

        Parameters
        ----------
        coordinates : Coordinate2D

        Returns
        -------
        equatorial_offsets : Coordinate2D
        """
        rotation = self.sky_angle
        offsets = coordinates.copy()
        if rotation != 0:
            offsets.rotate(rotation)
        offsets.invert_x()
        return offsets

    def equatorial_offsets_to_detector_coordinates(self, offsets):
        """
        Convert equatorial offsets to detector coordinates.

        Parameters
        ----------
        offsets : Coordinate2D

        Returns
        -------
        detector_coordinates : Coordinate2D
        """
        coordinates = offsets.copy()
        coordinates.invert_x()
        rotation = self.sky_angle
        if rotation != 0:
            coordinates.rotate(-rotation)
        return coordinates

    def detector_coordinates_to_equatorial(self, coordinates):
        """
        Convert detector coordinates to equatorial coordinates.

        Parameters
        ----------
        coordinates : Coordinate2D

        Returns
        -------
        equatorial : EquatorialCoordinates
        """
        equatorial = self.boresight_equatorial.copy()
        equatorial.add_offset(
            self.detector_coordinates_to_equatorial_offsets(coordinates))
        return equatorial

    def equatorial_to_detector_coordinates(self, equatorial):
        """
        Convert equatorial coordinates to detector coordinates.

        Parameters
        ----------
        equatorial : EquatorialCoordinates

        Returns
        -------
        detector_coordinates : Coordinate2D
        """
        offset = equatorial.get_offset_from(self.boresight_equatorial)
        return self.equatorial_offsets_to_detector_coordinates(offset)

    def find_pixel_positions(self, detector_xy):
        """
        Calculate the pixel positions based on inputs.

        Parameters
        ----------
        detector_xy : Coordinate2D
            The raw XY detector coordinates of shape (n, n_pixels) or
            (n, n_spexels, n_spaxels).

        Returns
        -------
        pixel_positions : Coordinate2D
        """
        x = np.atleast_2d(detector_xy.x.copy())
        y = np.atleast_2d(detector_xy.y.copy())
        if x.ndim == 3:
            x = x[..., self.spexel_indices, self.spaxel_indices]
            y = y[..., self.spexel_indices, self.spaxel_indices]

        boresight_trajectory = self.get_boresight_trajectory(x, y)
        x -= boresight_trajectory.x[:, None]
        y -= boresight_trajectory.y[:, None]
        dx = np.nanmean(x, axis=0)
        dy = np.nanmean(y, axis=0)

        positions = Coordinate2D([dx, dy])
        rotation = self.rotation
        if rotation != 0:
            if not isinstance(rotation, units.Quantity):
                rotation *= units.Unit('radian')
            positions.rotate(-rotation)
        return positions

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
        header['CROTA2'] = (to_header_float(-self.sky_angle, 'degree'),
                            'Rotation angle (deg)')
        header['CTYPE3'] = 'WAVE', 'Axis 3 type and projection'
        header['SPECSYS'] = 'BARYCENT', 'Spectral reference frame'
