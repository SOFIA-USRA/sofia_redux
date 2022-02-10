# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.time import Time
import numpy as np

from sofia_redux.scan.custom.hawc_plus.flags.frame_flags import (
    HawcPlusFrameFlags)
from sofia_redux.scan.custom.hawc_plus.frames import \
    hawc_plus_frame_numba_functions as hnf
from sofia_redux.scan.custom.sofia.frames.frames import SofiaFrames

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000

__all__ = ['HawcPlusFrames']


class HawcPlusFrames(SofiaFrames):

    flagspace = HawcPlusFrameFlags
    FITS_FLAG_NORMAL_OBSERVING = 0
    FITS_FLAG_LOS_REWIND = 1
    FITS_FLAG_IVCURVES = 2
    FITS_FLAG_BETWEEN_SCANS = 3

    def __init__(self):
        super().__init__()
        self.mce_serial = None
        self.hwp_angle = None
        self.los = None
        self.roll = None
        self.status = None

        # Special 2-D
        self.jump_counter = None

    @property
    def default_field_types(self):
        """
        Used to define the default values for data arrays.

        Returns a dictionary of structure {field: default_value}.  The default
        values have the following effects:

        type - empty numpy array of the given type.
        value - full numpy array of the given value.
        astropy.units.Unit - empty numpy array (float) in the given unit.
        astropy.units.Quantity - full numpy array of the given quantity.

        If a tuple is provided, the array will have additional axes appended
        such that the first element gives the type as above, and any additional
        integers give additional axes dimensions,  e.g. (0.0, 2, 3) would
        result in a numpy array filled with zeros of shape (self.size, 2, 3).

        Returns
        -------
        fields : dict
        """
        fields = super().default_field_types
        fields.update({
            'mce_serial': 0,
            'hwp_angle': 0.0 * units.Unit('deg'),
            'los': 0.0 * units.Unit('deg'),
            'roll': 0.0 * units.Unit('deg'),
            'status': 0,
        })
        return fields

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
        fields.update({'jump_counter': 0})
        return fields

    @property
    def readout_attributes(self):
        """Attributes that will be operated on by the `shift` method."""
        fields = super().readout_attributes
        fields.add('jump_counter')
        fields.add('chopper_position')
        fields.add('hwp_angle')
        fields.add('mjd')
        fields.add('mce_serial')
        return fields

    @property
    def info(self):
        """
        Return the scan info object.

        Returns
        -------
        HawcPlusInfo
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

    def configure_hdu_columns(self, hdu):
        """
        Given an HDU and a scan, return columns to be read.

        Will also update scan astrometry if necessary.

        Parameters
        ----------
        hdu : astropy.io.fits.hdu.table.BinTableHDU
            A data HDU containing "timestream" data.

        Returns
        -------
        columns : dict
        """
        columns = {
            'ts': 'Timestamp',
            'sn': 'FrameCounter',
            'jump': 'FluxJumps',
            'dac': 'SQ1Feedback',
            'hwp': 'hwpCounts',
            'stat': 'Flag',
            'az': 'AZ',
            'el': 'EL',
            'ra': 'RA',
            'dec': 'DEC',
            'ora': 'RA',  # may change
            'odec': 'DEC',  # may change
            'lst': 'LST',
            'avpa': 'SIBS_VPA',
            'tvpa': 'TABS_VPA',
            'cvpa': 'Chop_VPA',
            'lon': 'LON',
            'lat': 'LAT',
            'chopr': 'sofiaChopR',
            'chops': 'sofiaChopS',
            'pwv': 'PWV',
            'los': 'LOS',
            'roll': 'ROLL'
        }

        if self.scan.is_nonsidereal:
            columns['ora'] = 'NonSiderealRA'
            columns['odec'] = 'NonSiderealDec'

        if self.configuration.get_bool('lab'):
            log.info("Lab mode data reduction.  Ignoring telescope data...")
            for key, name in columns.items():
                if key not in ['ts', 'sn', 'jump', 'dac', 'hwp']:
                    columns[key] = None
                    continue

        for key, name in columns.items():
            if name not in hdu.columns.names:
                columns[key] = None
                log.warning(f"Missing {name} KEY in HDU")

        # Update astrometry if necessary
        for astrometry in [self.info.astrometry,
                           self.integration.info.astrometry]:
            if astrometry.equatorial is None:
                ra = hdu.data[columns['ra']][0] * units.Unit('hourangle')
                dec = hdu.data[columns['dec']][0] * units.Unit('deg')
                astrometry.equatorial = EquatorialCoordinates(
                    np.stack((ra, dec)), epoch=J2000)

        if columns['ora'] is not None or columns['odec'] is not None:
            if (np.isnan(hdu.data[columns['ora']][0])
                    or np.isnan(hdu.data[columns['odec']][0])):
                columns['ora'] = None
                columns['odec'] = None
                if self.scan.is_nonsidereal:
                    log.warning("Missing NonSiderealRA/NonSiderealDEC "
                                "columns. Forcing sidereal mapping.")
                    self.info.astrometry.is_nonsidereal = False

        return columns

    def read_hdus(self, hdus):
        """
        Populate the data from a list of HDUs.

        Parameters
        ----------
        hdus : list (astropy.io.fits.hdu.table.BinTableHDU)
            A list of data HDUs containing "timestream" data.

        Returns
        -------
        None
        """
        start_index = 0
        for hdu in hdus:
            end_index = start_index + hdu.header['NAXIS2']
            self.apply_hdu(hdu, start_index=start_index, end_index=end_index)
            start_index = end_index

    def apply_hdu(self, hdu, start_index=0, end_index=None):
        """
        Read a single data HDU.

        Parameters
        ----------
        hdu : astropy.io.fits.hdu.table.BinTableHDU
            A data HDU containing "timestream" data.
        start_index : int
            The index at which to start populating frame data.
        end_index : int
            The index at which to finish populating frame data.

        Returns
        -------
        None
        """
        deg = units.Unit('deg')
        hourangle = units.Unit('hourangle')

        table = hdu.data
        columns = self.configure_hdu_columns(hdu)
        dac = table[columns['dac']]
        jump = table[columns['jump']]
        log.debug(f"FITS HDU has {dac.shape[1]} x {dac.shape[2]} arrays.")

        if end_index is None:
            end_index = self.size
        indices = slice(start_index, end_index)
        n_frames = end_index - start_index
        row, col = self.channels.data.fits_row, self.channels.data.fits_col
        subarray_norm = self.channels.subarray_gain_renorm[
            self.channels.data.sub]

        log.debug("Reading data from HDU")
        self.data[indices] = dac[indices, row, col] / subarray_norm
        self.jump_counter[indices] = jump[indices, row, col]
        log.debug("...Done.")

        if columns['sn'] is not None:
            self.mce_serial[indices] = table[columns['sn']]

        self.utc[indices] = table[columns['ts']]
        self.mjd[indices] = Time(self.utc[indices], format='unix').mjd

        if columns['hwp'] is not None:
            self.hwp_angle[indices] = (
                table[columns['hwp']] * self.info.instrument.hwp_step
            ) - self.info.instrument.hwp_telescope_vertical

        is_lab = self.configuration.get_bool('lab') or None in [
            columns['ra'], columns['dec']]
        self.has_telescope_info[indices] = not is_lab
        if is_lab:
            self.valid[indices] = True
            return

        # Below is telescope information, ignored for 'lab' mode reductions.
        self.status[indices] = table[columns['stat']]

        if columns['pwv'] is not None:
            pwv = table[columns['pwv']]
            pwv[pwv == -9999] = np.nan
            self.pwv[indices] = pwv * units.Unit('um')

        have_site = None not in ([columns['lat'], columns['lon'],
                                  columns['lst']])
        if have_site:
            lon = table[columns['lon']] * deg
            lat = table[columns['lat']] * deg
            site = GeodeticCoordinates(np.stack((lon, lat)), copy=False)
            self.site[indices] = site
        else:
            site = None

        if columns['lst'] is not None:
            self.lst[indices] = table[columns['lst']] * hourangle

        ra = table[columns['ra']] * hourangle
        dec = table[columns['dec']] * deg
        equatorial = EquatorialCoordinates(
            np.stack((ra, dec)), epoch=self.info.telescope.epoch, copy=False)

        self.equatorial[indices] = equatorial

        if (self.scan.is_nonsidereal
                and None not in [columns['ora'], columns['odec']]):

            ora = (table[columns['ora']] * hourangle) % (24 * hourangle)
            odec = table[columns['odec']] * deg
            self.object_equatorial[indices] = EquatorialCoordinates(
                np.stack((ora, odec)),
                epoch=self.info.telescope.epoch,
                copy=False)
            reference = self.object_equatorial[indices]
        else:
            reference = self.info.astrometry.equatorial

        if columns['avpa'] is not None:
            self.instrument_vpa[indices] = table[columns['avpa']] * deg
        if columns['tvpa'] is not None:
            self.telescope_vpa[indices] = table[columns['tvpa']] * deg
        if columns['cvpa'] is not None:
            self.chop_vpa[indices] = table[columns['cvpa']] * deg
        if columns['los'] is not None:
            self.los[indices] = table[columns['los']] * deg
        if columns['roll'] is not None:
            self.roll[indices] = table[columns['roll']] * deg

        # Rotation from pixel coordinates to telescope coordinates
        self.set_rotation(self.instrument_vpa[indices]
                          - self.telescope_vpa[indices], indices=indices)

        # Rotation from telescope coordinates to equatorial
        # Sets cos_pa and sin_pa
        self.set_parallactic_angle(self.telescope_vpa[indices],
                                   indices=indices)

        horizontal_offset = equatorial.get_native_offset_from(reference)
        self.equatorial_native_to_horizontal_offset(
            horizontal_offset, indices=indices, in_place=True)
        self.horizontal_offset[indices] = horizontal_offset

        # Calculate the chopper positions
        chop_scale = self.info.chopping.volts_to_angle * units.Unit('volt')
        if columns['chops'] is not None:
            cx = table[columns['chops']] * -chop_scale
        else:
            cx = np.zeros(n_frames, dtype=float) * chop_scale.unit
        if columns['chopr'] is not None:
            cy = table[columns['chopr']] * -chop_scale
        else:
            cy = np.zeros(n_frames, dtype=float) * chop_scale.unit
        chopper_position = Coordinate2D(np.stack((cx, cy)), copy=False)

        if self.configuration.get_bool('chopper.invert'):
            chopper_position.invert()

        chopper_position.rotate(self.chop_vpa[indices]
                                - self.telescope_vpa[indices])
        self.chopper_position[indices] = chopper_position

        # If lat/lon site data is available, use it to calculate horizontal
        # coordinates rather than using noisy aircraft values.
        if have_site:
            apparent = equatorial.copy()
            self.info.astrometry.to_apparent.precess(apparent)
            horizontal = apparent.to_horizontal(site, self.lst[indices])
        else:
            horizontal = HorizontalCoordinates(np.stack(
                (table[columns['az']], table[columns['el']])),
                unit='degree', copy=False)

        self.horizontal[indices] = horizontal
        self.valid[indices] = True

    def add_data_from(self, other_frames, scaling=1.0, indices=None):
        """
        Add data from other frames to these.

        Parameters
        ----------
        other_frames : HawcPlusFrames
        scaling : float, optional
        indices : int or slice or numpy.ndarray (int or bool)
            The frame indices to add to.

        Returns
        -------
        None
        """
        super().add_data_from(other_frames, scaling=scaling, indices=indices)
        if scaling == 0:
            return
        if indices is None:
            self.hwp_angle += scaling * other_frames.hwp_angle
        else:
            if self.is_singular:
                self.hwp_angle = self.hwp_angle + (
                    scaling * other_frames.hwp_angle)
            else:
                self.hwp_angle[indices] += scaling * other_frames.hwp_angle

    def validate(self):
        """
        Validate frame data after read.

        Should set the `validated` (checked) attribute if necessary.

        Returns
        -------
        None
        """
        chopping = self.info.chopping.chopping
        if self.configuration.has_option('chopped'):
            chopping |= self.configuration.get_bool('chopped')
        elif 'chopped' in self.configuration.disabled:
            chopping = False

        hnf.validate(
            valid=self.valid,
            validated=self.validated,
            status=self.status,
            chop_length=self.chopper_position.length.to('arcsec').value,
            chopping=chopping,
            use_between_scans=self.configuration.is_configured('betweenscans'),
            normal_observing_flag=self.FITS_FLAG_NORMAL_OBSERVING,
            between_scan_flag=self.FITS_FLAG_BETWEEN_SCANS,
            transit_tolerance=self.info.chopping.transit_tolerance.to(
                'arcsec').value,
            chopper_amplitude=self.info.chopping.amplitude.to('arcsec').value,
            check_coordinates=self.has_telescope_info,
            non_sidereal=self.astrometry.is_nonsidereal,
            equatorial_null=self.equatorial.is_null(),
            equatorial_nan=self.equatorial.is_nan(),
            object_null=self.object_equatorial.is_null(),
            horizontal_nan=self.horizontal.is_nan(),
            chopper_nan=self.chopper_position.is_nan(),
            lst=self.lst,
            site_lon=self.site.longitude.value,
            site_lat=self.site.latitude.value,
            telescope_vpa=self.telescope_vpa.value,
            instrument_vpa=self.instrument_vpa.value)

        valid_mask = self.valid & self.has_telescope_info
        valid = np.nonzero(valid_mask)[0]
        invalid = np.nonzero(~valid_mask)[0]
        if valid.size > 0:

            chopper_position = self.chopper_position[valid]
            horizontal_offset = self.horizontal_offset[valid]
            horizontal = self.horizontal[valid]
            equatorial = self.equatorial[valid]

            horizontal_offset.add(chopper_position)
            horizontal.add_offset(chopper_position)

            equatorial_offset = self.horizontal_to_native_equatorial_offset(
                chopper_position, indices=valid, in_place=False)

            equatorial.add_native_offset(equatorial_offset)

            self.horizontal_offset[valid] = horizontal_offset
            self.horizontal[valid] = horizontal
            self.equatorial[valid] = equatorial
            self.hwp_angle[valid] += self.telescope_vpa[valid]

        if invalid.size > 0:
            self.chopper_position.nan(indices=invalid)
            self.horizontal_offset.nan(indices=invalid)
            self.equatorial.nan(indices=invalid)
            self.horizontal.nan(indices=invalid)
            self.hwp_angle[invalid] = np.nan

        if self.info.detector_array.dark_squid_correction:
            self.dark_correct()

        super().validate()

    def dark_correct(self):
        """
        Perform the squid dark correction for blind channels.

        Returns
        -------
        None
        """
        channels = self.scan.channels.data
        squid_lookup = self.info.detector_array.dark_squid_lookup

        blind_indices = channels.is_flagged('BLIND')
        blind_sub = channels.sub[blind_indices]
        blind_col = channels.col[blind_indices]

        squid_fixed_indices = squid_lookup[blind_sub, blind_col]

        # Cull is used to that the two indices are of the same shape.
        # Invalid indices are set to -1.
        frame_squid_indices = self.find_fixed_indices(
            squid_fixed_indices, cull=False)
        frame_channel_indices = self.find_channel_fixed_indices(
            channels.fixed_index[blind_indices], cull=False)

        hnf.dark_correct(
            data=self.data,
            valid_frame=self.valid,
            channel_indices=frame_channel_indices,
            squid_indices=frame_squid_indices)

    def set_from_downsampled(self, frames, start_indices, valid, window):
        """
        Set the data for these frames by downsampling higher-res frames.

        Parameters
        ----------
        frames : HawcPlusFrames
            The frames at a higher resolution.
        start_indices : numpy.ndarray (int)
            The start indices containing the first index of the high resolution
            frame for each convolution with the window function.  Should be
            of shape (self.size,).
        valid : numpy.ndarray (bool)
            A boolean mask indicating whether a downsampled frame could be
            determined from the higher resolution frames.  Should be of shape
            (self.size,).
        window : numpy.ndarray (float)
            The window function used for convolution of shape (n_window,).

        Returns
        -------
        None
        """
        super().set_from_downsampled(frames, start_indices, valid, window)

        if isinstance(frames.hwp_angle, units.Quantity):
            unit = frames.hwp_angle.unit
            hwp_angle = frames.hwp_angle.value
        else:
            unit = None
            hwp_angle = frames.hwp_angle

        low_res_hwp = hnf.downsample_hwp_angle(
            hwp_angle=hwp_angle,
            start_indices=start_indices,
            valid=valid,
            window=window)

        if unit is not None:
            low_res_hwp = low_res_hwp * unit

        # TODO: This is a blatant bug in CRUSH which we replicate here.
        # TODO: The correct line should be
        # self.hwp_angle = low_res_hwp
        self.hwp_angle = self.hwp_angle + low_res_hwp  # TODO: Change
