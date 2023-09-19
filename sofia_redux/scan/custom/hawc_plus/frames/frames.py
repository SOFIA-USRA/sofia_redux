# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.io import fits
from astropy.time import Time
import numpy as np

from sofia_redux.scan.custom.hawc_plus.flags.frame_flags import (
    HawcPlusFrameFlags)
from sofia_redux.scan.custom.hawc_plus.flags.polarimetry_flags import \
    HawcPlusPolarModulation
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
        """
        Initialize HAWC+ frames.

        HAWC+ frames contain the timestream data for HAWC+ integrations.
        The include additional information on the MCE serial, the HWP angle,
        line-of-sight (LOSS), aircraft roll, the observing status flag, and
        the detector jump count.
        """
        super().__init__()
        self.mce_serial = None
        self.hwp_angle = None
        self.los = None
        self.roll = None
        self.status = None
        self.unpolarized_gain = None
        self.q = None
        self.u = None

        # Special 2-D
        self.jump_counter = None

    def copy(self):
        """
        Return a copy of the HAWC+ frames.

        Returns
        -------
        HawcPlusFrames
        """
        return super().copy()

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
            'unpolarized_gain': 0.0,  # polarization parameters
            'q': 0.0,
            'u': 0.0
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
        """
        Attributes that will be operated on by the `shift` method.

        Returns
        -------
        set (str)
        """
        fields = super().readout_attributes
        fields.add('jump_counter')
        fields.add('chopper_position')
        fields.add('hwp_angle')
        fields.add('mjd')
        fields.add('mce_serial')
        fields.add('unpolarized_gain')
        fields.add('q')
        fields.add('u')
        return fields

    @property
    def info(self):
        """
        Return the scan info object.

        Returns
        -------
        sofia_redux.scan.custom.hawc_plus.info.info.HawcPlusInfo
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
            if name is None:
                continue
            if name not in hdu.columns.names:
                columns[key] = None
                log.warning(f"Missing {name} KEY in HDU")

        # Update astrometry if necessary
        for astrometry in [self.info.astrometry,
                           self.integration.info.astrometry]:
            if astrometry.equatorial is None:
                try:
                    ra = hdu.data[columns['ra']][0] * units.Unit('hourangle')
                    dec = hdu.data[columns['dec']][0] * units.Unit('deg')
                    astrometry.equatorial = EquatorialCoordinates(
                        np.stack((ra, dec)), epoch=J2000)
                except (KeyError, IndexError, ValueError):  # pragma: no cover
                    pass

        if columns['ora'] is not None or columns['odec'] is not None:
            try:
                obsra = hdu.data[columns['ora']][0]
                obsdec = hdu.data[columns['odec']][0]
            except (KeyError, IndexError, ValueError):
                obsra = obsdec = np.nan

            if np.isnan(obsra) or np.isnan(obsdec):
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
        self.validate_polarimetry()

    def get_polarization_theta(self, offset=5 * units.Unit('degree'),
                               telvpa_from_header=False):
        """
        Return the polarization theta angle from HawcPlusFrames.

        where::
            theta = vpa + (2 * (hwp_zero - offset)) + grid_angle

        The vpa is taken from the TELVPA key in the header if
        `telvpa_from_header`=`True`, or the instrument vpa otherwise.
        `hwp_zero` is taken from HWPINIT in the header if present and
        commanded, or the calculated HWP angle in the frame data.  The
        grid_angle is provided in the configuration from the
        polarization.grid_angle parameter.

        Parameters
        ----------
        offset : units.Quantity, optional
            The offset to apply to the correction.
        telvpa_from_header : bool, optional
            If `True`, get the telescope VPA from the header rather than using
            the actual instrument VPA.

        Returns
        -------
        units.Quantity
        """
        degree = units.Unit('degree')
        half = 180 * degree
        full = 360 * degree
        header = self.configuration.fits.header
        if header is None:
            header = fits.Header()

        hwp_angle = self.hwp_angle - self.telescope_vpa
        hwp_init = header.get('HWPINIT', -9999) * degree
        hwp_zero = header.get('HWPSTART', -9999) * degree
        hwp_tolerance = self.configuration.get_float(
            'polarization.hwp_tolerance', 5.0) * degree
        zero_option = self.configuration.get_string(
            'polarization.hwp_zero', 'actual').lower().strip()
        grid_angle = self.configuration.get_float(
            'polarization.gridangle', default=0.0) * degree

        mean_hwp = np.nanmean(hwp_init)
        mean_zero = np.nanmean(hwp_zero)

        if telvpa_from_header and 'TELVPA' in header:
            vpa = header.get('TELVPA') * degree
        else:
            vpa = self.instrument_vpa

        if zero_option == 'actual':
            log.warning("HWP method: actual")
            hwp_zero = offset
        elif zero_option == 'model':
            log.warning("HWP method: model")
            hwp_zero = hwp_angle
        elif zero_option == 'commanded':
            log.warning("HWP method: commanded")
            if hwp_zero == -9999 * degree:
                log.error("HWPSTART not in header.  Cannot apply commanded "
                          "HWP ZERO option")
                hwp_zero = offset
            else:
                if hwp_init == -9999 * degree:
                    hwp_init = np.nanmean(hwp_angle)
                if hwp_zero == -9999 * degree:
                    hwp_zero = hwp_init
                if hwp_init > half:
                    hwp_init -= full
                if hwp_zero > half:
                    hwp_init -= full
                if abs(hwp_zero - hwp_init) > hwp_tolerance:
                    msg = (f'Initial HWP angle is above the tolerance of '
                           f'{hwp_tolerance}.  HWP zero: {mean_zero}, '
                           f'HWP init: {mean_hwp}.')
                    log.warning(msg)
                hwp_zero = hwp_init
        else:
            raise ValueError(f"Invalid HWP zero option: {zero_option}")

        log.info(f"Determining polarization angle using offset={offset} "
                 f"hwp_zero={hwp_zero} grid_angle={grid_angle}")

        horizontal = 2 * (hwp_zero - offset)
        return vpa, horizontal, grid_angle

    # For testing...
    # def validate_polarimetry(self):
    #     """
    #     Populate the polarimetry values in the frame data.
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     degree = units.Unit('degree')
    #     vpa, horizontal, grid_angle = self.get_polarization_theta()
    #     eta_qh, eta_uh = self.get_instrumental_polarization()
    #
    #     # Fixed
    #     horizontal_polarization = False
    #     apply_analyzer_position = True
    #     analyzer_difference_angle = -45 * degree
    #     incidence_phase = 90 * degree
    #     counter_rotating = False
    #
    #     # Logic
    #     cos_i = np.cos(grid_angle).decompose().value
    #     if counter_rotating:
    #         horizontal *= -1
    #
    #     v_plate_angle = grid_angle + horizontal
    #
    #     projected = incidence_phase + np.arctan2(
    #         np.sin(v_plate_angle - incidence_phase),
    #         cos_i * np.cos(v_plate_angle - incidence_phase)
    #     )
    #
    #     v_pol_angle = (4 * projected)
    #     if apply_analyzer_position:
    #         v_pol_angle += 2 * analyzer_difference_angle
    #
    #     qh = np.cos(-v_pol_angle)
    #     uh = np.sin(-v_pol_angle)
    #
    #     if isinstance(qh, units.Quantity):
    #         qh = qh.decompose().value
    #         uh = uh.decompose().value
    #
    #     if horizontal_polarization:
    #         self.q = qh
    #         self.u = uh
    #         self.unpolarized_gain = 1.0 + (qh * eta_qh) + (uh * eta_uh)
    #     else:  # This is applying VPA (telescope_vpa)
    #         cos2pa = (self.cos_pa * self.cos_pa) - (self.sin_pa * self.sin_pa)
    #         sin2pa = (2 * self.sin_pa * self.cos_pa)
    #         self.q = (cos2pa * qh) - (sin2pa * uh)
    #         self.u = (sin2pa * qh) + (cos2pa * uh)
    #         eta_q = (cos2pa * eta_qh) - (sin2pa * eta_uh)
    #         eta_u = (sin2pa * eta_qh) + (cos2pa * eta_uh)
    #         self.unpolarized_gain = 1.0 + (self.q * eta_q) + (self.u * eta_u)

    def validate_polarimetry(self):
        """
        Populate the polarimetry values in the frame data.

        Returns
        -------
        None
        """
        degree = units.Unit('degree')
        vpa, horizontal, grid_angle = self.get_polarization_theta()
        eta_qh, eta_uh = self.get_instrumental_polarization()

        # Fixed
        horizontal_polarization = False
        analyzer_difference_angle = -45 * degree
        incidence_phase = 90 * degree

        # Logic
        cos_i = np.cos(grid_angle).decompose().value
        v_plate_angle = grid_angle + horizontal
        projected = incidence_phase + np.arctan2(
            np.sin(v_plate_angle - incidence_phase),
            cos_i * np.cos(v_plate_angle - incidence_phase)
        )
        v_pol_angle = (4 * projected) + (2 * analyzer_difference_angle)
        qh = np.cos(-v_pol_angle)
        uh = np.sin(-v_pol_angle)

        if isinstance(qh, units.Quantity):
            qh = qh.decompose().value
            uh = uh.decompose().value

        if horizontal_polarization:
            self.q = qh
            self.u = uh
            self.unpolarized_gain = 1.0 + (qh * eta_qh) + (uh * eta_uh)
        else:  # This is applying VPA (telescope_vpa)
            cos2pa = (self.cos_pa * self.cos_pa) - (self.sin_pa * self.sin_pa)
            sin2pa = (2 * self.sin_pa * self.cos_pa)
            self.q = (cos2pa * qh) - (sin2pa * uh)
            self.u = (sin2pa * qh) + (cos2pa * uh)
            eta_q = (cos2pa * eta_qh) - (sin2pa * eta_uh)
            eta_u = (sin2pa * eta_qh) + (cos2pa * eta_uh)
            self.unpolarized_gain = 1.0 + (self.q * eta_q) + (self.u * eta_u)

    def get_instrumental_polarization(self):
        """
        Return the instrument polarization.

        Returns
        -------
        q_inst, u_inst : float, float
        """
        method = self.configuration.get_string(
            'polarization.fileip', 'none')

        if method.lower() in ['none', 'uniform']:
            q_inst = self.configuration.get_float('polarization.qinst', 0.0)
            u_inst = self.configuration.get_float('polarization.uinst', 0.0)
            return q_inst, u_inst

        return self.read_instrumental_polarization_file(method)

    def read_instrumental_polarization_file(self, filename):
        """
        Read an instrumental polarization file.

        This will not currently work as the instrumental polarization needs
        to be applied in channel space, not frame space.  This makes things
        fairly complicated.  Uniform instrumental polarization is currently
        applied to the unpolarized frame gain by::

          unpolarized_gain = 1 + (q * qi) + (u * ui)

        where q and u are the Q/U gains, and qi/ui are the respective
        uniform instrumental polarization factors.

        This will need to be changed to unpolarized_gain = 1

        For the N Stokes map only, we will then need to apply the instrumental
        polarization at a few places::

          1) AstroModel2D.sync_integration
          2) AstroIntensityMap.calculate_coupling
          3) AstroModel2D.add_frames_from_integration
          4) AstroModel2D.sync_pixels

        However, this needs to be applied as a cross-product of the frame
        gains with the channel gains.  The correction for instrumental
        polarization will need to add (q * qi) + (u * uh) to the input gains
        in the above methods where q/u are in frame space, and qi/ui are in
        channel space.

        There is not a pretty way to do this, so it might be worth considering
        creating a child class of AstroIntensityMap for the Stokes parameter
        maps to keep these changes away from the main classes.

        Parameters
        ----------
        filename : str

        Returns
        -------
        q_inst, u_inst : (float, float) or (np.ndarray, np.ndarray)
        """
        raise NotImplementedError("This needs to be built somewhere else")
        # if os.path.isfile(filename):
        #     ip_file = filename
        # else:
        #     filenames = self.configuration.find_configuration_files(filename)
        #     if len(filenames) == 0:
        #         log.warning(f"Instrumental polarization file {filename} not "
        #                     f"found.  Will not correct for instrumental "
        #                     f"polarization.")
        #         return 0.0, 0.0
        #     # Get the highest_priority
        #     ip_file = filenames[-1]
        #
        # df = pd.read_csv(ip_file, names=['ch', 'qi', 'qei', 'ui', 'uie'],
        #                  comment='#', delimiter=r'\s+')

    def dark_correct(self):
        """
        Perform the squid dark correction for blind channels.

        The dark correction is applied for all frames by by subtracting the
        associated dark squid channel data from each non-blind channel over
        all valid frames.

        Returns
        -------
        None
        """
        channels = self.scan.channels.data
        squid_lookup = self.info.detector_array.dark_squid_lookup

        correct_indices = channels.is_unflagged('BLIND', indices=True)
        correct_sub = channels.sub[correct_indices]
        correct_col = channels.col[correct_indices]

        squid_fixed_indices = squid_lookup[correct_sub, correct_col]
        squid_indices = self.channels.find_fixed_indices(
            squid_fixed_indices, cull=False)

        hnf.dark_correct(
            data=self.data,
            valid_frame=self.valid,
            channel_indices=correct_indices,
            squid_indices=squid_indices)

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

        # Note: due to a bug the original added on the smoothed value to the
        # existing values (not replicated here)
        self.hwp_angle = low_res_hwp

    def get_source_gain(self, mode_flag):
        """
        Return the source gain.

        The basic frame class will only return a result for the TOTAL_POWER
        flag.  Polarimetry flags should derive from the PolarModulation class.

        Parameters
        ----------
        mode_flag : FrameFlagTypes or str or int or enum.Enum
            The gain mode flag type.

        Returns
        -------
        gain : numpy.ndarray (float)
            The source gains.
        """
        gain = super().get_source_gain(self.flagspace.flags.TOTAL_POWER)

        n_gain = HawcPlusPolarModulation.get_n_gain()
        qu_gain = HawcPlusPolarModulation.get_qu_gain()

        if mode_flag == HawcPlusPolarModulation.N:
            return self.unpolarized_gain * gain * n_gain
        elif mode_flag == HawcPlusPolarModulation.Q:
            return self.q * gain * qu_gain
        elif mode_flag == HawcPlusPolarModulation.U:
            return self.u * gain * qu_gain
        else:
            flag = self.flagspace.convert_flag(mode_flag)
            if flag == self.flagspace.flags.TOTAL_POWER:
                return gain
            else:  # pragma: no cover
                # maybe for future development
                return super().get_source_gain(mode_flag)
