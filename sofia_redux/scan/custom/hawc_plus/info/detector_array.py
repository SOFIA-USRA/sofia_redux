# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units, log
import numpy as np
import re

from sofia_redux.scan.custom.sofia.info.detector_array import (
    SofiaDetectorArrayInfo)
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['HawcPlusDetectorArrayInfo']


class HawcPlusDetectorArrayInfo(SofiaDetectorArrayInfo):

    pol_arrays = 2
    pol_subarrays = 2
    subarrays = pol_arrays * pol_subarrays
    subarray_cols = 32
    rows = 41
    subarray_pixels = rows * subarray_cols
    pol_cols = pol_subarrays * subarray_cols
    pol_array_pixels = rows * pol_cols
    pixels = pol_arrays * pol_array_pixels

    DARK_SQUID_ROW = rows - 1
    MCE_BIAS_LINES = 20
    FITS_ROWS = 41
    FITS_COLS = 128
    FITS_CHANNELS = FITS_ROWS * FITS_COLS
    JUMP_RANGE = 1 << 7

    R0 = 0
    R1 = 1
    T0 = 2
    T1 = 3
    R_ARRAY = 0
    T_ARRAY = 1
    POL_ID = ("R", "T")

    hwp_step = 0.25 * units.Unit('degree')
    default_boresight_index = Coordinate2D([33.5, 19.5], unit=None)

    def __init__(self):
        super().__init__()
        self.dark_squid_correction = False
        self.dark_squid_lookup = None
        self.hwp_telescope_vertical = np.nan
        self.subarray_gain_renorm = None
        self.subarrays_requested = ''
        self.hwp_angle = -1

        self.mce_subarray = np.full(self.subarrays, -1)
        self.has_subarray = np.full(self.subarrays, False)
        # offsets in channels following rotation
        self.subarray_offset = Coordinate2D(
            np.full((2, self.subarrays), np.nan))

        self.subarray_orientation = np.full(
            self.subarrays, np.nan) * units.Unit('deg')
        # Relative zoom of the polarization planes
        self.pol_zoom = np.full(self.pol_arrays, np.nan)
        self.pixel_sizes = Coordinate2D(unit='arcsec')

        # Determined from configuration HDU
        self.detector_bias = np.zeros(
            (self.subarrays, self.MCE_BIAS_LINES), dtype=int)

    def apply_configuration(self):
        """
        Apply the configuration to the detector array.

        Returns
        -------
        None
        """
        super().apply_configuration()
        options = self.options
        if options is None:
            return

        self.dark_squid_correction = self.configuration.has_option(
            'darkcorrect')

        mce_map = options.get_string("MCEMAP")
        self.mce_subarray.fill(-1)
        self.has_subarray.fill(False)
        if isinstance(mce_map, str):
            mces = [utils.get_int(x) for x in re.split(r'[\t,:;]', mce_map)]
            nmces = len(mces)

            for sub in range(min([self.subarrays, nmces])):
                mce = mces[sub]
                if mce >= 0:
                    self.has_subarray[sub] = True
                    self.mce_subarray[mce] = sub
                log.debug(f"Sub: {sub}, {self.has_subarray[sub]}")

        self.select_subarrays()
        self.set_hwp_header()

    def set_hwp_header(self):
        """
        Set the HWP angle of the detector array.

        Note that the angle is stored as an integer value indicating the number
        of HWP steps.

        Returns
        -------
        None
        """
        angle = self.configuration.get_int('hwp', default=None)
        if angle is None:
            return
        self.hwp_angle = angle

    def load_detector_configuration(self):
        """
        Apply the configuration to set various parameters for the detector.

        Returns
        -------
        None
        """
        deg = units.Unit('deg')
        self.subarray_orientation.fill(np.nan)
        self.subarray_orientation[self.R0] = self.configuration.get_float(
            'rotation.R0', default=0.0) * deg
        self.subarray_orientation[self.R1] = self.configuration.get_float(
            'rotation.R1', default=180.0) * deg
        self.subarray_orientation[self.T0] = self.configuration.get_float(
            'rotation.T0', default=0.0) * deg
        self.subarray_orientation[self.T1] = self.configuration.get_float(
            'rotation.T1', default=180.0) * deg

        self.subarray_offset[self.R0].set(
            self.configuration.get_float_list(
                'offset.R0', default=[np.nan, np.nan]), copy=False)

        self.subarray_offset[self.R1].set(
            self.configuration.get_float_list(
                'offset.R1', default=[67.03, -39.0]), copy=False)

        self.subarray_offset[self.T0].set(
            self.configuration.get_float_list(
                'offset.T0', default=[np.nan, np.nan]), copy=False)

        self.subarray_offset[self.T1].set(
            self.configuration.get_float_list(
                'offset.T1', default=[67.03, -39.0]), copy=False)

        self.pol_zoom.fill(np.nan)
        self.pol_zoom[self.R_ARRAY] = self.configuration.get_float(
            'zoom.R', default=1.0)
        self.pol_zoom[self.T_ARRAY] = self.configuration.get_float(
            'zoom.T', default=1.0)

        pixel_sizes = Coordinate2D(unit='arcsec')
        pixel_sizes.set([self.pixel_size, self.pixel_size])

        if 'pixelsize' in self.configuration:
            config_pixel_sizes = self.configuration.get_float_list(
                'pixelsize', delimiter=r'[ \t,:xX]',
                default=[self.pixel_size.value])

            if len(config_pixel_sizes) >= 1:
                pixel_sizes.x = config_pixel_sizes[0]
            if len(config_pixel_sizes) >= 2:
                pixel_sizes.y = config_pixel_sizes[0]
            else:
                pixel_sizes.y = pixel_sizes.x

        self.pixel_size = np.sqrt(pixel_sizes.x * pixel_sizes.y)
        self.pixel_sizes = pixel_sizes

    def set_boresight(self):
        """
        Set the boresight index of the detector array.

        Returns
        -------
        None
        """
        log.info(f"Boresight pixel from FITS is {self.boresight_index}")

        if 'pcenter' in self.configuration:
            boresight_override = self.configuration.get_float_list(
                'pcenter', default=[])
            if len(boresight_override) == 1:
                self.boresight_index.x = boresight_override[0]
                self.boresight_index.y = self.boresight_index.x
            elif len(boresight_override) == 2:
                self.boresight_index.set(boresight_override)
            else:
                raise ValueError(
                    f"Boresight override in configuration is wrong length "
                    f"({len(boresight_override)})")
            log.info(f"Boresight override --> {self.boresight_index}")
        elif self.boresight_index.is_nan():
            self.boresight_index = self.default_boresight_index.copy()
            log.warning(f"Missing FITS boresight --> {self.boresight_index}")

    def select_subarrays(self, specification=None):
        """
        Select the detector subarrays to be included in the detector array.

        Parameters
        ----------
        specification : str, optional
            A string specifying which subarrays to select.  If not supplied,
            will be extracted from the 'subarray' setting in the configuration.

        Returns
        -------
        None
        """
        if specification is None:
            specification = self.configuration.get_string('subarray')
        if specification is None:
            return

        subarrays = re.split(r'[\[\]\'\",; \t]', specification)
        subarrays = [x.upper().strip() for x in subarrays if x != '']
        if len(subarrays) == 0:
            return

        old_has_subarray = self.has_subarray.copy()
        self.has_subarray = np.full(self.subarrays, False)
        requested_subarrays = []

        for subarray in subarrays:
            pol = subarray[0]
            sub = int(subarray[1:]) if len(subarray) > 1 else None

            if pol == 'R':
                pol_array = self.R0
                requested_subarrays.append('R0')

            elif pol == 'T':
                pol_array = self.T0
                requested_subarrays.append('T0')

            else:
                pol_array = -1

            if pol_array < 0:
                log.warning(f"Invalid subarray selection: {subarray}")
                continue

            if sub is None:
                index = slice(pol_array, pol_array + self.pol_subarrays)
            else:
                index = pol_array + sub

            self.has_subarray[index] = old_has_subarray[index]

        self.subarrays_requested = ', '.join(requested_subarrays)

    def parse_configuration_hdu(self, hdu):
        """
        Parse the data from a configuration HDU and apply to the header data.

        Parameters
        ----------
        hdu : fits.BinTableHDU

        Returns
        -------
        None
        """
        self.detector_bias.fill(0)
        found = 0
        header = hdu.header
        for sub in range(self.subarrays):
            key = f"MCE{sub}_TES_BIAS"
            bias = header.get(key, None)
            if bias is not None:
                bias = [utils.get_int(x) for x in re.split(r'[\t,:;]', bias)]
                if len(bias) != self.MCE_BIAS_LINES:
                    log.warning(
                        f"Subarray {sub} requires {self.mce_subarray} bias "
                        f"lines (found {len(bias)})")
                    break
                self.detector_bias[sub] = bias
                found += 1
            else:
                if sub != 3:
                    log.warning(f"Missing TES bias values for subarray {sub}")
                break
        log.debug(f"Parsing HAWC+ TES bias. Found for {found} MCEs")

    def get_sibs_position(self, sub, row, col):
        """
        Given a subarray, row, and column, return the pixel position.

        The SIBS position are in tEl, tXel coordinates in units of the
        `pixel_xy_size` attribute.

        Parameters
        ----------
        sub : int or numpy.ndarray (int)
            The detector subarray index.
        row : int or float or numpy.ndarray (int or float)
            The channel/pixel detector row.
        col : int or float or numpy.ndarray (int or float)
            The channel/pixel detector column.

        Returns
        -------
        position : Coordinate2D
            The pixel (x, y) pixel positions.
        """
        position = Coordinate2D()
        position.set([col, 39.0 - row])
        position.rotate(self.subarray_orientation[sub])
        position.add(Coordinate2D(self.subarray_offset[sub]))

        # X is oriented like AZ (tXEL), whereas Y is oriented like -tEL
        position.scale_x(self.pixel_sizes.x)
        position.scale_y(-self.pixel_sizes.y)
        position.scale(self.pol_zoom[sub >> 1])
        return position

    def get_subarray_id(self, subarray):
        """
        Return the subarray string ID.

        Parameters
        ----------
        subarray : int

        Returns
        -------
        str
        """
        return self.POL_ID[subarray // 2] + str(subarray % 2)

    def create_dark_squid_lookup(self, channels):
        """
        Store dark squid pixels (blind channels) in a lookup array.

        The lookup array is of the form lookup[sub, col] = fixed_index.
        Invalid values are marked with values of -1 (good pixels).

        Parameters
        ----------
        channels : HawcPlusChannels

        Returns
        -------
        None
        """
        self.dark_squid_lookup = np.full(
            (self.subarrays, self.subarray_cols), -1)
        blind_idx = channels.data.is_flagged('BLIND', indices=True)

        self.dark_squid_lookup[channels.data.sub[blind_idx],
                               channels.data.col[blind_idx]] = \
            channels.data.fixed_index[blind_idx]

    def initialize_channel_data(self, data):
        """
        Apply this information to create and populate the channel data.

        Parameters
        ----------
        data : HawcPlusChannelData

        Returns
        -------
        None
        """
        index = np.arange(self.pixels)
        sub = index // self.subarray_pixels
        keep = self.has_subarray[sub]
        index = index[keep]

        data.fixed_index = index
        data.set_default_values()

        data.col = index % self.subarray_cols
        data.row = index // self.subarray_cols
        data.sub = index // self.subarray_pixels
        data.pol = data.sub >> 1

        data.fits_row = data.subrow = data.row % self.rows
        data.bias_line = np.right_shift(data.row, 1)

        data.fits_col = data.mux = (data.sub * self.subarray_cols) + data.col
        data.series_array = np.right_shift(data.mux, 2)

        data.fits_index = (data.fits_row * self.FITS_COLS) + data.fits_col

        data.flag = np.zeros(data.size, dtype=int)
        blind_flag = data.flagspace.flags.BLIND.value
        data.flag[data.subrow == self.DARK_SQUID_ROW] = blind_flag

        data.channel_id = np.array(
            [f'{self.POL_ID[x[0]]}{x[1] & 1}[{x[2]},{x[3]}]'
             for x in zip(data.pol, data.sub, data.subrow, data.col)])
