# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.scan.custom.hawc_plus.channels.gain_provider.pol_imbalance \
    import HawcPlusPolImbalance
from sofia_redux.scan.custom.hawc_plus.channels.mode.los_response import (
    LosResponse)
from sofia_redux.scan.custom.hawc_plus.channels.mode.roll_response import (
    RollResponse)
from sofia_redux.scan.custom.sofia.channels.camera import SofiaCamera
from sofia_redux.scan.channels.modality.coupled_modality import (
    CoupledModality)
from sofia_redux.scan.channels.modality.correlated_modality import (
    CorrelatedModality)
from sofia_redux.scan.channels.modality.modality import Modality

__all__ = ['HawcPlusChannels']


class HawcPlusChannels(SofiaCamera):

    def __init__(self, parent=None, info=None, size=0, name='hawc_plus'):
        super().__init__(name=name, parent=parent, info=info, size=size)
        self.subarray_groups = None
        self.subarray_gain_renorm = None

    @property
    def detector(self):
        """
        Return the detector info.

        Returns
        -------
        HawcPlusDetectorArrayInfo
        """
        return self.info.detector_array

    @property
    def band_id(self):
        """
        Return the HAWC_PLUS Band

        Returns
        -------
        band : str
        """
        return self.info.instrument.band_id

    @property
    def pixel_sizes(self):
        """
        Return the (x,y) pixel size.

        Returns
        -------
        numpy.ndarray (astropy.units.Quantity)
            An array of size 2 containing the x, y pixel size in arc seconds.
        """
        return self.info.detector_array.pixel_sizes

    @property
    def dark_squid_lookup(self):
        """
        Return the dark squid lookup array.

        The lookup array is of the form lookup[sub, col] = fixed_index.
        Invalid values are marked with values of -1 (good pixels).

        Returns
        -------
        lookup : numpy.ndarray (int)
        """
        return self.info.detector_array.dark_squid_lookup

    def init_divisions(self):
        """
        Initializes channel divisions.

        Divisions contain sets of channel groups.

        The HAWC_PLUS channel adds divisions consisting of groups where
        each contains a unique value of a certain data field.  For example,
        the "rows" division contains a group for row 1, a group for row 2, etc.

        Returns
        -------
        None
        """
        super().init_divisions()
        dead_blind = self.flagspace.flags.DEAD | self.flagspace.flags.BLIND

        for division_name, field in [
            ('polarrays', 'pol'),
            ('subarrays', 'sub'),
            ('bias', 'bias_line'),
            ('series', 'series_array')
        ]:
            self.add_division(self.get_division(
                name=division_name, field=field, discard_flag=dead_blind))

        if self.configuration.has_option('darkcorrect'):
            bad_mux_flag = self.flagspace.flags.DEAD
        else:
            bad_mux_flag = dead_blind

        mux_division = self.get_division(name='mux', field='mux',
                                         discard_flag=bad_mux_flag)

        # I don't know why, but order channels in pin (row) order.
        for group in mux_division.groups:
            group.indices = group.indices[np.argsort(group.row)]

        self.add_division(mux_division)
        self.add_division(self.get_division(
            name='rows', field='row', discard_flag=bad_mux_flag))

    def init_groups(self):
        """
        Initializes channel groups.

        Each group contains a subset of the channel data, referenced by index.

        The HAWC_PLUS groups contain additional groups based on sub-array.

        Returns
        -------
        None
        """
        super().init_groups()
        self.subarray_groups = [None] * self.info.detector_array.subarrays
        sub_index = 0
        for pol_array in range(self.info.detector_array.pol_arrays):
            for pol_sub_array in range(self.info.detector_array.pol_subarrays):
                indices = np.nonzero(self.data.sub == sub_index)[0]
                pol_id = self.info.detector_array.POL_ID[pol_array]
                group = self.create_channel_group(
                    indices=indices,
                    name=f'{pol_id}{pol_sub_array}')
                self.add_group(group)
                self.subarray_groups[sub_index] = group
                sub_index += 1

    def init_modalities(self):
        """
        Initializes channel modalities.

        A modality is based of a channel division and contains a mode for each
        channel group in the channel division.

        The HAWC_PLUS modalities simply contain additional correlated modes
        based on the additional channel fields.  A new coupled modality
        is also created according to polarization arrays.

        Returns
        -------
        None
        """
        super().init_modalities()

        obs_modality = self.modalities.get('obs-channels')
        if obs_modality is not None:
            self.add_modality(
                CoupledModality(modality=obs_modality,
                                name='polarrays',
                                identity='p',
                                gain_provider=HawcPlusPolImbalance()))

        flags = self.flagspace.flags
        builds = [('subarrays', 'S', 'subarrays', 'sub_gain', flags.SUB),
                  ('bias', 'b', 'bias', 'bias_gain', flags.BIAS),
                  ('series', 's', 'series', 'series_gain', flags.SERIES_ARRAY),
                  ('mux', 'm', 'mux', 'mux_gain', flags.MUX),
                  ('rows', 'r', 'rows', 'pin_gain', flags.ROW)]

        for name, identity, division_name, gain_field, gain_flag in builds:
            division = self.divisions.get(division_name)
            if division is None:
                log.warning(f"Channel division {division_name} not found.")
                continue
            modality = CorrelatedModality(name=name,
                                          identity=identity,
                                          channel_division=division,
                                          gain_provider=gain_field)
            modality.set_gain_flag(gain_flag)
            self.add_modality(modality)

        detector_division = self.divisions.get('detectors')

        los_response = Modality(name='los', identity='L',
                                channel_division=detector_division,
                                gain_provider='los_gain',
                                mode_class=LosResponse)
        los_response.set_gain_flag(flags.LOS_RESPONSE)
        self.add_modality(los_response)

        roll_response = Modality(name='roll', identity='R',
                                 channel_division=detector_division,
                                 gain_provider='roll_gain',
                                 mode_class=RollResponse)
        roll_response.set_gain_flag(flags.ROLL_RESPONSE)
        self.add_modality(roll_response)

    def load_channel_data(self):
        """
        Load the channel data.

        The pixel data and wiring data files should be defined in the
        configuration.

        Returns
        -------
        None
        """
        self.detector.load_detector_configuration()
        self.detector.initialize_channel_data(self.data)

        self.set_nominal_pixel_positions(self.info.detector_array.pixel_sizes)
        super().load_channel_data()
        if self.configuration.has_option('jumpdata'):
            self.read_jump_levels(self.configuration.get_filepath('jumpdata'))

    def read_jump_levels(self, filename):
        """
        Read the jump levels from a data file.

        Parameters
        ----------
        filename : str

        Returns
        -------
        None
        """
        if filename is None:
            return
        log.info(f"Loading jump levels from {filename}")
        hdul = fits.open(filename)
        self.data.read_jump_hdu(hdul[0])
        self.info.register_config_file(filename)
        hdul.close()

    def normalize_array_gains(self):
        """
        Normalize the relative channel gains in observing channels.

        Returns
        -------
        average_gain : float
            The average gain prior to normalization.
        """
        log.debug("Normalizing subarray gains.")
        sub_arrays = CorrelatedModality(
            name='subs', identity='S',
            channel_division=self.divisions.get('subarrays'),
            gain_provider='gain')
        self.subarray_gain_renorm = np.full(
            self.info.detector_array.subarrays, np.nan)
        for mode in sub_arrays.modes:
            sub_value = mode.channel_group.sub[0]
            self.subarray_gain_renorm[sub_value] = mode.normalize_gains()
            sub_id = self.info.detector_array.get_subarray_id(sub_value)
            sub_gain = self.subarray_gain_renorm[sub_value]
            log.debug(f"--> {sub_id} gain = {sub_gain:.3f}")
        return 1.0

    def set_nominal_pixel_positions(self, pixel_sizes):
        """
        Set the nominal pixel positions for the given pixel sizes.

        Parameters
        ----------
        pixel_sizes : Coordinate2D

        Returns
        -------
        None
        """
        self.detector.pixel_sizes = pixel_sizes
        self.detector.set_boresight()
        self.data.calculate_sibs_position()

        center = self.detector.get_sibs_position(
            sub=0,
            row=39 - self.detector.boresight_index.y,
            col=self.detector.boresight_index.x)
        self.set_reference_position(center)  # subtracts the center position.

    def max_pixels(self):
        """
        Return the maximum pixels in the detector array.

        Returns
        -------
        count : int
        """
        return self.n_store_channels

    def read_data(self, hdul):
        """
        Read a FITS HDU list to populate channel data.

        Parameters
        ----------
        hdul : fits.HDUList

        Returns
        -------
        None
        """
        for hdu in hdul:
            if (hdu.header.get('EXTNAME', '').strip().upper()
                    == 'CONFIGURATION'):
                self.info.detector_array.parse_configuration_hdu(hdu)

    def validate_scan(self, scan):
        """
        Validate the channels with a scan.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        pol_mask = 0
        for i in range(self.detector.subarrays):
            pol_mask |= (i & 2) + 1

        self.detector.dark_squid_correction = self.configuration.has_option(
            'darkcorrect')

        if pol_mask != 3:
            self.configuration.blacklist('correlated.polarrays')

        self.detector.initialize_channel_data(self.data)

        if not self.configuration.has_option('filter'):
            wavelength = self.info.instrument.wavelenth.to('um').value
            if (wavelength % 1) == 0:
                wavelength = int(wavelength)
            self.configuration.set_option('filter', f'{wavelength}um')
        log.info(f"HAWC+ Filter set to "
                 f"{self.configuration.get_string('filter')}")

        super().validate_scan(scan)
        self.create_dark_squid_lookup()

    def slim(self, reindex=True):
        """
        Remove all DEAD or DISCARD flagged channels.

        Will also update channel groups, divisions, and modalities.

        Parameters
        ----------
        reindex : bool, optional
            If `True`, reindex channels if slimmed.

        Returns
        -------
        slimmed : bool
            `True` if channels were discarded, `False` otherwise.
        """
        slimmed = super().slim(reindex=reindex)
        if slimmed:
            self.create_dark_squid_lookup()
        return slimmed

    def create_dark_squid_lookup(self):
        """
        Create the dark squid lookup array.

        The dark squid lookup array is stored in the info.detector_array
        object.

        Returns
        -------
        None
        """
        self.info.detector_array.create_dark_squid_lookup(self)

    def get_si_pixel_size(self):
        """
        Return the science instrument pixel size

        Returns
        -------
        x, y : Coordinate2D
            The (x, y) pixel sizes
        """
        return self.detector.pixel_sizes

    def write_flat_field(self, filename, include_nonlinear=False):
        """
        Write a flat field file used for chop-nod pipelines.

        Parameters
        ----------
        filename : str
            The filename to write to.
        include_nonlinear : bool, optional
            If `True`, include the nonlinear responses.

        Returns
        -------
        None
        """
        shape = self.detector.rows, self.detector.pol_cols

        # Set defaults
        gain_r = np.ones(shape, dtype=float)
        gain_t = np.ones(shape, dtype=float)
        flag_r = np.full(shape, 1, dtype=int)
        flag_t = np.full(shape, 2, dtype=int)
        if include_nonlinear:
            nonlinear_r = np.zeros(shape, dtype=float)
            nonlinear_t = np.zeros(shape, dtype=float)
        else:
            nonlinear_r = nonlinear_t = None

        data = self.data
        detector = self.detector
        flagged = self.data.is_flagged()

        gains = self.subarray_gain_renorm[data.sub] * data.gain * data.coupling
        inverse_gains = np.zeros_like(gains)
        nzi = gains != 0
        inverse_gains[nzi] = 1 / gains[nzi]
        all_cols = ((data.sub & 1) * detector.subarray_cols) + data.col

        for array in [detector.R_ARRAY, detector.T_ARRAY]:
            select = np.nonzero(data.pol == array)[0]
            if select.size == 0:
                continue

            if array == detector.R_ARRAY:
                gains, flags, nonlinear = gain_r, flag_r, nonlinear_r
            else:
                gains, flags, nonlinear = gain_t, flag_t, nonlinear_t

            rows = data.subrow[select]
            cols = all_cols[select]
            gains[rows, cols] = inverse_gains[select]
            if include_nonlinear:
                nonlinear[rows, cols] = data.nonlinearity[select]

            flags[rows, cols] *= flagged[select]

        hdul = fits.HDUList()
        hdul.append(fits.ImageHDU(gain_r, name='R array gain'))
        hdul.append(fits.ImageHDU(gain_t, name='T array gain'))
        hdul.append(fits.ImageHDU(flag_r, name='R bad pixel mask'))
        hdul.append(fits.ImageHDU(flag_t, name='T bad pixel mask'))
        if include_nonlinear:
            hdul.append(fits.ImageHDU(nonlinear_r,
                                      name='R array nonlinearity'))
            hdul.append(fits.ImageHDU(nonlinear_t,
                                      name='T array nonlinearity'))
        hdul.writeto(filename, overwrite=True)
        hdul.close()

        log.info(f"Written flat field to {filename}.")

    def add_hdu(self, hdul, hdu, extname):
        """
        Add a FITS HDU to the HDUList.

        Parameters
        ----------
        hdul : fits.HDUList
            The HDUList to append to.
        hdu : fits.ImageHDU or fits.PrimaryHDU or fits.BinTableHDU
            The fits HDU to append.
        extname : str
            The name of the HDU extension.

        Returns
        -------
        None
        """
        hdu.header['EXTNAME'] = extname, 'image content ID'
        self.info.edit_header(hdu.header)
        hdul.append(hdu)

    def get_table_entry(self, name):
        """
        Return a channel parameter for the given name.

        Parameters
        ----------
        name : str

        Returns
        -------
        value
        """
        if name == 'band':
            return self.band_id
        else:
            return super().get_table_entry(name)
