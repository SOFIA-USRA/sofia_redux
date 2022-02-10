# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
import numpy as np
from astropy import units, log

from sofia_redux.scan.custom.sofia.info import sofia_info_numba_functions \
    as sinf
from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000, Epoch
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.utilities import utils

__all__ = ['SofiaGyroDriftsInfo']


class SofiaGyroDriftsInfo(InfoBase):

    def __init__(self):
        """
        Initialize the GyroDrift object.
        """
        super().__init__()
        self.drifts = None

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        The log ID is used to extract certain information from table data.

        Returns
        -------
        str
        """
        return 'gyro'

    @property
    def n_drifts(self):
        """
        Return the number of drifts (utc ranges) available.

        Returns
        -------
        int
        """
        if self.drifts is None:
            return 0
        return len(self.drifts)

    @property
    def lengths(self):
        """
        Return the length of each drift (utc range).

        Returns
        -------
        astropy.units.Quantity (numpy.array)
            An array containing the length of each drift.
        """
        if self.n_drifts == 0:
            return np.empty(0, dtype=np.float64) * units.Unit('deg')
        return np.array(
            [d.length.to(units.Unit('deg')).value for d in self.drifts]
        ) * units.Unit('deg')

    def apply_configuration(self):
        """
        Apply the configuration to the information.

        Returns
        -------
        None
        """
        self.add_drifts()
        super().apply_configuration()

    def add_drifts(self):
        """
        Read the FITS header options and create drifts.

        Returns
        -------
        None
        """
        options = self.options
        if options is None:
            return
        self.drifts = []
        drift_index = 0
        while True:
            drift = GyroDrift(options.header, drift_index)
            if drift.valid:
                self.drifts.append(drift)
                log.debug(f"drift {drift.index}: {drift.length.to('arcsec')}")
            else:
                break
            drift_index += 1

    def get_max(self):
        """
        Return the maximum drift.

        Returns
        -------
        astropy.units.Quantity
            The maximum drift.
        """
        if self.n_drifts == 0:
            return np.nan * units.Unit('deg')
        return np.max(self.lengths)

    def get_rms(self):
        """
        Return the drift RMS.

        Returns
        -------
        astropy.units.Quantity
            The drift RMS.
        """
        if self.n_drifts == 0:
            return np.nan * units.Unit('deg')
        else:
            return np.sqrt(np.sum(self.lengths ** 2) / self.n_drifts)

    def get_drift_utc_ranges(self):
        """
        Return the drift UTC ranges.

        Returns
        -------
        numpy.ndarray (float)
        """
        utc_ranges = np.full((self.n_drifts, 2), np.nan)
        if self.n_drifts == 0:
            return utc_ranges

        for i, drift in enumerate(self.drifts):
            utc_ranges[i] = drift.utc_range.min, drift.utc_range.max
        return utc_ranges

    def get_drift_deltas(self):
        """
        Return the offsets for all drifts.

        Returns
        -------
        astropy.units.Quantity (numpy.ndarray)
        """
        offsets = np.zeros((self.n_drifts, 2)) * units.Unit('arcsec')
        if self.n_drifts == 0:
            return offsets
        for i, drift in enumerate(self.drifts):
            offsets[i, 0] = drift.delta.x
            offsets[i, 1] = drift.delta.y
        return offsets

    def validate_time_range(self, scan):
        """
        Ensure the UTC time ranges for each drift are correct.

        Returns
        -------
        None
        """
        if scan is None or self.n_drifts == 0:
            return
        t = scan.get_first_integration().frames.get_first_frame_value('utc')
        for drift in self.drifts:
            drift.utc_range.min = t
            t = drift.next_utc

    def correct(self, integration):
        """
        Apply the gyro drift corrections to an integration.

        Equatorial offsets are added to the integration frames.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        if self.n_drifts == 0:
            log.warning("Skipping gyro drift correction. No data...")
            return

        config = integration.configuration
        limit = config.get_float('gyrocorrect.max', default=np.nan
                                 ) * units.Unit('arcsec')

        if np.isfinite(limit) and (self.get_max() > limit):
            log.warning("Skipping gyro drift correction. "
                        "Drifts are too large.")
            return

        log.debug("Correcting for gyro drifts.")
        self.validate_time_range(integration.scan)

        drift_correction, extrapolate_frame = sinf.get_drift_corrections(
            frame_utc=integration.frames.utc,
            frame_valid=integration.frames.valid,
            drift_utc_ranges=self.get_drift_utc_ranges(),
            drift_deltas=self.get_drift_deltas().value)

        if extrapolate_frame >= 0:
            log.warning(f"Extrapolated drift correction after "
                        f"frame {extrapolate_frame}")

        offset = Coordinate2D(drift_correction.T, unit='arcsec')
        integration.frames.equatorial.add_offset(offset)
        integration.frames.equatorial_to_horizontal_offset(
            offset, in_place=True)
        integration.frames.horizontal_offset.add(offset)
        integration.frames.horizontal.add_offset(offset)

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
        if name == 'max':
            return self.get_max().to('arcsec')
        elif name == 'rms':
            return self.get_rms().to('arcsec')
        else:
            return None


class GyroDrift(ABC):

    def __init__(self, header, index):
        """
        Initializes a single drift.

        Parameters
        ----------
        header : fits.header.Header
        index : str or int
            The drift index identifier.
        """
        self.index = int(index)
        self.valid = False
        self.delta = Coordinate2D(unit='arcsec')
        self.before = None
        self.after = None
        self.epoch = None
        self.utc_range = Range()
        self.next_utc = None
        self.parse_header(header)

    def __str__(self):
        """
        Return a string representation of the GyroDrift.

        Returns
        -------
        str
        """
        s = ''
        for key, value in self.__dict__.items():
            s += f'{key}: {value}\n'
        return s

    @property
    def length(self):
        """
        Return the length of the drift.

        Returns
        -------
        astropy.units.Quantity
        """
        if self.delta is None:
            return np.nan * units.Unit('arcsec')
        return np.hypot(self.delta.x, self.delta.y).to(units.Unit('deg'))

    def parse_header(self, header):
        """
        Apply the fits options to the drift.

        Parameters
        ----------
        header : fits.header.Header

        Returns
        -------
        None
        """
        drift_values = {}
        for key in ['DBRA', 'DBDEC', 'DARA', 'DADEC']:
            key_index = f'{key}{self.index}'
            if key_index not in header:
                self.valid = False
                return
            drift_values[key] = header[key_index]

        if 'EQUINOX' in header:
            self.epoch = Epoch.get_epoch(header['EQUINOX'])
        else:
            self.epoch = J2000

        self.before = EquatorialCoordinates(epoch=self.epoch)
        self.after = EquatorialCoordinates(epoch=self.epoch)

        self.before.ra = utils.get_hms_time(
            drift_values['DBRA'], angle=True)
        self.before.dec = utils.get_dms_angle(drift_values['DBDEC'])
        self.after.ra = utils.get_hms_time(drift_values['DARA'], angle=True)
        self.after.dec = utils.get_dms_angle(drift_values['DADEC'])

        self.delta = self.after.get_offset_from(self.before)
        if np.isnan(self.delta.length):
            log.warning("Could not parse gyro drift values - will not apply.")
            self.valid = False
            return

        self.utc_range.max = header.get(f'DBTIME{self.index}', np.nan)
        self.next_utc = header.get(f'DATIME{self.index}', np.nan)
        self.valid = True
