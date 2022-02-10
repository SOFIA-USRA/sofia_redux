# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import units, log
from astropy.coordinates import Angle
import os
import numpy as np

from sofia_redux.scan.info.weather_info import WeatherInfo
from sofia_redux.scan.custom.sofia.info.instrument import SofiaInstrumentInfo
from sofia_redux.scan.custom.sofia.info.astrometry import SofiaAstrometryInfo
from sofia_redux.scan.custom.sofia.info.aircraft import SofiaAircraftInfo
from sofia_redux.scan.custom.sofia.info.chopping import SofiaChoppingInfo
from sofia_redux.scan.custom.sofia.info.detector_array import (
    SofiaDetectorArrayInfo)
from sofia_redux.scan.custom.sofia.info.dithering import SofiaDitheringInfo
from sofia_redux.scan.custom.sofia.info.environment import SofiaEnvironmentInfo
from sofia_redux.scan.custom.sofia.info.mapping import SofiaMappingInfo
from sofia_redux.scan.custom.sofia.info.mission import SofiaMissionInfo
from sofia_redux.scan.custom.sofia.info.mode import SofiaModeInfo
from sofia_redux.scan.custom.sofia.info.nodding import SofiaNoddingInfo
from sofia_redux.scan.custom.sofia.info.observation import SofiaObservationInfo
from sofia_redux.scan.custom.sofia.info.origination import SofiaOriginationInfo
from sofia_redux.scan.custom.sofia.info.processing import SofiaProcessingInfo
from sofia_redux.scan.custom.sofia.info.scanning import SofiaScanningInfo
from sofia_redux.scan.custom.sofia.info.spectroscopy import (
    SofiaSpectroscopyInfo)
from sofia_redux.scan.custom.sofia.info.telescope import SofiaTelescopeInfo
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.info.camera.info import CameraInfo

__all__ = ['SofiaInfo']


class SofiaInfo(WeatherInfo, CameraInfo):

    def __init__(self, configuration_path=None):
        """
        Initialize a SofiaInfo object.

        Parameters
        ----------
        configuration_path : str, optional
            An alternate directory path to the configuration tree to be
            used during the reduction.  The default is
            <package>/data/configurations.
        """
        super().__init__(configuration_path=configuration_path)
        self.name = 'sofia'
        self.history = []
        self.configuration_files = set()
        self.instrument = SofiaInstrumentInfo()
        self.astrometry = SofiaAstrometryInfo()
        self.aircraft = SofiaAircraftInfo()
        self.chopping = SofiaChoppingInfo()
        self.detector_array = SofiaDetectorArrayInfo()
        self.dithering = SofiaDitheringInfo()
        self.environment = SofiaEnvironmentInfo()
        self.mapping = SofiaMappingInfo()
        self.mission = SofiaMissionInfo()
        self.mode = SofiaModeInfo()
        self.nodding = SofiaNoddingInfo()
        self.observation = SofiaObservationInfo()
        self.origin = SofiaOriginationInfo()
        self.processing = SofiaProcessingInfo()
        self.scanning = SofiaScanningInfo()
        self.spectroscopy = SofiaSpectroscopyInfo()
        self.telescope = SofiaTelescopeInfo()

    def register_config_file(self, filename):
        """
        Register a configuration file in the history and for reference.

        Parameters
        ----------
        filename : str

        Returns
        -------
        None
        """
        super().register_config_file(filename)
        if filename is None:
            return
        self.configuration_files.add(filename)
        self.append_history_message(f'AUX: {filename}')

    def read_configuration(self, configuration_file='default.cfg',
                           validate=True):
        """
        Read and apply a configuration file.

        Parameters
        ----------
        configuration_file : str, optional
            Path to, or name of, a configuration file.
        validate : bool, optional
            If `True` (default), validate information read from the
            configuration file.

        Returns
        -------
        None
        """
        super().read_configuration(configuration_file=configuration_file,
                                   validate=validate)
        config_files = self.configuration.config_files
        if config_files is None:
            return
        config_files = list(np.unique(config_files))
        for config_file in config_files:
            self.append_history_message(f'AUX: {config_file}')

    def get_name(self):
        """
        Return the name of the information.

        Returns
        -------
        name : str
        """
        if self.instrument is None or self.instrument.name is None:
            return super().get_name()
        return self.instrument.name

    def apply_configuration(self):
        """
        Apply a configuration to the information.

        Returns
        -------
        None
        """
        super().apply_configuration()
        log.info(f"[{self.observation.source_name}] "
                 f"of AOR {self.observation.aor_id}")
        log.info(f"Observed on {self.astrometry.date} "
                 f"at {self.astrometry.start_time} "
                 f"by {self.origin.observer}")

        if self.astrometry.equatorial is not None:
            log.info(f"Equatorial: {self.astrometry.equatorial}")

        if self.telescope.boresight_equatorial is not None:
            log.info(f"Boresight: {self.telescope.boresight_equatorial}")

        if self.astrometry.requested_equatorial is not None:
            log.info(f"Requested: {self.astrometry.requested_equatorial}")

        kft = (self.aircraft.altitude.midpoint.to(self.aircraft.kft)).value
        temp = self.environment.ambient_t.to(
            units.K, equivalencies=units.temperature()).value
        log.info(f"Altitude: {kft:.2f} kft, Tamb: {temp:.3f} K")

        if self.telescope.focus_t is not None:
            log.info(f"Focus: {self.telescope.focus_t}")

        hwp = self.configuration.get_float('hwp', default=np.nan)
        if not np.isnan(hwp):
            hwp_header = hwp, 'Actual value of the initial HWP angle (degree)'
            self.configuration.fits.preserved_cards['HWPINIT'] = hwp_header
            self.configuration.fits.header['HWPINIT'] = hwp_header
            self.configuration.fits.reread()
            self.configuration.merge_fits_options()

        self.parse_history(self.configuration.fits.header)

    def append_history_message(self, message):
        """
        Add a FITS history message for later addition to a FITS header.

        Parameters
        ----------
        message : str
            The history message to add.

        Returns
        -------
        None
        """
        if message is None:
            return
        if self.history is None:
            self.history = []
        if isinstance(message, str):
            if message in self.history:
                return
            self.history.append(message)
        elif isinstance(message, list):
            for msg in message:
                self.append_history_message(msg)

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
        if scans is None:
            return

        mjds = [scan.mjd for scan in scans]
        first_scan = scans[np.argmin(mjds)]
        last_scan = scans[np.argmax(mjds)]

        aors = [scan.info.observation.aor_id for scan in scans]
        aors = [aor for aor in aors if aor is not None]
        mission_ids = [scan.info.mission.mission_id for scan in scans]
        mission_ids = [mid for mid in mission_ids if mid is not None]
        freqs = [scan.info.instrument.frequency.decompose().value
                 for scan in scans]

        # SOFIA date and time keys
        header['DATE-OBS'] = first_scan.info.astrometry.time_stamp
        header.comments['DATE-OBS'] = 'Start of observation'

        utc_range = [first_scan.info.astrometry.utc.start,
                     last_scan.info.astrometry.utc.end]
        for i, utc in enumerate(utc_range):
            if isinstance(utc, units.Quantity) and utc.unit == 'hour':
                utc_range[i] = utc.value * units.Unit('hourangle')

        utc_str = ['00:00:NaN'] * 2
        if not np.isnan(utc_range[0]):
            utc_str[0] = Angle(utc_range[0]).to_string(
                sep=':', pad=True, precision=3)
        if not np.isnan(utc_range[1]):
            utc_str[1] = Angle(utc_range[1]).to_string(
                sep=':', pad=True, precision=3)

        header['UTCSTART'] = utc_str[0], 'UTC start of first scan'
        header['UTCEND'] = utc_str[1], 'UTC end of last scan'

        # SOFIA observation keys
        first_scan.info.observation.edit_header(header)

        # SOFIA mission keys
        first_scan.info.mission.edit_header(header)

        # SOFIA origination keys
        origin = first_scan.info.origin.copy()
        if self.configuration.has_option('organization'):
            origin.organization = self.configuration.get_string('organization')
        origin.creator = 'sofscan'
        origin.filename = None  # FILENAME fills automatically at writing.
        origin.edit_image_header(header)

        # SOFIA environmental keys
        environment = first_scan.info.environment.copy()
        environment.merge(last_scan.info.environment)
        environment.edit_header(header)

        # SOFIA aircraft keys
        aircraft = first_scan.info.aircraft.copy()
        aircraft.merge(last_scan.info.aircraft)
        aircraft.edit_header(header)

        # SOFIA telescope keys
        telescope = first_scan.info.telescope.copy()
        telescope.merge(last_scan.info.telescope)
        telescope.edit_header(header)

        # SOFIA INSTRUMENT keys
        self.instrument.exposure_time = self.get_total_exposure_time(
            scans=scans)

        # SOFIA array keys
        if self.detector_array is not None and len(scans) == 1:
            self.detector_array.boresight_index = (
                first_scan.info.detector_array.boresight_index)

        self.edit_header(header)

        # SOFIA collection keys
        first_scan.info.mode.edit_header(header)
        if first_scan.info.mode.is_chopping:
            first_scan.info.chopping.edit_header(header)
        if first_scan.info.mode.is_nodding:
            first_scan.info.nodding.edit_header(header)
        if first_scan.info.mode.is_dithering:
            dither = first_scan.info.dithering.copy()
            if len(scans) > 1:
                dither.index = utils.UNKNOWN_INT_VALUE
            dither.edit_header(header)

        if first_scan.info.mode.is_mapping:
            first_scan.info.mapping.edit_header(header)

        if first_scan.info.mode.is_scanning:
            scanning = first_scan.info.scanning.copy()
            scanning.merge(last_scan.info.scanning)
            scanning.edit_header(header)

        # SOFIA data processing keys
        processing = self.processing.get_processing(
            is_calibrated=self.configuration.has_option('calibrated'),
            dims=header.get('NAXIS', 0),
            quality_level=self.get_lowest_quality(scans))
        processing.associated_aors = aors
        processing.associated_mission_ids = mission_ids
        processing.associated_frequencies = freqs
        processing.edit_header(header)

        first_scan.info.configuration.add_preserved_header_keys(header)

    @staticmethod
    def has_tracking_error(scans):
        """
        Report whether any scan in a set contains a telescope tracking error.

        Parameters
        ----------
        scans : list (SofiaScan)
            A list of scans.

        Returns
        -------
        tracking_error : bool
            `True` if any scan contains a telescope tracking error.  `False`
            otherwise.
        """
        if scans is None:
            return False
        for scan in scans:
            if scan.info.telescope.has_tracking_error:
                return True
        else:
            return False

    def edit_header(self, header):
        """
        Edit a scan header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.

        Returns
        -------
        None
        """
        self.observation.edit_header(header)
        self.mission.edit_header(header)
        self.origin.edit_header(header)
        self.environment.edit_header(header)
        self.aircraft.edit_header(header)
        self.telescope.edit_header(header)
        self.instrument.edit_header(header)
        self.mode.edit_header(header)
        if self.chopping is not None:
            self.chopping.edit_header(header)
        if self.nodding is not None:
            self.nodding.edit_header(header)
        if self.dithering is not None:
            self.dithering.edit_header(header)
        if self.mapping is not None:
            self.mapping.edit_header(header)
        if self.scanning is not None:
            self.scanning.edit_header(header)

        self.processing.edit_header(header)

    @staticmethod
    def get_total_exposure_time(scans=None):
        """
        Return the total integration time for a list of scans.

        Parameters
        ----------
        scans : list (Scan), optional
            A list of scans from which to get the total integration time.

        Returns
        -------
        time : astropy.units.Quantity
        """
        time = 0.0 * units.Unit('second')
        if scans is None:
            return time
        for scan in scans:
            time += scan.info.instrument.exposure_time
        return time

    @staticmethod
    def get_lowest_quality(scans):
        """
        Return the lowest quality processing flag from a list of scans.

        Parameters
        ----------
        scans : list (Scan)
            A list of scans from which to determine the lowest quality
            processing level.

        Returns
        -------
        QualityFlagTypes
        """
        flag_values = [scan.info.processing.quality_level.value
                       for scan in scans]
        min_index = np.argmin(flag_values)
        return scans[min_index].info.processing.quality_level

    def add_history(self, header, scans=None):
        """
        Add HISTORY messages to a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The header to update with HISTORY messages.
        scans : list (Scan), optional
            A list of scans to add HISTORY messages from if necessary.

        Returns
        -------
        None
        """
        super().add_history(header, scans=scans)
        self.validate_configuration_registration()
        self.append_history_message(f'PWD: {os.getcwd()}')
        # Add obs-ID for all input scans
        if scans is not None:
            if not isinstance(scans, list):
                scans = [scans]
            for i, scan in enumerate(scans):
                self.append_history_message(
                    f' OBS-ID[{i + 1}]: {scan.get_id()}')

        if self.history is not None:
            for message in self.history:
                header['HISTORY'] = f' {message}'

    def parse_history(self, header):
        """
        Parse all history messages in the header to the scan info.

        Parameters
        ----------
        header : astropy.io.fits.Header

        Returns
        -------
        None
        """
        self.history = []
        if 'HISTORY' in header:
            self.history = list(header['HISTORY'])

        if len(self.history) > 0:
            log.debug(f"Processing History: "
                      f"{len(self.history)} entries found.")

    def get_ambient_kelvins(self):
        """
        Get the ambient temperature in Kelvins.

        Returns
        -------
        kelvins : units.Quantity
        """
        return self.environment.ambient_t.to(
            'Kelvin', equivalencies=units.temperature())

    def get_ambient_pressure(self):
        """
        Get the ambient pressure.

        Returns
        -------
        pressure : units.Quantity
        """
        return np.nan * units.Unit('Pascal')

    def get_ambient_humidity(self):
        """
        Get the ambient humidity.

        Returns
        -------
        humidity : units.Quantity
        """
        return np.nan * units.Unit('gram/m3')

    def get_wind_direction(self):
        """
        Return the wind direction.

        Returns the tail vs. head wind.

        Returns
        -------
        direction : units.Quantity
        """
        if self.aircraft.ground_speed > self.aircraft.air_speed:
            return -180.0 * units.Unit('degree')
        else:
            return 0.0 * units.Unit('degree')

    def get_wind_speed(self):
        """
        Return the wind speed.

        Returns
        -------
        speed : units.Quantity
        """
        return np.abs(self.aircraft.ground_speed - self.aircraft.air_speed)

    def get_wind_peak(self):
        """
        Return the wind peak.

        Returns
        -------
        speed : units.Quantity
        """
        return np.nan * units.Unit('m/second')

    def validate_scans(self, scans):
        """
        Validate a list of scans specific to the instrument

        Parameters
        ----------
        scans : list (SofiaScan)
            A list of scans.

        Returns
        -------
        None
        """
        if scans is None or len(scans) == 0 or scans[0] is None:
            super().validate_scans(scans)
            return

        if len(scans) == 1:
            first_scan = scans[0]
            if first_scan.get_observing_time() < 3.3 * units.Unit('minute'):
                self.set_pointing(first_scan)
        super().validate_scans(scans)

    @abstractmethod
    def get_si_pixel_size(self):
        """
        Get the science instrument pixel size.

        Returns
        -------
        size : Coordinate2D
            The (x, y) pixel sizes, each of which is a units.Quantity.
        """
        pass

    @abstractmethod
    def get_file_id(self):
        """
        Return the file ID.

        Returns
        -------
        str
        """
        pass

    @staticmethod
    def get_plate_scale(angular_size, physical_size):
        """
        Return plate scaling.

        The plate scaling is in radians/m for the focal plane projected through
        the telescope.

        Parameters
        ----------
        angular_size : list (astropy.units.Quantity or float)
            x, y angles representing the projected angular size on the sky.  If
            float values are used, must be supplied in radians.
        physical_size : list (astropy.units.Quantity or float)
            x, y physical/geometric size on the focal plane unit.  If float
            values are used, must be in meters.

        Returns
        -------
        plate_scale : astropy.units.Quantity
        """
        rpm = units.Unit('radian/meter')
        result = angular_size[0] * angular_size[1]
        result /= physical_size[0] * physical_size[1]
        result = np.sqrt(result)

        if isinstance(result, units.Quantity):
            return result.to(rpm)
        else:
            return result * rpm
