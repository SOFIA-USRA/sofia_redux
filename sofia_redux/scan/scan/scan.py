# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC, abstractmethod
from astropy import log, units
from astropy.io import fits
from astropy.table import vstack as table_vstack
from astropy.time import Time
from copy import deepcopy
import numpy as np

from sofia_redux.scan.utilities import utils, numba_functions
from sofia_redux.scan.utilities.class_provider import get_scan_class
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.epoch.epoch import J2000
from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
from sofia_redux.scan.info.weather_info import WeatherInfo
from sofia_redux.scan.flags.mounts import Mount
from sofia_redux.scan.source_models.beams.elliptical_source import \
    EllipticalSource
from sofia_redux.scan.coordinate_systems.offset_2d import Offset2D
from sofia_redux.scan.coordinate_systems.celestial_coordinates import \
    CelestialCoordinates
from sofia_redux.scan.coordinate_systems.focal_plane_coordinates import \
    FocalPlaneCoordinates
from sofia_redux.scan.source_models.beams.instant_focus import InstantFocus
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.utilities.class_provider import get_integration_class

__all__ = ['Scan']


class Scan(ABC):

    def __init__(self, channels, reduction=None):
        self.serial_number = -1
        self.filename = None
        self.map_range = None
        self.weight = 1.0
        self.source_points = 0
        self.is_split = False
        self.pointing = None
        self.source_model = None
        self.integrations = None
        self.gain = 1.0
        self.map_range = np.zeros(4, dtype=float) * units.Unit('arcsec')
        self.pointing_correction = Coordinate2D()

        self.channels = None
        self.reduction = reduction
        self.set_channels(channels)

    @property
    def referenced_attributes(self):
        """
        Return the names of attributes that are referenced during a copy.

        Returns
        -------
        attribute_names : set (str)
        """
        return {'channels', 'reduction', 'source_model', 'integrations'}

    def copy(self):
        """
        Return a copy of the scan.  Certain attributes will be referenced only.

        Returns
        -------
        scan : Scan
        """
        new = self.__class__(channels=self.channels)
        referenced = self.referenced_attributes
        for attribute, value in self.__dict__.items():
            if attribute in referenced:
                setattr(new, attribute, value)
            elif hasattr(value, 'copy'):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, deepcopy(value))

        return new

    def set_channels(self, channels):
        """
        Set the instrument channels for the scan.

        Parameters
        ----------
        channels : Channels

        Returns
        -------
        None
        """
        if channels is None:
            return
        self.channels = channels.copy()
        self.channels.set_parent(self)

    def __len__(self):
        """
        Return the number of integrations in the scan.

        Returns
        -------
        n_integrations : int
        """
        if self.integrations is None:
            return 0
        return len(self.integrations)

    def __getitem__(self, index):
        """
        Return an integration(s) at the correct index.

        Parameters
        ----------
        index : int or slice

        Returns
        -------
        integration : Integration or list (Integration)
        """
        return self.integrations[index]

    def __setitem__(self, index, integration):
        """
        Set the integration(s) for a given index.

        Parameters
        ----------
        index : int or slice
        integration : Integration or list (Integration)
            The integration(s) to set.

        Returns
        -------
        None
        """
        self.integrations[index] = integration

    @property
    def info(self):
        """
        Return the information object for the scan.

        The information object contains the reduction configuration and various
        parameters pertaining the this scan.

        Returns
        -------
        Info
        """
        if self.channels is None:
            return None
        return self.channels.info

    @property
    def astrometry(self):
        """
        Return the scan astrometry information.

        Returns
        -------
        info : AstrometryInfo
        """
        if self.info is None:
            return None
        return self.info.astrometry

    @property
    def channel_flagspace(self):
        """
        Return the flagspace for the scan channels.

        Returns
        -------
        flagspace : ChannelFlags
        """
        if self.channels is None:
            return None
        return self.channels.flagspace

    @property
    def instrument_name(self):
        """
        Return the name of the instrument in the scan.

        Returns
        -------
        instrument : str
        """
        return self.info.instrument.name

    @property
    def frame_flagspace(self):
        """
        Return the flagspace for the scan frames.

        Returns
        -------
        flagspace : FrameFlags
        """
        if self.size == 0:
            return None
        return self.integrations[0].flagspace

    @classmethod
    def class_from_instrument_name(cls, name):
        """
        Return the appropriate scan instance for an instrument name.

        Parameters
        ----------
        name : str
            The instrument name.

        Returns
        -------
        Scan
            A scan instance.
        """
        return get_scan_class(name)

    @property
    def size(self):
        """
        Return the number of integrations in the scan.

        Returns
        -------
        n_integrations : int
        """
        return self.__len__()

    @property
    def mjd(self):
        """
        Get the scan Modified Julian Date (MJD).

        Returns
        -------
        mjd : float
        """
        if self.astrometry is None:
            return None
        return self.astrometry.mjd

    @mjd.setter
    def mjd(self, value):
        """
        Set the Modified Julian Date (MJD) and apply configuration options.

        Parameters
        ----------
        value : float

        Returns
        -------
        None
        """
        if self.astrometry is None:
            return
        self.astrometry.set_mjd(value)

    @property
    def lst(self):
        """
        Get the scan Local Sidereal Time (LST).

        Returns
        -------
        lst : astropy.units.Quantity
            The LST as an hour angle.
        """
        if self.astrometry is None:
            return None
        return self.astrometry.lst

    @lst.setter
    def lst(self, value):
        """
        Set the scan Local Sidereal Time (LST).

        Parameters
        ----------
        value : astropy.units.Quantity
            The value (in hourangle units) to set.

        Returns
        -------
        None
        """
        if self.astrometry is None:
            return
        self.astrometry.lst = value

    @property
    def source_name(self):
        """
        Return the name of the observation source.

        Returns
        -------
        name : str
        """
        if self.configuration is None:
            return None
        else:
            return self.info.observation.source_name

    @source_name.setter
    def source_name(self, value):
        """
        Set the name of the source.

        Parameters
        ----------
        value : str

        Returns
        -------
        None
        """
        if self.configuration is None:
            return
        self.info.observation.set_source(value)

    @property
    def configuration(self):
        """
        Return the scan configuration.

        Returns
        -------
        Configuration
        """
        if self.info is None:
            return None
        return self.info.configuration

    @property
    def is_nonsidereal(self):
        """
        Return whether the observation is non-sidereal.

        Returns
        -------
        non_sidereal : bool
        """
        if self.astrometry is None:
            return False
        return self.astrometry.is_nonsidereal

    @is_nonsidereal.setter
    def is_nonsidereal(self, value):
        """
        Set the non-sidereal flag for the observation.

        Parameters
        ----------
        value : bool

        Returns
        -------
        None
        """
        if self.astrometry is None:
            return
        self.astrometry.is_nonsidereal = utils.get_bool(value)

    @property
    def equatorial(self):
        """
        Return the scan equatorial position.

        Returns
        -------
        equatorial_coordinate : EquatorialCoordinates
        """
        if self.astrometry is None:
            return None
        return self.astrometry.equatorial

    @equatorial.setter
    def equatorial(self, value):
        """
        Set the scan equatorial position.

        Parameters
        ----------
        value : EquatorialCoordinates

        Returns
        -------
        None
        """
        if self.astrometry is None:
            return
        self.astrometry.equatorial = value

    @property
    def horizontal(self):
        """
        Get the scan horizontal coordinate position.

        Returns
        -------
        horizontal_coordinate : HorizontalCoordinate
        """
        if self.astrometry is None:
            return None
        return self.astrometry.horizontal

    @horizontal.setter
    def horizontal(self, value):
        """
        Set the scan horizontal coordinate position.

        Parameters
        ----------
        value : HorizontalCoordinates

        Returns
        -------
        None
        """
        if self.astrometry is None:
            return
        self.astrometry.horizontal = value

    @property
    def site(self):
        """
        Return the location of the site of the observation.

        Returns
        -------
        site_location : GeodeticCoordinates
        """
        if self.astrometry is None:
            return None
        return self.astrometry.site

    @site.setter
    def site(self, value):
        """
        Set the coordinates of the site location for the observation.

        Parameters
        ----------
        value : GeodeticCoordinates

        Returns
        -------
        None
        """
        if self.astrometry is None:
            return
        self.astrometry.site = value

    @property
    def apparent(self):
        """
        Return the apparent equatorial coordinate of the observation.

        Returns
        -------
        apparent_coordinate : EquatorialCoordinate
        """
        if self.astrometry is None:
            return None
        return self.astrometry.apparent

    @apparent.setter
    def apparent(self, value):
        """
        Set the apparent equatorial coordinate of the observation.

        Parameters
        ----------
        value : EquatorialCoordinates

        Returns
        -------
        None
        """
        if self.astrometry is None:
            return
        self.astrometry.apparent = value

    @property
    def is_tracking(self):
        """
        Return whether the telescope is tracking.

        Returns
        -------
        tracking : bool
        """
        if self.info is None:
            return False
        return self.info.telescope.is_tracking

    @property
    def serial(self):
        """
        Return the scan serial number.

        Returns
        -------
        serial_number : int
        """
        return self.serial_number

    @serial.setter
    def serial(self, value):
        """
        Set the scan serial number.

        Parameters
        ----------
        value : int

        Returns
        -------
        None
        """
        self.serial_number = value
        self.info.configuration.set_serial(value)

    def has_option(self, option):
        """
        Check whether an option is set in the configuration.

        Parameters
        ----------
        option : str
            The configuration option.

        Returns
        -------
        bool
        """
        if self.configuration is None:
            return False
        return self.configuration.is_configured(option)

    def have_equatorial(self):
        """
        Return whether equatorial coordinates exist for the scan.

        Returns
        -------
        bool
        """
        return self.equatorial is not None and self.equatorial.size != 0

    def have_horizontal(self):
        """
        Return whether horizontal coordinates exist for the scan.

        Returns
        -------
        bool
        """
        return self.horizontal is not None and self.horizontal.size != 0

    def have_site(self):
        """
        Return whether site coordinates exist for the scan.

        Returns
        -------
        bool
        """
        return self.site is not None and self.site.size != 0

    def have_apparent(self):
        """
        Return whether horizontal coordinates exist for the scan.

        Returns
        -------
        bool
        """
        return self.apparent is not None and self.apparent.size != 0

    def validate(self):
        """
        Validate the scan after a read.

        Returns
        -------
        None
        """
        if self.integrations is None:
            log.warning('No integrations to validate')
            return

        log.info("Processing scan data:")

        if self.configuration.get_bool('subscan.merge'):
            self.merge_integrations()

        if self.has_option('segment'):
            segment_time = self.configuration.get_float(
                'segment', default=60.0) * units.Unit('second')
            self.segment_to(segment_time)

        self.is_nonsidereal |= self.has_option('moving')

        first_frame = self.integrations[0].get_first_frame()
        last_frame = self.integrations[-1].get_last_frame()
        if np.isnan(self.mjd):  # pragma: no cover
            # this doesn't seem reachable under normal conditions
            self.mjd = first_frame.mjd
        if np.isnan(self.lst):
            self.lst = 0.5 * (first_frame.lst + last_frame.lst)

        if not self.configuration.get_bool('lab'):

            if not self.have_equatorial():
                self.calculate_equatorial()

            # Use J2000 coordinates
            if self.equatorial.epoch != J2000:
                self.precess(J2000)

            log.info(f"Equatorial: {self.equatorial}")

            # Calculate apparent and approximate horizontal coordinates
            if not self.have_apparent():
                self.calculate_apparent()

            if not self.have_horizontal() and self.have_site():
                self.calculate_horizontal()

            if self.have_horizontal():
                log.info(f"Horizontal: {self.horizontal}")

        for i, integration in enumerate(self.integrations):
            log.info(f"Processing integration {i + 1}:")
            integration.validate()
            n_valid = integration.get_frame_count(match_flag=0)
            t = integration.get_exposure_time()
            log.debug(f"Integration has {n_valid} valid frames.")
            log.debug(f"Total exposure time: {t}")

        if not self.have_valid_integrations():
            log.warning("No valid integrations exist")
            return

        if self.has_option('jackknife'):
            self.source_name += '-JK'

        if self.has_option('pointing'):
            pointing_options = self.configuration.get_branch('pointing')
            correction = self.get_pointing_correction(pointing_options)
            self.pointing_at(correction)

        self.channels.calculate_overlaps(point_size=self.get_point_size())

    def validate_integrations(self):
        """
        Remove any invalid integrations from the scan.

        Returns
        -------
        None
        """
        if self.integrations is None:
            return
        self.integrations = [x for x in self.integrations if x.is_valid]

    def is_valid(self):
        """
        Return whether the scan contains any valid integrations.

        Returns
        -------
        bool
        """
        self.validate_integrations()
        return self.size > 0

    def have_valid_integrations(self):
        """
        Return whether valid integrations exist in the scan.

        Returns
        -------
        bool
        """
        if self.integrations is None:
            return False
        for integration in self.integrations:
            if integration.is_valid:
                return True
        return False

    def set_iteration(self, iteration, rounds=None):
        r"""
        Set the configuration for a given iteration

        Parameters
        ----------
        iteration : float or int or str
            The iteration to set.  A positive integer defines the exact
            iteration number.  A negative integer defines the iteration
            relative to the last (e.g. -1 is the last iteration). A float
            represents a fraction (0->1) of the number of rounds, and
            a string may be parsed in many ways such as last, first, a float,
            integer, or percentage (if suffixed with a %).
        rounds : int, optional
            The maximum number of iterations.

        Returns
        -------
        None
        """
        if rounds is not None:
            self.configuration.iterations.max_iteration = rounds
        self.configuration.set_iteration(iteration)

        self.channels.calculate_overlaps(self.get_point_size())
        if self.integrations is not None:
            for integration in self.integrations:
                if integration.channels is not self.channels:
                    integration.set_iteration(iteration, rounds=rounds)

    def get_short_date_string(self):
        """
        Return a short date representation of the MJD time.

        Returns
        -------
        date : str
            The MJD date in YYYY-MM-DD format.
        """
        return Time(self.mjd, format='mjd', scale='utc').isot.split('T')[0]

    def get_pointing_correction(self, options):
        """
        Return the pointing corrections.

        Parameters
        ----------
        options : dict
            The pointing options.  Relevant keys are 'value' and 'offset'.

        Returns
        -------
        correction : Coordinate2D
            Contains the (x, y) pointing offsets.
        """
        size_unit = self.info.instrument.get_size_unit()
        if isinstance(options, dict):
            correction = Coordinate2D(
                options.get('value', [0.0, 0.0]), unit=size_unit)
            offset = Coordinate2D(
                options.get('offset', [0.0, 0.0]), unit=size_unit)
            correction.add(offset)
        elif len(options) == 2:
            correction = Coordinate2D([float(x) for x in options],
                                      unit=size_unit)
        else:
            correction = Coordinate2D(np.zeros(2, dtype=float), unit=size_unit)
        return correction

    def pointing_at(self, correction):
        """
        Apply a pointing correction.

        If a new pointing correction is provided, it will be added to the
        existing pointing correction.

        Parameters
        ----------
        correction : Coordinate2D
            The pointing (x, y) offset.

        Returns
        -------
        None
        """
        if not isinstance(correction, Coordinate2D):
            return
        log.info(f"Adjusting pointing by {correction}")
        if self.integrations is None:
            return
        for integration in self.integrations:
            integration.pointing_at(correction)

        if (self.pointing_correction is None
                or self.pointing_correction.size) == 0:
            self.pointing_correction = correction
        else:
            self.pointing_correction.add(correction)

    def precess(self, epoch):
        """
        Precess coordinates to a new epoch and update integrations.

        The scan and all integration frames will be precessed to a new epoch
        if necessary.

        Parameters
        ----------
        epoch : Epoch
            The new epoch.

        Returns
        -------
        None
        """
        self.info.astrometry.precess(epoch, scan=self)

    def calculate_equatorial(self):
        """
        Calculate the equatorial coordinates of the scan.

        Returns
        -------
        None
        """
        self.info.astrometry.calculate_equatorial()

    def calculate_apparent(self):
        """
        Calculate the apparent equatorial coordinates of the scan.

        Returns
        -------
        None
        """
        self.info.astrometry.calculate_apparent()

    def calculate_horizontal(self):
        """
        Calculate the horizontal coordinates of the scan.

        Returns
        -------
        None
        """
        self.info.astrometry.calculate_horizontal()

    def get_native_coordinates(self):
        """
        Return the native coordinates of the scan.

        Returns
        -------
        SphericalCoordinates
        """
        return self.info.astrometry.get_native_coordinates()

    def get_position_reference(self, system=None):
        """
        Return position reference in the defined coordinate frame.

        By default, the equatorial coordinates are returned, but many other
        frame systems may be specified.  All astropy coordinate frames may
        be used but may raise conversion errors depending on the type.  If
        an error is encountered during conversion, or the frame system is
        unavailable, equatorial coordinates will be returned.

        Parameters
        ----------
        system : str
            Name of the coordinate frame.  Available values are:
            {'horizontal', 'native', 'focalplane'} and all Astropy frame
            type names.  Note that focalplane is not currently implemented and
            will raise an error.

        Returns
        -------
        coordinates : SphericalCoordinates
            Coordinates of the specified type.
        """
        return self.info.astrometry.get_position_reference(system=system)

    def get_first_integration(self):
        """
        Return the first integration of the scan.

        Returns
        -------
        integration : Integration or None
            Will be `None` if no integrations exist.
        """
        if self.size == 0:
            return None
        return self.integrations[0]

    def get_last_integration(self):
        """
        Return the last integration of the scan.

        Returns
        -------
        integration : Integration or None
            Will be `None` if no integrations exist.
        """
        if self.size == 0:
            return None
        return self.integrations[-1]

    def get_first_frame(self):
        """
        Return the first frame of the first integration.

        Returns
        -------
        Frames
        """
        return self.get_first_integration().get_first_frame()

    def get_last_frame(self):
        """
        Return the last frame of the last integration.

        Returns
        -------
        Frames
        """
        return self.get_last_integration().get_last_frame()

    @abstractmethod
    def read(self, filename, read_fully=True):
        """
        Read a filename to populate the scan.

        The read should validate the channels before instantiating integrations
        for reading.

        Parameters
        ----------
        filename : str
            The name of the file to read.
        read_fully : bool, optional
            If `True`, perform a full read (default)

        Returns
        -------
        None
        """
        pass  # pragma: no cover

    def get_integration_instance(self):
        """
        Return an integration instance of the correct type for the scan.

        Returns
        -------
        integration : Integration
        """
        integration_class = get_integration_class(self.instrument_name)
        integration = integration_class(self)
        return integration

    def merge_integrations(self):
        """
        Merge integrations as necessary.

        Returns
        -------
        None
        """
        if not self.configuration.get_bool("subscan.merge"):
            return
        if self.size < 2:
            return
        log.info(f"Merging {self.size} integrations.")

        dt = self.info.instrument.sampling_interval
        max_discontinuity = self.configuration.get_float(
            "subscan.merge.maxgap", default=np.nan) * units.Unit('second')

        if np.isnan(max_discontinuity):
            max_gap = np.inf
        else:
            max_gap = int(np.ceil(max_discontinuity / dt).decompose().value)

        new_integrations = []
        merged = self.integrations[0]
        day = units.Unit('day')
        # Remove invalid frames from the end
        merged.trim(start=False, end=True)
        last_mjd = merged.frames.get_last_frame_value('mjd') * day

        for integration in self.integrations[1:]:
            integration.trim(start=True, end=True)
            next_mjd = integration.frames.get_first_frame_value('mjd') * day
            gap = int(np.round(((next_mjd
                                 - last_mjd - dt) / dt).decompose().value))
            last_mjd = integration.frames.get_last_frame_value('mjd') * day

            if gap > 0:
                if gap < max_gap:
                    log.debug(f"  > Padding with {gap} frames before "
                              f"integration {integration.get_id()}.")
                    insert_indices = np.zeros(gap, dtype=int)
                    integration.frames.insert_blanks(insert_indices)
                else:
                    log.debug(f"  > Large gap before "
                              f"integration {integration.get_id()}. "
                              f"Starting new merge.")
                    new_integrations.append(merged)
                    merged = integration
                    continue

            # Merge integration frames onto the merged integration.
            merged.merge(integration)

        merged.reindex()
        new_integrations.append(merged)
        self.integrations = new_integrations
        self.sort_integrations()
        log.info("  > New total exposure times:")
        for integration in self.integrations:
            int_id, t = integration.get_id(), integration.get_exposure_time()
            log.info(f"  > Integration {int_id}: {t}")

    def get_pa(self):
        """
        Return the position angle derived from mid point values of frames.

        The position angle is the mean of the position angle from the first
        frame of the first integration, and the last frame of the last
        integration.

        Returns
        -------
        angle : astropy.units.Quantity
            The position angle in degrees.
        """
        angle1 = np.arctan2(self[0].frames.get_first_frame_value('sin_pa'),
                            self[0].frames.get_first_frame_value('cos_pa'))

        angle2 = np.arctan2(self[-1].frames.get_last_frame_value('sin_pa'),
                            self[-1].frames.get_last_frame_value('cos_pa'))

        return ((angle1 + angle2) / 2.0 * units.Unit('radian')).to('degree')

    def get_summary_hdu(self, configuration=None):
        """
        Create a FITS HDU from the scan given a configuration.

        Parameters
        ----------
        configuration : Configuration, optional
            The configuration from which to create the HDU.  If not supplied
            defaults to the scan configuration.

        Returns
        -------
        astropy.io.fits.hdu.table.BinTableHDU
        """
        if configuration is None:
            configuration = self.configuration
        get_details = configuration.get_bool('write.scandata.details')
        full_table = None
        for integration in self.integrations:
            table = integration.get_fits_data()
            if get_details:
                table = integration.add_details(table)
            if full_table is None:
                full_table = table
            else:
                full_table = table_vstack((full_table, table))

        hdu = fits.BinTableHDU(data=full_table)
        self.edit_scan_header(hdu.header)
        self.configuration.configuration_difference(configuration).edit_header(
            hdu.header)

        return hdu

    def edit_scan_header(self, header):
        """
        Edit scan FITS header information.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The header to edit.

        Returns
        -------
        None
        """
        header['EXTNAME'] = f"Scan-{self.get_id()}", 'Scan data'
        header['INSTRUME'] = (f"{self.info.instrument.name}",
                              'The instrument name')
        header['SCANID'] = self.get_id(), 'Scan ID.'

        if self.serial_number > 0:
            header['SCANNO'] = self.serial_number, 'Serial number for the scan'
        if self.info.origin.descriptor is not None:
            header['SCANSPEC'] = (self.info.origin.descriptor,
                                  'Scan descriptor')
        if self.info.origin.observer is not None:
            header['OBSERVER'] = (self.info.origin.observer,
                                  'Name(s) of the observer(s)')
        if self.info.observation.project is not None:
            header['PROJECT'] = (self.info.observation.project,
                                 'Description of the project')
        if self.info.origin.creator is not None:
            header['CREATOR'] = (self.info.origin.creator,
                                 "Software that wrote the scan data.")
        if self.info.astrometry.time_stamp is not None:
            header['DATE-OBS'] = (self.info.astrometry.time_stamp,
                                  "Start of observation")

        header['OBJECT'] = (self.info.observation.source_name,
                            'Object catalog name')

        system = 'FK5' if self.equatorial.epoch.is_julian else 'FK4'
        header['RADESYS'] = system, 'World coordinate system id'

        if np.isfinite(self.equatorial.ra):
            header['RA'] = (
                str(self.equatorial.ra.to('hourangle').round(2).value),
                'Human Readable Right Ascension')

        if np.isfinite(self.equatorial.dec):
            header['DEC'] = (
                str(self.equatorial.dec.to('degree').round(1).value),
                'Human Readable Declination')

        header['EQUINOX'] = (self.equatorial.epoch.equinox.value,
                             'Precession epoch')
        if np.isfinite(self.mjd):
            header['MJD'] = self.mjd, 'Modified Julian Day'

        if self.astrometry.ground_based:
            if np.isfinite(self.lst):
                header['LST'] = (self.lst.to('hourangle').value,
                                 'Local Sidereal Time (hours)')
            if self.horizontal is not None:
                if np.isfinite(self.horizontal.az):
                    header['AZ'] = (self.horizontal.az.to('degree').value,
                                    'Azimuth (deg).')
                if np.isfinite(self.horizontal.el):
                    header['EL'] = (self.horizontal.el.to('degree').value,
                                    'Elevation (deg).')
            if self.site is not None:
                header['SITELON'] = (
                    self.site.longitude.to('degree').round(1).value,
                    'Geodetic longitude of the observing site (deg)')
                header['SITELAT'] = (
                    self.site.latitude.to('degree').round(1).value,
                    'Geodetic latitude of the observing site (deg)')

            position_angle = self.get_pa()
            if np.isfinite(position_angle):
                header['PA'] = (position_angle.to('degree').value,
                                'Direction of zenith w.r.t. North (deg)')

        header['WEIGHT'] = self.weight, 'Relative source weight of the scan'
        header['TRACKIN'] = (
            self.is_tracking,
            'Was the telescope tracking during the observation?')

        if self.pointing is not None:
            self.edit_pointing_header_info(header)
        self.channels.edit_scan_header(header)

    def edit_pointing_header_info(self, header):
        """
        Edit pointing information in a header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.

        Returns
        -------
        None
        """
        if self.pointing is None:
            return

        relative = self.get_native_pointing_increment(self.pointing)

        header['COMMENT'] = "<------ Fitted Pointing / " \
                            "Calibration Info ------>"
        header['PNT_DX'] = (relative.x.value,
                            f'({relative.unit}) pointing offset in native X.')
        header['PNT_DY'] = (relative.y.value,
                            f'({relative.unit}) pointing offset in native Y.')

        self.pointing.edit_header(
            header, size_unit=self.info.instrument.get_size_unit())

    def set_source_model(self, model):
        """
        Set the source model for the scan.

        Parameters
        ----------
        model : SourceModel

        Returns
        -------
        None
        """
        self.source_model = model

    def get_observing_time(self):
        """
        Return the scan observing time in seconds.

        Returns
        -------
        observing_time : astropy.units.Quantity
            The observing time in seconds.
        """
        observing_time = 0.0 * units.Unit('s')
        if self.size == 0:
            return observing_time

        for integration in self.integrations:
            discard_flag = ~(integration.flagspace.flags.CHOP_LEFT
                             | integration.flagspace.flags.CHOP_RIGHT)
            n_frames = integration.get_frame_count(discard_flag=discard_flag)
            t = n_frames * integration.info.instrument.integration_time
            observing_time += t

        return observing_time

    def get_frame_count(self, keep_flag=None, discard_flag=None,
                        match_flag=None):
        """
        Return the number of frames in the scan.

        Optionally provide flags to indicate which

        Parameters
        ----------
        keep_flag : int or FrameFlagTypes
        discard_flag : int or FrameFlagTypes
        match_flag : int or FrameFlagTypes

        Returns
        -------
        n_frames : int
            The number of frames in the scan.
        """
        if self.size == 0:
            return 0
        n_frames = 0
        for integration in self.integrations:
            n_frames += integration.get_frame_count(keep_flag=keep_flag,
                                                    discard_flag=discard_flag,
                                                    match_flag=match_flag)
        return n_frames

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
        if not self.have_horizontal() and (self.have_equatorial()
                                           and self.have_site()):
            self.horizontal = self.equatorial.to_horizontal(
                self.site, self.lst)

        if self.source_model is None:
            source_type = 'model'
        else:
            source_type = self.source_model.logging_id

        if name.startswith('?'):
            name = name[1:].lower()

            value = self.configuration.get(name)
            if value is None:
                return None

            if len(value) == 0:
                # The configuration value exists, but is not set
                return True

            return value

        source_key = f'{source_type}.'
        if name.startswith(source_key):
            if self.source_model is None:
                return None
            source_entry = ''.join(name.split(source_key)[1:])
            return self.source_model.get_table_entry(source_entry)

        point_key = 'pnt.'
        if name.startswith(point_key):
            if self.pointing is None:
                return None
            return self.get_pointing_data().get(name[len(point_key):])

        astro_key = 'src.'
        if name.startswith(astro_key):
            if self.pointing is None:
                return None
            if not isinstance(self.source_model,
                              AstroIntensityMap):  # pragma: no cover
                # not reachable under normal circumstances
                return None

            astro_entry = name[len(astro_key):]
            pointing = self.pointing.get_representation(
                self.source_model.map.grid)
            pointing_data = pointing.get_data(
                self.source_model.map,
                size_unit=self.info.instrument.get_size_unit())
            return pointing_data.get(astro_entry)

        info = self.info
        if name == 'object':
            return self.source_name
        if name == 'id':
            return self.get_id()
        if name == 'serial':
            return self.serial_number
        if name == 'MJD':
            return self.mjd
        if name == 'UT':
            return self.mjd - np.floor(self.mjd)
        if name == 'UTh':
            return (self.mjd - np.floor(self.mjd)) * 24
        if name == 'PA':
            return self.get_pa().value
        if name == 'PAd':
            return self.get_pa().to('degree').value
        if name == 'AZ':
            if self.horizontal is None:
                return None
            else:
                return self.horizontal.az.decompose().value
        if name == 'EL':
            if self.horizontal is None:
                return None
            else:
                return self.horizontal.el.decompose().value
        if name == 'RA':
            return self.equatorial.ra.to('hourangle').value
        if name == 'DEC':
            return self.equatorial.dec.to('degree').value
        if name == 'AZd':
            if self.horizontal is None:
                return None
            else:
                return self.horizontal.az.to('degree').value
        if name == 'ELd':
            if self.horizontal is None:
                return None
            else:
                return self.horizontal.el.to('degree').value
        if name == 'RAd':
            return self.equatorial.ra.to('degree').value
        if name == 'RAh':
            two_pi = 2 * np.pi * units.Unit('radian')
            ra_h = (self.equatorial.ra + two_pi).to('hourangle').value
            return ra_h % 24
        if name == 'DECd':
            return self.equatorial.dec.to('degree').value
        if name == 'epoch':
            return str(self.equatorial.epoch)
        if name == 'epochY':
            return self.equatorial.epoch.year
        if name == 'LST':
            return self.lst.decompose().value
        if name == 'LSTh':
            return self.lst.to('hourangle').value
        if name == 'date':
            return Time(self.mjd, format='mjd', scale='utc').isot
        if name == 'obstime':
            return self.get_observing_time().to('second').value
        if name == 'obsmins':
            return self.get_observing_time().to('minute').value
        if name == 'obshours':
            return self.get_observing_time().to('hour').value
        if name == 'weight':
            return self.weight
        if name == 'frames':
            return self.get_frame_count(match_flag=0)
        if name == 'project':
            return info.observation.project
        if name == 'observer':
            return info.origin.observer
        if name == 'creator':
            return info.origin.creator
        if name == 'integrations':
            return self.size
        if name == 'generation':
            return self.get_source_generation()

        if isinstance(info, WeatherInfo):
            if name == 'Tamb':
                return info.get_ambient_kelvins().to(
                    'Celsius', equivalencies=units.temperature()).value
            if name == 'humidity':
                return info.get_ambient_humidity().value
            if name == 'pressure':
                return info.get_ambient_pressure().to('hPa').value
            if name == 'windspeed':
                return info.get_wind_speed().to('m/s').value
            if name == 'windpeak':
                return info.get_wind_peak().to('m/s').value
            if name == 'winddir':
                return info.get_wind_direction().to('degree').value

        if self.source_model is not None:
            model_entry = self.source_model.get_table_entry(name)
            if model_entry is not None:
                return model_entry

        if self.size > 0:
            return self.get_first_integration().get_table_entry(name)

        return None

    def get_source_generation(self):
        """
        Return the source generation.

        Returns
        -------
        source_generation : int
        """
        max_generation = 0
        if self.integrations is None:
            return max_generation
        for integration in self.integrations:
            if integration.source_generation > max_generation:
                max_generation = integration.source_generation
        return max_generation

    def write_products(self):
        """
        Write the scan information to file.

        Returns
        -------
        None
        """
        if self.size > 0:
            for integration in self.integrations:
                integration.write_products()

        if not self.configuration.get_bool('lab'):
            self.report_focus()
            self.report_pointing()

    def report_focus(self):
        """
        Report the focus information for the scan.

        Returns
        -------
        None
        """
        if self.pointing is not None:
            log.info(f"Instant Focus Results for Scan {self.get_id()}:\n\n"
                     f"{self.get_focus_string()}")
        elif (self.has_option('pointing')
              and self.source_model.is_valid()):
            method = self.configuration.get_string('pointing')
            if method in ['suggest', 'auto']:
                log.warning(f"Cannot suggest focus for scan {self.get_id()}.")

    def report_pointing(self):
        """
        Return the pointing results for the scan.

        Returns
        -------
        None
        """
        if self.pointing is not None:
            log.info(f"Pointing Results for Scan {self.get_id()}:\n\n"
                     f"{self.get_pointing_string()}")
        elif self.source_model.is_valid() and self.has_option('pointing'):
            method = self.configuration.get_string('pointing')
            if method in ['suggest', 'auto']:
                log.warning(
                    f"Cannot suggest pointing for scan {self.get_id()}.")

    def split(self):
        """
        Split this scan into multiple scans.

        Returns a list of scans each containing a single integration.

        Returns
        -------
        list (Scan)
        """
        log.info("Splitting subscans into separate scans.")
        if self.size <= 1:
            return [self]

        scans = []
        for integration in self.integrations:
            scan = self.copy()
            scan.integrations = []
            scan.channels = integration.channels
            integration.scan = scan
            scan.integrations = [integration]
            scans.append(scan)

        return scans

    def update_gains(self, modality_name):
        """
        Update all gains in the scan for a given modality.

        Parameters
        ----------
        modality_name : str
            The name of the modality

        Returns
        -------
        None
        """
        if not self.configuration.get_bool('gains'):
            return
        if self.has_option(f'correlated.{modality_name}.nogains'):
            return
        robust = self.configuration.get_string('gains.estimator') == 'median'
        have_gains = False
        gains = np.zeros(self.channels.size, dtype=np.float64)
        gain_weights = np.zeros(self.channels.size, dtype=np.float64)
        for integration in self.integrations:
            modality = integration.channels.modalities.get(modality_name)
            if modality is None:
                continue
            if modality.trigger is not None:
                if not self.configuration.check_trigger(modality.trigger):
                    continue
            modality.average_gains(integration, gains, gain_weights,
                                   robust=robust)
            have_gains = True

        if not have_gains:
            return

        # Apply the gain increment
        for integration in self.integrations:
            modality = integration.channels.modalities.get(modality_name)
            if modality is None:  # pragma: no cover
                # not reachable under normal conditions
                continue
            flagged = modality.apply_gains(integration, gains, gain_weights)
            if flagged:
                integration.channels.census()
                integration.comments.append(
                    f'{integration.channels.n_mapping_channels}')

    def decorrelate(self, modality_name):
        """
        Decorrelate a modality.

        Parameters
        ----------
        modality_name : str

        Returns
        -------
        None
        """
        if not self.configuration.get_bool(f'correlated.{modality_name}'):
            log.debug(f"correlated.{modality_name} is not configured: will "
                      f"not decorrelate.")
            return

        robust = self.configuration.get_string('estimator') == 'median'
        if (self.configuration.get_bool(f'correlated.{modality_name}.span')
                or self.configuration.get_bool('gains.span')):
            for integration in self.integrations:
                integration.decorrelate_signals(modality_name, robust=robust)
            self.update_gains(modality_name)
        else:
            for integration in self.integrations:
                integration.decorrelate(modality_name, robust=robust)
        for integration in self.integrations:
            if integration.comments is None or len(integration.comments) == 0:
                integration.comments = [' ']
            elif integration.comments[-1] != ' ':
                integration.comments.append(' ')

    def perform(self, task):
        """
        Perform a reduction task on the scan.

        Parameters
        ----------
        task : str
            The name of the task.

        Returns
        -------
        None
        """
        if task.startswith('correlated.'):
            self.decorrelate(task.split('.')[1])
        else:
            for integration in self.integrations:
                integration.perform(task)

    def get_id(self):
        """
        Return the scan simple ID.

        Returns
        -------
        str
        """
        if self.serial_number is None:
            return '1'
        else:
            return str(self.serial_number)

    def get_pointing_data(self):
        """
        Return pointing data information.

        Returns
        -------
        data : dict
        """
        if self.pointing is None:
            raise ValueError(f"No pointing data for scan {self.get_id()}")

        relative = self.get_native_pointing_increment(self.pointing)
        absolute = self.get_native_pointing(self.pointing)
        data = {}

        if isinstance(relative, SphericalCoordinates):
            name_x = relative.longitude_axis.short_label
            name_y = relative.latitude_axis.short_label
        else:
            name_x = 'X'
            name_y = 'Y'

        size_unit = self.info.instrument.get_size_unit()
        data['dX'] = relative.x.to(size_unit)
        data['dY'] = relative.y.to(size_unit)
        data[f'd{name_x}'] = relative.x.to(size_unit)
        data[f'd{name_y}'] = relative.y.to(size_unit)
        data[name_x] = absolute.x.to(size_unit)
        data[name_y] = absolute.y.to(size_unit)

        # Print Nasmyth offsets if applicable
        mount = self.info.instrument.mount
        if mount == Mount.LEFT_NASMYTH or mount == Mount.RIGHT_NASMYTH:
            nasmyth = self.get_nasmyth_offset(relative)
            data['dNasX'] = nasmyth.x.to(size_unit)
            data['dNasY'] = nasmyth.y.to(size_unit)
            nasmyth = self.get_nasmyth_offset(absolute)
            data['NasX'] = nasmyth.x.to(size_unit)
            data['NasY'] = nasmyth.y.to(size_unit)

        percent = units.Unit('percent')
        asymmetry = self.get_source_asymmetry(self.pointing)
        data['asymX'] = 100 * asymmetry.x * percent
        data['asymY'] = 100 * asymmetry.y * percent
        data['dasymX'] = 100 * asymmetry.x_rms * percent
        data['dasymY'] = 100 * asymmetry.y_rms * percent

        if isinstance(self.pointing, EllipticalSource):
            ellipse = self.pointing
            data['elong'] = 100 * percent * ellipse.elongation
            data['delong'] = 100 * percent * ellipse.elongation_rms
            data['angle'] = ellipse.position_angle.to('degree')
            data['dangle'] = ellipse.position_angle.to('degree')
            (elongation_x,
             elongation_rms) = self.get_source_elongation_x(ellipse)
            data['elongX'] = 100 * percent * elongation_x
            data['delongX'] = 100 * percent * elongation_rms

        return data

    def get_pointing_string(self):
        """
        Return a string representing the pointing information.

        Returns
        -------
        info : str
        """
        info = ['']
        if isinstance(self.source_model, AstroIntensityMap):
            info.extend(self.pointing.pointing_info(self.source_model.map))

        increment = self.get_native_pointing_increment(self.pointing)
        info.append(self.get_pointing_string_from_increment(increment))
        return '\n'.join(info)

    def get_source_asymmetry(self, region):
        """
        Return the source model asymmetry.

        Parameters
        ----------
        region : CircularRegion

        Returns
        -------
        Asymmetry2D
        """
        source = self.source_model
        if not isinstance(source, AstroIntensityMap):
            return None
        radial_range = Range()
        point_size = self.info.instrument.get_point_size()
        radial_range.min = point_size
        radial_range.max = self.configuration.get_float(
            'focus.r', default=2.5) * point_size

        if (not source.grid.is_equatorial()) or source.grid.is_horizontal():
            return region.get_asymmetry_2d(
                image=source.map,
                angle=0.0 * units.Unit('radian'),
                radial_range=radial_range)

        if self.astrometry.ground_based and source.grid.is_equatorial():
            angle = self.get_pa()
        else:
            angle = 0.0 * units.Unit('radian')

        return region.get_asymmetry_2d(image=source.map.get_significance(),
                                       angle=angle,
                                       radial_range=radial_range)

    def get_source_elongation_x(self, ellipse):
        """
        Return the elliptical source elongation in x.

        Parameters
        ----------
        ellipse : EllipticalSource

        Returns
        -------
        elongation_x, elongation_x_weight : float, float
        """
        elongation = ellipse.elongation
        weight = ellipse.elongation_weight
        angle = ellipse.position_angle

        if self.astrometry.ground_based and isinstance(
                self.pointing.coordinates, EquatorialCoordinates):
            angle -= self.get_pa()

        factor = np.cos(2 * angle)
        if isinstance(factor, units.Quantity):
            factor = factor.value
        elongation *= factor
        weight /= factor ** 2
        return elongation, weight

    def get_focus_string(self, asymmetry=None, elongation=None, weight=None):
        """
        Return a string representing focus.

        Parameters
        ----------
        asymmetry : Asymmetry2D, optional
            The source asymmetry.  If not supplied, will be determined from the
            scan pointing.
        elongation : float, optional
            The elongation of the source in x.  If not supplied, will be
            determined from the pointing elongation.
        weight : float, optional
            The weight of the elongation.  If not supplied will be determined
            from the pointing elongation.

        Returns
        -------
        str
        """
        if asymmetry is None:
            asymmetry = self.get_source_asymmetry(self.pointing)

        if elongation is None or weight is None:
            if isinstance(self.pointing, EllipticalSource):
                e, w = self.get_source_elongation_x(self.pointing)
            else:
                e = w = None
            elongation = e if elongation is None else elongation
            weight = w if weight is None else weight

        result = "" if asymmetry is None else f"{asymmetry}\n"
        if elongation is not None:
            result += f"  Elongation: {elongation * 100:.3f}%\n"

        relative_fwhm = (
            self.pointing.fwhm / self.get_point_size()).decompose().value
        force_focus = self.has_option('focus')
        if force_focus or (0.8 < relative_fwhm <= 2.0):
            focus = InstantFocus()
            focus.derive_from(self.configuration,
                              asymmetry=asymmetry,
                              elongation=elongation,
                              elongation_weight=weight)
            result += self.info.instrument.get_focus_string(focus)
        else:
            if relative_fwhm <= 0.8:
                log.warning("Source FWHM unrealistically low")
            else:
                log.warning("Source is either too extended or too defocused.\n"
                            "No focus correction is suggested. You can force\n"
                            "calculate suggested focus values by setting\n"
                            "'focus' option when running SOFSCAN.")

        return result

    def get_point_size(self):
        """
        Return the point size of the scan.

        The point size will be the maximum of either the scan or source model
        (if available).

        Returns
        -------
        astropy.units.Quantity
            The point size.
        """
        point_size = self.info.instrument.get_point_size()
        if self.source_model is None:
            return point_size
        return max(point_size, self.source_model.get_point_size())

    def get_pointing_string_from_increment(self, increment):
        """
        Return a pointing string given an increment offset.

        Parameters
        ----------
        increment : Offset2D

        Returns
        -------
        str
        """
        if increment is None:
            return ''
        size_unit = self.info.instrument.get_size_unit()
        try:
            coordinates = increment.get_coordinate_class()()
            system = coordinates.local_coordinate_system
            name_x = system.axes[0].short_label
            name_y = system.axes[1].short_label
        except Exception as err:
            log.warning(f"Could not retrieve local coordinate system.  Using "
                        f"(x,y) default: {err}")
            name_x = 'x'
            name_y = 'y'

        result = f"  Offset: {increment.x.to(size_unit):.3f}, "
        result += f"{increment.y.to(size_unit):.3f} ({name_x}, {name_y})"

        # Also print Nasmyth offsets if applicable
        if (self.info.instrument.mount == Mount.LEFT_NASMYTH
                or self.info.instrument.mount == Mount.RIGHT_NASMYTH):
            nasmyth = self.get_nasmyth_offset(increment)
            result += f"\n  Offset: {nasmyth.x.to(size_unit)}, "
            result += f"{nasmyth.y.to(size_unit)} (nasmyth)"

        return result

    def get_equatorial_pointing(self, source):
        """
        Return the equatorial pointing.

        Parameters
        ----------
        source : GaussianSource

        Returns
        -------
        coordinates : Coordinate2D
        """
        source_coordinates = source.coordinates
        if (source_coordinates.__class__
                != self.source_model.reference.__class__):
            raise ValueError("Pointing source is in a different coordinate "
                             "system from source model.")

        if isinstance(source_coordinates, EquatorialCoordinates):
            reference = self.source_model.reference
            if source_coordinates.epoch != self.equatorial.epoch:
                source_coordinates.precess(self.equatorial.epoch)
        else:
            equatorial_coordinates = self.equatorial.copy()
            reference = self.equatorial.copy()
            source_coordinates.to_equatorial(equatorial_coordinates)
            self.source_model.reference.to_equatorial(reference)

        return source_coordinates.get_offset_from(reference)

    def get_native_pointing(self, source):
        """
        Get the native pointing from a Gaussian source.

        Parameters
        ----------
        source : GaussianSource

        Returns
        -------
        Offset2D
        """
        pointing = self.get_native_pointing_increment(source)
        if (self.pointing_correction is not None
                and self.pointing_correction.size > 0):
            pointing.add(self.pointing_correction)
        return pointing

    def get_native_pointing_increment(self, source):
        """
        Return the native pointing increment of the map from the pointing.

        Parameters
        ----------
        source : GaussianSource

        Returns
        -------
        Offset2D
        """
        if (source.coordinates.__class__
                != self.source_model.reference.__class__):

            raise ValueError("pointing source is in a different coordinate "
                             "system from the source model.")

        source_coordinates = source.coordinates
        native_coordinates = self.get_native_coordinates()
        reference = self.source_model.get_reference()

        if source_coordinates.__class__ == native_coordinates.__class__:
            return Offset2D(
                reference,
                coordinates=source_coordinates.get_offset_from(reference))

        elif isinstance(source_coordinates, EquatorialCoordinates):
            source_offsets = source_coordinates.get_offset_from(reference)
            source_offset = Offset2D(reference, coordinates=source_offsets)
            return self.get_native_offset_of(source_offset)

        elif isinstance(source_coordinates, CelestialCoordinates):
            source_equatorial = source_coordinates.to_equatorial()
            reference_equatorial = reference.to_equatorial()
            source_offsets = source_equatorial.get_offset_from(
                reference_equatorial)
            source_offset = Offset2D(
                reference_equatorial, coordinates=source_offsets)
            return self.get_native_offset_of(source_offset)

        elif isinstance(source_coordinates, FocalPlaneCoordinates):
            offset = source_coordinates.get_offset_from(reference)
            first_frame = self.get_first_integration().get_first_frame()
            last_frame = self.get_last_integration().get_last_frame()
            first_angle = np.atleast_1d(first_frame.get_rotation())
            last_angle = np.atleast_1d(last_frame.get_rotation())
            angle = 0.5 * (first_angle[0] + last_angle[0])
            Coordinate2D.rotate_offsets(offset, -angle)
            return Offset2D(FocalPlaneCoordinates([0.0, 0.0]), offset)

        else:  # pragma: no cover
            # not reachable under normal circumstances
            return None

    def get_native_offset_of(self, equatorial_offset):
        """
        Get the native offset of equatorial offsets.

        Parameters
        ----------
        equatorial_offset : Offset2D
            The equatorial (x, y) offsets w.r.t. a reference position.

        Returns
        -------
        Offset2D
        """

        if not isinstance(equatorial_offset.get_coordinate_class()(),
                          EquatorialCoordinates):
            raise ValueError("Not an equatorial offset.")

        offset = Coordinate2D(equatorial_offset)
        from_offset = self.get_first_frame().equatorial_to_native_offset(
            equatorial_offset.copy())
        to_offset = self.get_last_frame().equatorial_to_native_offset(
            equatorial_offset.copy())
        offset.set_x(0.5 * (from_offset.x + to_offset.x), copy=False)
        offset.set_y(0.5 * (from_offset.y + to_offset.y), copy=False)
        return Offset2D(self.get_native_coordinates(), coordinates=offset)

    def get_nasmyth_offset(self, pointing):
        """
        Return the Nasmyth pointing offset w.r.t the reference position.

        Parameters
        ----------
        pointing : Offset2D

        Returns
        -------
        Coordinate2D
        """
        reference = self.get_native_coordinates()
        if pointing.get_coordinate_class() != reference.__class__:
            raise ValueError("Non-native pointing offset.")
        if self.info.instrument.mount == Mount.LEFT_NASMYTH:
            sin_a = -reference.sin_lat
        else:
            sin_a = reference.sin_lat
        cos_a = reference.cos_lat
        nasmyth = Coordinate2D()
        nasmyth.set_x((cos_a * pointing.x) + (sin_a * pointing.y), copy=False)
        nasmyth.set_y((cos_a * pointing.y) - (sin_a * pointing.x), copy=False)
        return nasmyth

    def __str__(self):
        """
        Return a string representation of the scan.

        Returns
        -------
        str
        """
        return f'Scan {self.get_id()}'

    def segment_to(self, segment_time):
        """
        Split integrations such that each are approximately the same length.

        Integrations are merged together, and then split such that each
        integration is of length `segment_time`.  All previous dependents,
        signals, and filter objects will unset.

        Parameters
        ----------
        segment_time : astropy.units.Quantity
            The maximum time for an integration.

        Returns
        -------
        None
        """
        if self.size > 1:
            self.merge_integrations()
        merged = self.integrations[0]
        n_frames = merged.frames_for(segment_time)
        n_segments = numba_functions.roundup_ratio(merged.size, n_frames)
        if n_segments <= 1:
            return

        log.info(f"Segmenting into {n_segments} integrations.")
        integrations = []
        for i in range(n_segments):
            start_frame = i * n_frames
            end_frame = np.clip(start_frame + n_frames, None, merged.size)
            new_integration = merged.clone()
            new_integration.frames = merged.frames[start_frame:end_frame]
            new_integration.frames.integration = new_integration
            new_integration.integration_number = i
            new_integration.reindex()

            integrations.append(new_integration)

        self.integrations = integrations

    @staticmethod
    def time_order_scans(scans):
        """
        Return a list or scans in time order (by MJD).

        Parameters
        ----------
        scans : list (Scan)
            A list of scans.

        Returns
        -------
        list (Scan)
        """
        result = []
        indices = np.argsort([scan.mjd for scan in scans])
        print([scan.mjd for scan in scans])
        print(indices)
        for index in indices:
            result.append(scans[index])
        return result

    def calculate_precessions(self, epoch):
        """
        Calculate the precessions to and from the apparent coordinates.

        epoch : int or float or str or Epoch

        Returns
        -------
        None
        """
        self.info.astrometry.calculate_precessions(epoch)

    def frame_midpoint_value(self, frame_field):
        """
        Return the midpoint value of a given frame field.

        The midpoint is defined as the mean of the first valid frame value from
        the first integration, and the last frame of the last integration.

        Parameters
        ----------
        frame_field : str
            The name of the frame field.

        Returns
        -------
        midpoint
        """
        frames1 = self.get_first_integration().frames
        frames2 = self.get_last_integration().frames
        value1 = getattr(frames1, frame_field, None)
        if value1 is None:
            raise ValueError(f"{frames1} does not contain a "
                             f"{frame_field} field.")
        value2 = getattr(frames1, frame_field, None)
        value1 = value1[frames1.get_first_frame_index()]
        value2 = value2[frames2.get_last_frame_index()]
        return (value1 + value2) / 2

    def sort_integrations(self):
        """
        Sort scan integrations based on MJD.

        The integration IDs are also updated.

        Returns
        -------
        None
        """
        mjds = [x.frames.get_first_frame_value('mjd')
                for x in self.integrations]
        sort_idx = list(np.argsort(mjds))
        new_integrations = []
        for i, idx in enumerate(sort_idx):
            integration = self.integrations[idx]
            integration.integration_number = i
            new_integrations.append(integration)
        self.integrations = new_integrations
