# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from abc import abstractmethod
from astropy import log, units
from astropy.coordinates import FK5, SkyCoord
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
import os
import psutil

from sofia_redux.toolkit.utilities import multiprocessing
from sofia_redux.scan.source_models.source_model import SourceModel
from sofia_redux.scan.utilities import numba_functions
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D
from sofia_redux.scan.source_models import source_numba_functions as snf
from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection
from sofia_redux.scan.coordinate_systems.grid.spherical_grid import \
    SphericalGrid
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.index_2d import Index2D
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector

__all__ = ['AstroModel2D']


class AstroModel2D(SourceModel):

    DEFAULT_PNG_SIZE = 300
    MAX_X_OR_Y_SIZE = 5000

    def __init__(self, info, reduction=None):
        super().__init__(info=info, reduction=reduction)
        self.grid = SphericalGrid()
        self.smoothing = 0.0 * units.Unit('arcsec')
        self.allow_indexing = True
        self.index_shift_x = 0
        self.index_mask_y = 0
        self.max_x_or_y_size = 5000
        if self.has_option('grid'):
            resolution = self.configuration.get_float(
                'grid') * self.info.instrument.get_size_unit()
        else:
            resolution = 0.2 * self.info.instrument.resolution

        self.grid.set_resolution(resolution)

    @property
    def projection(self):
        """
        Return the grid projection.

        Returns
        -------
        SphericalProjection
        """
        if self.grid is None:
            return None
        return self.grid.projection

    @projection.setter
    def projection(self, value):
        """
        Set the grid projection.

        Parameters
        ----------
        value : SphericalProjection

        Returns
        -------
        None
        """
        self.grid.projection = value

    @property
    def wcs(self):
        """
        Return the WCS projection.

        Returns
        -------
        WCS
        """
        if self.grid is None:
            return None
        return self.grid.wcs

    @wcs.setter
    def wcs(self, value):
        """
        Set the WCS projection.

        Parameters
        ----------
        value : WCS

        Returns
        -------
        None
        """
        if self.grid is None:
            return
        self.grid.wcs = value

    @property
    def reference(self):
        """
        Return the reference pixel of the source model WCS.

        Returns
        -------
        numpy.ndarray (float)
            The (x, y) reference position of the source model.
        """
        if self.grid is None:
            return None
        return self.grid.reference

    @reference.setter
    def reference(self, value):
        """
        Set the reference pixel of the source model WCS.

        Parameters
        ----------
        value : numpy.ndarray (float)
            The (x, y) reference position of the source model.

        Returns
        -------
        None
        """
        if self.grid is None:
            return
        self.grid.reference = value

    @abstractmethod
    def is_empty(self):
        """
        Return whether the source map is empty.

        Returns
        -------
        bool
        """
        pass

    @abstractmethod
    def get_pixel_footprint(self):
        """
        Return the pixel footprint of the source model.

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def base_footprint(self, pixels):
        """
        Return the base footprint of the source model.

        Parameters
        ----------
        pixels : int

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def process_final(self):
        """
        Don't know

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def write_fits(self, filename):
        """
        Write the results to a FITS file.

        Parameters
        ----------
        filename : str

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_map_2d(self):
        """
        Return a 2-D map.

        Returns
        -------
        Map
        """
        pass

    @abstractmethod
    def merge_accumulate(self, other):
        """
        Merge another source with this one.

        Parameters
        ----------
        other : AstroModel2D

        Returns
        -------
        None
        """
        pass

    @property
    def shape(self):
        """
        Return the data shape.

        Returns
        -------
        tuple (int)
        """
        return 0, 0  # Not implemented here.

    @property
    def size_x(self):
        """
        Return the size of the data in the x-direction.

        Returns
        -------
        int
        """
        return self.shape[1]

    @property
    def size_y(self):
        """
        Return the size of the data in the y-direction.

        Returns
        -------
        int
        """
        return self.shape[0]

    @abstractmethod
    def set_data_shape(self, shape):
        """
        Set the data shape.

        Parameters
        ----------
        shape : tuple (int)

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def covariant_points(self):
        """
        Return the covariant points.

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def add_points(self, frames, pixels, frame_gains, source_gains):
        """
        Add points to the source model.

        Parameters
        ----------
        frames : Frames
            The frames to add to the source model.
        pixels : ChannelGroup
            The channels (pixels) to add to the source model.
        frame_gains : numpy.ndarray (float)
            The gain values for all frames of shape (n_frames,).
        source_gains : numpy.ndarray (float)
            The channel source gains for all channels of shape (all_channels,).

        Returns
        -------
        mapping_frames : int
            The number of valid mapping frames added for the model.
        """
        pass

    @abstractmethod
    def sync_source_gains(self, frames, pixels, frame_gains, source_gains,
                          sync_gains):
        """
        Remove the source from all but the blind channels.

        This is the same as sync(exposure, pixel, index, fg, sourcegain,
        integration.sourcesyncgain).

        Parameters
        ----------
        frames : Frames
        pixels : Channels or ChannelGroup
        frame_gains : numpy.ndarray (float)
        source_gains : numpy.ndarray (float)
        sync_gains : numpy.ndarray (float)

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def calculate_coupling(self, integration, pixels, source_gains,
                           sync_gains):
        """
        Don't know

        Parameters
        ----------
        integration
        pixels
        source_gains
        sync_gains

        Returns
        -------
        None
        """
        pass

    def get_memory_footprint(self, pixels):
        """
        Return the memory footprint.

        Parameters
        ----------
        pixels : int

        Returns
        -------
        bytes : int
        """
        return (pixels * self.get_pixel_footprint()
                + self.base_footprint(pixels))

    def get_reduction_footprint(self, pixels):
        """
        Return the reduction memory footprint.

        The reduction memory footprint includes the composite map, plus one
        copy for each thread, plus the base image.

        Parameters
        ----------
        pixels : int

        Returns
        -------
        bytes : int
        """
        return (self.reduction.max_jobs + 1) * self.get_memory_footprint(
            pixels) + self.base_footprint(pixels)

    def pixels(self):
        """
        Return the number of pixels in the source model.

        Returns
        -------
        int
        """
        return int(np.prod(self.shape))

    def is_adding_to_master(self):
        """
        Don't know

        Returns
        -------
        bool
        """
        return False

    def get_reference(self):
        """
        Return the reference pixel of the source model WCS.

        Returns
        -------
        numpy.ndarray (float)
            The (x, y) reference position of the source model.
        """
        return self.reference

    def reset_processing(self):
        """
        Reset the source processing.

        Returns
        -------
        None
        """
        super().reset_processing()
        self.update_smoothing()

    def is_valid(self):
        """
        Return whether the source model is valid.

        Returns
        -------
        bool
        """
        return not self.is_empty()

    def get_default_file_name(self):
        """
        Return the default file name for the FITS source model output.

        Returns
        -------
        str
        """
        return os.path.join(self.reduction.work_path,
                            f'{self.get_source_name()}.fits')

    def get_core_name(self):
        """
        Return the core name for the source model.

        Returns
        -------
        str
        """
        if self.has_option('name'):
            name = os.path.expandvars(self.configuration.get_string('name'))
            if name.endswith('.fits'):
                name = name[:-5]
            return name
        return self.get_default_core_name()

    def create_from(self, scans, assign_scans=True):
        """
        Initialize model from scans.

        Sets the model scans to those provided, and the source model for each
        scan as this.  All integration gains are normalized to the first scan.
        If the first scan is non-sidereal, the system will be forced to an
        equatorial frame.

        Parameters
        ----------
        scans : list (Scan)
            A list of scans from which to create the model.
        assign_scans : bool, optional
            If `True`, assign the scans to this source model.  Otherwise,
            there will be no hard link between the scans and source model.

        Returns
        -------
        None
        """
        super().create_from(scans, assign_scans=assign_scans)
        log.info("\nInitializing Source Map.\n")

        projection = SphericalProjection.for_name(
            self.configuration.get_string('projection', default='gnomonic'))
        projection.set_reference(self.get_first_scan().get_position_reference(
            self.configuration.get_string('system', default='equatorial')))

        self.projection = projection
        self.set_size()

        if self.allow_indexing and self.configuration.get_bool('indexing'):
            self.index()

    def set_size(self):
        """
        Set the size of the source model.

        Returns
        -------
        None
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            map_range = self.search_corners()

        dx = (map_range.x[1] - map_range.x[0]).round(1)
        dy = (map_range.y[1] - map_range.y[0]).round(1)
        log.debug(f"Map range: {dx.value} x {dy}")

        delta = Coordinate2D(unit=self.info.instrument.get_size_unit())
        if self.has_option('grid'):
            delta_values = self.configuration.get_float_list('grid')
            if len(delta_values) == 1:
                delta.y = delta.x = delta_values[0]
            else:
                delta.set(delta_values[:2])
        else:
            delta.y = delta.x = self.info.instrument.get_point_size() / 5.0

        # Make the reference fall on pixel boundaries
        self.grid.set_resolution(delta)
        x_min, x_max = map_range.x
        y_min, y_max = map_range.y

        ref_x = 0.5 - np.round((x_min / delta.x).decompose().value)
        ref_y = 0.5 - np.round((y_min / delta.y).decompose().value)
        self.grid.reference_index = Coordinate2D([ref_x, ref_y])

        lower_corner_index = self.grid.offset_to_index(
            map_range[0], in_place=False)
        log.debug(f"near corner: {lower_corner_index}")
        upper_corner_index = self.grid.offset_to_index(
            map_range[1], in_place=False)
        log.debug(f"far corner: {upper_corner_index}")

        x_size = 1 + int(np.ceil(self.grid.reference_index.x
                                 + (x_max / delta.x).decompose().value))
        y_size = 1 + int(np.ceil(self.grid.reference_index.y
                                 + (y_max / delta.y).decompose().value))

        log.debug(f"Map pixels: {x_size} x {y_size} (nx, ny)")
        if x_size < 0 or y_size < 0:
            raise ValueError(f"Negative image size: {x_size} x {y_size}")

        if not self.has_option('large'):
            if (x_size >= self.MAX_X_OR_Y_SIZE
                    or y_size >= self.MAX_X_OR_Y_SIZE):
                raise ValueError("Map too large.  Use 'large' option.")

        self.set_data_shape((y_size, x_size))

    def search_corners(self):
        """
        Determine the extent of the source model.

        If the map range is specified in the configuration via 'map.size',
        this will be returned, but also result in integration frame samples
        being flagged with the SAMPLE_SKIP flag if outside this range.

        Otherwise, the map range will be determined from the span of sample
        coordinates projected onto the model grid over all scans and
        integrations in the model.  An attempt will be made to determine this
        using the perimeter pixels (near the edge of the detector array).

        This will also set the scan map range for each scan if 'map.size' is
        not set.

        Returns
        -------
        map_range : astropy.units.Quantity (numpy.ndarray)
            An array of shape (4,) containing [min(x), min(y), max(x), max(y)]
            in units of arcseconds.
        """
        map_range = Coordinate2D(unit='arcsec')
        fix_size = self.has_option('map.size')

        n_integrations = 0
        integrations = []
        scan_integration_map = []
        for scan in self.scans:
            integration_map = []
            for integration in scan.integrations:
                integration_map.append(n_integrations)
                integrations.append(integration)
                n_integrations += 1
            scan_integration_map.append(integration_map)

        if self.reduction.is_sub_reduction:
            jobs = self.reduction.parent_reduction.available_reduction_jobs
        else:
            jobs = self.reduction.available_reduction_jobs

        if fix_size:
            map_size = self.configuration.get_float_list(
                'map.size', delimiter='[ \t,:xX]', default=[])
            map_width = np.asarray(map_size) * 0.5
            map_range.x = -map_width[0], map_width[0]
            map_range.y = -map_width[1], map_width[1]

            args = integrations, self.projection, map_range
            kwargs = None
            multiprocessing.multitask(
                self.parallel_safe_flag_outside, range(n_integrations),
                args, kwargs, jobs=jobs, max_nbytes=None, force_threading=True,
                logger=log)

        else:

            args = integrations, self.projection
            kwargs = None
            integration_ranges = multiprocessing.multitask(
                self.parallel_safe_integration_range, range(n_integrations),
                args, kwargs, jobs=jobs, max_nbytes=None, force_threading=True,
                logger=log)

            map_range.x = np.inf, -np.inf
            map_range.y = np.inf, -np.inf
            scan_range = Coordinate2D(unit='arcsec')

            for scan_index, scan in enumerate(self.scans):
                scan_range.x = np.inf, -np.inf
                scan_range.y = np.inf, -np.inf
                for integration_index, integration in enumerate(
                        scan.integrations):
                    integration_number = scan_integration_map[
                        scan_index][integration_index]
                    min_x, max_x, min_y, max_y = integration_ranges[
                        integration_number]

                    for range_array in [scan_range, map_range]:
                        if min_x < range_array.x[0]:
                            range_array.x[0] = min_x
                        if min_y < range_array.y[0]:
                            range_array.y[0] = min_y
                        if max_x > range_array.x[1]:
                            range_array.x[1] = max_x
                        if max_y > range_array.y[1]:
                            range_array.y[1] = max_y

                scan.range = scan_range

        return map_range

    @classmethod
    def parallel_safe_integration_range(cls, args, integration_number):
        """
        Return the range of the integration coordinates for all integrations.

        Parameters
        ----------
        args : 2-tuple
            args[0] = list of integrations (list (Integration)).
            args[1] = projection (Projection2D).
        integration_number : int
            The integration number for which to find the range.

        Returns
        -------
        integration_range : 4-tuple
            A tuple of the for (min_x, max_x, min_y, max_y).
        """
        integrations, projection = args
        integration = integrations[integration_number]
        channels = integration.channels
        perimeter_pixels = channels.get_perimeter_pixels()
        integration_range = integration.search_corners(
            perimeter_pixels, projection)
        min_x, max_x = integration_range.x
        min_y, max_y = integration_range.y
        return min_x, max_x, min_y, max_y

    @classmethod
    def parallel_safe_flag_outside(cls, args, integration_number):
        """
        Flag points that are outside specified map range for an integration.

        This function is safe for use with :func:`multiprocessing.multitask`.

        Parameters
        ----------
        args : 3-tuple
            args[0] = integrations (list (Integration))
            args[1] = projection (Projection2D)
            args[2] = map_range (Coordinate2D)
        integration_number : int
            The index of the integration to flag in args[0].

        Returns
        -------
        None
        """
        integrations, projection, map_range = args
        integration = integrations[integration_number]
        cls.flag_outside(projection, integration, map_range)

    @classmethod
    def flag_outside(cls, projection, integration, map_range):
        """
        Flag points that are outside of the specified map range.

        Parameters
        ----------
        projection : Projection2D
        integration : Integration
        map_range : Coordinate2D
            The map range containing the minimum (x, y) and maximum (x, y)
            coordinates of the map of shape (n_frames, n_channels).

        Returns
        -------
        None
        """
        channels = integration.channels.get_mapping_pixels(
            discard_flag=integration.channels.flagspace.sourceless_flags())

        projector = AstroProjector(projection)
        integration.frames.project(channels.data.position, projector)
        map_range.change_unit(projector.offset.unit)

        skip_flag = integration.flagspace.flags.SAMPLE_SKIP
        snf.flag_outside(
            sample_coordinates=projector.offset.coordinates.value,
            valid_frames=integration.frames.valid,
            channel_indices=channels.indices,
            sample_flags=integration.frames.sample_flag,
            skip_flag=skip_flag.value,
            map_range=map_range.coordinates.value)

    def index(self):
        """
        Store the map indices for fast lookup later.

        Returns
        -------
        None
        """
        max_usage = self.configuration.get_float(
            'indexing.saturation', default=0.5)
        log.debug(f"Indexing maps (up to {100 * max_usage}% "
                  f"of RAM saturation).")

        max_available = (psutil.virtual_memory().total
                         - self.get_reduction_footprint(self.pixels()))

        max_used = int(max_available * max_usage)
        for scan in self.scans:
            for integration in scan.integrations:
                if psutil.virtual_memory().used > max_used:
                    return
                self.create_lookup(integration)

    def create_lookup(self, integration):
        """
        Create the source indices for integration frames.

        The source indices contain 1-D lookup values for the pixel indices
        of a sample on the source model.  The map indices are stored in the
        integration frames as the 1-D `source_index` attribute, and the
        2-D `map_index` attribute.

        Parameters
        ----------
        integration : Integration
            The integration for which to create source indices.

        Returns
        -------
        None
        """
        pixels = integration.channels.get_mapping_pixels(
            discard_flag=integration.channel_flagspace.sourceless_flags())
        log.debug(f"lookup.pixels {pixels.size} : {integration.channels.size} "
                  f"(source pixels : total channels)")
        self.index_shift_x = numba_functions.log2ceil(self.size_y)
        self.index_mask_y = (1 << self.index_shift_x) - 1

        frames = integration.frames
        channels = integration.channels

        if frames.source_index is None:
            frames.source_index = np.full((frames.size, channels.size), -1)
        else:
            frames.source_index.fill(-1)

        if frames.map_index is None:
            frames.map_index = Index2D(
                np.full((2, frames.size, channels.size), -1))
        else:
            frames.map_index.coordinates.fill(-1)

        projector = AstroProjector(self.projection)
        offsets = frames.project(pixels.position, projector)
        map_indices = self.grid.offset_to_index(offsets)
        map_indices.coordinates = np.round(map_indices.coordinates).astype(int)

        bad_samples = snf.validate_pixel_indices(
            indices=map_indices.coordinates,
            x_size=self.size_x,
            y_size=self.size_y,
            valid_frame=frames.valid)

        if bad_samples > 0:
            log.warning(f"{bad_samples} samples have bad map indices")

        frames.map_index.coordinates[..., pixels.indices] = (
            map_indices.coordinates)
        source_indices = self.pixel_index_to_source_index(
            map_indices.coordinates)
        frames.source_index[..., pixels.indices] = source_indices

    def pixel_index_to_source_index(self, pixel_indices):
        """
        Return the 1-D source index for a pixel index.

        Parameters
        ----------
        pixel_indices : numpy.ndarray (int)
            The pixel indices of shape (2, n_frames, n_channels) containing
            the (x_index, y_index) pixel indices.

        Returns
        -------
        source_indices : numpy.ndarray (int)
            The 1-D source indices of shape (n_frames, n_channels).
        """
        return (pixel_indices[0] << self.index_shift_x) | pixel_indices[1]

    def source_index_to_pixel_index(self, source_indices):
        """
        Convert 1-D source indices to 2-D pixel indices.

        Parameters
        ----------
        source_indices : numpy.ndarray (int)
            The source indices of shape (n_frames, n_channels).

        Returns
        -------
        pixel_indices : numpy.ndarray (int)
            The pixel indices of shape (n_frames, n_channels, 2) containing the
            (x_index, y_index) pixel indices.
        """
        px = source_indices >> self.index_shift_x
        py = source_indices & self.index_mask_y

        invalid = np.nonzero(source_indices < 0)
        px[invalid] = -1
        py[invalid] = -1
        return np.stack((px, py), axis=-1)

    def find_outliers(self, max_distance):
        """
        Find and return outlier scans based on distance from median position.

        Returns all scans that deviate from the median equatorial position
        of all scans by >= `max_distance`.

        Parameters
        ----------
        max_distance : astropy.units.Quantity
            The maximum angular separation from a scan from the median
            equatorial position of all scans.

        Returns
        -------
        outlier_scans : list (Scan)
            A list of outlier scans.
        """
        if self.n_scans == 1:
            return []  # Can't check this

        ra = np.empty(self.n_scans, dtype=float
                      ) * self.grid.get_coordinate_unit()
        dec = np.empty(self.n_scans, dtype=float
                       ) * self.grid.get_coordinate_unit()

        scan_equatorial_list = []
        for scan_index, scan in enumerate(self.scans):
            equatorial = scan.equatorial.copy().transform_to(
                FK5(equinox='J2000'))
            scan_equatorial_list.append(equatorial)
            ra[scan_index] = equatorial.ra
            dec[scan_index] = equatorial.dec

        median_ra = np.median(ra)
        median_dec = np.median(dec)

        check = SkyCoord(
            ra=median_ra, dec=median_dec, equinox='J2000')

        outlier_scans = []
        for scan_index, equatorial in enumerate(scan_equatorial_list):
            if abs(check.separation(equatorial)) > max_distance:
                outlier_scans.append(self.scans[scan_index])

        return outlier_scans

    def find_slewing(self, max_distance):
        """
        Return all scans that have a slew greater than a certain distance.

        Parameters
        ----------
        max_distance : astropy.units.Quantity
            The maximum angular slew.  Any scans that have a map span greater
            than this limit will be added to the output scan list.

        Returns
        -------
        slewing_scans : list (Scan)
            The scans with a slew greater than `max_distance`.
        """
        slews = []
        cos_lat = np.cos(self.grid.reference.data.lat).value
        for scan in self.scans:
            dx = scan.map_range[2] - scan.map_range[0]
            dy = scan.map_range[3] - scan.map_range[1]
            slew_distance = np.hypot(dx * cos_lat, dy)
            if slew_distance > max_distance:
                slews.append(scan)

        return slews

    def add_integration(self, integration, signal_mode=None):
        """
        Add an integration to the source model.

        Parameters
        ----------
        integration : Integration
        signal_mode : str or int or FrameFlagTypes, optional
            The signal mode for which to extract source gains from integration
            frames.  Typically, TOTAL_POWER.

        Returns
        -------
        None
        """
        if signal_mode is None:
            signal_mode = self.signal_mode

        # For jackknifed maps, indicate sign.
        if self.id not in [None, '']:
            integration.comments.append(f'Map.{self.id}')
        else:
            integration.comments.append('Map')

        # Proceed only if there are enough pixels to do the job
        if not self.check_pixel_count(integration):
            return

        # Calculate the effective source NEFD based on the latest weights and
        # the current filtering.
        integration.calculate_source_nefd()

        # For the first source generation, apply the point source correction
        # directly to the signals.
        average_filtering = integration.channels.get_average_filtering()

        signal_correction = integration.source_generation == 0

        mapping_frames = self.add_frames_from_integration(
            integration=integration,
            pixels=integration.channels.get_mapping_pixels(keep_flag=0),
            source_gains=integration.channels.get_source_gains(
                filter_corrected=signal_correction),
            signal_mode=signal_mode)

        log.debug(f"Mapping frames: {mapping_frames} --> "
                  f"map points: {self.count_points()}")

        if signal_correction:
            if average_filtering != 0:
                integration.comments.append(f'[C~{1 / average_filtering:.2f}]')
            else:
                integration.comments.append(f'[C~{np.inf}]')

        integration.comments.append(' ')

    def add_frames_from_integration(self, integration, pixels, source_gains,
                                    signal_mode=None):
        """
        Add frames from an integration to the source model.

        Parameters
        ----------
        integration : Integration
            The integration to add.
        pixels : ChannelGroup
            The channels (pixels) to add to the source model.
        source_gains : numpy.ndarray (float)
            The source gains for the all channels (pixels).  Should be of
            shape (all_channels,).
        signal_mode : FrameFlagTypes, optional
            The signal mode flag, indicating which signal should be used to
            extract the frame source gains.  Typically, TOTAL_POWER and will
            be taken as the signal mode of this map if not supplied.

        Returns
        -------
        mapping_frames : int
            The number of frames that contributed towards mapping.
        """
        log.debug(f"add.pixels {pixels.size} : {integration.channels.size}")

        if self.is_adding_to_master():
            source_model = self
        else:
            source_model = self.get_recycled_clean_local_copy()

        if integration.frames.map_index is None or np.allclose(
                integration.frames.map_index.coordinates, -1):
            source_model.create_lookup(integration)

        if signal_mode is None:
            signal_mode = self.signal_mode

        frames = integration.frames
        frame_gains = integration.gain * frames.get_source_gain(signal_mode)

        mapping_frames = source_model.add_points(
            frames, pixels, frame_gains, source_gains)

        if not self.is_adding_to_master():
            self.merge_accumulate(source_model)

        return mapping_frames

    def add_pixels_from_integration(self, integration, pixels, source_gains,
                                    signal_mode):
        """
        Add pixels to the source model.

        As far as I can tell, this does exactly the same thing as `add_frames`.

        Parameters
        ----------
        integration : Integration
            The integration to add.
        pixels : ChannelGroup
            The channels (pixels) to add to the source model.
        source_gains : numpy.ndarray (float)
            The source gains for the all channels (pixels).  Should be of
            shape (all_channels,).
        signal_mode : FrameFlagTypes
            The signal mode flag, indicating which signal should be used to
            extract the frame source gains.  Typically, TOTAL_POWER.

        Returns
        -------
        mapping_frames : int
            The number of frames that contributed towards mapping.
        """
        return self.add_frames_from_integration(
            integration, pixels, source_gains, signal_mode)

    def get_sample_points(self, frames, pixels, frame_gains, source_gains):
        """
        Return the sample gains for integration frames and given pixels.

        Parameters
        ----------
        frames : Frames
            The integration frames for which to generate sample data.
        pixels : ChannelGroup
            The channel group for which to derive sample gains.  The size of
            the group should be n_channels.
        frame_gains : numpy.ndarray (float)
            The gain values for all frames of shape (n_frames,).
        source_gains : numpy.ndarray (float)
            The channel source gains for all channels of shape (all_channels,).

        Returns
        -------
        mapping_frames, data gains, weights, indices : 5-tuple (int, array+)
            The total number of mapping frames and the cross product of
            integration frame gains and source gains as an array of shape
            (n_frames, n_channels).  Any invalid frames/sample flags will
            result in a NaN gain value for the sample in question.  Invalid
            indices will be represented by -1 values, and invalid weights will
            be zero-valued.  The shape of the output indices will be
            (n_dimensions, n_frames, n_channels).
        """
        n, data, gains, weights, indices = snf.get_sample_points(
            frame_data=frames.data,
            frame_gains=frame_gains,
            frame_weights=frames.relative_weight,
            source_gains=source_gains,
            channel_variance=pixels.variance,
            valid_frames=frames.is_unflagged('SOURCE_FLAGS') & frames.valid,
            map_indices=frames.map_index.coordinates,
            channel_indices=pixels.indices,
            sample_flags=frames.sample_flag,
            exclude_sample_flag=self.exclude_samples.value)

        return n, data, gains, weights, indices

    def set_sync_gains(self, integration, pixels, source_gains):
        """
        Set the sync gains for the source model (set to current source gains).

        Parameters
        ----------
        integration : Integration
        pixels : ChannelGroup
            The channels (pixels) for which to set sync gains.
        source_gains : numpy.ndarray (float)
            The source gains to set as sync gains.  Should be of shape
            (all_channels,).

        Returns
        -------
        None
        """
        if (integration.source_sync_gain is None
                or integration.source_sync_gain.size != source_gains.size):
            integration.source_sync_gain = np.zeros(
                source_gains.size, dtype=float)

        integration.source_sync_gain[pixels.indices] = source_gains[
            pixels.indices]

    def sync_integration(self, integration, signal_mode=None):
        """
        Remove source from integration frame data.

        Parameters
        ----------
        integration : Integration
        signal_mode : FrameFlagTypes, optional
            The signal mode flag, indicating which signal should be used to
            extract the frame source gains.  Typically, TOTAL_POWER.

        Returns
        -------
        None
        """
        if signal_mode is None:
            signal_mode = self.signal_mode

        source_gains = integration.channels.get_source_gains(
            filter_corrected=False)

        if (integration.source_sync_gain is None
                or integration.source_sync_gain.size != source_gains.size):
            integration.source_sync_gain = np.zeros(
                source_gains.size, dtype=float)

        pixels = integration.channels.get_mapping_pixels(
            discard_flag=integration.channel_flagspace.sourceless_flags())

        if self.has_option('source.coupling'):
            self.calculate_coupling(
                integration=integration,
                pixels=pixels,
                source_gains=source_gains,
                sync_gains=integration.source_sync_gain)
            # Get updated source gains
            source_gains = integration.channels.get_source_gains(
                filter_corrected=False)

        self.sync_pixels(
            integration=integration,
            pixels=pixels,
            source_gains=source_gains,
            signal_mode=signal_mode)

        # Do an approximate accounting of the source dependence.
        n_points = min(self.count_points(), integration.scan.source_points)
        n_points /= self.covariant_points()
        frames = integration.frames

        parms = integration.get_dependents('source')

        frame_dp, pixel_dp = snf.get_delta_sync_parms(
            channel_source_gains=source_gains,
            channel_indices=pixels.indices,
            channel_flags=pixels.flag,
            channel_variance=pixels.variance,
            frame_weight=frames.relative_weight,
            frame_source_gains=frames.get_source_gain(signal_mode),
            frame_valid=frames.valid,
            frame_flags=frames.flag,
            source_flag=frames.flagspace.convert_flag('SOURCE_FLAGS').value,
            n_points=n_points)

        if integration.configuration.get_bool(
                'crushbugs'):  # TODO: remove once resolved

            # Only the first channel is applied in the clear operation
            i0 = pixels.indices[0]
            p0 = parms.for_channel[pixels.indices[0]]
            parms.for_channel.fill(0.0)
            parms.for_channel[i0] = p0
            parms.clear(channels=pixels, start=0, end=integration.size)

            # Only the last channel is applied in the apply operation
            i1 = pixels.indices[-1]
            parms.for_channel[i1] = pixel_dp[-1]

            # The frame parms are applied multiple times: once for each pixel.
            parms.for_frame[...] = frame_dp * pixels.size
            parms.apply(channels=pixels)

            # Put in the actual correct frame parms
            parms.for_frame[...] = frame_dp

        else:

            # The correct way to do things...
            parms.clear(channels=pixels, start=0, end=integration.size)
            parms.add_async(pixels, pixel_dp)
            parms.add_async(frames, frame_dp)
            parms.apply(channels=pixels)

        self.set_sync_gains(integration=integration,
                            pixels=pixels,
                            source_gains=source_gains)

    def sync_pixels(self, integration, pixels, source_gains, signal_mode=None):
        """
        Remove the source from pixels.

        Parameters
        ----------
        integration : Integration
        pixels : ChannelGroup
        source_gains : numpy.ndarray (float)
            The integration channel source gains of shape (all_channels,)
        signal_mode : FrameFlagTypes, optional
            The signal mode flag, indicating which signal should be used to
            extract the frame source gains.  Typically, TOTAL_POWER.

        Returns
        -------
        None
        """
        if signal_mode is None:
            signal_mode = self.signal_mode

        log.debug(f"sync.pixels {pixels.size} : {integration.channels.size}")

        frames = integration.frames
        if frames.map_index is None:
            self.create_lookup(integration)
        # grid_indices = self.get_index(integration.frames, pixels, self.grid)

        frame_gains = integration.gain * frames.get_source_gain(signal_mode)

        # Remove source from all but blind channels
        self.sync_source_gains(
            frames=integration.frames,
            pixels=pixels,
            frame_gains=frame_gains,
            source_gains=source_gains,
            sync_gains=integration.source_sync_gain)

    def get_table_entry(self, name):
        """
        Return a parameter value for a given name.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        value
        """
        if name == 'smooth':
            if not isinstance(self.smoothing, units.Quantity):
                return self.smoothing
            return self.smoothing.to(self.info.size_unit)
        if name == 'system':
            if not isinstance(self.grid, SphericalGrid):
                return None
            if self.grid.reference is None:
                return None
            return self.grid.reference.two_letter_code
        return super().get_table_entry(name)

    def write(self, path):
        """
        Write the source to file.

        Parameters
        ----------
        path : str
            The file path to write to.

        Returns
        -------
        None
        """
        # Remove the intermediate image file
        intermediate = os.path.join(path, f'intermediate.{self.id}.fits')
        if os.path.isfile(intermediate):
            os.remove(intermediate)

        if self.id not in [None, '']:
            file_name = os.path.join(
                path, f'{self.get_core_name()}.{self.id}.fits')
        else:
            file_name = os.path.join(path, f'{self.get_core_name()}.fits')

        if self.is_empty():
            source_name = ((self.id + ' ')
                           if self.id not in [None, ''] else '')
            log.warning(f"Source {source_name} is empty. Skipping")
            if os.path.isfile(file_name):
                os.remove(file_name)
            return

        self.process_final()
        self.write_fits(file_name)
        if self.configuration.get_bool('write.png'):
            self.write_png(self.get_map_2d(), file_name)

    def write_png(self, map_2d, file_name):
        """
        Write a PNG of the map.

        Parameters
        ----------
        map_2d : Fits2D
        file_name : str
            The file path to write the PNG to.

        Returns
        -------
        None
        """
        if not self.configuration.get_bool('write.png'):
            return

        smooth = self.configuration.get_string('write.png.smooth')
        if smooth is not None:
            fwhm = self.get_smoothing(smooth)
            if fwhm == 0:
                fwhm = 0.5 * self.info.instrument.get_point_size()
            map_2d.smooth_to(fwhm)

        if self.has_option('write.png.crop'):

            if self.configuration.get_string('write.png.crop') == 'auto':
                offsets = []
            else:
                offsets = self.configuration.get_float_list(
                    'write.png.crop', default=[])

            n = len(offsets)
            if n == 0 or np.isnan(offsets[0]):
                map_2d.auto_crop()
                return

            if n >= 4:
                dx_min, dy_min, dx_max, dy_max = offsets[:4]
            elif n == 1:
                dx_max = dy_max = offsets[0]
                dx_min = dy_min = -dx_max
            elif n == 2:
                dx_max, dy_max = offsets[0]
                dx_min, dy_min = -dx_max, -dy_max
            elif n == 3:
                dx_min, dy_min, dx_max = offsets
                dy_max = -dy_min
            else:
                map_2d.auto_crop()
                return

            size_unit = self.info.instrument.get_size_unit()
            map_2d.crop(dx_min * size_unit,
                        dy_min * size_unit,
                        dx_max * size_unit,
                        dy_max * size_unit)

        if isinstance(map_2d, Observation2D) and self.has_option(
                'write.png.plane'):
            plane = self.configuration.get_string('write.png.plane').lower()
            if plane in ['s2n', 's/n']:
                values = map_2d.get_significance()
            elif plane == 'time':
                values = map_2d.get_exposures()
            elif plane in ['noise', 'rms']:
                values = map_2d.get_noise()
            elif plane == 'weight':
                values = map_2d.get_weights()
            else:
                values = map_2d
        else:
            values = map_2d

        self.write_image_png(values.get_image(), file_name)

    def write_image_png(self, image, file_name):
        """
        Write a PNG of an image.

        Parameters
        ----------
        image : Image
        file_name : str
           The file path to write the PNG to.

        Returns
        -------
        None
        """
        try:
            import matplotlib.pyplot as plt
        except Exception as err:
            log.warning(f"Could not import matplotlib: will not create png: "
                        f"{err}")
            return

        width = height = self.DEFAULT_PNG_SIZE
        if self.has_option('write.png.size'):
            sizes = self.configuration.get_int_list(
                'write.png.size', delimiter='[ \t,:xX\\*]', default=[])
            if len(sizes) > 0:
                width = sizes[0]
                height = width if len(sizes) == 1 else sizes[1]

        if not self.has_option('write.png.crop'):
            image = image.copy()
            image.auto_crop()

        if not file_name.endswith('.png'):
            png_file = f"{file_name}.png"
        else:
            png_file = file_name

        if self.configuration.has_option('write.png.color'):
            color = self.configuration.get_string('write.png.color')
        else:
            color = None

        interactive = plt.isinteractive()
        if interactive:
            plt.ioff()
        fig = plt.figure(frameon=False)
        # Set a max size of 10 inches
        dimensions = np.asarray([width, height])
        # Use a dpi of 100
        dpi = 100
        fig_size = (dimensions / dpi).astype(int)
        fig.set_size_inches(fig_size)
        data = image.data.copy()
        data[~image.valid] = np.nan
        plt.imshow(data, cmap=color)
        plt.savefig(png_file, dpi=dpi, format='png')
        log.info(f"Saved png image to {png_file}")
        plt.close()
        if interactive:
            plt.ion()

    def get_smoothing(self, smooth):
        """
        Get the smoothing FWHM given a smoothing specification.

        Either a float value or one of the following strings may be supplied:

        'beam': 1 * instrument FWHM
        'halfbeam': 0.5 * instrument FWHM
        '2/3beam': instrument FWHM / 1.5
        'minimal': 0.3 * instrument FWHM
        'optimal': either (1 * instrument FWHM), or the smooth.optimal value

        If the pixelization FWHM is greater than the calculated FWHM, it will
        be returned instead.  The pixelization FWHM is defined as:

        pix_FWHM = sqrt(pixel_area * (8 ln(2)) / 2pi)

        Parameters
        ----------
        smooth : str or float
            The FWHM for which to smooth to in units of the instrument size
            unit (float), or one of the following string: {'beam', 'halfbeam',
            '2/3beam', 'minimal', 'optimal'}.

        Returns
        -------
        FWHM : astropy.units.Quantity
            The FWHM of the Gaussian smoothing kernel.
        """
        size_unit = self.info.instrument.get_size_unit()
        beam = self.info.instrument.get_point_size()
        pixel_smoothing = self.get_pixelization_smoothing()
        if smooth == 'beam':
            fwhm = beam
        elif smooth == 'halfbeam':
            fwhm = 0.5 * beam
        elif smooth == '2/3beam':
            fwhm = beam / 1.5
        elif smooth == 'minimal':
            fwhm = 0.3 * beam
        elif smooth == 'optimal':
            optimal = self.configuration.get_float('smooth.optimal',
                                                   default=np.nan)
            fwhm = beam if np.isnan(optimal) else (optimal * size_unit)
        else:
            try:
                fwhm = float(smooth) * size_unit
            except (TypeError, ValueError):
                fwhm = 0 * size_unit

        if pixel_smoothing > fwhm:
            fwhm = pixel_smoothing
        return fwhm.to(size_unit)

    def set_smoothing(self, smoothing):
        """
        Set the model smoothing.

        Parameters
        ----------
        smoothing : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.smoothing = smoothing

    def update_smoothing(self):
        """
        Update the model smoothing from the configuration.

        Returns
        -------
        None
        """
        if not self.has_option('smooth'):
            return
        self.set_smoothing(self.get_smoothing(
            self.configuration.get_string('smooth')))

    def get_requested_smoothing(self, smooth_spec=None):
        """
        Get the requested smoothing for a given specification.

        Parameters
        ----------
        smooth_spec : str or float, optional
            The type of smoothing to retrieve.

        Returns
        -------
        fwhm : astropy.units.Quantity
            The FWHM of the Gaussian smoothing kernel.
        """
        if smooth_spec in [None, '']:
            return self.smoothing
        return self.get_smoothing(smooth_spec)

    def get_pixelization_smoothing(self):
        """
        Return the pixelation smoothing FWHM.

        The returned value is:

        sqrt(pixel_area * (8 ln(2)) / 2pi)

        Returns
        -------
        pixel_fwhm : astropy.units.Quantity
        """
        # Used to convert FWHM^2 to beam integral
        gaussian_area = 2 * np.pi * (gaussian_fwhm_to_sigma ** 2)
        return np.sqrt(self.grid.get_pixel_area() / gaussian_area)

    def get_point_size(self):
        """
        Return the point size of the source model.

        Returns
        -------
        astropy.units.Quantity
        """
        smoothing = self.get_requested_smoothing(
            smooth_spec=self.configuration.get_string('smooth'))
        return np.hypot(self.info.instrument.get_point_size(), smoothing)

    def get_source_size(self):
        """
        Return the source size of the source model.

        Returns
        -------
        astropy.units.Quantity
        """
        smoothing = self.get_requested_smoothing(
            smooth_spec=self.configuration.get_string('smooth'))
        return np.hypot(super().get_source_size(), smoothing)
