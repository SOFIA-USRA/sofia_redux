# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
import warnings

from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_1d import Coordinate1D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.source_models.maps.image_2d1 import Image2D1
from sofia_redux.scan.source_models.maps.observation_2d1 import Observation2D1
from sofia_redux.scan.coordinate_systems.index_3d import Index3D
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector
from sofia_redux.scan.coordinate_systems.grid.spherical_grid_2d1 import \
    SphericalGrid2D1
from sofia_redux.scan.utilities.utils import round_values
from sofia_redux.scan.source_models import source_numba_functions as snf

__all__ = ['SpectralCube']


class SpectralCube(AstroIntensityMap):
    default_map_class = Observation2D1
    default_image_class = Image2D1
    MAX_Z_SIZE = 5000
    DEFAULT_GRID_CLASS = SphericalGrid2D1

    def __init__(self, info, reduction=None):
        """
        Initialize a spectral cube.

        The spectral cube map represents the source as an
        :class:`Observation2D1` map containing data, noise, and exposure
        values. It also contains a base image containing the results of the
        previous reduction iteration in order to calculate gain and coupling
        increments.  The (x, y) coordinates are spatial, and the 3rd z
        coordinate is along the spectral dimension in which all (x, y) planes
        are equivalent.

        Parameters
        ----------
        info : sofia_redux.scan.info.info.Info
            The Info object which should belong to this source model.
        reduction : sofia_redux.scan.reduction.reduction.Reduction, optional
            The reduction for which this source model should be applied.
        """
        super().__init__(info, reduction=reduction)
        xy_unit = self.info.instrument.get_size_unit()
        z_unit = self.info.instrument.get_spectral_unit()

        self.smoothing = Coordinate2D1([0.0 * xy_unit,
                                        0.0 * xy_unit,
                                        0.0 * z_unit])

    def copy(self, with_contents=True):
        """
        Return a copy of the source model.

        Parameters
        ----------
        with_contents : bool, optional
            If `True`, return a true copy of the map.  Otherwise, just return
            a map with basic metadata.

        Returns
        -------
        SpectralCube
        """
        return super().copy(with_contents=with_contents)

    def create_grid(self):
        """
        Create the grid instance for the spectral cube model.

        Returns
        -------
        None
        """
        self.grid = self.DEFAULT_GRID_CLASS()
        if self.has_option('grid'):
            resolution = self.configuration.get_float_list('grid')
        else:
            resolution = None

        grid_resolution = self.info.instrument.resolution.copy()
        xy_unit = self.info.instrument.get_size_unit()
        z_unit = self.info.instrument.get_spectral_unit()
        if resolution is None or len(resolution) == 0:
            grid_resolution.scale(0.2)
        elif len(resolution) == 1:
            grid_resolution.x = grid_resolution.y = resolution[0] * xy_unit
            grid_resolution.z_coordinates.scale(0.2)
        elif len(resolution) == 2:
            grid_resolution.x = grid_resolution.y = resolution[0] * xy_unit
            grid_resolution.z = resolution[1] * z_unit
        else:
            grid_resolution.x, grid_resolution.y = resolution[:2] * xy_unit
            grid_resolution.z = resolution[2] * z_unit
        self.grid.set_resolution(grid_resolution)

    @property
    def reference(self):
        """
        Return the reference coordinate for the source model grid.

        Returns
        -------
        Coordinate2D1
            The reference coordinates for the source model grid.
        """
        return super().reference

    @reference.setter
    def reference(self, value):
        """
        Set the reference coordinate for the source model grid.

        Parameters
        ----------
        value : Coordinate2D1 or None
            The new reference coordinates for the source model grid.

        Returns
        -------
        None
        """
        if self.grid is None:
            return
        if value is not None and not isinstance(value, Coordinate2D1):
            raise ValueError(f"Reference coordinates must be "
                             f"{Coordinate2D1}")

        xy = value.xy_coordinates
        if xy is not None and not isinstance(xy, SphericalCoordinates):
            raise ValueError(f"Reference xy coordinates must be "
                             f"{SphericalCoordinates}.")

        self.grid.reference = value

    @property
    def size_x(self):
        """
        Return the size of the data in the x-direction.

        Returns
        -------
        int
        """
        return self.shape[2]

    @property
    def size_y(self):
        """
        Return the size of the data in the y-direction.

        Returns
        -------
        int
        """
        return self.shape[1]

    @property
    def size_z(self):
        """
        Return the size of the data in the z-direction.

        Returns
        -------
        int
        """
        return self.shape[0]

    def get_reference(self):
        """
        Return the reference position of the source model WCS.

        Returns
        -------
        Coordinate2D1
            The reference position of the source model.
        """
        return self.reference

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
        ref_z = self.get_first_scan().info.instrument.wavelength
        self.grid.reference.z = ref_z

    def create_map(self):
        """
        Create the source model map.

        Returns
        -------
        None
        """
        super().create_map()
        self.grid.z.reference = self.info.instrument.wavelength
        self.map = self.default_map_class()
        self.map.set_grid(self.grid)
        self.map.set_validating_flags(~self.mask_flag)
        self.map.add_local_unit(self.get_native_unit())
        self.map.set_display_grid_unit(self.info.instrument.get_size_unit())
        self.map.fits_properties.set_instrument_name(
            self.info.instrument.name)
        self.map.fits_properties.set_copyright(
            self.map.fits_properties.default_copyright)
        if self.reduction is not None:
            self.map.set_parallel(self.reduction.max_jobs)
            self.map.fits_properties.set_creator_name(
                self.reduction.__class__.__name__)

    def set_size(self):
        """
        Set the size of the source model.

        Determines the grid resolution, the size of the grid dimensions in
        (x, y) pixels, and the reference index of the grid.  The grid
        resolution is determined from the 'grid' keyword value in the
        configuration or one fifth of the instrument point size.  The map
        size is determined from the span of coordinates over all scan
        integrations.

        Returns
        -------
        None
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            map_range = self.search_corners()

        dx = (map_range.x[1] - map_range.x[0]).round(1)
        dy = (map_range.y[1] - map_range.y[0]).round(1)
        dz = (map_range.z[1] - map_range.z[0]).round(1)
        log.debug(f"Map range: ({dx.value}x{dy}) x {dz}")

        xy_unit = self.info.instrument.get_size_unit()
        z_unit = self.info.instrument.get_spectral_unit()

        delta = Coordinate2D1(xy_unit=xy_unit, z_unit=z_unit)
        resolution = self.configuration.get_float_list('grid', default=None)
        if resolution is None or len(resolution) == 0:
            point_size = self.info.instrument.get_point_size()
            rx = 0.2 * point_size.x
            ry = 0.2 * point_size.y
            rz = 0.2 * point_size.z
        elif len(resolution) == 1:
            rx = ry = resolution[0] * xy_unit
            rz = 0.2 * self.info.instrument.resolution.z
        elif len(resolution) == 2:
            rx = ry = resolution[0] * xy_unit
            rz = resolution[1] * z_unit
        else:
            rx, ry = resolution[:2] * xy_unit
            rz = resolution[2] * z_unit
        delta.set([rx, ry, rz])

        # Make the reference fall on pixel boundaries
        self.grid.set_resolution(delta)
        x_min, x_max = map_range.x
        y_min, y_max = map_range.y
        z_min, z_max = map_range.z

        ref_x = 0.5 - round_values((x_min / delta.x).decompose().value)
        ref_y = 0.5 - round_values((y_min / delta.y).decompose().value)
        ref_z = 0.5 - round_values((z_min / delta.z).decompose().value)
        self.grid.reference_index = Coordinate2D1([ref_x, ref_y, ref_z])

        lower_corner_index = self.grid.offset_to_index(
            map_range.min, in_place=False)
        log.debug(f"near corner: {lower_corner_index}")
        upper_corner_index = self.grid.offset_to_index(
            map_range.max, in_place=False)
        log.debug(f"far corner: {upper_corner_index}")

        x_size = 1 + int(np.ceil(self.grid.reference_index.x
                                 + (x_max / delta.x).decompose().value))
        y_size = 1 + int(np.ceil(self.grid.reference_index.y
                                 + (y_max / delta.y).decompose().value))
        z_size = 1 + int(np.ceil(self.grid.reference_index.z
                                 + (z_max / delta.z).decompose().value))

        log.debug(f"Map pixels: {x_size} x {y_size} x {z_size} (nx, ny, nz)")
        if x_size < 0 or y_size < 0 or z_size < 0:
            raise ValueError(f"Negative image size: "
                             f"{x_size} x {y_size} x {z_size}")

        if not self.configuration.get_bool('large'):
            if (x_size >= self.MAX_X_OR_Y_SIZE or
                    y_size >= self.MAX_X_OR_Y_SIZE or
                    z_size >= self.MAX_Z_SIZE):
                raise ValueError("Map too large.  Use 'large' option.")

        self.set_data_shape((z_size, y_size, x_size))

    def search_corners(self, determine_scan_range=False):
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

        Parameters
        ----------
        determine_scan_range : bool, optional
            If `True`, only determine the map ranges for all scans rather than
            check for the 'map.size' configuration instruction.  I.e., do not
            perform any flagging.

        Returns
        -------
        map_range : Coordinate2D1
            A coordinate of shape (2,) containing the minimum and maximum
            x, y, and z coordinates.
        """
        xy_range = super().search_corners(
            determine_scan_range=determine_scan_range)
        z_unit = self.info.instrument.get_spectral_unit()

        if determine_scan_range:
            fix_size = False
            z_span = 0.0
        else:
            fix_size = self.has_option('map.size')
            map_size = self.configuration.get_float_list(
                'map.size', delimiter='[ \t,:xX]', default=[])
            if len(map_size) < 3:
                fix_size = False
                z_span = 0.0
            else:
                z_span = abs(map_size[2])

        z_ref = self.grid.z.reference.x

        if fix_size:
            z_width = 0.5 * z_span * z_unit
            min_z = (z_ref - z_width).to(z_unit)
            max_z = (z_ref + z_width).to(z_unit)
            min_zf = min_z.to(z_unit).value  # float value for numba
            max_zf = max_z.to(z_unit).value  # float value for numba

            for scan in self.scans:
                for integration in scan.integrations:
                    sf = integration.channels.flagspace.sourceless_flags()
                    skip_flag = integration.flagspace.flags.SAMPLE_SKIP.value
                    channels = integration.channels.get_mapping_pixels(
                        discard_flag=sf)
                    snf.flag_z_outside(
                        z=channels.data.wavelength.to(z_unit).value,
                        min_z=min_zf,
                        max_z=max_zf,
                        valid_frames=integration.frames.valid,
                        channel_indices=channels.indices,
                        sample_flags=integration.frames.sample_flag,
                        skip_flag=skip_flag)
            z_range = Coordinate1D([-z_width, z_width])

        else:
            min_z = np.inf * z_unit
            max_z = -np.inf * z_unit
            for scan in self.scans:
                xy_scan_range = scan.map_range
                scan_min_z = np.inf * z_unit
                scan_max_z = -np.inf * z_unit
                for integration in scan.integrations:
                    sf = integration.channels.flagspace.sourceless_flags()
                    mapping_pixels = integration.channels.get_mapping_pixels(
                        discard_flag=sf)
                    i_min_z = np.nanmin(mapping_pixels.wavelength) - z_ref
                    i_max_z = np.nanmax(mapping_pixels.wavelength) - z_ref
                    if i_min_z < scan_min_z:
                        scan_min_z = i_min_z
                    if i_max_z > scan_max_z:
                        scan_max_z = i_max_z
                z_scan_range = Coordinate1D([scan_min_z, scan_max_z])
                scan.map_range = Coordinate2D1(
                    [xy_scan_range.x, xy_scan_range.y, z_scan_range.x])

                if scan_min_z < min_z:
                    min_z = scan_min_z
                if scan_max_z > max_z:
                    max_z = scan_max_z
            z_range = Coordinate1D([min_z, max_z])

        map_range = Coordinate2D1([xy_range.x, xy_range.y, z_range.x])
        return map_range

    def create_lookup(self, integration):
        """
        Create the source indices for integration frames.

        The source indices contain 1-D lookup values for the pixel indices
        of a sample on the source model.  The map indices are stored in the
        integration frames as the 1-D `source_index` attribute, and the
        2D+1 `map_index` attribute.

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

        frames = integration.frames
        channels = integration.channels
        if frames.source_index is None:
            frames.source_index = np.full((frames.size, channels.size), -1)
        else:
            frames.source_index.fill(-1)

        if frames.map_index is None:
            frames.map_index = Index3D(
                np.full((3, frames.size, channels.size), -1))
        else:
            frames.map_index.coordinates.fill(-1)

        projector = AstroProjector(self.projection)
        xy_offsets = frames.project(pixels.position, projector)
        z_offsets = Coordinate1D(pixels.wavelength, copy=True)
        z_offsets.coordinates -= self.reference.z

        offsets = Coordinate2D1(xy=xy_offsets, z=z_offsets)

        indices = self.grid.offset_to_index(offsets)

        zi = np.broadcast_to(indices.z, xy_offsets.shape)
        indices = Index3D([indices.x, indices.y, zi])

        bad_samples = snf.validate_voxel_indices(
            indices=indices.coordinates,
            x_size=self.size_x,
            y_size=self.size_y,
            z_size=self.size_z,
            valid_frame=frames.valid)

        if bad_samples > 0:  # pragma: no cover
            log.warning(f"{bad_samples} samples have bad map indices")

        frames.map_index.coordinates[..., pixels.indices] = (
            indices.coordinates)
        source_indices = self.pixel_index_to_source_index(
            indices.coordinates)
        frames.source_index[..., pixels.indices] = source_indices

    def pixel_index_to_source_index(self, pixel_indices):
        """
        Return the 1-D source index for a pixel index.

        Parameters
        ----------
        pixel_indices : numpy.ndarray (int)
            The pixel indices of shape (3, shape,) containing
            the (x_index, y_index, z_index) pixel indices.

        Returns
        -------
        source_indices : numpy.ndarray (int)
            The 1-D source indices of shape (shape,).
        """
        xi, yi, zi = pixel_indices
        gi = (xi != -1) & (yi != -1) & (zi != -1)
        source_indices = np.full(xi.shape, -1)
        source_indices[gi] = np.ravel_multi_index(
            (zi[gi], yi[gi], xi[gi]), self.shape)
        return source_indices

    def source_index_to_pixel_index(self, source_indices):
        """
        Convert 1-D source indices to 3-D pixel indices.

        Parameters
        ----------
        source_indices : numpy.ndarray (int)
            The source indices of shape (shape,).

        Returns
        -------
        pixel_indices : numpy.ndarray (int)
            The pixel indices of shape (3, shape,) containing the
            (x_index, y_index) pixel indices.
        """
        bi = source_indices < 0
        if bi.any():
            correct = True
            source_indices = source_indices.copy()
            source_indices[bi] = 0
        else:
            correct = False

        z, y, x = np.unravel_index(source_indices, self.shape)
        if correct:
            z[bi] = -1
            y[bi] = -1
            x[bi] = -1

        return np.stack((x, y, z), axis=0)

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
        smooth : str or Coordinate2D1
            The FWHM for which to smooth to in units of the instrument size
            unit (float), or one of the following string: {'beam', 'halfbeam',
            '2/3beam', 'minimal', 'optimal'}.

        Returns
        -------
        FWHM : Coordinate2D1
            The FWHM of the Gaussian smoothing kernel for the spatial and
            spectral dimensions.
        """
        xy_unit = self.info.instrument.get_size_unit()
        z_unit = self.info.instrument.get_spectral_unit()
        point_size = self.info.instrument.get_point_size().copy()

        pixel_smoothing = self.get_pixelization_smoothing()
        if smooth == 'beam':
            pass
        elif smooth == 'halfbeam':
            point_size.scale(0.5)
        elif smooth == '2/3beam':
            point_size.scale(1 / 1.5)
        elif smooth == 'minimal':
            point_size.scale(0.3)
        elif smooth == 'optimal':
            if 'smooth.optimal' in self.configuration:
                optimal = self.configuration.get_float_list('smooth.optimal',
                                                            default=None)
                if optimal is not None and len(optimal) != 0:
                    if len(optimal) == 1:
                        point_size.x = point_size.y = optimal[0] * xy_unit
                    elif len(optimal) == 2:
                        point_size.x = point_size.y = optimal[0] * xy_unit
                        point_size.z = optimal[1] * z_unit
                    else:
                        point_size.x = optimal[0] * xy_unit
                        point_size.y = optimal[1] * xy_unit
                        point_size.z = optimal[2] * z_unit
            else:
                pass  # Keep beam setting
        elif str(smooth).lower().strip() == 'none':
            point_size.zero()
        elif isinstance(smooth, Coordinate2D1):
            point_size = smooth
        else:
            point_size.zero()

        if point_size.x < pixel_smoothing.x:
            point_size.x = pixel_smoothing.x
        if point_size.y < pixel_smoothing.y:
            point_size.y = pixel_smoothing.y
        if point_size.z < pixel_smoothing.z:
            point_size.z = pixel_smoothing.z

        return point_size

    def set_smoothing(self, smoothing):
        """
        Set the model smoothing.

        Parameters
        ----------
        smoothing : Coordinate2D1

        Returns
        -------
        None
        """
        self.smoothing = smoothing

    def get_requested_smoothing(self, smooth_spec=None):
        """
        Get the requested smoothing for a given specification.

        Parameters
        ----------
        smooth_spec : str or float, optional
            The type of smoothing to retrieve.

        Returns
        -------
        fwhm : Coordinate2D1
            The FWHM of the Gaussian smoothing kernel.
        """
        return super().get_requested_smoothing(smooth_spec=smooth_spec)

    def get_pixelization_smoothing(self):
        """
        Return the pixelation smoothing FWHM.

        The returned value is:

        sqrt(pixel_area * (8 ln(2)) / 2pi)

        Returns
        -------
        pixel_fwhm : Coordinate2D1
        """
        # Used to convert FWHM^2 to beam integral
        gaussian_area = 2 * np.pi * (gaussian_fwhm_to_sigma ** 2)
        xy_fwhm = np.sqrt(self.grid.get_pixel_area() / gaussian_area)
        z_fwhm = self.grid.get_pixel_size().z / np.sqrt(gaussian_area)
        return Coordinate2D1([xy_fwhm, xy_fwhm, z_fwhm])

    def get_point_size(self):
        """
        Return the point size of the source model.

        Returns
        -------
        Coordinate2D1
        """
        smoothing = self.get_requested_smoothing(
            smooth_spec=self.configuration.get_string('smooth'))

        beam_size = self.info.instrument.get_point_size()
        x_fwhm = np.hypot(beam_size.x, smoothing.x)
        y_fwhm = np.hypot(beam_size.y, smoothing.y)
        z_fwhm = np.hypot(beam_size.z, smoothing.z)
        return Coordinate2D1([x_fwhm, y_fwhm, z_fwhm])

    def get_source_size(self):
        """
        Return the source size of the source model.

        Returns
        -------
        Coordinate2D1
        """
        smoothing = self.get_requested_smoothing(
            smooth_spec=self.configuration.get_string('smooth'))

        resolution = self.info.instrument.get_source_size()
        x_fwhm = np.hypot(resolution.x, smoothing.x)
        y_fwhm = np.hypot(resolution.y, smoothing.y)
        z_fwhm = np.hypot(resolution.z, smoothing.z)
        return Coordinate2D1([x_fwhm, y_fwhm, z_fwhm])

    def stand_alone(self):
        """
        Create a stand alone base image.

        Returns
        -------
        None
        """
        self.base = Image2D1(x_size=self.size_x, y_size=self.size_y,
                             z_size=self.size_z, dtype=float)

    def post_process_scan(self, scan):
        """
        Perform post-processing steps on a scan.

        For FIFI-LS, pointing is not necessary.  But a skeleton template
        is commented out below if it's ever necessary.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        pass
        # if self.is_empty():
        #     return
        #
        # if self.configuration.get_bool('pointing.suggest'):
        #     # Smooth before checking pointing
        #     optimal = self.configuration.get_float_list(
        #         'smooth.optimal', default=None)
        #
        #     if optimal is None or len(optimal) == 0:
        #         optimal_x = optimal_y = scan.info.instrument.get_point_size()
        #         optimal_z = scan.info.instrument.get_spectral_size()
        #     else:
        #         xy_unit = self.info.instrument.get_size_unit()
        #         z_unit = self.info.instrument.get_spectral_unit()
        #         if len(optimal) == 1:
        #             optimal_x = optimal_y = optimal[0] * xy_unit
        #             optimal_z = scan.info.instrument.get_spectral_size()
        #         elif len(optimal) == 2:
        #             optimal_x = optimal_y = optimal[0] * xy_unit
        #             optimal_z = optimal[1] * z_unit
        #         else:
        #             optimal_x = optimal[0] * xy_unit
        #             optimal_y = optimal[1] * xy_unit
        #             optimal_z = optimal[2] * z_unit
        #
        #     optimal = Coordinate2D1([optimal_x, optimal_y, optimal_z])
        #     self.map.smooth_to(optimal)
        #
        # if self.has_option('pointing.exposureclip'):
        #     exposure = self.map.get_exposures()
        #     limit = self.configuration.get_float(
        #         'pointing.exposureclip') * exposure.select(0.9)
        #     valid_range = Range(min_val=limit)
        #     exposure.restrict_range(valid_range)
        #
        # # Robust weight before restricting to potentially tiny search area
        # self.map.reweight(robust=True)
        #
        # if self.has_option('pointing.radius'):
        #     # In case pointing.radius is None, use default of infinity
        #     radius = self.configuration.get_float(
        #         'pointing.radius', default=np.inf) * units.Unit('arcsec')
        #     if np.isfinite(radius):
        #         iz, iy, ix = np.indices(self.map.shape)
        #         xy_indices = Coordinate2D([ix, iy])
        #         z_indices = Coordinate1D(iz)
        #         map_indices = Coordinate2D1(xy=xy_indices, z=z_indices)
        #         offsets = self.map.grid.index_to_offset(map_indices)
        #         distance = offsets.xy_coordinates.length
        #         self.map.discard(distance > radius)
        #
        # spline_degree = self.configuration.get_int('pointing.degree',
        #                                            default=3)
        # reduce_degrees = self.configuration.get_bool(
        #     'pointing.reduce_degrees')
        # # Sets a GaussianSource or None
        # scan.pointing = self.get_peak_source(degree=spline_degree,
        #                                      reduce_degrees=reduce_degrees)

    # def get_peak_source(self, degree=3, reduce_degrees=False):
    #     """
    #     Return the peak source model.
    #
    #     Parameters
    #     ----------
    #     degree : int, optional
    #         The spline degree used to fit the peak map value.
    #     reduce_degrees : bool, optional
    #         If `True`, allow the spline fit to reduce the number of degrees
    #         in cases where there are not enough points available to perform
    #         the spline fit of `degree`.  If `False`, a ValueError will be
    #         raised if such a fit fails.
    #
    #     Returns
    #     -------
    #     GaussianSource
    #     """
    #     self.map.level(robust=True)
    #     beam = self.map.get_image_beam()
    #     peak_source = EllipticalSource(gaussian_model=beam)
    #     peak_source.set_positioning_method(
    #         self.configuration.get_string(
    #             'pointing.method', default='centroid'))
    #     peak_source.position = self.get_peak_coords()
    #
    #     if self.configuration.get_bool('pointing.lsq'):
    #         log.debug("Fitting peak source using LSQ method.")
    #         try:
    #             peak_source.fit_map_least_squares(
    #                 self.map, degree=degree, reduce_degrees=reduce_degrees)
    #         except Exception as err:
    #             log.warning(f"Could not fit using LSQ method: {err}")
    #             log.warning("Attempting standard fitting...")
    #             try:
    #                 peak_source.fit_map(self.map, degree=degree,
    #                                     reduce_degrees=reduce_degrees)
    #             except Exception as err:
    #                 log.warning(f"Could not fit peak: {err}")
    #                 return None
    #     else:
    #         try:
    #             peak_source.fit_map(self.map, degree=degree,
    #                                 reduce_degrees=reduce_degrees)
    #         except Exception as err:
    #             log.warning(f"Could not fit peak: {err}")
    #             return None
    #
    #     peak_source.deconvolve_with(self.map.smoothing_beam)
    #     critical_s2n = self.configuration.get_float(
    #         'pointing.significance', default=5.0)
    #
    #     if peak_source.peak_significance < critical_s2n:
    #         return None
    #     else:
    #         return peak_source

    def get_peak_index(self):
        """
        Return the peak index.

        Returns
        -------
        index : Index3D
            The peak (x, y, z) coordinate.
        """
        sign = self.configuration.get_sign('source.sign')
        s2n = self.get_significance()

        if sign > 0:
            z, y, x = np.unravel_index(np.nanargmax(s2n.data), s2n.shape)
        elif sign < 0:
            z, y, x = np.unravel_index(np.nanargmin(s2n.data), s2n.shape)
        else:
            z, y, x = np.unravel_index(
                np.nanargmax(np.abs(s2n.data)), s2n.shape)
        return Index3D([x, y, z])

    def get_peak_coords(self):
        """
        Return the coordinates of the peak value.

        Returns
        -------
        peak_coordinates : Coordinate2D1
            The (x, y, z) peak coordinate.
        """
        projector = AstroProjector(self.projection)
        offset = self.grid.get_offset(self.get_peak_index())
        projector.offset = offset.xy_coordinates
        projector.deproject()
        xy = projector.coordinates
        z = offset.z_coordinates
        z.add(self.grid.z.reference_value)
        return Coordinate2D1(xy=xy, z=z)

    def mask_integration_samples(self, integration, flag='SAMPLE_SKIP'):
        """
        Set the sample flag in an integration for masked map elements.

        Parameters
        ----------
        integration : Integration
        flag : str or int or Enum, optional
            The name, integer identifier, or actual flag by which to flag
            samples.

        Returns
        -------
        None
        """
        masked = self.is_masked()
        if not masked.any():
            return

        pixels = integration.channels.get_mapping_pixels()
        flag_value = integration.frames.flagspace.convert_flag(flag).value

        if integration.frames.map_index is None or np.allclose(
                integration.frames.map_index.x, -1):
            self.create_lookup(integration)

        xi, yi, zi = integration.frames.map_index.coordinates[
            ..., pixels.indices]
        masked_samples = masked[zi, yi, xi]
        if not masked_samples.any():
            return

        sample_flags = integration.frames.sample_flag[..., pixels.indices]
        sample_flags[masked_samples] |= flag_value
        integration.frames.sample_flag[..., pixels.indices] = sample_flags

    def get_map_2d(self):
        """
        Return the 2D map.

        Returns
        -------
        Map2D1
        """
        return super().get_map_2d()

    def get_data(self):
        """
        Return the map data.

        Returns
        -------
        Observation2D1
        """
        return super().get_data()

    def smooth_to(self, fwhm):
        """
        Smooth the map to a given FWHM.

        Parameters
        ----------
        fwhm : Coordinate2D1 or Gaussian2D1

        Returns
        -------
        None
        """
        self.map.smooth_to(fwhm)

    def filter_source(self, filter_fwhm, filter_blanking=None, use_fft=False):
        """
        Apply source filtering.

        Parameters
        ----------
        filter_fwhm : Coordinate2D1
            The filtering FWHM for the source.
        filter_blanking : float, optional
            Only apply filtering within the optional range -filter_blanking ->
            +filter_blanking.
        use_fft : bool, optional
            If `True`, use FFTs to perform the filtering.  Otherwise, do full
            convolutions.

        Returns
        -------
        None
        """
        super().filter_source(filter_fwhm, filter_blanking=filter_blanking,
                              use_fft=use_fft)

    def set_filtering(self, fwhm):
        """
        Set the map filtering FWHM.

        Parameters
        ----------
        fwhm : Coordinate2D1

        Returns
        -------
        None
        """
        self.map.update_filtering(fwhm)

    def get_average_resolution(self):
        """
        Return the average resolution.

        The average resolution is given by::

           r = sqrt(sum(wg^2 * integration_resolution^2) / sum(wg^2))

        where the sum occurs over all integrations in all scans, w is the
        weight of the scan belonging to each integration, and g is the gain
        factor for each integration.  If the denominator is zero, then the
        resolution returned by the source model instrument info is returned
        instead.

        Returns
        -------
        resolution : Coordinate2D1
            The average resolution.
        """
        xy_value = 0.0 * self.info.instrument.get_size_unit() ** 2
        z_value = 0.0 * self.info.instrument.get_spectral_unit() ** 2
        weight = 0.0
        for scan in self.scans:
            for integration in scan.integrations:
                if integration.info is not self.info:
                    xy_resolution = integration.info.instrument.xy_resolution
                    z_resolution = integration.info.instrument.z_resolution
                    wg2 = scan.weight * integration.gain ** 2
                    xy_value += wg2 * xy_resolution ** 2
                    z_value += wg2 * z_resolution ** 2
                    weight += wg2

        if weight > 0:
            xy_resolution = np.sqrt(xy_value / weight)
            z_resolution = np.sqrt(z_value / weight)
            return Coordinate2D1([xy_resolution, xy_resolution, z_resolution])
        else:
            return self.info.instrument.resolution.copy()
