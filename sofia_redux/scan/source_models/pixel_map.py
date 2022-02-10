# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
import os
import numpy as np

from sofia_redux.scan.source_models.astro_model_2d import AstroModel2D
from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
from sofia_redux.scan.coordinate_systems.index_2d import Index2D
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector
from sofia_redux.scan.source_models import source_numba_functions as snf
from sofia_redux.scan.utilities import numba_functions
from sofia_redux.toolkit.utilities import multiprocessing

__all__ = ['PixelMap']


class PixelMap(AstroModel2D):

    def __init__(self, info, reduction=None):
        super().__init__(info, reduction=reduction)
        self.pixel_maps = {}
        self.template = AstroIntensityMap(info, reduction=reduction)

    def is_adding_to_master(self):
        """
        Return whether this map is adding to a master map during accumulation.

        Returns
        -------
        bool
        """
        return True

    @property
    def referenced_attributes(self):
        """
        Return attributes that are referenced with a standard copy operation.

        Returns
        -------
        set (str)
        """
        attributes = super().referenced_attributes
        attributes.add('pixel_maps')
        return attributes

    @property
    def shape(self):
        """
        Return the shape of the map.

        Note that this is in numpy (y, x) order.

        Returns
        -------
        tuple of int
        """
        if self.template is None:
            return ()
        return self.template.shape

    @shape.setter
    def shape(self, new_shape):
        """
        Set the shape of the map.

        Parameters
        ----------
        new_shape : tuple of int
            The shape should be supplied in numpy format (y, x).

        Returns
        -------
        None
        """
        self.template.shape = new_shape

    def copy(self, with_contents=True):
        """
        Return a copy of the pixel map.

        Parameters
        ----------
        with_contents : bool, optional
            If `True`, include all pixel maps in the copy.

        Returns
        -------
        PixelMap
        """
        new = super().copy()
        pixel_maps = {}
        if with_contents:
            for pixel, pixel_map in self.pixel_maps.items():
                if isinstance(pixel_map, AstroIntensityMap):
                    pixel_maps[pixel] = pixel_map.copy()
        new.pixel_maps = pixel_maps
        return new

    def clear_process_brief(self):
        """
        Remove all process brief information.

        Returns
        -------
        None
        """
        super().clear_process_brief()
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                pixel_map.clear_process_brief()

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
        log.info("Initializing pixel maps.")

        # Clear all channel position data and set all channels as independent.
        for scan in scans:
            for integration in scan.integrations:
                skip = integration.channels.flagspace.sourceless_flags()
                pixels = integration.channels.get_mapping_pixels(
                    discard_flag=skip)
                # This is a data group, so we need to overwrite references
                position = pixels.position
                position.zero()
                pixels.position = position
                independent = pixels.independent
                independent[:] = True
                pixels.independent = independent

        self.template = AstroIntensityMap(self.info, reduction=self.reduction)
        self.template.create_from(scans, assign_scans=False)
        self.pixel_maps = {}
        super().create_from(scans, assign_scans=assign_scans)

    def reset_processing(self):
        """
        Reset the source processing.

        Returns
        -------
        None
        """
        super().reset_processing()
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                pixel_map.reset_processing()

    def clear_content(self):
        """
        Clear the data.

        Returns
        -------
        None
        """
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                pixel_map.clear_content()

    def create_lookup(self, integration):
        """
        Create the source indices for integration frames.

        The source indices contain 1-D lookup values for the pixel indices
        of a sample on the source model.  The map indices are stored in the
        integration frames as the 1-D `source_index` attribute, and the
        2-D `map_index` attribute.

        Unlike other types of source maps, the PixelMap does not take pixel
        positions into account, so we are solely dealing with timestream
        positions relative to a (0, 0) pixel position for all channels.  This
        means we only need to calculate positions for a single channel with
        position (0, 0)

        Parameters
        ----------
        integration : Integration
            The integration for which to create source indices.

        Returns
        -------
        None
        """
        log.debug("pixel map: lookup for single pixel at (0, 0)")
        self.index_shift_x = numba_functions.log2ceil(self.size_y)
        self.index_mask_y = (1 << self.index_shift_x) - 1

        frames = integration.frames

        if frames.source_index is None:
            frames.source_index = np.full((frames.size, 1), -1)
        else:
            frames.source_index.fill(-1)

        if frames.map_index is None:
            frames.map_index = Index2D(
                np.full((2, frames.size, 1), -1))
        else:
            frames.map_index.coordinates.fill(-1)

        projector = AstroProjector(self.projection)
        central_position = integration.channels.data.position[0:1].copy()
        central_position.zero()
        offsets = frames.project(central_position, projector)

        map_indices = self.grid.offset_to_index(offsets)
        map_indices.coordinates = np.round(map_indices.coordinates).astype(int)
        frames.map_index.coordinates = map_indices.coordinates
        frames.source_index = self.pixel_index_to_source_index(
            map_indices.coordinates)

        bad_samples = snf.validate_pixel_indices(
            indices=map_indices.coordinates,
            x_size=self.size_x,
            y_size=self.size_y,
            valid_frame=frames.valid)

        if bad_samples > 0:
            log.warning(f"{bad_samples} frames have bad map indices")

    def add_model_data(self, other, weight=1.0):
        """
        Add an increment source model data onto the current model.

        Parameters
        ----------
        other : PixelMap
            The source model increment.
        weight : float, optional
            The weight of the source model increment.

        Returns
        -------
        None
        """
        if not isinstance(other, PixelMap):
            raise ValueError(f"Cannot add {other} to {self}.")

        for pixel, pixel_map in other.pixel_maps.items():
            if not isinstance(pixel_map, AstroIntensityMap):
                continue
            this_map = self.pixel_maps.get(pixel)
            if this_map is None:
                self.pixel_maps[pixel] = pixel_map
            elif not isinstance(this_map, AstroIntensityMap):
                continue
            else:
                this_map.add_model_data(pixel_map, weight=weight)

    def add_points(self, frames, pixels, frame_gains, source_gains):
        """
        Add points to the pixel maps.

        This may be an extremely time consuming and memory intensive endeavour.
        A map is created and stored for each mapping pixel in the detector
        channels.

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
        dt = pixels.info.instrument.sampling_interval
        if isinstance(dt, units.Quantity):
            dt = dt.to('second').value
        else:
            dt = float(dt)

        n, frame_data, sample_gains, sample_weights, sample_indices = (
            self.get_sample_points(frames, pixels, frame_gains, source_gains))

        n_pixels = pixels.size

        msg = f"Updating {n_pixels} pixel maps"
        jobs = self.reduction.available_reduction_jobs
        if jobs > 1:
            msg += f" in parallel using {jobs} threads"
        msg += ' (This may take a while).'
        log.debug(msg)

        args = (self.template, self.pixel_maps, pixels.fixed_index, frame_data,
                sample_gains, sample_weights, dt, sample_indices)
        kwargs = None

        pixel_maps = multiprocessing.multitask(
            self.parallel_safe_add_points, range(n_pixels), args, kwargs,
            jobs=jobs, logger=log, force_threading=True)

        for (fixed_index, pixel_map) in pixel_maps:
            self.pixel_maps[fixed_index] = pixel_map

        return n

    @classmethod
    def parallel_safe_add_points(cls, args, pixel_number):
        """
        Add points for a single pixel map.

        This function is safe for :func:`multitask`.

        Parameters
        ----------
        args : 7-tuple
            A tuple of arguments where:
                args[0] - template (AstroIntensityMap)
                args[1] - pixel_maps (dict)
                args[2] - fixed_indices (array) (n_pixels,)
                args[2] - frame_data (array) (n_frames, n_pixels)
                args[3] - sample_gains (array) (n_frames, n_pixels)
                args[4] - sample_weights (array) (n_frames, n_pixels)
                args[5] - dt (float)
                args[6] - sample_indices (array) (2, n_frames, n_pixels)
        pixel_number : int
            The pixel index for the pixel map.

        Returns
        -------
        pixel_maps : 2-tuple
            A tuple of the form (fixed_index, pixel_map).
        """
        (template, pixel_maps, fixed_indices, frame_data, sample_gains,
         sample_weights, dt, sample_indices) = args

        i = pixel_number
        fixed_index = fixed_indices[i]

        pixel_map = pixel_maps.get(fixed_index)
        if pixel_map is None:
            pixel_map = template.copy(with_contents=False)
            pixel_map.id = f'{fixed_index}'
            pixel_map.set_data_shape(template.shape)

        pixel_map.map.accumulate_at(
            image=frame_data[:, i:i + 1],
            gains=sample_gains[:, i:i + 1],
            weights=sample_weights[:, i:i + 1],
            times=dt,
            indices=sample_indices[..., i:i + 1])

        return fixed_index, pixel_map

    def calculate_coupling(self, integration, pixels, source_gains,
                           sync_gains):
        """Not implemented for pixel maps"""
        pass

    def count_points(self):
        """
        Return the number of points in the model.

        Returns
        -------
        int
        """
        points = 0
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                points += pixel_map.count_points()
        return points

    def covariant_points(self):
        """
        Return the number of points in the smoothing beam of all maps.

        Returns
        -------
        float
        """
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                return pixel_map.covariant_points()
        return 1.0

    def get_pixel_footprint(self):
        """
        Returns the Number of bytes per pixel over all pixel maps.

        This is probably no longer relevant and just copies CRUSH.

        Returns
        -------
        int
        """
        return len(self.pixel_maps) * self.template.get_pixel_footprint()

    def base_footprint(self, pixels):
        """
        Returns the base footprint over all pixel maps.

        Parameters
        ----------
        pixels : int

        Returns
        -------
        int
        """
        return len(self.pixel_maps) * self.template.base_footprint(pixels)

    def set_data_shape(self, shape):
        """
        Set the shape of the map.

        Parameters
        ----------
        shape : tuple (int)

        Returns
        -------
        None
        """
        self.template.set_data_shape(shape)
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                pixel_map.set_data_shape(shape)

    def sync_source_gains(self, frames, pixels, frame_gains, source_gains,
                          sync_gains):
        """
        Remove the map source from frame data for all pixel maps.

        In addition to source removal, samples are also flagged if the map
        is masked at that location.

        For a given sample at frame i and channel j, frame data d_{i,j} will be
        decremented by dg where:

        dg = fg * ( (gain(source) * map[index]) - (gain(sync) * base[index]) )

        Here, fg is the frame gain and index is the index on the map of sample
        (i,j).

        Any masked map value will result in matching samples being flagged.

        Parameters
        ----------
        frames : Frames
            The frames for which to remove the source gains.
        pixels : Channels or ChannelGroup
            The channels for which to remove the source gains.
        frame_gains : numpy.ndarray (float)
            An array of frame gains of shape (n_frames,).
        source_gains : numpy.ndarray (float)
            An array of channel source gains for all channels of shape
            (all_channels,).
        sync_gains : numpy.ndarray (float)
            an array of channel sync gains for all channels of shape
            (all_channels,).  The sync gains should contain the prior source
            gain.

        Returns
        -------
        None
        """
        n_pixels = pixels.size
        msg = f"Syncing gains in {n_pixels} pixel maps"
        jobs = self.reduction.available_reduction_jobs
        if jobs > 1:
            msg += f" in parallel using {jobs} threads"
        msg += ' (This may take a while).'
        log.debug(msg)

        args = (self.pixel_maps, frames, pixels, frame_gains, source_gains,
                sync_gains)
        kwargs = None

        frame_data = multiprocessing.multitask(
            self.parallel_safe_sync_source_gains, range(n_pixels),
            args, kwargs, jobs=jobs, logger=log, force_threading=True)
        for channel, data in enumerate(frame_data):
            frames.data[:, channel] = data

    @classmethod
    def parallel_safe_sync_source_gains(cls, args, pixel_number):
        """
        Remove the source from all pixel maps.

        This function is safe for :func:`multitask`.

        Parameters
        ----------
        args : 6-tuple
            A tuple of arguments where:
                args[0] - pixel_maps (dict)
                args[1] - frames (Frames)
                args[2] - pixels (ChannelGroup)
                args[3] - frame_gains (array) (n_frames,)
                args[4] - source_gains (array) (all_channels,)
                args[5] - sync_gains (array) (all_channels,)
        pixel_number : int
            The pixel index for the pixel map.

        Returns
        -------
        channel_frame_data : numpy.ndarray (float)
            The synced frame data for `pixel_number` of shape (n_frames,).
        """
        # TODO: test to see if this can be updated in place with threading.
        (pixel_maps, frames, pixels, frame_gains,
         source_gains, sync_gains) = args
        fixed_index = pixels.fixed_index[pixel_number]
        pixel_map = pixel_maps[fixed_index]
        channel_indices = np.full(1, pixel_number)
        snf.sync_map_samples(
            frame_data=frames.data,
            frame_valid=frames.valid,
            frame_gains=frame_gains,
            channel_indices=channel_indices,
            map_values=pixel_map.map.data,
            map_valid=pixel_map.map.valid,
            map_masked=pixel_map.is_masked(),
            map_indices=frames.map_index.coordinates,
            base_values=pixel_map.base.data,
            base_valid=pixel_map.base.valid,
            source_gains=source_gains,
            sync_gains=sync_gains,
            sample_flags=frames.sample_flag,
            sample_blank_flag=frames.flagspace.convert_flag(
                'SAMPLE_SOURCE_BLANK').value)
        return frames.data[:, pixel_number]

    def set_base(self):
        """
        Set the base to the map (copy of).

        Returns
        -------
        None
        """
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                pixel_map.set_base()

    def process_scan(self, scan):
        """
        Process a scan.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                pixel_map.process_scan(scan)

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
        if self.has_option('pixelmap.writemaps'):
            try:
                pixels = self.configuration.get_int_list('pixelmap.writemaps')
            except ValueError:
                pixels = None
            if pixels is None:
                if self.configuration.get_bool('pixelmap.writemaps'):
                    pixels = list(self.pixel_maps.keys())
            for pixel in pixels:
                pixel_map = self.pixel_maps.get(pixel)
                if isinstance(pixel_map, AstroIntensityMap):
                    pixel_map.write(path)
        self.calculate_pixel_data(smooth=False)
        self.write_pixel_data()

    def write_fits(self, filename):
        """
        Write the results to a FITS file.

        Not implemented for the single PixelMap.

        Parameters
        ----------
        filename : str

        Returns
        -------
        None
        """
        return

    def process(self):
        """
        Process the source model.

        Returns
        -------
        None
        """
        process = self.configuration.get_bool('pixelmap.process')
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                if process:
                    pixel_map.process()
                else:
                    pixel_map.map.end_accumulation()
                    self.next_generation()

    def calculate_pixel_data(self, smooth=False):
        """
        Calculate pixel position offsets and couplings.

        Pixel positions and couplings will be stored in the channel data of the
        first scan in the source model.

        Parameters
        ----------
        smooth : bool, optional
            If `True`, smooth the map to the instrument peak resolution before
            fitting.

        Returns
        -------
        None
        """
        mapping_pixels = self.get_first_scan().channels.get_mapping_pixels(
            keep_flag=0)

        n_pixels = len(self.pixel_maps)
        msg = f"Fitting {n_pixels} pixel positions to source map"
        jobs = self.reduction.available_reduction_jobs
        if jobs > 1:
            msg += f" in parallel using {jobs} threads"
        msg += '.'
        log.debug(msg)

        point_size = self.info.instrument.get_point_size()
        args = self.pixel_maps, smooth, point_size
        kwargs = None

        peaks = multiprocessing.multitask(
            self.parallel_safe_calculate_pixel_data,
            mapping_pixels.fixed_index, args, kwargs,
            jobs=jobs, logger=log, force_threading=True)

        peak_values = {}
        pixel_positions = mapping_pixels.position

        for pixel_index, fixed_index in enumerate(mapping_pixels.fixed_index):
            peak = peaks[pixel_index]
            if peak is None:
                continue
            position = self.projection.project(peak.coordinates)
            position.invert()
            pixel_positions[pixel_index] = position
            peak_values[pixel_index] = peak.peak

        if len(peak_values) == 0:
            return

        mapping_pixels.position = pixel_positions
        mean_peak = np.median(list(peak_values.values()))

        couplings = mapping_pixels.coupling
        for pixel_index, peak in peak_values.items():
            couplings[pixel_index] *= peak / mean_peak
        mapping_pixels.coupling = couplings

    @classmethod
    def parallel_safe_calculate_pixel_data(cls, args, pixel_number):
        """
        Fit a peak to a single pixel.

        This function is safe for :func:`multitask`.

        Parameters
        ----------
        args : 7-tuple
            A tuple of arguments where:
                args[0] - pixel_maps (dict)
                args[1] - smooth (bool)
                args[2] - point_size (units.Quantity)
        pixel_number : int
            The pixel fixed index for the pixel map.

        Returns
        -------
        gaussian_model : EllipticalSource or None
        """
        pixel_maps, smooth, point_size = args

        beam_map = pixel_maps.get(pixel_number)
        if beam_map is None or not beam_map.is_valid():
            return None
        obs_map = beam_map.map
        if smooth:
            obs_map.smooth_to(point_size)
        source = beam_map.get_peak_source()
        return source

    def write_pixel_data(self):
        """
        Write the pixel data to file.

        Returns
        -------
        None
        """
        channels = self.get_first_scan().channels
        source_gain = channels.get_source_gains(filter_corrected=False)
        filename = os.path.join(self.reduction.work_path,
                                f'{self.get_default_core_name()}.rcp')
        channels.data.coupling = source_gain / channels.data.gain

        hdr = self.get_first_scan().get_first_integration().get_ascii_header()
        file_contents = channels.print_pixel_rcp(header=hdr)

        with open(filename, 'w') as f:
            f.write(file_contents)
        log.info(f"Written RCP contents to {filename}.")

    def set_parallel(self, threads):
        """
        Set the number of parallel operations for the pixel map.

        Not applicable for the SkyDip model.

        Parameters
        ----------
        threads : int

        Returns
        -------
        None
        """
        for pixel_map in self.pixel_maps.values():
            if pixel_map is not None:
                pixel_map.set_parallel(threads)

    def no_parallel(self):
        """
        Disable parallel processing for each pixel map.

        Returns
        -------
        None
        """
        for pixel_map in self.pixel_maps.values():
            if pixel_map is not None:
                pixel_map.no_parallel()

    def get_parallel(self):
        """
        Return the number of parallel threads for the pixel maps.

        Returns
        -------
        threads : int
        """
        for pixel_map in self.pixel_maps.values():
            if pixel_map is not None:
                return pixel_map.get_parallel()
        return 1

    def merge_accumulate(self, other):
        """
        Merge another pixel map with this one.

        Parameters
        ----------
        other : PixelMap

        Returns
        -------
        None
        """
        if not isinstance(other, PixelMap):
            raise ValueError(f"Cannot add {other} to {self}.")

        args = self.pixel_maps, other.pixel_maps
        kwargs = None
        jobs = self.reduction.available_reduction_jobs

        fixed_indices = list(self.pixel_maps.keys())

        updated = multiprocessing.multitask(
            self.parallel_safe_merge_accumulate,
            fixed_indices, args, kwargs,
            jobs=jobs, logger=log, force_threading=True)

        for fixed_index, pixel_map in updated:
            self.pixel_maps[fixed_index] = pixel_map

    @classmethod
    def parallel_safe_merge_accumulate(cls, args, pixel_index):
        """
        Merge a single pixel map onto another.

        This function is safe for :func:`multitask`.

        Parameters
        ----------
        args : 2-tuple
            A tuple of arguments where:
                args[0] - pixel_maps (dict)
                args[1] - other_maps (dict)
        pixel_index : int
            The pixel fixed index for the pixel map to merge

        Returns
        -------
        fixed_index, pixel_map : int, AstroIntensityMap
            The fixed pixel index and merged pixel map.
        """
        pixel_maps, other_maps = args
        pixel_map = pixel_maps.get(pixel_index)
        other_map = other_maps.get(pixel_index)

        if not isinstance(pixel_map, AstroIntensityMap) or not isinstance(
                other_map, AstroIntensityMap):
            return pixel_index, pixel_map
        pixel_map.merge_accumulate(other_map)
        return pixel_index, pixel_map

    def get_source_name(self):
        """
        Return the source name for the pixel map.

        Returns
        -------
        name : str
        """
        return self.template.get_source_name()

    def get_unit(self):
        """
        Return the map data unit.

        Returns
        -------
        units.Quantity or units.Unit
            If a Quantity is returned, the value represents a scaling factor.
        """
        return self.template.get_unit()

    def is_empty(self):
        """
        Return whether this pixel map has no data.

        Returns
        -------
        bool
        """
        if len(self.pixel_maps) == 0:
            return True
        for pixel_map in self.pixel_maps.values():
            if isinstance(pixel_map, AstroIntensityMap):
                return False
        else:
            return True

    def process_final(self):
        """
        Perform the final processing steps for each pixel map.

        Returns
        -------
        None
        """
        updated = multiprocessing.multitask(
            self.parallel_safe_process_final, list(self.pixel_maps.keys()),
            (self.pixel_maps,), None,
            jobs=self.reduction.available_reduction_jobs,
            logger=log,
            force_threading=True)

        for fixed_index, pixel_map in updated:
            self.pixel_maps[fixed_index] = pixel_map

    @classmethod
    def parallel_safe_process_final(cls, args, pixel_index):
        """
        Perform the final processing steps for a single pixel map.

        Parameters
        ----------
        args : 2-tuple
            A tuple of arguments where:
                args[0] - pixel_maps (dict)
        pixel_index : int
            The pixel fixed index for the pixel map to merge

        Returns
        -------
        pixel_index, pixel_map : int, AstroIntensityMap
            The pixel index and updated pixel map.
        """
        pixel_maps = args[0]
        pixel_map = pixel_maps.get(pixel_index)
        if not isinstance(pixel_map, AstroIntensityMap):
            return pixel_index, pixel_map
        pixel_map.process_final()
        return pixel_index, pixel_map

    def get_map_2d(self):
        """
        Return the 2D map.

        This is not valid for a pixel map.

        Returns
        -------
        None
        """
        return None
