# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np

from sofia_redux.scan.source_models.astro_data_2d import AstroData2D
from sofia_redux.scan.source_models.maps.image_2d import Image2D
from sofia_redux.scan.source_models.maps.observation_2d import Observation2D
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.source_models.beams.elliptical_source import (
    EllipticalSource)
from sofia_redux.scan.source_models import source_numba_functions as snf
from sofia_redux.scan.coordinate_systems.index_2d import Index2D
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector
from sofia_redux.scan.coordinate_systems.grid.spherical_grid import \
    SphericalGrid

__all__ = ['AstroIntensityMap']


class AstroIntensityMap(AstroData2D):

    def __init__(self, info, reduction=None):
        self.map = Observation2D()
        self.base = None  # referenced, not copied
        super().__init__(info, reduction=reduction)
        self.create_map()

    @property
    def flagspace(self):
        return self.map.flagspace

    @property
    def shape(self):
        """
        Return the shape of the map.

        Note that this is in numpy (y, x) order.

        Returns
        -------
        tuple of int
        """
        if self.map is None:
            return ()
        return self.map.shape

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
        self.map.shape = new_shape

    @property
    def referenced_attributes(self):
        """
        Return attributes that should be referenced during a copy.

        Returns
        -------
        set (str)
        """
        attributes = super().referenced_attributes
        attributes.add('base')
        return attributes

    def set_info(self, info):
        """
        Set the channels for the source. (setInstrument)

        Parameters
        ----------
        info : Info

        Returns
        -------
        None
        """
        super().set_info(info)
        if self.map is not None:
            self.map.fits_properties.set_instrument_name(
                self.info.instrument.name)
            self.map.fits_properties.set_telescope_name(
                self.info.telescope.telescope)

    def get_jansky_unit(self):
        """
        Return the Jansky unit for the model.

        Returns
        -------
        astropy.units.Quantity
        """
        # astropy.modeling.functional_models.Gaussian2D
        beam = self.map.underlying_beam
        if beam is None:
            beam_area = 0 * units.Unit('degree2')
        else:
            area_factor = 2 * np.pi * (gaussian_fwhm_to_sigma ** 2)
            beam_area = beam.x_fwhm * beam.y_fwhm * area_factor
        jy = units.Unit('Jy')
        return beam_area * self.info.instrument.jansky_per_beam() * jy

    def add_model_data(self, source_model, weight=1.0):
        """
        Add an increment source model data onto the current model.

        Parameters
        ----------
        source_model : AstroIntensityMap
            The source model increment.
        weight : float, optional
            The weight of the source model increment.

        Returns
        -------
        None
        """
        self.map.accumulate(source_model.map, weight=weight)

    def merge_accumulate(self, other):
        """
        Merge another source with this one.

        Parameters
        ----------
        other : AstroIntensityMap

        Returns
        -------
        None
        """
        self.map.merge_accumulate(other.map)

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
        AstroIntensityMap
        """
        new = super().copy(with_contents=with_contents)
        if self.grid is not None and isinstance(new, AstroIntensityMap):
            new.map.set_grid(self.grid)
        return new

    def stand_alone(self):
        """
        Create a stand alone base image.

        Returns
        -------
        None
        """
        self.base = Image2D(x_size=self.size_x, y_size=self.size_y,
                            dtype=float)

    def create_map(self):
        """
        Create the source model map.

        Returns
        -------
        None
        """
        self.map = Observation2D()
        self.map.set_grid(self.grid)
        self.map.set_validating_flags(~self.mask_flag)
        self.map.add_local_unit(self.get_native_unit())
        self.map.add_local_unit(self.get_jansky_unit())
        self.map.add_local_unit(self.get_kelvin_unit())
        self.map.set_display_grid_unit(self.info.instrument.get_size_unit())
        self.map.fits_properties.set_instrument_name(
            self.info.instrument.name)
        self.map.fits_properties.set_copyright('LOL')
        if self.reduction is not None:
            self.map.set_parallel(self.reduction.max_jobs)
            self.map.fits_properties.set_creator_name(
                self.reduction.__class__.__name__)

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
        self.create_map()
        super().create_from(scans, assign_scans=assign_scans)
        properties = self.map.fits_properties
        properties.set_object_name(self.get_first_scan().source_name)
        self.map.set_underlying_beam(self.get_average_resolution())
        log.info("\n".join(self.map.get_info()))
        self.base = Image2D(x_size=self.map.shape[1], y_size=self.map.shape[0],
                            dtype=float)

    def post_process_scan(self, scan):
        """
        Perform post-processing steps on a scan.

        At this stage, the map should have already been added to the main
        reduction map and we are left with a throw-away map that can be used
        to update settings in other objects.  The intensity map may be used to
        update the pointing for a given scan.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        super().post_process_scan(scan)
        if self.is_empty():
            return

        if self.configuration.get_bool('pointing.suggest'):
            optimal = self.configuration.get_float('smooth.optimal')
            if np.isnan(optimal):
                optimal = scan.info.instrument.get_point_size()
            else:
                optimal = optimal * self.info.instrument.get_size_unit()
            self.map.smooth_to(optimal)

        if self.has_option('pointing.exposureclip'):
            exposure = self.map.get_exposures()
            limit = self.configuration.get_float(
                'pointing.exposureclip') * exposure.select(0.9)
            valid_range = Range(min_val=limit)
            exposure.restrict_range(valid_range)

        # Robust weight before restricting to potentially tiny search area
        self.map.reweight(robust=True)

        if self.has_option('pointing.radius'):
            # In case pointing.radius is None, use default of infinity
            radius = self.configuration.get_float(
                'pointing.radius', default=np.inf) * units.Unit('arcsec')
            if np.isfinite(radius):
                iy, ix = np.indices(self.map.shape)
                map_indices = Coordinate2D([ix, iy])
                distance = self.map.grid.index_to_offset(map_indices).length
                self.map.discard(distance > radius)

        scan.pointing = self.get_peak_source()  # A GaussianSource or None

    def get_peak_index(self):
        """
        Return the peak index.

        Returns
        -------
        index : Index2D
            The peak (x, y) coordinate.
        """
        sign = self.configuration.get_sign('source.sign')
        s2n = self.get_significance()

        if sign > 0:
            y, x = np.unravel_index(np.nanargmax(s2n.data), s2n.shape)
        elif sign < 0:
            y, x = np.unravel_index(np.nanargmin(s2n.data), s2n.shape)
        else:
            y, x = np.unravel_index(np.nanargmax(np.abs(s2n.data)), s2n.shape)
        return Index2D([x, y])

    def get_peak_coords(self):
        """
        Return the coordinates of the peak value.

        Returns
        -------
        peak_coordinates : Coordinate2D
            The (x, y) peak coordinate.
        """
        projector = AstroProjector(self.projection)
        self.grid.get_offset(self.get_peak_index(), offset=projector.offset)
        projector.deproject()
        return projector.coordinates

    def get_peak_source(self):
        """
        Return the peak source model.

        Returns
        -------
        GaussianSource
        """
        self.map.level(robust=True)
        beam = self.map.get_image_beam()
        peak_source = EllipticalSource(gaussian_model=beam)
        peak_source.set_positioning_method(
            self.configuration.get_string(
                'pointing.method', default='centroid'))
        peak_source.position = self.get_peak_coords()

        if self.configuration.get_bool('pointing.lsq'):
            log.debug("Fitting peak source using LSQ method.")
            try:
                peak_source.fit_map_least_squares(self.map)
            except Exception as err:
                log.warning(f"Could not fit using LSQ method: {err}")
                log.warning("Attempting standard fitting...")
                try:
                    peak_source.fit_map(self.map)
                except Exception as err:
                    log.warning(f"Could not fit peak: {err}")
                    return None
        else:
            try:
                peak_source.fit_map(self.map)
            except Exception as err:
                log.warning(f"Could not fit peak: {err}")
                return None

        peak_source.deconvolve_with(self.map.smoothing_beam)
        critical_s2n = self.configuration.get_float(
            'pointing.significance', default=5.0)

        if peak_source.peak_significance < critical_s2n:
            return None
        else:
            return peak_source

    def update_mask(self, blanking_level=None, min_neighbors=2):
        """
        Update the map mask based on significance levels and valid neighbors.

        If a blanking level is supplied, significance values above or equal to
        the blanking level will be masked.  If the configuration


        Parameters
        ----------
        blanking_level : float, optional
            The significance level used to mark the map.  If not supplied,
            significance is irrelevant.  See above for more details.
        min_neighbors : int, optional
            The minimum number of neighbors including the pixel itself.
            Therefore, the default of 2 excludes single pixels as this would
            require a single valid pixel and one valid neighbor.

        Returns
        -------
        None
        """
        if blanking_level is None or np.isnan(blanking_level):
            blanking_range = np.array([-np.inf, np.inf])
            check_blanking = False
        else:
            blanking_range = np.array([-blanking_level, blanking_level])
            sign = self.configuration.get_sign(self.source_option('sign'))
            check_blanking = True
            if sign < 0:
                blanking_range[1] = np.inf
            elif sign > 0:
                blanking_range[0] = -np.inf

        neighbors = self.map.get_neighbors()
        mask = neighbors >= min_neighbors  # neighbors includes center pix
        mask &= self.map.valid

        if check_blanking:
            significance = self.map.significance_values()
            blank = significance < blanking_range[0]
            blank |= significance >= blanking_range[1]
            mask &= blank

        self.map.set_flags(self.mask_flag, mask)
        self.map.unflag(self.mask_flag, ~mask)

    def merge_mask(self, other_map):
        """
        Merge the mask from another map onto this one.

        Parameters
        ----------
        other_map : FlaggedArray

        Returns
        -------
        None
        """
        other_mask_flag = other_map.flagspace.convert_flag(self.FLAG_MASK)
        self.map.set_flags(
            self.mask_flag, indices=other_map.is_flagged(other_mask_flag))

    def is_masked(self):
        """
        Return the map mask.

        Returns
        -------
        numpy.ndarray (bool)
        """
        return self.map.is_flagged(self.mask_flag)

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
        dt = pixels.info.instrument.sampling_interval
        if isinstance(dt, units.Quantity):
            dt = dt.to('second').value
        else:
            dt = float(dt)

        n, frame_data, sample_gains, sample_weights, sample_indices = (
            self.get_sample_points(frames, pixels, frame_gains, source_gains))

        self.map.accumulate_at(image=frame_data,
                               gains=sample_gains,
                               weights=sample_weights,
                               times=dt,
                               indices=sample_indices)
        return n

    def mask_samples(self, flag):
        """
        Mask samples in all integrations in all scans with a given pattern.

        Parameters
        ----------
        flag : str or int or Enum
           The name, integer identifier, or actual flag by which to mask
           samples.

        Returns
        -------
        None
        """
        for scan in self.scans:
            for integration in scan.integrations:
                self.mask_integration_samples(integration, flag)

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
        sample_flag = integration.frames.flagspace.convert_flag(flag).value

        if integration.frames.source_index is None:
            self.create_lookup(integration)

        map_indices = self.source_index_to_pixel_index(
            integration.frames.source_index[:, pixels.indices])

        # Yes, this is correct (map indices is in (x, y) format).
        xi, yi = map_indices[..., 0].flat, map_indices[..., 1].flat
        masked_samples = masked[yi, xi]
        if not masked_samples.any():
            return

        sample_flags = integration.frames.sample_flag.flat
        sample_flags[masked_samples] |= sample_flag

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
            extract the frame source gains.  Typically, TOTAL_POWER.

        Returns
        -------
        mapping_frames : int
            The number of frames that contributed towards mapping.
        """
        good_frames = super().add_frames_from_integration(
            integration=integration, pixels=pixels, source_gains=source_gains,
            signal_mode=signal_mode)
        self.add_integration_time(
            good_frames * integration.info.instrument.sampling_interval)
        return good_frames

    def sync_source_gains(self, frames, pixels, frame_gains, source_gains,
                          sync_gains):
        """
        Remove the map source from frame data.

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
        snf.sync_map_samples(
            frame_data=frames.data,
            frame_valid=frames.valid,
            frame_gains=frame_gains,
            channel_indices=pixels.indices,
            map_values=self.map.data,
            map_valid=self.map.valid,
            map_masked=self.is_masked(),
            map_indices=frames.map_index.coordinates,
            base_values=self.base.data,
            base_valid=self.base.valid,
            source_gains=source_gains,
            sync_gains=sync_gains,
            sample_flags=frames.sample_flag,
            sample_blank_flag=frames.flagspace.convert_flag(
                'SAMPLE_SOURCE_BLANK').value)

    def calculate_coupling(self, integration, pixels, source_gains,
                           sync_gains):
        """
        Don't know

        Parameters
        ----------
        integration : Integration
        pixels : Channels or ChannelGroup
            The pixels for which to calculate coupling.
        source_gains : numpy.ndarray (float)
            The source gains for all frames in the integration of shape
            (n_frames,).
        sync_gains : numpy.ndarray (float)
            The sync gains for all channels in the integration of shape
            (all_channels,).

        Returns
        -------
        None
        """
        if self.has_option('source.coupling.s2n'):
            s2n_range = self.configuration.get_range('source.coupling.s2n')
        else:
            s2n_range = Range(min_val=5.0)

        if integration.frames.map_index is None:
            self.create_lookup(integration)

        frame_gains = integration.gain * integration.frames.get_source_gain(
            self.signal_mode)

        coupling_increment = snf.calculate_coupling_increment(
            map_indices=integration.frames.map_index.coordinates,
            base_values=self.base.data,
            map_values=self.map.data,
            map_noise=self.map.get_noise().data,
            sync_gains=sync_gains,  # Frame space
            source_gains=source_gains,  # Channel space
            frame_data=integration.frames.data,
            frame_weight=integration.frames.relative_weight,
            frame_gains=frame_gains,
            frame_valid=integration.frames.valid,
            sample_flags=integration.frames.sample_flag,
            channel_indices=pixels.indices,
            min_s2n=s2n_range.min,
            max_s2n=s2n_range.max,
            exclude_flag=self.exclude_samples.value)

        coupling = integration.channels.data.coupling
        new_coupling = coupling + (coupling * coupling_increment)

        integration.channels.data.coupling = new_coupling

        # Normalize the couplings to 1.0
        integration.channels.modalities.get(
            'coupling').modes[0].normalize_gains()

        # If the coupling falls out of range, revert to the default of 1.0
        if self.has_option('source.coupling.range'):
            coupling_range = self.configuration.get_range(
                'source.coupling.range')
            snf.flag_out_of_range_coupling(
                channel_indices=pixels.indices,
                coupling_values=pixels.data.coupling,
                min_coupling=coupling_range.min,
                max_coupling=coupling_range.max,
                flags=pixels.data.flag,
                blind_flag=pixels.flagspace.convert_flag('BLIND').value)

    def process_final(self):
        """
        Perform the final processing steps.

        The additional steps performed for the AstroIntensityMap are
        map leveling (if not extended or deep) and map re-weighting.
        The map may also be resampled if re-griding is enabled.

        Returns
        -------
        None
        """
        super().process_final()
        if not (self.configuration.get_bool('extended')
                | self.configuration.get_bool('deep')):
            if self.enable_level:
                self.map.level(robust=True)
            if self.enable_weighting:
                self.map.reweight(robust=True)

        if self.has_option('regrid'):
            size_unit = self.info.instrument.get_size_unit()
            resolution = self.configuration.get_float('regrid') * size_unit
            self.map.resample(resolution)

    def get_table_entry(self, name):
        """
        Return a parameter value for a given name.

        Parameters
        ----------
        name : str, optional

        Returns
        -------
        value
        """
        if name == 'system':
            if not isinstance(self.grid, SphericalGrid):
                return None
            if self.grid.reference is None:
                return None
            return self.grid.reference.two_letter_code
        if name.startswith('map.'):
            return self.map.get_table_entry(name[4:])
        return super().get_table_entry(name)

    def get_pixel_footprint(self):
        """
        Returns the Number of bytes per pixel.

        This is probably no longer relevant and just copies CRUSH.

        Returns
        -------
        int
        """
        return 32

    def base_footprint(self, pixels):
        """
        Returns the base footprint.

        Parameters
        ----------
        pixels : int

        Returns
        -------
        int
        """
        return 8 * pixels

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
        self.map.set_data_shape(shape)

    def set_base(self):
        """
        Set the base to the map (copy of).

        Returns
        -------
        None
        """
        if self.base is None:
            self.stand_alone()
        self.base.paste(self.map, report=False)

    def reset_processing(self):
        """
        Reset the source processing.

        Returns
        -------
        None
        """
        super().reset_processing()
        self.map.reset_processing()

    def covariant_points(self):
        """
        Return the number of points in the smoothing beam of the map.

        Returns
        -------
        float
        """
        return self.map.get_points_per_smoothing_beam()

    def get_map_2d(self):
        """
        Return the 2D map.

        Returns
        -------
        Map2D
        """
        return self.map

    def get_source_name(self):
        """
        Return the source name.

        Returns
        -------
        str
        """
        return self.map.fits_properties.object_name

    def get_unit(self):
        """
        Return the map data unit.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.map.unit

    def get_data(self):
        """
        Return the map data.

        Returns
        -------
        Observation2D
        """
        return self.map

    def add_base(self):
        """
        Add the base to the observation map.

        Returns
        -------
        None
        """
        if self.base is not None:
            self.map.add(self.base)

    def smooth_to(self, fwhm):
        """
        Smooth the map to a given FWHM.

        Parameters
        ----------
        fwhm : astropy.units.Quantity or Gaussian2D

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
        filter_fwhm : astropy.units.Quantity
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
        if filter_blanking is not None and np.isfinite(filter_blanking):
            significance = self.map.get_significance()
            valid = significance.valid
            significance_values = significance.data
            valid &= significance_values <= filter_blanking
            valid &= significance_values >= -filter_blanking
        else:
            valid = None

        if use_fft:
            log.debug(f"Filtering source above {filter_fwhm} (FFT).")
            self.map.fft_filter_above(filter_fwhm, valid=valid)
        else:
            log.debug(f"Filtering source above {filter_fwhm}.")
            self.map.filter_above(filter_fwhm, valid=valid)

        self.map.set_filter_blanking(filter_blanking)

    def set_filtering(self, fwhm):
        """
        Set the map filtering FWHM.

        Parameters
        ----------
        fwhm : astropy.units.Quantity

        Returns
        -------
        None
        """
        self.map.update_filtering(fwhm)

    def reset_filtering(self):
        """
        Reset the map filtering.

        Returns
        -------
        None
        """
        self.map.reset_filtering()

    def filter_beam_correct(self):
        """
        Apply beam filter correction.

        Returns
        -------
        None
        """
        self.map.filter_beam_correct()

    def mem_correct(self, lg_multiplier):
        """
        Apply maximum entropy correction to the map.

        Parameters
        ----------
        lg_multiplier : float

        Returns
        -------
        None
        """
        self.map.mem_correct_observation(None, lg_multiplier)

    def get_clean_local_copy(self, full=False):
        """
        Get an unprocessed copy of the source model.

        Parameters
        ----------
        full : bool, optional
            If True, copy additional parameters for stand-alone reductions
            that would otherwise be referenced.

        Returns
        -------
        SourceModel
        """
        model = super().get_clean_local_copy(full=full)
        if not full:
            return model
        model.base = model.base.copy()
        model.base.clear()
        return model
