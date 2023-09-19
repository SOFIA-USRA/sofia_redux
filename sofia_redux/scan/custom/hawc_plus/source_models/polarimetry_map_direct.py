# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.custom.hawc_plus.source_models\
    .source_model_numba_functions import combine_rt_map_data
from sofia_redux.scan.custom.hawc_plus.source_models\
    .polarimetry_map import HawcPlusPolarimetryMap

__all__ = ['HawcPlusPolarimetryMapDirect']


class HawcPlusPolarimetryMapDirect(HawcPlusPolarimetryMap):

    def __init__(self, info, reduction=None):
        """
        Initialize a polarization map for HAWC+.

        Parameters
        ----------
        info : sofia_redux.scan.info.info.Info
            The Info object which should belong to this source model.
        reduction : sofia_redux.scan.reduction.reduction.Reduction, optional
            The reduction for which this source model should be applied.
        """
        super().__init__(info, reduction=reduction)

    def copy(self, with_contents=True):
        """
        Return a copy of the polarimetry model.

        Parameters
        ----------
        with_contents : bool, optional
            If `True`, return a true copy of the map.  Otherwise, just return
            a map with basic metadata.

        Returns
        -------
        HawcPlusPolarimetryMapDirect
        """
        return super().copy(with_contents=with_contents)

    def add_integration(self, integration, signal_mode=None):
        """
        Add an integration to the polarimetry model.

        Accumulate integration data into the source model.  Note that the
        source model should already contain (if not empty), weighted data
        products.  I.e., the source model map values should contain
        weight * data values.

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
        if self.configuration.get_bool('polarization.aggregate'):
            self.add_integration_aggregate(
                integration, signal_mode=signal_mode)
        else:
            self.add_integration_separate(
                integration, signal_mode=signal_mode)

    def add_integration_separate(self, integration, signal_mode=None):
        """
        Add an integration to the polarimetry model.

        Accumulate integration data into the source model.  Note that the
        source model should already contain (if not empty), weighted data
        products.  I.e., the source model map values should contain
        weight * data values.

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
        initial_channel_flags = integration.channels.data.flag.copy()
        t_channels = integration.channels.data.sub >= 2
        t_indices = np.nonzero(t_channels)[0]
        r_indices = np.nonzero(~t_channels)[0]

        # Populate map indices now, in case it's overwritten...
        if integration.frames.map_index is None or np.allclose(
                integration.frames.map_index.x, -1):
            self.create_lookup(integration)

        stokes_maps = [self.n]
        if self.use_polarization:
            stokes_maps += [self.q, self.u]

        for stokes_map in stokes_maps:

            # R subarray map
            integration.channels.data.flag = initial_channel_flags.copy()
            integration.channels.data.set_flags('DEAD', indices=t_indices)
            integration.comments.append('R')
            r_map = stokes_map.copy()
            r_map.clear_content()
            r_map.add_integration(integration)

            # T subarray map
            integration.channels.data.flag = initial_channel_flags.copy()
            integration.channels.data.set_flags('DEAD', indices=r_indices)
            integration.comments.append('T')
            t_map = stokes_map.copy()
            t_map.clear_content()
            t_map.add_integration(integration)

            r_map.map.validate()
            t_map.map.validate()

            if stokes_map.signal_mode == self.polarimetry_flags.N:
                sign = 1  # N
            else:
                sign = -1  # Q/U

            discard_flag = stokes_map.flagspace.convert_flag('DISCARD').value

            rt, rt_weight, rt_exposure, rt_flag = combine_rt_map_data(
                r=r_map.map.data,
                r_weight=r_map.map.weight.data,
                r_exposure=r_map.map.exposure.data,
                r_valid=r_map.map.is_valid(),
                t=t_map.map.data,
                t_weight=t_map.map.weight.data,
                t_exposure=t_map.map.exposure.data,
                t_valid=t_map.map.is_valid(),
                bad_flag=discard_flag,
                sign=sign
            )

            rt_map = r_map.copy()
            rt_map.map.data = rt
            rt_map.map.weight.data = rt_weight
            rt_map.map.exposure.data = rt_exposure
            rt_map.map.flag = rt_flag
            rt_map.end_accumulation()
            rt_map.map.validate()
            stokes_map.add_model(rt_map)

        integration.channels.data.flag = initial_channel_flags

    def add_integration_aggregate(self, integration, signal_mode=None):
        """
        Add an integration to the polarimetry model.

        Accumulate integration data into the source model.  Note that the
        source model should already contain (if not empty), weighted data
        products.  I.e., the source model map values should contain
        weight * data values.

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
        # Populate map indices now, in case it's overwritten...
        if integration.frames.map_index is None or np.allclose(
                integration.frames.map_index.x, -1):
            self.create_lookup(integration)

        self.standard_add_integration(integration)

    def add_points(self, frames, pixels, frame_gains, source_gains):
        """
        Add points to the source model for Stokes N, Q, and U.

        Accumulates the combined frame and channel data to the source map
        for each frame/channel sample.  If a given sample maps to a single map
        pixel, that pixel is incremented by::

            i = frame_data * weights * gains
            w = weights * gains^2
            weights = frame_weight / channel_variance
            gains = frame_gain * channel_gain

        where i is the weighted data increment, and w is the weight increment.
        The exposure values are also added to by simply incrementing the time
        at any pixel by the sampling interval multiplied by the number of
        samples falling within that pixel.

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

        integration_gain = frames.integration.gain
        # N map
        frame_gains = integration_gain * frames.get_source_gain(
            self.polarimetry_flags.N)
        n, frame_data, sample_gains, sample_weights, sample_indices = (
            self.get_sample_points(frames, pixels, frame_gains, source_gains))

        self.n.map.accumulate_at(image=frame_data,
                                 gains=sample_gains,
                                 weights=sample_weights,
                                 times=dt,
                                 indices=sample_indices)

        if not self.use_polarization:
            return n

        # For Q/U maps T array gains are negative (R-T)
        t_channels = frames.integration.channels.data.sub >= 2
        t_indices = np.nonzero(t_channels)[0]
        source_gains[t_indices] *= -1

        frame_gains = integration_gain * frames.get_source_gain(
            self.polarimetry_flags.Q)
        n_q, frame_data, sample_gains, sample_weights, sample_indices = (
            self.get_sample_points(frames, pixels, frame_gains, source_gains))
        self.q.map.accumulate_at(image=frame_data,
                                 gains=sample_gains,
                                 weights=sample_weights,
                                 times=dt,
                                 indices=sample_indices)

        frame_gains = integration_gain * frames.get_source_gain(
            self.polarimetry_flags.U)
        n_u, frame_data, sample_gains, sample_weights, sample_indices = (
            self.get_sample_points(frames, pixels, frame_gains, source_gains))
        self.u.map.accumulate_at(image=frame_data,
                                 gains=sample_gains,
                                 weights=sample_weights,
                                 times=dt,
                                 indices=sample_indices)
        return n

    # def apply_stokes_gain_scaling_to(self, stokes_map, normalized=False):
    #     """
    #     Apply Stokes gain scaling to a given map.
    #
    #     For polarization maps, the rotation of the Stokes parameters occurs
    #     at the timestream level before being accumulated into map space.
    #     This is done at the gain calculation phase, and are applied via::
    #
    #         gain_N = unpolarized_gain * frame_gain * N_scaling
    #         gain_Q = Q_rotation_gain * frame_gain * Q_scaling
    #         gain_U = U_rotation_gain * frame_gain * U_scaling
    #
    #     Typically, the N, Q, and U scaling factors are 0.5, but for HAWC+
    #     HWP observations, may be different.
    #
    #     For the maps where Stokes maps are updated directly, scaling are
    #     applied to weight (w) and accumulated data (d = weight * data) values
    #     as::
    #
    #         {x}_weight_scaling = (4 * {x}_scaling)^(-2)
    #         dN_scaling = weight_scaling * N_scaling
    #         d{Q or U}_scaling = dN_scaling / sqrt(8)
    #
    #     Notes
    #     -----
    #     Currently the actual value of {N,Q,U} scaling is irrelevant to the
    #     process and will not change the output values.  This is because
    #     *this* correction factor is applied post processing.  However, there
    #     is more than one way to accumulate/sync/process the Stokes parameters
    #     from the HAWC+ R and T data.  One such method is to apply *this*
    #     correction factor during accumulation, in which case the definition of
    #     these scaling factors becomes more important.  The gain scaling
    #     factors are defined in
    #     `sofia_redux.scan.custom.flags.polarimetry_flags`.
    #
    #     Parameters
    #     ----------
    #     stokes_map : AstroIntensityMap
    #         The Stokes map for which to rescale according to the polarization
    #         gain scaling factor.
    #     normalized : bool, optional
    #         If `True`, indicates that the `stokes_map` data values have already
    #         been normalized from `weight * data` to `weight`.  If so, they
    #         must first be converted back to the accumulated state before
    #         scaling, and reconverted to normalized data once complete.
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     gain_scale = self.polarimetry_flags.get_gain_factor(
    #         stokes_map.signal_mode)
    #
    #     weight_scale = (1.0 / gain_scale) ** 2  # 0.25 / ...
    #     value_scale = gain_scale
    #     if stokes_map.signal_mode != self.polarimetry_flags.N:
    #         value_scale *= 1.0  # 0.25  # For Q/U
    #
    #     value_scale *= weight_scale
    #
    #     if normalized:
    #         stokes_map.map.data *= stokes_map.map.weight.data
    #
    #     stokes_map.map.data *= value_scale
    #     stokes_map.map.weight.data *= weight_scale
    #
    #     if normalized:
    #         stokes_map.end_accumulation()
