# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
import numpy as np

from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
from sofia_redux.scan.custom.hawc_plus.source_models\
    .polarimetry_map import HawcPlusPolarimetryMap

__all__ = ['HawcPlusPolarimetryMapSeparateRt']


class HawcPlusPolarimetryMapSeparateRt(HawcPlusPolarimetryMap):

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
        self.nr = AstroIntensityMap(info, reduction=reduction)
        self.qr = AstroIntensityMap(info, reduction=reduction)
        self.ur = AstroIntensityMap(info, reduction=reduction)
        self.nt = AstroIntensityMap(info, reduction=reduction)
        self.qt = AstroIntensityMap(info, reduction=reduction)
        self.ut = AstroIntensityMap(info, reduction=reduction)
        self.cov_qi = AstroIntensityMap(info, reduction=reduction)
        self.cov_ui = AstroIntensityMap(info, reduction=reduction)
        self.cov_qu = AstroIntensityMap(info, reduction=reduction)
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
        HawcPlusPolarimetryMapSeparateRt
        """
        new = super().copy(with_contents=with_contents)
        new.nr = self.nr.copy(with_contents=with_contents)
        new.qr = self.qr.copy(with_contents=with_contents)
        new.ur = self.ur.copy(with_contents=with_contents)
        new.nt = self.nt.copy(with_contents=with_contents)
        new.qt = self.qt.copy(with_contents=with_contents)
        new.ut = self.ut.copy(with_contents=with_contents)
        new.cov_qi = self.cov_qi.copy(with_contents=with_contents)
        new.cov_ui = self.cov_ui.copy(with_contents=with_contents)
        new.cov_qu = self.cov_qu.copy(with_contents=with_contents)
        return new

    def set_info(self, info):
        """
        Set the Info object for the source model.

        This sets the provided `info` as the primary Info object containing
        the configuration and reduction information for the source model.
        The source model will also take ownership of the `info` and set
        various parameters from the contents.

        Parameters
        ----------
        info : sofia_redux.info.info.Info

        Returns
        -------
        None
        """
        self.nr.set_info(info)
        self.nt.set_info(info)
        self.qr.set_info(info)
        self.qt.set_info(info)
        self.ur.set_info(info)
        self.ut.set_info(info)
        self.cov_qi.set_info(info)
        self.cov_ui.set_info(info)
        self.cov_qu.set_info(info)
        super().set_info(info)

    def set_scans(self, scans):
        """
        Set the scans for this model by reference.

        Nothing fancy, but important for child classes.

        Parameters
        ----------
        scans : list (Scan)

        Returns
        -------
        None
        """
        super().set_scans(scans)
        self.nr.set_scans(scans)
        self.nt.set_scans(scans)
        self.qr.set_scans(scans)
        self.qt.set_scans(scans)
        self.ur.set_scans(scans)
        self.ut.set_scans(scans)
        self.cov_qi.set_scans(scans)
        self.cov_ui.set_scans(scans)
        self.cov_qu.set_scans(scans)

    def purge_artifacts(self):
        """
        Generally used to remove all data but that relevant to the model.

        Returns
        -------
        None
        """
        self.nr.purge_artifacts()
        self.nt.purge_artifacts()
        self.qr.purge_artifacts()
        self.qt.purge_artifacts()
        self.ur.purge_artifacts()
        self.ut.purge_artifacts()
        self.cov_qi.purge_artifacts()
        self.cov_ui.purge_artifacts()
        self.cov_qu.purge_artifacts()
        super().purge_artifacts()

    @property
    def referenced_attributes(self):
        """
        Return attributes that should be referenced during a copy.

        Bypasses the standard copy algorithm for speed by referencing I,Q,U
        maps which are individually copied by the above copy method.

        Returns
        -------
        set (str)
        """
        referenced = super().referenced_attributes
        referenced.add('nr')
        referenced.add('qr')
        referenced.add('ur')
        referenced.add('nt')
        referenced.add('qt')
        referenced.add('ut')
        referenced.add('cov_qi')
        referenced.add('cov_ui')
        referenced.add('cov_qu')
        return referenced

    def create_from(self, scans, assign_scans=True):
        """
        Initialize model from scans.

        In addition to the standard maps, also creates maps for each Stokes
        Q, U, and N parameter, and each R and T HAWC+ subarray for a total
        of 6 extra maps.

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

        # Applies weighting to N, Q, and U
        # with Q and U weight scaling derived from N
        self.enable_weighting = False  # I map

        enable_stokes_weighting = False
        self.n.enable_weighting = enable_stokes_weighting
        self.q.enable_weighting = enable_stokes_weighting
        self.u.enable_weighting = enable_stokes_weighting

        enable_component_weighting = False

        self.nr = self.n.copy()
        self.nr.stand_alone()
        self.nr.id = 'NR'
        self.nr.enable_weighting = enable_component_weighting

        self.nt = self.n.copy()
        self.nt.stand_alone()
        self.nt.id = 'NT'
        self.nt.enable_weighting = enable_component_weighting

        self.qr = self.q.copy()
        self.qr.stand_alone()
        self.qr.id = 'QR'
        self.qr.enable_weighting = enable_component_weighting

        self.qt = self.q.copy()
        self.qt.stand_alone()
        self.qt.id = 'QT'
        self.qt.enable_weighting = enable_component_weighting

        self.ur = self.u.copy()
        self.ur.stand_alone()
        self.ur.id = 'UR'
        self.ur.enable_weighting = enable_component_weighting

        self.ut = self.u.copy()
        self.ut.stand_alone()
        self.ut.id = 'UT'
        self.ut.enable_weighting = enable_component_weighting

        self.cov_qi = self.n.copy()
        self.cov_qi.stand_alone()
        self.cov_qi.id = 'COV QI'
        self.cov_qi.enable_weighting = False

        self.cov_ui = self.n.copy()
        self.cov_ui.stand_alone()
        self.cov_ui.id = 'COV UI'
        self.cov_ui.enable_weighting = False

        self.cov_qu = self.n.copy()
        self.cov_qu.stand_alone()
        self.cov_qu.id = 'COV QU'
        self.cov_qu.enable_weighting = False

    def clear_process_brief(self):
        """
        Remove all process brief information.

        Returns
        -------
        None
        """
        super().clear_process_brief()
        self.nr.clear_process_brief()
        self.qr.clear_process_brief()
        self.ur.clear_process_brief()
        self.nt.clear_process_brief()
        self.qt.clear_process_brief()
        self.ut.clear_process_brief()
        self.cov_qi.clear_process_brief()
        self.cov_ui.clear_process_brief()
        self.cov_qu.clear_process_brief()

    def add_model_data(self, source_model, weight=1.0):
        """
        Add an increment source model data onto the current model.

        Parameters
        ----------
        source_model : HawcPlusPolarimetryMapSeparateRt
            The source model increment.
        weight : float, optional
            The weight of the source model increment.

        Returns
        -------
        None
        """
        self.nr.add_model(source_model.nr, weight=weight)
        self.nt.add_model(source_model.nt, weight=weight)
        if self.use_polarization:
            self.qr.add_model(source_model.qr, weight=weight)
            self.qt.add_model(source_model.qt, weight=weight)
            self.ur.add_model(source_model.ur, weight=weight)
            self.ut.add_model(source_model.ut, weight=weight)
            self.has_polarization = True

    def set_base(self):
        """
        Set the base to the map (copy of).

        Returns
        -------
        None
        """
        super().set_base()
        self.nr.set_base()
        self.nt.set_base()
        if self.use_polarization:
            self.qr.set_base()
            self.qt.set_base()
            self.ur.set_base()
            self.ut.set_base()
            self.set_base_main_map()

    def process_scan(self, scan):
        """
        Process a scan.

        Parameters
        ----------
        scan : sofia_redux.scan.scan.scan.Scan

        Returns
        -------
        None
        """
        self.nr.process_scan(scan)
        self.nt.process_scan(scan)
        if self.use_polarization:
            self.qr.process_scan(scan)
            self.qt.process_scan(scan)
            self.ur.process_scan(scan)
            self.ut.process_scan(scan)

    def reset_processing(self):
        """
        Reset the source processing.

        Returns
        -------
        None
        """
        super().reset_processing()
        self.nr.reset_processing()
        self.nt.reset_processing()
        if self.use_polarization:
            self.qr.reset_processing()
            self.qt.reset_processing()
            self.ur.reset_processing()
            self.ut.reset_processing()
            self.cov_qi.reset_processing()
            self.cov_ui.reset_processing()
            self.cov_qu.reset_processing()

    def clear_content(self):
        """
        Clear the data.

        Returns
        -------
        None
        """
        super().clear_content()
        self.nr.clear_content()
        self.nt.clear_content()
        if self.use_polarization:
            self.qr.clear_content()
            self.qt.clear_content()
            self.ur.clear_content()
            self.ut.clear_content()
            self.cov_qi.clear_content()
            self.cov_ui.clear_content()
            self.cov_qu.clear_content()

    def get_executor(self):
        """
        Return the source map parallel executor.

        The executor is not currently implemented in any way.

        Returns
        -------
        executor : object
        """
        for x in [self.n, self.q, self.u, self.nr, self.qr, self.ur,
                  self.nt, self.qt, self.ut, self.cov_qi, self.cov_ui,
                  self.cov_qu]:
            executor = x.get_executor()
            if executor is not None:
                return executor
        return None

    def set_executor(self, executor):
        """
        Set the parallel executor for the source.

        The executor is not currently implemented in any way.

        Parameters
        ----------
        executor : object

        Returns
        -------
        None
        """
        super().set_executor(executor)
        self.nr.set_executor(executor)
        self.qr.set_executor(executor)
        self.ur.set_executor(executor)
        self.nt.set_executor(executor)
        self.qt.set_executor(executor)
        self.ut.set_executor(executor)
        self.cov_qi.set_executor(executor)
        self.cov_ui.set_executor(executor)
        self.cov_qu.set_executor(executor)

    def get_parallel(self):
        """
        Get the number of parallel operations for the source model.

        Returns
        -------
        threads : int
        """
        for x in [self.n, self.q, self.u, self.nr, self.qr, self.ur,
                  self.nt, self.qt, self.ut, self.cov_qi, self.cov_ui,
                  self.cov_qu]:
            threads = x.get_parallel()
            if threads is not None:
                return threads
        return 1

    def set_parallel(self, threads):
        """
        Set the number of parallel operations for the source model.

        Parameters
        ----------
        threads : int

        Returns
        -------
        None
        """
        super().set_parallel(threads)
        self.nr.set_parallel(threads)
        self.qr.set_parallel(threads)
        self.ur.set_parallel(threads)
        self.nt.set_parallel(threads)
        self.qt.set_parallel(threads)
        self.ut.set_parallel(threads)
        self.cov_qi.set_parallel(threads)
        self.cov_ui.set_parallel(threads)
        self.cov_qu.set_parallel(threads)

    def count_points(self):
        """
        Return the number of points in the source map.

        Returns
        -------
        points : int
        """
        c = 0
        for x in [self.nr, self.nt, self.qr, self.qt, self.ur, self.ut]:
            c += x.count_points()
        return c

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
        if name.startswith('NR.'):
            return self.n.get_table_entry(name[3:])
        elif name.startswith('QR.'):
            return self.q.get_table_entry(name[3:])
        elif name.startswith('UR.'):
            return self.u.get_table_entry(name[3:])
        elif name.startswith('NT.'):
            return self.n.get_table_entry(name[3:])
        elif name.startswith('QT.'):
            return self.q.get_table_entry(name[3:])
        elif name.startswith('UT.'):
            return self.u.get_table_entry(name[3:])
        elif name.startswith('CovQI.'):
            return self.cov_qi.get_table_entry(name[3:])
        elif name.startswith('CovUI.'):
            return self.cov_ui.get_table_entry(name[3:])
        elif name.startswith('CovQU.'):
            return self.cov_qu.get_table_entry(name[3:])
        else:
            return super().get_table_entry(name)

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
        initial_channel_flags = integration.channels.data.flag.copy()
        t_channels = integration.channels.data.sub >= 2
        t_indices = np.nonzero(t_channels)[0]
        r_indices = np.nonzero(~t_channels)[0]

        # Populate map indices now, in case it's overwritten...
        if integration.frames.map_index is None or np.allclose(
                integration.frames.map_index.x, -1):
            self.create_lookup(integration)

        stokes_maps = [(self.nr, self.nt)]
        if self.use_polarization:
            stokes_maps += [(self.qr, self.qt),
                            (self.ur, self.ut)]

        for r_map, t_map in stokes_maps:

            # R subarray map
            integration.channels.data.flag = initial_channel_flags.copy()
            integration.channels.data.set_flags('DEAD', indices=t_indices)
            integration.comments.append('R')
            r_map.add_integration(integration)

            # T subarray map
            integration.channels.data.flag = initial_channel_flags.copy()
            integration.channels.data.set_flags('DEAD', indices=r_indices)
            integration.comments.append('T')
            t_map.add_integration(integration)

        integration.channels.data.flag = initial_channel_flags

    def process(self):
        """
        Process the source model.

        The processing step will normalize source map data from `weight * data`
        values to `data` values.  This is usually followed by smoothing and
        filtering steps if required, and also masking certain areas of the
        source due to things like exposure time and S2N ranges.

        Returns
        -------
        None
        """
        self.add_process_brief('[NR]')
        self.nr.process()
        self.add_process_brief('[NT]')
        self.nt.process()
        if self.use_polarization:
            self.add_process_brief('[QR]')
            self.qr.process()
            self.add_process_brief('[QT]')
            self.qt.process()
            self.add_process_brief('[UR]')
            self.ur.process()
            self.add_process_brief('[UT]')
            self.ut.process()

        self.merge_all_flags(include_main=True)

    def process_on_stokes_i(self):
        """
        Process the source model.

        The processing step will normalize source map data from `weight * data`
        values to `data` values.  This is usually followed by smoothing and
        filtering steps if required, and also masking certain areas of the
        source due to things like exposure time and S2N ranges.

        Returns
        -------
        None
        """
        components = [self.nr, self.nt, self.qr, self.qt, self.ur, self.ut]
        for component in components:
            component.end_accumulation()

        self.aggregate_stokes_i()  # Stores stokes I as main map
        self.add_process_brief('[I]')
        self.map.data *= self.map.weight.data
        self.process_main_map()
        for component in components:
            component.map.data *= component.map.weight.data
            self.add_process_brief(f'[{component.id}]')
            component.process()
        self.merge_all_flags(include_main=True)

    def aggregate_stokes_i(self, variance_method=False, add_np_weights=False):
        """
        Aggregate the Stokes components to create the main Stokes I base map.

        Creates the Stokes I map from the individual components and updates
        the main source map with the values.

        Returns
        -------
        None
        """
        # Create N, Q, and U
        self.create_stokes(variance_method=variance_method)
        stokes_i = self.get_i(allow_invalid=True,
                              add_np_weights=add_np_weights)
        self.map.data = stokes_i.map.data
        self.map.weight.data = stokes_i.map.weight.data
        self.map.exposure.data = stokes_i.map.exposure.data
        self.map.flag = stokes_i.map.flag

    def merge_all_flags(self, include_main=False):
        """
        Merge masking flags for all maps

        Sets the map flags for all Stokes (N, Q, U) maps equal to the union
        of the flags of all Stokes maps.

        Parameters
        ----------
        include_main : bool, optional
            If `True`, include the flags present in the main source model
            (Stokes I).

        Returns
        -------
        None
        """
        all_maps = [self.nr, self.nt]
        if self.use_polarization:
            all_maps += [self.qr, self.qt, self.ur, self.ut]

        flags = np.zeros(self.shape, dtype=int)
        for x in all_maps:
            flags |= x.map.flag
        for x in all_maps:
            x.map.flag = flags.copy()

    def post_process_scan(self, scan):
        """
        Perform post-processing steps on a scan.

        Performs the post processing steps on the unpolarized N Stokes map.
        This typically involves determining the pointing correction for the
        source. Processing steps may affect properties of *this* map, but
        should not impact the primary reduction source polarization model.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        self.create_stokes_n()  # Process on N
        super().post_process_scan(scan)

    def sync_integration(self, integration, signal_mode=None):
        """
        Remove source from integration frame data.

        Performs the synchronization for all 6 {Q,U,N} x {R,T} maps.

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
        self.sync_integration_all_components(
            integration, signal_mode=signal_mode)

    def sync_integration_on_stokes(self, integration, signal_mode=None):
        """
        Remove source from integration frame data.

        Performs the synchronization on N, Q, and U

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
        self.aggregate_stokes_i()
        self.n.sync_integration(integration)
        if self.use_polarization:
            self.q.sync_integration(integration)
            self.u.sync_integration(integration)

    def sync_integration_all_components(self, integration, signal_mode=None):
        """
        Remove source from integration frame data.

        Performs the synchronization for all 6 {Q,U,N} x {R,T} maps.

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
        initial_channel_flags = integration.channels.data.flag.copy()
        t_channels = integration.channels.data.sub >= 2
        t_indices = np.nonzero(t_channels)[0]
        r_indices = np.nonzero(~t_channels)[0]

        # R subarrays
        integration.channels.data.flag = initial_channel_flags.copy()
        integration.channels.data.set_flags('DEAD', indices=t_indices)
        integration.comments.append('R')

        self.nr.sync_integration(integration)
        if self.use_polarization:
            self.qr.sync_integration(integration)
            self.ur.sync_integration(integration)

        # T subarrays
        integration.channels.data.flag = initial_channel_flags.copy()
        integration.channels.data.set_flags('DEAD', indices=r_indices)
        integration.comments.append('T')

        self.nt.sync_integration(integration)
        if self.use_polarization:
            self.qt.sync_integration(integration)
            self.ut.sync_integration(integration)

        integration.channels.data.flag = initial_channel_flags.copy()

    def process_final(self):
        """
        Perform the final processing steps.

        The final processing steps are performed on each {Q,U,N} x {R,T} map
        before calculating the final combined Stokes maps.

        Returns
        -------
        None
        """
        # Correct weight by just unpolarized
        chi2_r = self.nr.map.get_chi2(robust=True)
        chi2_t = self.nt.map.get_chi2(robust=True)
        if np.isfinite(chi2_r) and chi2_r != 0:
            self.nr.map.weight.data /= chi2_r
            self.qr.map.weight.data /= chi2_r
            self.ur.map.weight.data /= chi2_r
        if np.isfinite(chi2_t) and chi2_t != 0:
            self.nt.map.weight.data /= chi2_t
            self.qt.map.weight.data /= chi2_t
            self.ut.map.weight.data /= chi2_t

        # First time to create I, N, Q, and U
        self.aggregate_stokes_i(add_np_weights=True)

        # Process final (no weighting should occur)
        self.n.process_final()
        self.q.process_final()
        self.u.process_final()
        self.process_final_main_map()

        n_chi2 = self.n.map.get_chi2(robust=True)
        i_chi2 = self.map.get_chi2(robust=True)

        # Manual scaling of components
        if np.isfinite(i_chi2) and i_chi2 != 0:
            for stokes in [self.n.map, self.q.map, self.u.map]:
                stokes.weight.data *= i_chi2

        # Manual scaling of I based on previous N chi_2
        if np.isfinite(n_chi2) and n_chi2 != 0:
            self.map.weight.data *= n_chi2

        # Manual re-scaling of R and T components
        for rt_map in [self.nr.map, self.qr.map, self.ur.map,
                       self.nt.map, self.qt.map, self.ut.map]:
            rt_map.weight.data *= n_chi2 / i_chi2

        self.generate_covariance_maps_from_final_processed_maps()

    def populate_stokes_map(self, stokes_map, r_map, t_map,
                            variance_method=False):
        """
        Populate a Stokes map for the given R and T maps.

        Parameters
        ----------
        stokes_map : AstroIntensityMap
        r_map : AstroIntensityMap
        t_map : AstroIntensityMap
        variance_method : bool, optional
            If `True`, use the variance method for adding weights.

        Returns
        -------
        None
        """
        sign = 1 if stokes_map.signal_mode == self.polarimetry_flags.N else -1
        stokes_map.clear_content()

        r = r_map.map.data.copy()
        t = t_map.map.data.copy()
        wr = r_map.map.weight.data.copy()
        wt = t_map.map.weight.data.copy()

        valid = r_map.map.is_valid() & t_map.map.is_valid()
        valid &= (wr != 0) & (wt != 0)
        invalid = np.logical_not(valid)

        exposure = r_map.map.exposure.data + t_map.map.exposure.data
        for arr in [r, t, wr, wt, exposure]:
            arr[invalid] = 0.0

        d = 0.5 * (r + t) if sign == 1 else 0.25 * (r - t)

        if variance_method:
            vr = wr.copy()
            vr[valid] = 1.0 / vr[valid]
            vt = wt.copy()
            vt[valid] = 1.0 / vt[valid]
            v = 0.25 * (vr + vt)
            w = v.copy()
            w[valid] = 1.0 / w[valid]
        else:  # Active method
            w = wr + wt

        w[~valid] = 0.0
        dw = d * w

        stokes_map.map.clear(indices=invalid)
        stokes_map.map.data = dw
        stokes_map.map.weight.data = w
        stokes_map.map.exposure.data = exposure

        stokes_map.process()
        stokes_map.map.validate()

    def create_stokes(self, variance_method=False):
        """
        Create the full Stokes N, Q, and U maps.

        Returns
        -------
        None
        """
        self.create_stokes_n(variance_method=variance_method)
        if self.use_polarization:
            self.create_stokes_q(variance_method=variance_method)
            self.create_stokes_u(variance_method=variance_method)

    def create_stokes_n(self, variance_method=False):
        """
        Create the Stokes N map from the R and T subarray maps.

        The R and T (N polarization) maps should have been normalized (not
        at the accumulated stage).

        Returns
        -------
        None
        """
        self.populate_stokes_map(self.n, self.nr, self.nt,
                                 variance_method=variance_method)

    def create_stokes_q(self, variance_method=False):
        """
        Create the Stokes N map from the R and T subarray maps.

        Returns
        -------
        None
        """
        self.populate_stokes_map(self.q, self.qr, self.qt,
                                 variance_method=variance_method)

    def create_stokes_u(self, variance_method=False):
        """
        Create the Stokes N map from the R and T subarray maps.

        Returns
        -------
        None
        """
        self.populate_stokes_map(self.u, self.ur, self.ut,
                                 variance_method=variance_method)

    def scale_qu_weights_from_n(self):
        """
        Scale the Q and U maps based on N.

        Returns
        -------
        None
        """
        chi2 = self.n.map.get_chi2(robust=True)

        self.n.map.weight.data /= chi2
        self.q.map.weight.data /= chi2
        self.u.map.weight.data /= chi2
        print(f'\n\n\nchi2 N: {chi2}\n\n\n')

    def set_smoothing(self, smoothing):
        """
        Set the model smoothing.

        Parameters
        ----------
        smoothing : units.Quantity

        Returns
        -------
        None
        """
        super().set_smoothing(smoothing)
        self.nr.set_smoothing(smoothing)
        self.nt.set_smoothing(smoothing)
        if self.use_polarization:
            self.qr.set_smoothing(smoothing)
            self.qt.set_smoothing(smoothing)
            self.ur.set_smoothing(smoothing)
            self.ut.set_smoothing(smoothing)

    def is_valid(self):
        """
        Return whether the polarimetry model is valid.

        Returns
        -------
        bool
        """
        if not self.nr.is_valid() or not self.nt.is_valid():
            return False
        if self.has_polarization:
            for x in [self.qr, self.qt, self.ur, self.ut]:
                if not x.is_valid():
                    return False
        return True

    def generate_covariance_maps_from_final_processed_maps(self):
        """
        Calculate Cov(Q,I), Cov(U,I) and Cov(Q,U) from final processed maps.

        Returns
        -------
        None
        """
        if not self.use_polarization:
            return None
        log.info("Generating covariance maps")
        valid = self.map.is_valid()
        for cov_map in (self.cov_qi, self.cov_ui, self.cov_qu):
            cov_map.reset_processing()
            cov_map.clear_content()

        qr = self.qr.copy()
        qt = self.qt.copy()
        ur = self.ur.copy()
        ut = self.ut.copy()
        nr = self.nr.copy()
        nt = self.nt.copy()

        var_nr = nr.map.get_noise().data ** 2
        var_nt = nt.map.get_noise().data ** 2
        var_qr = qr.map.get_noise().data ** 2
        var_qt = qt.map.get_noise().data ** 2
        var_ur = ur.map.get_noise().data ** 2
        var_ut = ut.map.get_noise().data ** 2

        var_ir = var_nr + 0.25 * (var_qr + var_ur)
        var_it = var_nt + 0.25 * (var_qt + var_ut)

        # Negative version
        cov_qu = var_ur - var_ut - var_qr + var_qt
        cov_qi = var_ir - var_it - var_qr + var_qt
        cov_ui = var_ir - var_it - var_ur + var_ut

        # Cov(Q,U) looks valid when compared with previous method
        self.cov_qu.map.add(cov_qu)
        self.cov_qu.map.get_weights().add(16 * valid)
        self.cov_qu.map.get_exposures().add(self.map.get_exposure_image().data)
        self.cov_qu.map.end_accumulation()
        self.cov_qu.map.validate()

        # Cov(Q, I)
        self.cov_qi.map.add(cov_qi)
        self.cov_qi.map.get_weights().add(16 * valid)
        self.cov_qi.map.get_exposures().add(self.map.get_exposure_image().data)
        self.cov_qi.map.end_accumulation()
        self.cov_qi.map.validate()

        # Cov(U, I)
        self.cov_ui.map.add(cov_ui)
        self.cov_ui.map.get_weights().add(16 * valid)
        self.cov_ui.map.get_exposures().add(self.map.get_exposure_image().data)
        self.cov_ui.map.end_accumulation()
        self.cov_ui.map.validate()

    def generate_covariance_maps(self):
        """
        Test method for covariance maps (not used)

        Returns
        -------
        None
        """
        if not self.use_polarization:
            return None, None, None

        log.info("Generating covariance maps")

        self.cov_qi.reset_processing()
        self.cov_qi.clear_content()
        self.cov_ui.reset_processing()
        self.cov_ui.clear_content()
        self.cov_qu.reset_processing()
        self.cov_qu.clear_content()

        cov_qnr = self.cov_qu.get_clean_local_copy()
        cov_qnt = self.cov_qu.get_clean_local_copy()
        cov_unr = self.cov_qu.get_clean_local_copy()
        cov_unt = self.cov_qu.get_clean_local_copy()
        cov_qur = self.cov_qu.get_clean_local_copy()
        cov_qut = self.cov_qu.get_clean_local_copy()

        stokes_maps = [(self.nr, self.nt),
                       (self.qr, self.qt),
                       (self.ur, self.ut)]

        valid = np.full(self.map.shape, True)
        for r_map, t_map in stokes_maps:
            valid &= r_map.map.is_valid()
            valid &= t_map.map.is_valid()

        chi2_r = self.nr.map.get_chi2(robust=True)
        chi2_t = self.nt.map.get_chi2(robust=True)

        var_nr = valid * chi2_r * self.nr.map.get_noise().data ** 2
        var_nt = valid * chi2_t * self.nt.map.get_noise().data ** 2
        var_qr = valid * chi2_r * self.qr.map.get_noise().data ** 2
        var_qt = valid * chi2_t * self.qt.map.get_noise().data ** 2
        var_ur = valid * chi2_r * self.ur.map.get_noise().data ** 2
        var_ut = valid * chi2_t * self.ut.map.get_noise().data ** 2

        cov_qnr.map.add(valid * (var_qr - var_nr))
        cov_qnt.map.add(valid * (var_qt - var_nt))
        cov_unr.map.add(valid * (var_ur - var_nr))
        cov_unt.map.add(valid * (var_ut - var_nt))
        cov_qur.map.add(valid * (var_qr - var_ur))
        cov_qut.map.add(valid * (var_qt - var_ut))
        for component in [cov_qnr, cov_qnt, cov_unr, cov_unt, cov_qur,
                          cov_qut]:
            component.map.get_weights().add(valid * 2.0)

    def generate_qu_covariance_map(self):
        """
        Generate the (Q, U) covariance map.  (Not used)

        Returns
        -------
        None
        """
        log.info("Generating Cov(Q,U) map")
        self.cov_qu.reset_processing()
        self.cov_qu.clear_content()
        vq = self.q.map.get_noise().data ** 2
        vu = self.u.map.get_noise().data ** 2
        self.cov_qu.map.add(vq - vu)
        self.cov_qu.map.get_weight_image().add(4.0)
        self.cov_qu.map.get_exposure_image().add(
            self.q.map.exposure.data + self.u.map.exposure.data)
        self.cov_qu.map.end_accumulation()

    def get_covariance_maps(self):
        """
        Return the covariance HDUs for the final write.

        Returns
        -------
        covariance_hdus : dict
            The name key should be one of {'QI':HDU, 'UI':HDU, 'QU':HDU,
            'valid':bool}.  If 'valid' is set to `True`, then these should
            be written to disk if requested.
        """
        return {
            'QI': self.cov_qi,
            'UI': self.cov_ui,
            'QU': self.cov_qu,
            'valid': True
        }
