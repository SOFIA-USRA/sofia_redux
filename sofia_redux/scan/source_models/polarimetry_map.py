# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units

from sofia_redux.scan.flags.polarimetry_flags import PolarModulation
from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
from sofia_redux.scan.source_models.source_numba_functions import (
    calculate_polarized_power, calculate_total_power,
    calculate_polarized_fraction, calculate_polarization_angles)


class PolarimetryMap(AstroIntensityMap):

    polarimetry_flags = PolarModulation

    def __init__(self, info, reduction=None):
        """
        Initialize a polarization map.

        Parameters
        ----------
        info : sofia_redux.scan.info.info.Info
            The Info object which should belong to this source model.
        reduction : sofia_redux.scan.reduction.reduction.Reduction, optional
            The reduction for which this source model should be applied.
        """
        self.n = AstroIntensityMap(info, reduction=reduction)
        self.q = AstroIntensityMap(info, reduction=reduction)
        self.u = AstroIntensityMap(info, reduction=reduction)
        self.has_polarization = False
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
        PolarimetryMap
        """
        new = super().copy(with_contents=with_contents)
        new.n = new.n.copy(with_contents=with_contents)
        new.q = new.q.copy(with_contents=with_contents)
        new.u = new.u.copy(with_contents=with_contents)
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
        self.n.set_info(info)
        self.q.set_info(info)
        self.u.set_info(info)
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
        self.n.set_scans(scans)
        self.q.set_scans(scans)
        self.u.set_scans(scans)

    def purge_artifacts(self):
        """
        Generally used to remove all data but that relevant to the model.

        Returns
        -------
        None
        """
        self.n.purge_artifacts()
        self.q.purge_artifacts()
        self.u.purge_artifacts()
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
        referenced.add('n')
        referenced.add('q')
        referenced.add('u')
        return referenced

    @property
    def use_polarization(self):
        """
        Return whether to use polarization from the configuration.

        Returns
        -------
        bool
        """
        if self.configuration is None:
            return False
        return self.configuration.get_bool('source.polar')

    def get_map_instance(self):
        """
        Return an initialized AstroIntensityMap.

        Returns
        -------
        AstroIntensityMap
        """
        return AstroIntensityMap(info=self.info, reduction=self.reduction)

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
        self.n = self.get_map_instance()
        self.n.create_from(scans, assign_scans=False)
        self.n.signal_mode = self.polarimetry_flags.N
        self.n.enable_level = True
        self.n.enable_bias = True
        self.n.enable_weighting = True

        self.n.id = 'N'

        self.q = self.n.copy()
        self.q.signal_mode = self.polarimetry_flags.Q
        self.q.stand_alone()
        self.q.enable_level = False
        self.q.enable_bias = False
        self.q.enable_weighting = True
        self.q.id = 'Q'

        self.u = self.q.copy()
        self.u.signal_mode = self.polarimetry_flags.U
        self.u.stand_alone()
        self.u.id = 'U'

    def is_valid(self):
        """
        Return whether the polarimetry model is valid.

        Returns
        -------
        bool
        """
        if not self.n.is_valid():
            return False
        if self.has_polarization:
            if not self.q.is_valid() or not self.u.is_valid():
                return False
        return True

    def clear_process_brief(self):
        """
        Remove all process brief information.

        Returns
        -------
        None
        """
        super().clear_process_brief()
        self.n.clear_process_brief()
        self.q.clear_process_brief()
        self.u.clear_process_brief()

    def add_model_data(self, source_model, weight=1.0):
        """
        Add an increment source model data onto the current model.

        Parameters
        ----------
        source_model : PolarimetryMap
            The source model increment.
        weight : float, optional
            The weight of the source model increment.

        Returns
        -------
        None
        """
        self.n.add_model(source_model.n, weight=weight)
        if self.use_polarization:
            self.q.add_model(source_model.q, weight=weight)
            self.u.add_model(source_model.u, weight=weight)
            self.has_polarization = True

    def set_base(self):
        """
        Set the base to the map (copy of).

        Returns
        -------
        None
        """
        self.n.set_base()
        if self.use_polarization:
            self.q.set_base()
            self.u.set_base()

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
        self.n.process_scan(scan)
        if self.use_polarization:
            self.q.process_scan(scan)
            self.u.process_scan(scan)

    def process_main_map(self):
        """
        Perform the standard processing steps on the main map.

        Returns
        -------
        None
        """
        super().process()

    def sync_integration_main_map(self, integration, signal_mode=None):
        """
        Remove source from integration frame data using the main map.

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
        super().sync_integration(integration, signal_mode=signal_mode)

    def set_base_main_map(self):
        """
        Set the base to the map (copy of) for the main source map.

        Returns
        -------
        None
        """
        super().set_base()

    def process(self):
        """
        Process the source model.

        Returns
        -------
        None
        """
        self.add_process_brief('[N]')
        self.n.process()
        # TODO: propagate masks from N to U/Q
        if self.use_polarization:
            self.add_process_brief('[Q]')
            self.q.process()
            self.add_process_brief('[U]')
            self.u.process()
            self.n.merge_mask(self.u.map)
            self.n.merge_mask(self.q.map)
            self.u.merge_mask(self.n.map)
            self.q.merge_mask(self.n.map)

    def reset_processing(self):
        """
        Reset the source processing.

        Returns
        -------
        None
        """
        super().reset_processing()
        self.n.reset_processing()
        if self.use_polarization:
            self.q.reset_processing()
            self.u.reset_processing()

    def clear_content(self):
        """
        Clear the data.

        Returns
        -------
        None
        """
        self.n.clear_content()
        if self.use_polarization:
            self.q.clear_content()
            self.u.clear_content()

    def merge_accumulate(self, other):
        """
        Merge another source with this one.

        Parameters
        ----------
        other : PolarimetryMap

        Returns
        -------
        None
        """
        self.n.map.merge_accumulate(other.n.map)
        self.q.map.merge_accumulate(other.q.map)
        self.u.map.merge_accumulate(other.u.map)

    def post_process_scan_main(self, scan):
        """
        Perform post-processing steps on scan using main map.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        super().post_process_scan(scan)

    def post_process_scan(self, scan):
        """
        Perform post-processing steps on a scan.

        At this stage, perform the post-processing using the total-power map.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        super().post_process_scan(scan)
        self.n.post_process_scan(scan)

    def get_source_name(self):
        """
        Return the source name.

        Returns
        -------
        str
        """
        return self.n.get_source_name()

    def get_unit(self):
        """
        Return the map data unit.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.n.get_unit()

    def get_executor(self):
        """
        Return the source map parallel executor.

        The executor is not currently implemented in any way.

        Returns
        -------
        executor : object
        """
        for x in [self.n, self.q, self.u]:
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
        self.n.set_executor(executor)
        self.q.set_executor(executor)
        self.u.set_executor(executor)

    def get_parallel(self):
        """
        Get the number of parallel operations for the source model.

        Returns
        -------
        threads : int
        """
        for x in [self.n, self.q, self.u]:
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
        self.n.set_parallel(threads)
        self.q.set_parallel(threads)
        self.u.set_parallel(threads)

    def count_points(self):
        """
        Return the number of points in the source map.

        Returns
        -------
        points : int
        """
        c = 0
        for x in [self.n, self.q, self.u]:
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
        if name.startswith('N.'):
            return self.n.get_table_entry(name[2:])
        elif name.startswith('Q.'):
            return self.q.get_table_entry(name[2:])
        elif name.startswith('U.'):
            return self.u.get_table_entry(name[2:])
        else:
            return super().get_table_entry(name)

    def get_reference(self):
        """
        Return the reference pixel of the source model WCS.

        Returns
        -------
        numpy.ndarray (float)
            The (x, y) reference position of the source model.
        """
        return self.n.get_reference()

    def get_p(self, debias=True):
        """
        Return the polarized power map.

        Returns
        -------
        AstroIntensityMap
        """
        p_map = self.n.copy()
        p_map.clear_content()
        p = p_map.map
        discard_flag = p.flagspace.convert_flag('DISCARD').value

        calculate_polarized_power(p=p.data,
                                  p_weight=p.weight.data,
                                  p_flag=p.flag,
                                  q=self.q.map.data,
                                  q_weight=self.q.map.weight.data,
                                  q_valid=self.q.map.is_valid(),
                                  u=self.u.map.data,
                                  u_weight=self.u.map.weight.data,
                                  u_valid=self.u.map.is_valid(),
                                  bad_flag=discard_flag)
        p_map.id = 'P'
        p.validate()
        return p_map

    def get_i(self, p=None, allow_invalid=False, add_np_weights=True):
        """
        Return the total power map.

        Parameters
        ----------
        p : AstroIntensityMap, optional
            The polarized power map.  If not supplied, defaults to the
            polarized power map derived from this data.
        allow_invalid : bool, optional
            If `True`, does not pay attention to invalid Q/U map points.  This
            is important if they have been marked as invalid due to previous
            clipping operations that are irrelevant when creating the
            total intensity map.
        add_np_weights : bool, optional
           If `True`, calculate I weights by aggregating N and P weights.
           Otherwise, just use N weights (copy).

        Returns
        -------
        AstroIntensityMap
        """
        if p is None:
            p = self.get_p()
        i = self.n.copy()
        i.clear_content()
        discard_flag = p.flagspace.convert_flag('DISCARD').value

        calculate_total_power(
            polarized_power=p.map.data,
            unpolarized_power=self.n.map.data,
            polarized_valid=p.map.is_valid(),
            unpolarized_valid=self.n.map.is_valid(),
            polarized_weight=p.map.weight.data,
            unpolarized_weight=self.n.map.weight.data,
            exposure=self.n.map.exposure.data,
            total_power=i.map.data,
            total_power_weight=i.map.weight.data,
            total_power_flags=i.map.flag,
            total_power_exposure=i.map.exposure.data,
            bad_flag=discard_flag,
            allow_invalid=allow_invalid,
            add_np_weights=True
        )

        i.id = 'I'
        i.map.validate()
        return i

    def get_polarized_fraction(self, polarized_power, total_power,
                               accuracy=None):
        """
        Return the polarized fraction.

        Parameters
        ----------
        polarized_power : AstroIntensityMap
            The polarized power map.
        total_power : AstroIntensityMap
            The total power map.
        accuracy : float, optional
            The fractional accuracy.  Anything less than this will be flagged
            in the output product.  The default is 3 percent.

        Returns
        -------
        polarized_fraction : AstroIntensityMap
        """
        fraction = polarized_power.copy()
        fraction.clear_content()
        discard_flag = fraction.flagspace.convert_flag('DISCARD').value

        if accuracy is None:
            accuracy = self.configuration.get_float(
                'source.polar.fraction.rmsclip', default=0.03)

        calculate_polarized_fraction(
            total_power=total_power.map.data,
            total_power_weight=total_power.map.weight.data,
            total_power_valid=total_power.map.is_valid(),
            polarized=polarized_power.map.data,
            polarized_weight=polarized_power.map.weight.data,
            polarized_valid=polarized_power.map.is_valid(),
            exposure=total_power.map.exposure.data,
            fraction=fraction.map.data,
            fraction_weight=fraction.map.weight.data,
            fraction_exposure=fraction.map.exposure.data,
            fraction_flags=fraction.map.flag,
            bad_flag=discard_flag,
            accuracy=accuracy)

        fraction.id = 'F'
        fraction.enable_level = False
        fraction.enable_weighting = False
        fraction.enable_bias = False
        fraction.map.set_unit(units.dimensionless_unscaled)
        fraction.map.validate()
        return fraction

    def get_angles(self, polarized_power, polarized_fraction):
        """
        Return the polarization angle map.

        The angles are measured East of North.

        Parameters
        ----------
        polarized_power : AstroIntensityMap
            The polarized power map.
        polarized_fraction : AstroIntensityMap
            The polarized fraction map.

        Returns
        -------
        angles : AstroIntensityMap
        """
        angles = self.n.copy()
        angles.clear_content()
        discard_flag = angles.flagspace.convert_flag('DISCARD').value

        calculate_polarization_angles(
            polarized_power=polarized_power.map.data,
            fraction_valid=polarized_fraction.map.is_valid(),
            stokes_q=self.q.map.data,
            stokes_q_weight=self.q.map.weight.data,
            stokes_u=self.u.map.data,
            stokes_u_weight=self.u.map.weight.data,
            angles=angles.map.data,
            angles_weight=angles.map.weight.data,
            angles_flag=angles.map.flag,
            bad_flag=discard_flag)

        angles.id = 'A'
        angles.map.exposure = polarized_fraction.map.exposure.copy()
        angles.map.validate()
        angles.enable_level = False
        angles.enable_weighting = False
        angles.enable_bias = False
        angles.map.set_unit(units.Unit('degree'))
        return angles

    def write(self, path):
        """
        Write the source to file.

        Performing a write operation will write various products to the
        `path` directory.  If any intermediate.<id>.fits file is found
        it will be delete

        Parameters
        ----------
        path : str
            The directory to write to.
        accuracy : float, optional

        Returns
        -------
        None
        """
        self.n.write(path)
        if not self.has_polarization:
            log.warning(
                "No polarization products available.  Consider settings the "
                "'source.polarization' option to create Q, U, P, and I "
                "images (and optionally F).")
            return

        self.q.write(path)  # Stokes Q
        self.u.write(path)  # Stokes U
        p = self.get_p()
        p.write(path)  # Polarized power
        i = self.get_i(p=p)
        i.write(path)  # Total power

        get_fraction = self.configuration.get_bool('source.polar.fraction')
        get_angles = self.configuration.get_bool('source.polar.angles')
        if not get_fraction and not get_angles:
            return

        f = self.get_polarized_fraction(p, i)
        if get_fraction:
            f.write(path)  # Polarized fraction

        if get_angles:
            a = self.get_angles(p, f)
            a.write(path)  # Polarization angle

    def add_integration(self, integration, signal_mode=None):
        """
        Add an integration to the polarimetry model.

        The integration NEFD is calculated, and then frames are added via
        :func:`AstroModel2D.add_frames_from_integration`.  A filter correction
        is applied on the first source generation only.

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
        self.n.add_integration(integration)
        if self.use_polarization:
            self.q.add_integration(integration)
            self.u.add_integration(integration)

    def standard_add_integration(self, integration, signal_mode=None):
        """
        Add an integration to the polarimetry model.

        The integration NEFD is calculated, and then frames are added via
        :func:`AstroModel2D.add_frames_from_integration`.  A filter correction
        is applied on the first source generation only.

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
        super().add_integration(integration, signal_mode=signal_mode)

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
        self.n.sync_integration(integration)
        if self.use_polarization:
            self.q.sync_integration(integration)
            self.u.sync_integration(integration)

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
        self.smoothing = smoothing
        self.n.set_smoothing(smoothing)
        if self.use_polarization:
            self.q.set_smoothing(smoothing)
            self.u.set_smoothing(smoothing)

    def smooth(self):
        """
        Smooth the source model.

        Returns
        -------
        None
        """
        super().smooth()
        self.n.smooth()
        if self.use_polarization:
            self.q.smooth()
            self.u.smooth()

    def add_integration_time(self, time):
        """
        Add integration time to the source model.

        Parameters
        ----------
        time : astropy.units.Quantity
            The time to add.

        Returns
        -------
        None
        """
        self.n.integration_time += time
        self.q.integration_time += time
        self.u.integration_time += time
        super().add_integration_time(time)
