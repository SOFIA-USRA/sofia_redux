# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
import numpy as np
import os
import warnings

from sofia_redux.scan.source_models.source_model import SourceModel
from sofia_redux.scan.source_models.sky_dip_model import SkyDipModel
from sofia_redux.scan.source_models import source_numba_functions as snf
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.utilities.utils import round_values

__all__ = ['SkyDip']


class SkyDip(SourceModel):

    def __init__(self, info, reduction=None):
        """
        Initialize a skydip model.

        The SkyDip source model is used to measure and store a model of the
        instrument response and atmospheric parameters with respect to the
        elevation of an observation.

        Parameters
        ----------
        info : sofia_redux.scan.info.info.Info
            The Info object which should belong to this source model.
        reduction : sofia_redux.scan.reduction.reduction.Reduction, optional
            The reduction for which this source model should be applied.
        """
        super().__init__(info=info, reduction=reduction)
        self.data = None
        self.weight = None
        self.resolution = 0.0
        self.tamb = 0.0 * units.Unit('Kelvin')
        self.tamb_weight = 0.0 * units.Unit('second')
        self.signal_name = 'obs-channels'
        self.signal_index = 0
        self.model = SkyDipModel()

    def copy(self, with_contents=True):
        """
        Return a copy of the SkyDip.

        Parameters
        ----------
        with_contents : bool, optional
            If `True`, return a true copy of the source model.  Otherwise, just
            return a basic version with the necessary meta data.

        Returns
        -------
        SkyDip
        """
        copy = super().copy(with_contents=with_contents)
        return copy

    def clear_all_memory(self):
        """
        Clear all memory references prior to deletion.

        Returns
        -------
        None
        """
        super().clear_all_memory()
        self.data = None
        self.weight = None

    @property
    def logging_id(self):
        """
        Return the logging ID.

        Returns
        -------
        str
        """
        return 'skydip'

    def get_unit(self):
        """
        Return the unit for the model

        Returns
        -------
        units.Unit
        """
        return units.Unit('Kelvin')

    def get_table_entry(self, name):
        """
        Return a parameter value for a given name.

        Parameters
        ----------
        name : str

        Returns
        -------
        value
        """
        parameters = self.model.parameters
        errors = self.model.errors
        if parameters is None:
            parameters = {}
        if errors is None:
            errors = {}

        if name == 'tau':
            return parameters.get('tau')
        elif name == 'dtau':
            return errors.get('tau')
        elif name == 'kelvin':
            return parameters.get('kelvin')
        elif name == 'dkelvin':
            return errors.get('kelvin')
        elif name == 'tsky':
            return parameters.get('tsky')
        elif name == 'dtsky':
            return errors.get('tsky')
        else:
            return super().get_table_entry(name)

    def clear_content(self):
        """
        Clear the SkyDip data contents.

        Returns
        -------
        None
        """
        if isinstance(self.data, np.ndarray):
            self.data.fill(0.0)
        else:
            self.data = None
        if isinstance(self.weight, np.ndarray):
            self.weight.fill(0.0)
        else:
            self.weight = None
        self.tamb = 0.0 * units.Unit('Kelvin')
        self.tamb_weight = 0.0 * units.Unit('second')

    def average_temp_with(self, temp, weight):
        """
        Average the existing temperature with another.

        Parameters
        ----------
        temp : units.Quantity
            The temperature in Kelvins to average with the current temperature.
        weight : units.Quantity
            The weight value of `temp` in seconds.

        Returns
        -------
        None
        """
        if np.isnan(temp) or np.isnan(weight) or weight == 0:
            return

        wd_sum = self.tamb * self.tamb_weight
        wd_sum += temp * weight
        w_sum = self.tamb_weight + weight
        self.tamb = wd_sum / w_sum if w_sum > 0 else wd_sum / units.Unit('s')
        self.tamb_weight = w_sum

    def create_from(self, scans, assign_scans=True):
        """
        Initialize a skydip model from scans.

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
        if self.has_option('skydip.grid'):
            resolution = self.configuration.get_float('skydip.grid')
            self.resolution = resolution * units.Unit('arcsec')
        else:
            self.resolution = 900 * units.Unit('arcsec')  # 1/4 degree

        if self.has_option('skydip.signal'):
            self.signal_name = self.configuration.get_string('skydip.signal')
        if self.has_option('skydip.mode'):
            self.signal_index = self.configuration.get_int('skydip.mode')

        n_data = ((90 * units.Unit('degree')) / self.resolution).decompose()
        n_data = int(np.ceil(n_data))
        self.data = np.zeros(n_data, dtype=float)
        self.weight = np.zeros(n_data, dtype=float)

    def get_bin(self, elevation):
        """
        Return the bin for the given elevation.

        Parameters
        ----------
        elevation : units.Quantity

        Returns
        -------
        bins : int or numpy.ndarray (int)
        """
        singular = elevation.shape == ()
        if singular and (np.isnan(elevation) or self.resolution == 0):
            return -1

        float_bins = (elevation / self.resolution).decompose().value
        if singular:
            if not np.isfinite(float_bins):
                return -1
            else:
                return int(round_values(float_bins))

        float_bins[~np.isfinite(float_bins)] = -1
        return round_values(float_bins)

    def get_elevation(self, bins):
        """
        Return the elevation for a given bin or bins.

        Parameters
        ----------
        bins : int or numpy.ndarray (int)

        Returns
        -------
        elevation : units.Quantity
        """
        return (bins + 0.5) * self.resolution

    def add_model_data(self, skydip, weight=1.0):
        """
        Add an increment source model data onto the current model.

        Parameters
        ----------
        skydip : SkyDip
            The skydip increment.
        weight : float, optional
            The weight of the model increment.  Not applicable for the skydip
            model.

        Returns
        -------
        None
        """
        # Average the temperature
        self.average_temp_with(skydip.tamb, skydip.tamb_weight)

        # Average the data
        if skydip.data is None:
            return

        if self.data is None:
            self.data = skydip.data.copy()
            self.weight = skydip.weight.copy()
            return

        if self.data.shape != skydip.data.shape:
            raise ValueError(f"SkyDip data shapes do not match: "
                             f"{self.data.shape} != {skydip.data.shape}")

        wd_sum = self.data * self.weight
        wd_sum += skydip.data * skydip.weight
        w_sum = self.weight + skydip.weight
        nzi = w_sum > 0
        self.data[nzi] = wd_sum[nzi] / w_sum[nzi]
        self.data[~nzi] = wd_sum[~nzi]
        self.weight = w_sum

    def add_integration(self, integration):
        """
        Add an integration to the sky dip.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        integration.comments.append("[Dip]")
        modality = integration.channels.modalities.get(self.signal_name)
        if modality is None:
            raise ValueError(f"{self.signal_name} not found in integration "
                             f"channel modalities.")
        try:
            mode = modality.modes[self.signal_index]
        except IndexError:
            raise ValueError(f"Cannot retrieve signal "
                             f"index {self.signal_index} "
                             f"from integration channel {self.signal_name} "
                             f"modality modes.")

        signal = integration.get_signal(mode)
        if signal is None:
            signal_class = integration.get_correlated_signal_class()
            signal = signal_class(integration, mode)
            try:
                signal.update(robust=False)
            except Exception as err:
                log.error(f"Cannot decorrelate sky channels: {err}")
            signal = integration.get_signal(mode)

        valid_frames = integration.frames.is_unflagged('SOURCE_FLAGS')
        valid_frames &= integration.frames.valid

        frame_indices = np.arange(integration.frames.size)

        snf.add_skydip_frames(
            data=self.data,
            weight=self.weight,
            signal_values=signal.value_at(frame_indices),
            signal_weights=signal.weight_at(frame_indices),
            frame_weights=integration.frames.relative_weight,
            frame_valid=valid_frames,
            data_bins=self.get_bin(integration.frames.horizontal.el))

    def set_base(self):
        """
        Set the base map of the model.

        This is unused for the SkyDip class.

        Returns
        -------
        None
        """
        pass

    def get_source_name(self):
        """
        Return the source name for the sky dip model.

        Returns
        -------
        str
        """
        return 'SkyDip'

    def end_accumulation(self):
        """
        End model accumulation by scaling with inverse weights.

        Returns
        -------
        None
        """
        if self.data is None or self.weight is None:
            return
        nzi = self.weight != 0
        self.data[nzi] /= self.weight[nzi]

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
        self.end_accumulation()
        if hasattr(scan.info, 'get_ambient_kelvins'):
            ambient_temp = scan.info.get_ambient_kelvins()
            if not np.isnan(ambient_temp):
                obs_time = scan.get_observing_time()
                self.average_temp_with(ambient_temp, obs_time)

    def count_points(self):
        """
        Return the number of points in the skydip model.

        Returns
        -------
        int
        """
        if self.weight is None:
            return 0
        return int(np.sum(self.weight > 0))

    def sync_integration(self, integration, signal_mode=None):
        """
        Synchronize the sky dip model with an integration.

        This is not used for the :class:`SkyDip` class.

        Parameters
        ----------
        integration : Integration
        signal_mode : FrameFlagTypes
            The signal mode flag, indicating which signal should be used to
            extract the frame source gains.  Typically, TOTAL_POWER.

        Returns
        -------
        None
        """
        pass

    def get_signal_range(self):
        """
        Return the signal range of the SkyDip model.

        Returns
        -------
        Range
        """
        signal_range = Range()
        if self.data is None or self.weight is None:
            return signal_range
        values = self.data[self.weight > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            signal_range.min = np.nanmin(values)
            signal_range.max = np.nanmax(values)
        return signal_range

    def get_elevation_range(self):
        """
        Return the elevation range of the model.

        Returns
        -------
        Range
        """
        el_range = Range()
        valid = np.nonzero(self.weight > 0)[0]
        elevations = self.get_elevation(valid)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            el_range.min = np.nanmin(elevations)
            el_range.max = np.nanmax(elevations)
        return el_range

    def get_air_mass_range(self):
        """
        Return the range of air masses in the model.

        Returns
        -------
        Range
        """
        el_range = self.get_elevation_range()
        lower = 1.0 / np.sin(el_range.max)
        upper = 1.0 / np.sin(el_range.min)
        if isinstance(lower, units.Quantity):
            lower = lower.decompose().value
            upper = upper.decompose().value

        am_range = Range(min_val=lower, max_val=upper)
        return am_range

    def fit(self, model):
        """
        Fit the sky dip data to a model.

        Parameters
        ----------
        model : SkyDipModel

        Returns
        -------
        None
        """
        model.set_configuration(self.configuration)
        model.fit(self)

    def write(self, path=None):
        """
        Write the sky dip model to file.

        Parameters
        ----------
        path : str, optional
            The directory in which to write the results.  The default is the
            reduction work path.

        Returns
        -------
        None
        """
        self.model = SkyDipModel()
        if path is None:
            path = self.configuration.work_path

        initial_values = self.model.initial_guess
        initial_values['kelvin'] = self.info.kelvin.to(
            "Kelvin", equivalencies=units.temperature()).value
        self.model.data_unit = self.info.instrument.get_data_unit()
        self.fit(self.model)

        if self.model.has_converged:
            msg = ['Skydip result:',
                   '=================================================',
                   f'{self.model}',
                   '=================================================']
            log.info('\n'.join(msg))
        else:  # pragma: no cover
            log.warning("Skydip fit did not converge...")

        name = self.configuration.get_string(
            'name', default=self.get_default_core_name())
        core_name = os.path.join(path, name)
        filename = f'{core_name}.dat'
        header = [x.strip() for x in str(self.model).split('\n')]
        header = [f'# {x}' for x in header if x != '']
        header.extend(['#', '# EL\tobs\tmodel'])

        body = []
        n_data = self.data.size
        elevations = self.get_elevation(np.arange(n_data)).to('degree')
        fit = self.model.fit_elevation(elevations)
        el = elevations.value
        for i in range(self.data.size):
            line = f'{el[i]:.3f}\t'
            line += '...\t' if self.weight[i] == 0 else f'{self.data[i]:.3e}\t'
            line += f'{fit[i]:.3e}'
            body.append(line)

        with open(filename, 'w') as f:
            for line in (header + body):
                print(line, file=f)

        log.info(f"Written {filename}")

        self.create_plot(core_name)

    def create_plot(self, core_name):
        """
        Write a plot of the model and data to file.

        Parameters
        ----------
        core_name : str
            The name of the file to write to including the full file path but
            excluding the file extension.

        Returns
        -------
        None
        """
        pass

    def no_parallel(self):
        """
        Do not use parallel processing.

        Not applicable for the SkyDip model.

        Returns
        -------
        None
        """
        pass

    def set_parallel(self, threads):
        """
        Set the number of parallel operations for the skydip model.

        Not applicable for the SkyDip model.

        Parameters
        ----------
        threads : int

        Returns
        -------
        None
        """
        pass

    def process(self):
        """
        Process the skydip model.

        Not applicable for the SkyDip model.

        Returns
        -------
        None
        """
        pass

    def is_valid(self):
        """
        Return whether the skydip model is valid or not.

        Returns
        -------
        bool
        """
        return self.count_points() > 0

    def set_executor(self, executor):
        """
        Set the parallel task executor.

        Not applicable for the SkyDip model.

        Parameters
        ----------
        executor : object

        Returns
        -------
        None
        """
        pass

    def get_parallel(self):
        """
        Get the number of parallel operations for the source model.

        Not applicable for the SkyDip model.

        Returns
        -------
        threads : int
        """
        return 1

    def get_reference(self):
        """
        Return the reference (x, y) coordinate.

        Not applicable for the SkyDip model.

        Returns
        -------
        None
        """
        return None
