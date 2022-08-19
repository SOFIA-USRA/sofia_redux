# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import abstractmethod
from astropy import log
import os
import numpy as np

from sofia_redux.scan.source_models.astro_model_2d import AstroModel2D
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.flags.array_flags import ArrayFlags

__all__ = ['AstroData2D']


class AstroData2D(AstroModel2D):

    FLAG_MASK = ArrayFlags.flags.MASK

    def __init__(self, info, reduction=None):
        """
        Initialize an astronomical 2D model of the source data.

        The AstroData2D is an abstract class that extends the
        :class:`AstroModel2D` to operate on the
        :class:`sofia_redux.scan.source_models.map.observation_2d.Observaton2D`
        class.

        Parameters
        ----------
        info : sofia_redux.scan.info.info.Info
            The Info object which should belong to this source model.
        reduction : sofia_redux.scan.reduction.reduction.Reduction, optional
            The reduction for which this source model should be applied.
        """
        super().__init__(info=info, reduction=reduction)

    def create_from(self, scans, assign_scans=True):
        """
        Initialize model from scans.

        Sets the model scans to those provided, and the source model for each
        scan as this.  All integration gains are normalized to the first scan.
        If the first scan is non-sidereal, the system will be forced to an
        equatorial frame.

        Parameters
        ----------
        scans : list (sofia_redux.scan.scan.scan.Scan)
            A list of scans from which to create the model.
        assign_scans : bool, optional
            If `True`, assign the scans to this source model.  Otherwise,
            there will be no hard link between the scans and source model.

        Returns
        -------
        None
        """
        super().create_from(scans, assign_scans=assign_scans)
        if self.has_option('unit'):
            self.get_data().set_unit(self.configuration.get_string('unit'))

    @property
    def flagspace(self):
        """
        Return the flagspace for this source model.

        Returns
        -------
        ArrayFlags
        """
        return ArrayFlags

    @property
    def mask_flag(self):
        """
        Return the masking flag for this source model.

        Returns
        -------
        flag : enum.Enum
        """
        return self.flagspace.convert_flag(self.FLAG_MASK)

    def get_weights(self):
        """
        Return the weight map for the source model.

        Returns
        -------
        weight_map : sofia_redux.scan.source_models.maps.weight_map.WeightMap
        """
        return self.get_data().get_weights()

    def get_noise(self):
        """
        Return the noise map for the source model.

        Returns
        -------
        noise_map : sofia_redux.scan.source_models.maps.noise_map.NoiseMap
        """
        return self.get_data().get_noise()

    def get_significance(self):
        """
        Return the data significance map (signal-to-noise).

        Returns
        -------
        sofia_redux.scan.source_models.maps.significance_map.SignificanceMap
        """
        return self.get_data().get_significance()

    def get_exposures(self):
        """
        Return the exposure map for the source model.

        Returns
        -------
        sofia_redux.scan.source_models.maps.exposure_map.ExposureMap
        """
        return self.get_data().get_exposures()

    def end_accumulation(self):
        """
        End map accumulation (typically scale by inverse weights).

        Returns
        -------
        None
        """
        self.get_data().end_accumulation()

    def get_executor(self):
        """
        Return the source map parallel executor.

        The executor is not currently implemented in any way.

        Returns
        -------
        executor : object
        """
        return self.get_data().executor

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
        self.get_data().set_executor(executor)

    def get_parallel(self):
        """
        Get the number of parallel operations for the source model.

        Returns
        -------
        threads : int
        """
        return self.get_data().parallelism

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
        self.get_data().set_parallel(threads)

    def no_parallel(self):
        """
        Disable parallel processing for the model.

        Returns
        -------
        None
        """
        self.get_data().set_parallel(0)

    def clear_content(self):
        """
        Clear the data.

        Returns
        -------
        None
        """
        self.get_data().clear()

    def is_empty(self):
        """
        Return whether source map is empty.

        Returns
        -------
        bool
        """
        return self.get_data().count_points() == 0

    def count_points(self):
        """
        Return the number of points in the source map.

        Returns
        -------
        points : int
        """
        return self.get_data().count_points()

    def get_chi2(self, robust=False):
        """
        Get the Chi-squared statistic.

        Parameters
        ----------
        robust : bool, optional
            If `True`, use the robust (median) method for determining variance.
            Otherwise, use a weighted mean.

        Returns
        -------
        chi2 : float
        """
        return self.get_significance().variance(robust=robust)

    def smooth(self):
        """
        Smooth the source model.

        Returns
        -------
        None
        """
        self.smooth_to(self.smoothing)

    def filter(self, allow_blanking=False):
        """
        Apply filtering to the source.

        Parameters
        ----------
        allow_blanking : bool, optional
            If `True`, allow the blanking value to be determined from the
            'source.filter.blank' configuration value.  Otherwise, the filter
            blanking value will be set to NaN.  The filter blanking level
            defines the range of values prior to filter smoothing that are
            permitted during the convolution.  Any values > filter_level, or
            values < filter_level will be marked as invalid and not included.

        Returns
        -------
        None
        """
        if (not self.configuration.get_bool('source.filter')
                or self.get_source_size() <= 0):
            self.reset_filtering()
            return

        if allow_blanking:
            filter_blanking = self.configuration.get_float(
                'source.filter.blank', default=np.nan)
        else:
            filter_blanking = np.nan

        filter_fwhm = self.get_filter_scale()
        use_fft = self.configuration.get_string('source.filter.type') == 'fft'
        self.filter_source(
            filter_fwhm=filter_fwhm,
            filter_blanking=filter_blanking,
            use_fft=use_fft)

    def get_filter_scale(self):
        """
        Return the filter scale.

        Returns
        -------
        filter_fwhm : astropy.units.Quantity
            The FWHM of the filter scale.
        """
        directive = self.configuration.get_string('source.filter.fwhm',
                                                  default='auto')
        try:
            fwhm = float(directive) * self.info.instrument.get_size_unit()
        except ValueError:
            fwhm = 5 * self.get_source_size()

        return fwhm

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
        data = self.get_data()
        self.end_accumulation()
        self.add_base()

        if self.enable_level:
            self.level(robust=True)

        if self.configuration.get_bool('source.despike'):
            despike_level = self.configuration.get_float(
                'source.despike.level', default=10)
            data.despike(despike_level)

        self.filter(allow_blanking=False)

        scan.weight = 1.0
        if self.configuration.get_bool('weighting.scans'):
            method = self.configuration.get_string(
                'weighting.scans.method', default='rms')
            scan.weight = 1.0 / self.get_chi2(robust=(method == 'robust'))
            if not np.isfinite(scan.weight):
                scan.weight = 0.0

        if self.configuration.get_bool('scanmaps'):
            file_name = os.path.join(self.reduction.work_path,
                                     f'scan-{scan.mjd}-{scan.get_id()}.fits')
            self.write_fits(file_name)

    def level(self, robust=False):
        """
        Level the source model data.

        Parameters
        ----------
        robust : bool, optional
            If `True`, use the robust (weighted median) method to level data.
            Otherwise, use a weighted mean.

        Returns
        -------
        None
        """
        self.get_data().level(robust=robust)

    def process(self):
        """
        Process the source model.

        Returns
        -------
        None
        """
        self.end_accumulation()
        self.next_generation()  # increment the map generation
        self.info.instrument.resolution = self.get_average_resolution()

        if self.enable_level:
            self.add_process_brief('{level} ')

        if self.configuration.get_bool('source.despike'):
            self.add_process_brief('{despike} ')

        if (self.configuration.get_bool('source.filter')
                and self.get_source_size() > 0):
            self.add_process_brief('{filter} ')

        if (self.enable_weighting
                and self.configuration.get_bool('weighting.scans')):
            for scan in self.scans:
                if scan.weight != 0:
                    message = f'{{{1.0 / scan.weight:.2f}x}}'
                else:
                    message = "{inf}"
                self.add_process_brief(message)

        if self.has_option('source.redundancy'):
            self.add_process_brief('(check) ')
            redundancy = self.configuration.get_int('source.redundancy')
            min_integration_time = (
                self.info.instrument.integration_time * redundancy)
            exposures = self.get_exposures()
            min_integration_time = min_integration_time.to('second').value
            exposures.restrict_range(Range(min_val=min_integration_time))

        smooth = self.configuration.get_string(
            'smooth', default='None').lower().strip()
        if smooth != 'none' and not self.configuration.get_bool(
                'smooth.external'):
            self.add_process_brief('(smooth) ')
            self.smooth()

        # Apply the filtering to the final map, to reflect the correct blanking
        # level
        if self.configuration.get_bool('source.filter'):
            self.add_process_brief('(filter) ')
            self.filter(allow_blanking=True)
            self.filter_beam_correct()

        # Noise and exposure clip after smoothing for evened-out coverage.
        if self.has_option('exposureclip'):
            self.add_process_brief('(exposureclip) ')
            clip_level = self.configuration.get_float('exposureclip')
            exposures = self.get_exposures()
            min_clip = clip_level * exposures.select(fraction=0.95)
            exposures.restrict_range(Range(min_val=min_clip))

        if self.has_option('noiseclip'):
            self.add_process_brief('(noiseclip) ')
            rms = self.get_noise()
            clip_level = self.configuration.get_float('noiseclip')
            max_clip = clip_level * rms.select(fraction=0.05)
            rms.restrict_range(Range(min_val=0, max_val=max_clip))

        if self.enable_bias and self.has_option('clip'):
            clip_level = self.configuration.get_float('clip')
            self.add_process_brief(f'(clip:{clip_level}) ')
            sign = self.configuration.get_sign('source.sign', default=0)
            s2n_reject = Range(-clip_level, clip_level)
            if sign > 0:
                s2n_reject.min = -np.inf
            elif sign < 0:
                s2n_reject.max = np.inf
            self.get_significance().discard_range(s2n_reject)

        if self.configuration.get_bool('source.mem'):
            self.add_process_brief('(MEM) ')
            multiplier = self.configuration.get_float(
                'source.mem.lambda', default=0.1)
            self.mem_correct(lg_multiplier=multiplier)

        if self.configuration.get_bool('source.intermediates'):
            file_name = os.path.join(self.reduction.work_path,
                                     'intermediate.fits')
            self.write_fits(file_name)

        # Coupled with blanking
        if not self.configuration.get_bool('source.nosync'):
            if self.enable_bias and self.has_option('blank'):
                blanking_level = self.get_blanking_level()
                self.add_process_brief(f'blank:{blanking_level}) ')
                self.update_mask(blanking_level=blanking_level,
                                 min_neighbors=2)
            else:
                self.update_mask(blanking_level=np.nan, min_neighbors=2)

    def process_final(self):
        """
        Runs any final processing steps.

        Returns
        -------
        None
        """
        self.get_data().clear_history()

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
        hdu_list = self.get_data().create_fits()
        for hdu in hdu_list:
            self.edit_header(hdu.header)
            hdu.header['FILENAME'] = filename, 'Name at creation'

        self.add_scan_hdus_to(hdu_list)
        self.hdul = hdu_list
        if not self.configuration.get_bool('write.source'):
            return

        log.info(f'Writing to {filename}')
        hdu_list.writeto(filename, overwrite=True)
        hdu_list.close()

    @abstractmethod
    def get_data(self):  # pragma: no cover
        """
        Return the data for the source model.

        Returns
        -------
        data : sofia_redux.scan.source_models.maps.observation_2d.Observation2D
        """
        pass

    @abstractmethod
    def add_base(self):  # pragma: no cover
        """
        Add a base to the source model data.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def smooth_to(self, fwhm):  # pragma: no cover
        """
        Smooth the map using a Gaussian kernel of a given FWHM.

        Parameters
        ----------
        fwhm : astropy.units.Quantity

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def filter_source(self, filter_fwhm, filter_blanking=None, use_fft=False
                      ):  # pragma: no cover
        """
        Filter (smooth) the source above a given FWHM.

        Parameters
        ----------
        filter_fwhm : astropy.units.Quantity
            The filter FWHM scale to filter above.
        filter_blanking : float, optional
        use_fft : bool, optional
            If `True`, use FFT filtering.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def set_filtering(self, fwhm):  # pragma: no cover
        """
        Set the filtering FWHM.

        Parameters
        ----------
        fwhm : astropy.units.Quantity

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def reset_filtering(self):  # pragma: no cover
        """
        Reset the source filtering parameters.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def filter_beam_correct(self):  # pragma: no cover
        """
        Apply the filter beam correction.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def mem_correct(self, lg_multiplier):  # pragma: no cover
        """
        Apply maximum entropy correction to the map.

        Parameters
        ----------
        lg_multiplier : float
           The Lagrange multiplier (lambda).

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def update_mask(self, blanking_level=np.nan, min_neighbors=None
                    ):  # pragma: no cover
        """
        Update the map mask based on significance levels and valid neighbors.

        If a blanking level is supplied, significance values above or equal to
        the blanking level will be masked.


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
        pass
