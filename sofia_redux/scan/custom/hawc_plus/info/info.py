# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log, units
from astropy.io import fits
import numpy as np
import os

from sofia_redux.scan.custom.hawc_plus.info.astrometry import (
    HawcPlusAstrometryInfo)
from sofia_redux.scan.custom.hawc_plus.info.chopping import (
    HawcPlusChoppingInfo)
from sofia_redux.scan.custom.hawc_plus.info.detector_array import (
    HawcPlusDetectorArrayInfo)
from sofia_redux.scan.custom.hawc_plus.info.instrument import (
    HawcPlusInstrumentInfo)
from sofia_redux.scan.custom.hawc_plus.info.telescope import (
    HawcPlusTelescopeInfo)
from sofia_redux.scan.custom.hawc_plus.info.observation import (
    HawcPlusObservationInfo)
from sofia_redux.scan.custom.sofia.info.info import SofiaInfo
from sofia_redux.scan.custom.sofia.info.gyro_drifts import SofiaGyroDriftsInfo
from sofia_redux.scan.custom.sofia.info.extended_scanning import (
    SofiaExtendedScanningInfo)
from sofia_redux.scan.utilities.utils import insert_info_in_header
from sofia_redux.toolkit.utilities import multiprocessing

__all__ = ['HawcPlusInfo']


class HawcPlusInfo(SofiaInfo):

    def __init__(self, configuration_path=None):
        """
        Initialize a HawcPlusInfo object.

        Parameters
        ----------
        configuration_path : str, optional
            An alternate directory path to the configuration tree to be
            used during the reduction.  The default is
            <package>/data/configurations.
        """
        super().__init__(configuration_path=configuration_path)
        self.name = 'hawc_plus'
        self.astrometry = HawcPlusAstrometryInfo()  #
        self.gyro_drifts = SofiaGyroDriftsInfo()  #
        self.chopping = HawcPlusChoppingInfo()  #
        self.detector_array = HawcPlusDetectorArrayInfo()
        self.instrument = HawcPlusInstrumentInfo()  #
        self.spectroscopy = None  #
        self.telescope = HawcPlusTelescopeInfo()  #
        self.observation = HawcPlusObservationInfo()  #
        self.scanning = SofiaExtendedScanningInfo()
        self.hwp_grouping_angle = 2 * units.Unit('degree')

    @classmethod
    def get_file_id(cls):
        """
        Return the file ID.

        Returns
        -------
        str
        """
        return 'HAW'

    def edit_header(self, header):
        """
        Edit an image header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.

        Returns
        -------
        None
        """
        super().edit_header(header)
        self.detector_array.edit_header(header)

        info = [('COMMENT', "<------ HAWC+ Header Keys ------>"),
                ('SMPLFREQ', (1.0 / self.sampling_interval).to('Hz').value,
                 "(Hz) Detector readout rate.")]

        requested_subarrays = self.detector_array.subarrays_requested
        if requested_subarrays not in ['', None]:
            info.append(('SUBARRAY', requested_subarrays,
                         'Subarrays in image.'))

        if self.detector_array.hwp_angle not in [None, -1]:
            info.append(
                ('DETHWPAG', self.detector_array.hwp_angle,
                 "The determine half wave plate angle from the file group"))
        insert_info_in_header(header, info, delete_special=True)

    def validate_scans(self, scans):
        """
        Validate a list of scans specific to the instrument.

        Parameters
        ----------
        scans : list (HawcPlusScan)
            A list of scans.  Scans are culled in-place if they do not meet
            certain criteria.

        Returns
        -------
        None
        """
        if scans is None or len(scans) < 2 or scans[0] is None:
            super().validate_scans(scans)
            return

        n_scans = len(scans)

        first_scan = scans[0]
        wavelength = first_scan.info.instrument.wavelength
        instrument_config = first_scan.info.instrument.instrument_config
        keep_scans = np.full(n_scans, True)

        for i in range(1, n_scans):
            scan = scans[i]
            if scan.info.instrument.wavelength != wavelength:
                log.warning(f"Scan {scan.get_id()} in a different band. "
                            f"Removing from set.")
                keep_scans[i] = False
            elif scan.info.instrument.instrument_config != instrument_config:
                log.warning(f"Scan {scan.get_id()} is in a different "
                            f"instrument configuration. Removing from set.")
                keep_scans[i] = False
            else:
                limit = scan.configuration.get_float(
                    'gyrocorrect.max', default=np.nan) * units.Unit('arcsec')
                if np.isnan(limit):
                    continue

                if scan.info.gyro_drifts.get_max() > limit:
                    log.warning(f"Scan {scan.get_id()} has too large gyro "
                                f"drifts. Removing from set.")
                    keep_scans[i] = False

        for i in range(n_scans - 1, 0, -1):
            if not keep_scans[i]:
                del scans[i]

        super().validate_scans(scans)

    def max_pixels(self):
        """
        Return the maximum number of pixels.

        Returns
        -------
        count : int
        """
        return self.instrument.n_store_channels

    def get_si_pixel_size(self):
        """
        Get the science instrument pixel size.

        Returns
        -------
        size : Coordinate2D
            The (x, y) pixel sizes, each of which is a units.Quantity.
        """
        return self.detector_array.pixel_sizes

    def perform_reduction(self, reduction, filenames):
        """
        Fully reduce a given reduction and set of files.

        While it is possible for the reduction object to fully reduce a set of
        files, certain special considerations may be required for certain
        instruments.  Therefore, the instrument specific Info object is given
        control of how a reduction should progress.

        HAWC+ requires special processing for scan polarimetry data.  When
        multiple files are provided, files will be grouped by half-wave-plate
        (HWP) angle and each group will then be reduced for the R0 and T0
        subarrays.  Note that a standard reduction will be performed if
        multiple HWP angles are not detected.

        Parameters
        ----------
        reduction : Reduction
            The reduction object.
        filenames : str or list (str)
            A single file (str) or list of files to be included in the
            reduction.

        Returns
        -------
        None
        """
        if isinstance(filenames, str) or len(filenames) <= 1:
            super().perform_reduction(reduction, filenames)
            return

        reduction.update_parallel_config(reset=True)
        file_groups = self.group_files_by_hwp(
            filenames, jobs=reduction.parallel_read, force_threading=True)
        if len(file_groups) <= 1:
            super().perform_reduction(reduction, filenames)
            return

        self.split_reduction(reduction, file_groups)

        reduction.read_scans()
        reduction.validate()

        # Can now add the default name in if necessary
        base_name = reduction.configuration.get_string('name', default=None)
        if base_name is None:
            for sub_reduction in reduction.sub_reductions:
                config = sub_reduction.configuration
                suffix = config.get_string('name')
                default_name = sub_reduction.source.get_default_core_name()
                if default_name.endswith('.fits'):
                    default_name = default_name[:-5]

                config.put('name', f'{default_name}_{suffix}')

        reduction.reduce()

    def group_files_by_hwp(self, filenames, jobs=1, force_threading=False):
        """
        Group HAWC+ files by HWP angle.

        Parameters
        ----------
        filenames : list (str)
            A list of HAWC+ FITS files to group.
        jobs : int
            The number of parallel jobs used to determine the grouping.
        force_threading : bool
            If `True`, force parallel processing using threads.

        Returns
        -------
        file_groups : dict
            The files grouped by HWP angle {angle : [files]}
        """
        n_files = len(filenames)
        read_jobs = int(np.clip(n_files, 1, jobs))

        msg = f"Grouping {n_files} HAWC_PLUS files by HWP angles"
        if jobs > 1:
            msg += f" using {read_jobs} parallel threads."
        log.debug(msg)
        file_groups = {}
        if isinstance(filenames, str):
            filenames = [filenames]

        da = self.hwp_grouping_angle
        hwp_step = self.instrument.hwp_step

        args = filenames, hwp_step
        kwargs = None
        hwp_angles = multiprocessing.multitask(
            self.parallel_safe_determine_hwp_angle, range(n_files), args,
            kwargs, jobs=read_jobs, max_nbytes=None,
            force_threading=force_threading, logger=log)

        for filename, hwp_angle in zip(filenames, hwp_angles):
            if np.isnan(hwp_angle):
                if None in file_groups:
                    file_groups[hwp_angle].append(filename)
                    continue

            for angle, angle_files in file_groups.items():
                if (angle - da) <= hwp_angle <= (angle + da):
                    angle_files.append(filename)
                    break
            else:
                file_groups[hwp_angle] = [filename]

        log.info(f"{len(file_groups)} HWP groups will be reduced.")
        return file_groups

    @classmethod
    def parallel_safe_determine_hwp_angle(cls, args, file_index):
        """
        Return the HWP (half-wave-plate) angle for a single file.

        This function is safe for multiprocessing using
        :func:`multiprocessing.multitask`.

        Parameters
        ----------
        args : 2-tuple
            args[0] : list (str)
                A list of FITS file names.
            args[1] : units.Quantity
                The HWP step used to convert HWP counts to a HWP angle.
        file_index : int
            The index of the file in args[0] for which to determine the HWP
            angle.

        Returns
        -------
        hwp_angle: units.Quantity
        """
        filenames, hwp_step = args
        filename = filenames[file_index]
        return cls.determine_hwp_angle(filename, hwp_step)

    @classmethod
    def determine_hwp_angle(cls, filename, hwp_step):
        """
        Determine the mean HWP angle in the given file.

        Parameters
        ----------
        filename : str
            The FITS file to search.
        hwp_step : units.Quantity
            The HWP step size for each HWP count.

        Returns
        -------
        hwp_angle : units.Quantity
            The average HWP angle in the file.
        """
        if not os.path.isfile(filename):
            log.error(f"Could not locate file: {filename}")
            return np.nan * units.Unit('degree')

        hdul = fits.open(filename)
        hwp_counts = np.empty(0, dtype=int)
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU):
                extname = hdu.header.get('EXTNAME')
                if extname is None or extname.lower().strip() != 'timestream':
                    continue
                if 'hwpCounts' not in hdu.columns.names:
                    continue
                hwp_counts = np.concatenate(
                    (hwp_counts, hdu.data['hwpCounts'].ravel()))

        hdul.close()

        if hwp_counts.size == 0:
            log.warning(f"Could not find HWP angles for file: {filename}")
            return np.nan * units.Unit('degree')

        return np.nanmean(hwp_counts) * hwp_step

    @staticmethod
    def split_reduction(reduction, file_groups):
        """
        Split the reduction based on files grouped by HWP angle.

        Parameters
        ----------
        reduction : Reduction
        file_groups : dict

        Returns
        -------
        None
        """
        base_config = reduction.configuration
        base_name = base_config.get_string('name', default=None)
        base_subarray = base_config.get_string_list(
            'subarray', delimiter=',', default=None)
        reductions = []

        for angle, file_group in file_groups.items():
            if isinstance(angle, units.Quantity):
                angle = angle.to('degree').value
            angle_string = f'{angle:.2f}'

            r0 = reduction.blank_copy()
            t0 = reduction.blank_copy()

            for subarray, configuration in zip(
                    ['R0', 'T0'], [r0.configuration, t0.configuration]):

                configuration.recall('name')  # in case it was disabled
                new_name = f'{angle_string}{subarray}.fits'
                if base_name is not None:
                    new_name = f'{base_name}.{new_name}'
                configuration.put('name', new_name)

                configuration.recall('subarray')
                if base_subarray is None:
                    configuration.put('subarray', subarray)
                    configuration.lock('subarray')
                else:
                    subarrays = ','.join(np.unique(base_subarray + [subarray]))
                    configuration.put('subarray', subarrays)

                configuration.recall('hwp')
                configuration.put('hwp', angle_string)

            reductions.append((r0, file_group))
            reductions.append((t0, file_group))

        reduction.sub_reductions = []
        for i, (sub_reduction, file_group) in enumerate(reductions):
            reduction.sub_reductions.append(sub_reduction)
            sub_reduction.parent_reduction = reduction
            sub_reduction.reduction_files = file_group
            sub_reduction.reduction_number = i + 1
