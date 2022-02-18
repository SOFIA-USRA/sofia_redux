# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from copy import deepcopy
import json
import numpy as np
import os
import platform
import psutil
import re
import shutil
import sys
import tempfile
import time

from sofia_redux.scan.reduction.version import ReductionVersion
from sofia_redux.scan.info.info import Info
from sofia_redux.scan.utilities import utils
from sofia_redux.scan.pipeline.pipeline import Pipeline
from sofia_redux.scan.utilities.utils import insert_info_in_header
from sofia_redux.toolkit.utilities import multiprocessing

__all__ = ['Reduction']


class Reduction(ReductionVersion):

    global log
    from astropy import log

    def __init__(self, instrument, configuration_file='default.cfg',
                 configuration_path=None):
        """
        Initialize the reduction object.

        Parameters
        ----------
        instrument : str or None
            The name of the instrument for which the reduction applies.
        configuration_file : str, optional
            An optional configuration file to use.
        configuration_path : str, optional
            An alternate directory path to the configuration tree to be used
            during the reduction.  The default is
            <package>/data/configurations.
        """
        super().__init__()
        self.scans = []
        self.pipeline = None
        self.queue = []
        self.max_jobs = 1
        self.max_cores = 1
        self.available_reduction_jobs = 1
        self.parallel_reductions = 1
        self.parallel_scans = 1
        self.parallel_tasks = 1
        self.parallel_read = 1
        self.source = None
        self.source_executor = None
        self.channels = None
        self.sub_reductions = None
        self.parent_reduction = None
        self.reduction_files = None
        self.reduction_number = 0
        self.pickle_reduction_directory = None
        self.pickle_pipeline_directory = None
        self.jobs_assigned = False
        self.read_start_time = None
        self.read_end_time = None
        self.reduce_start_time = None
        self.reduce_end_time = None
        self.stored_user_configuration = None

        if instrument is None:
            return

        info = Info.instance_from_instrument_name(
            instrument, configuration_path=configuration_path)
        info.read_configuration(configuration_file=configuration_file,
                                validate=True)
        self.channels = info.get_channels_instance()
        self.set_outpath()
        self.channels.set_parent(self)
        log.info(f"Instrument is {self.info.get_name().upper()}")

    def blank_copy(self):
        """
        Return a base copy of the reduction.

        There will be no scans, source or pipelines loaded.  Only the basic
        channel information and info/configuration will be available.

        Returns
        -------
        Reduction
        """
        new = Reduction(None)

        if self.info is None:
            return

        new.channels = self.channels.copy()
        new.info.unlink_configuration()
        new.channels.set_parent(new)
        new.max_jobs = self.max_jobs
        new.parallel_tasks = self.parallel_tasks
        new.parallel_scans = self.parallel_scans
        return new

    @property
    def configuration(self):
        """
        Return the reduction configuration object.

        Returns
        -------
        Configuration
        """
        if self.info is None:
            return None
        return self.info.configuration

    @property
    def instrument(self):
        """
        Return the instrument name for the reduction

        Returns
        -------
        instrument_name : str
        """
        if self.info is None:
            return None
        return self.info.instrument.name

    @property
    def size(self):
        """
        Return the number of scans in the reduction

        Returns
        -------
        n_scans : int
        """
        if self.scans is None:
            return 0
        else:
            return len(self.scans)

    @property
    def name(self):
        """
        Return the name (typically instrument) of the reduction.

        Returns
        -------
        name : str
        """
        if self.info is None:
            return None
        return self.info.name

    @property
    def rounds(self):
        """
        Return the maximum number of rounds (iterations) in the reduction.

        Returns
        -------
        rounds : int
        """
        if self.configuration is None:
            return 0
        return self.configuration.iterations.max_iteration

    @rounds.setter
    def rounds(self, value):
        """
        Set the number of rounds in the reduction.

        Parameters
        ----------
        value : int

        Returns
        -------
        None
        """
        if self.configuration is None:
            raise ValueError("Cannot set rounds for non-initialized "
                             "configuration")
        self.configuration.iterations.max_iteration = value

    @property
    def info(self):
        """
        Return the info object for the reduction.

        Returns
        -------
        Info
        """
        if self.channels is None:
            return None
        return self.channels.info

    @info.setter
    def info(self, value):
        """
        Set the info object for the reduction.

        Parameters
        ----------
        value : Info

        Returns
        -------
        None
        """
        if self.channels is None:
            raise ValueError("Cannot set info for non-initialized channels")
        self.channels.info = value

    @property
    def total_reductions(self):
        """
        Return the total number of reductions to be processed.

        This is of importance for polarimetry HAWC_PLUS reductions, where
        separate source maps are generated for each sub-reduction.  Otherwise,
        it is expected for there to only be a single reduction.

        Returns
        -------
        int
        """
        # If there is no parent reduction, this is the reduction, or the
        # length of the number of sub reductions.
        if self.parent_reduction is None:
            n_sub = 0 if self.sub_reductions is None else len(
                self.sub_reductions)
            return int(np.clip(n_sub, 1, None))
        return self.parent_reduction.total_reductions

    @property
    def is_sub_reduction(self):
        """
        Return whether this reduction is a sub-reduction of a parent reduction.

        Returns
        -------
        bool
        """
        return self.parent_reduction is not None

    @property
    def reduction_id(self):
        """
        Return a unique identifier for this reduction.

        Returns
        -------
        id : str
        """
        self_id = f"{id(self)}.{self.reduction_number}"
        if self.parent_reduction is None:
            return self_id
        return f'{self.parent_reduction.reduction_id}-{self_id}'

    def iteration(self):
        """
        Return the current iteration.

        Returns
        -------
        iteration : int
        """
        if self.configuration is None:
            return 0
        elif self.configuration.iterations is None:
            return 0
        return self.configuration.iterations.current_iteration

    def read_scan(self, filename, read_fully=True):
        """
        Given a filename, read it and return a Scan instance.

        Scans are initialized based on the instrument name using the default
        configuration.  The Configuration (owned by each scan) is then updated
        using information from the supplied file, and any necessary information
        will be extracted.

        Parameters
        ----------
        filename : str
            The path to a FITS file.
        read_fully : bool, optional
            If `False`, do not fully read the scan (definition depends on the
            instrument).

        Returns
        -------
        Scan
        """
        self.update_runtime_config()
        log.info(f"Reading scan: {filename}")
        scan = self.channels.read_scan(filename, read_fully=read_fully)
        return scan

    def read_scans(self, filenames=None):
        """
        Read a list of FITS files to create scans.

        Parameters
        ----------
        filenames : list (str) or list (list (str)), optional
            A list of scan FITS file names.  If not supplied, defaults to
            the files stored in the `reduction_files` attribute.  If there
            are multiple sub-reductions, filenames must be a contain a list
            of filenames for each sub-reduction.  i.e., filenames[0] contains
            the list of files to read for the first sub-reduction.

        Returns
        -------
        None
        """
        self.read_start_time = time.time()
        self.update_runtime_config()
        self.scans = []
        if filenames is not None:
            self.assign_reduction_files(filenames)

        # Update the number of files to read in parallel
        self.update_parallel_config(reset=True)

        if self.sub_reductions is not None:
            self.read_sub_reduction_scans()
            return

        if self.reduction_files is None or len(self.reduction_files) == 0:
            log.warning('No files to read.')
            return
        n_scans = len(self.reduction_files)
        parallel_read = int(np.clip(self.parallel_read, 1, n_scans))

        msg = f"Reading {n_scans} files"
        if parallel_read > 1:
            msg += f" in parallel using {parallel_read} jobs"
        log.debug(msg)
        split_scans = self.configuration.get_bool('subscan.split')

        args = (self.reduction_files, self.channels)
        kwargs = None

        scans = multiprocessing.multitask(
            self.parallel_safe_read_scan, range(n_scans), args, kwargs,
            jobs=parallel_read, logger=log)

        for scan in scans:
            if scan is None:
                continue
            if split_scans:
                self.scans.extend(scan.split())
            else:
                self.scans.append(scan)

    @classmethod
    def parallel_safe_read_scan(cls, args, file_number):
        """
        Read a single scan.

        This function is safe for :func:`multitask`.

        Parameters
        ----------
        args : 2-tuple
            A tuple of arguments where:
                args[0] - list (str) of all filenames
                args[1] - The reduction Channels object.
        file_number : int
            The index of the file to read in all of the supplied filenames
            (args[0]).

        Returns
        -------
        scan : Scan or str
            A Scan object if no pickling is required, or a string filename
            pointing to the pickled scan object.
        """
        filenames, channels = args
        filename = filenames[file_number]
        return cls.return_scan_from_read_arguments(filename, channels)

    @classmethod
    def return_scan_from_read_arguments(cls, filename, channels):

        if isinstance(channels, str):
            channels, _ = multiprocessing.unpickle_file(channels)

        log.info(f"Reading scan: {filename}")
        scan = channels.read_scan(filename, read_fully=True)

        if scan.size == 0:
            log.warning(f"Scan {scan.get_id()} contains no valid data. "
                        f"Skipping")
            return None

        scan.validate()

        if scan.size == 0:
            log.warning(f"Scan {scan.get_id()} contains no valid data. "
                        f"Skipping")
            return None

        log.info(f"Successfully read scan {scan.get_id()}.\n")

        return scan

    def read_sub_reduction_scans(self):
        """
        Read scans for each sub-reduction.

        Reduction files MUST have already been assigned to each sub-reduction.

        Returns
        -------
        None
        """
        if self.sub_reductions is None:
            n_reductions = 0
        else:
            n_reductions = len(self.sub_reductions)

        if n_reductions == 0:
            raise ValueError("No sub-reductions exist.")

        file_read_args = []
        reduction_map = []

        pickle_directory = tempfile.mkdtemp(prefix='sub_reduction_read')

        for reduction_number, sr in enumerate(self.sub_reductions):
            sr.read_start_time = time.time()
            sr.update_runtime_config()
            sr.scans = []
            if sr.reduction_files is not None:
                for file_no, reduction_file in enumerate(sr.reduction_files):
                    channel_file = os.path.join(
                        pickle_directory,
                        f'SOFSCAN_channels_{reduction_number}_{file_no}.p')
                    channel_file = multiprocessing.pickle_object(
                        sr.channels, channel_file)

                    file_read_args.append([reduction_file, channel_file])
                    reduction_map.append(reduction_number)

        n_read = len(reduction_map)
        parallel_jobs = int(np.clip(n_read, 1, self.parallel_read))
        parallel = parallel_jobs > 1

        args = file_read_args, pickle_directory
        kwargs = None

        msg = f"Reading {n_read} files from {n_reductions} sub-reductions"
        if parallel:  # pragma: no cover
            msg += f' in parallel using {parallel_jobs} processes'
        msg += '.'
        log.debug(msg)

        scans = multiprocessing.multitask(
            self.parallel_safe_read_all_files, range(n_read), args, kwargs,
            jobs=parallel_jobs, force_processes=True, max_nbytes=None,
            logger=log)

        if parallel or (
                len(scans) > 0 and any(isinstance(x, str) for x in scans)):
            multiprocessing.unpickle_list(scans, delete=True)

        try:
            shutil.rmtree(pickle_directory)
        except Exception as err:  # pragma: no cover
            log.error(f"Could not delete pickle directory {pickle_directory}: "
                      f"{err}")

        for i, scan in enumerate(scans):
            if scan is None:
                continue
            sub_reduction = self.sub_reductions[reduction_map[i]]
            scan.reduction = sub_reduction
            if sub_reduction.configuration.get_bool('subscan.split'):
                sub_reduction.scans.extend(scan.split())
            else:
                sub_reduction.scans.append(scan)

    @classmethod
    def parallel_safe_read_all_files(cls, args, file_number):
        """
        Read a single file from a list of many and return a Scan.

        This function is safe for :func:`multitask`.

        Parameters
        ----------
        args : 2-tuple
            A tuple of arguments where:
                args[0] - A list (list (list)) of all read arguments.
                args[1] - pickle directory (str) in which to pickle the scan.
        file_number : int
            The index of the file to read in all of the supplied filenames
            (args[0]).

        Returns
        -------
        scan : Scan or str
            A Scan object if no pickling is required, or a string filename
            pointing to the pickled scan object.
        """
        all_read_arguments, pickle_directory = args
        read_arguments = all_read_arguments[file_number]
        log.debug(f"Reading file {file_number}")
        scan = cls.return_scan_from_read_arguments(*read_arguments)

        if pickle_directory is not None:
            pickle_file = multiprocessing.pickle_object(
                scan, os.path.join(pickle_directory,
                                   f'{file_number}-{id(scan)}.p'))
            del scan
            return pickle_file

        return scan

    @classmethod
    def parallel_safe_read_sub_reduction_scans(cls, args, reduction_number):
        """
        Read all files in a single reduction to create Scan objects.

        Parameters
        ----------
        args : 1-tuple
            A single tuple where args[0] is a list (Reduction).
        reduction_number : int
            The reduction for which to read files out of all the supplied
            reductions (args[0]).

        Returns
        -------
        reduction : Reduction
            A reduction where the `scans` attribute has been populated with a
            list of read and validated Scan objects.
        """
        sub_reductions = args[0]
        sub_reduction = sub_reductions[reduction_number]

        sub_reduction, sub_reduction_file = multiprocessing.unpickle_file(
            sub_reduction)

        reduction_files = sub_reduction.reduction_files
        if reduction_files is None or len(reduction_files) == 0:
            log.warning(f"No reduction files exist for sub-reduction "
                        f"{reduction_number}.  Sub-reduction will "
                        f"be excluded.")
            return None
        sub_reduction.read_scans()

        sub_reduction = multiprocessing.pickle_object(
            sub_reduction, sub_reduction_file)

        return sub_reduction

    def assign_reduction_files(self, filenames):
        """
        Assign reduction files to a reduction or sub-reductions.

        Parameters
        ----------
        filenames : list (str) or list (list (str))
            A list of files for a single reduction, or a list of files for
            each sub-reduction.

        Returns
        -------
        None
        """
        if self.sub_reductions is None:
            if isinstance(filenames, str):
                filenames = [x for x in re.split(r'[;, \t]', filenames) if
                             x != '']
            self.reduction_files = filenames.copy()
            return

        n_groups, n_sub = len(filenames), len(self.sub_reductions)
        if n_groups != n_sub:
            raise ValueError(
                f"Number of file groups ({n_groups}) does not "
                f"match number of sub-reductions ({n_sub}).")
        for group, sub_reduction in zip(filenames, self.sub_reductions):
            if isinstance(group, str):
                group = [group]
            sub_reduction.reduction_files = group

    def validate(self):
        """
        Validate scans following a read.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If there are no scans to reduce.
        """
        if self.sub_reductions is not None:
            self.validate_sub_reductions()
            return

        if self.size == 0:
            log.warning("No scans to reduce. Exiting.")
            raise ValueError("No scans to reduce.")

        self.info.validate_scans(self.scans)
        self.scans = [scan for scan in self.scans if scan is not None]
        if not self.is_valid():
            if self.is_sub_reduction:
                reduction_name = f'Sub-reduction {self.reduction_number}'
            else:
                reduction_name = 'Reduction'
            log.warning(f"{reduction_name} contains no valid scans.")
            return

        self.set_observing_time_options()

        # Make the global options derive from those of the first scan.
        # That way any options activated conditionally for that scan become
        # global starters as well.

        self.channels = self.scans[0].channels.copy()
        self.info.unlink_configuration()

        # Keep only the non-specific global options
        options = self.configuration.options
        for scan in self.scans:
            options = utils.dict_intersection(
                options, scan.configuration.options)
        self.configuration.options = options

        # Remove empty scans
        self.scans = [scan for scan in self.scans if scan.size > 0]

        # If this is a collection of reductions, the source model needs
        # to be consistent.
        self.read_end_time = time.time()

        is_sub_reduction = self.parent_reduction is not None
        if is_sub_reduction:
            return

        if None not in [self.read_start_time, self.read_end_time]:
            dt = self.read_end_time - self.read_start_time
            log.debug(f"Total read time = {dt} seconds")

        if not self.configuration.get_bool('lab'):
            self.init_source_model()

        self.update_parallel_config(reset=True)
        self.init_pipelines()

    def validate_scans(self):
        """
        Remove any invalid scans from the reduction scans.

        Returns
        -------
        None
        """
        if self.scans is None or len(self.scans) == 0:
            return
        self.scans = [x for x in self.scans if x.is_valid()]

    def is_valid(self):
        """
        Return whether the reduction contains any valid scans.

        Note that this also removes any invalid integrations from all scans.

        Returns
        -------
        bool
        """
        self.validate_scans()
        if self.scans is None or len(self.scans) == 0:
            return False
        return True

    def validate_sub_reductions(self):
        """
        Validate all sub reductions, then initialize models and pipelines.

        Returns
        -------
        None
        """
        if self.sub_reductions is None:
            raise ValueError("Reduction does not contain sub-reductions.")

        self.sub_reductions = [sr for sr in self.sub_reductions if sr.size > 0]
        self.sub_reductions = sorted(self.sub_reductions,
                                     key=lambda x: x.reduction_number)
        self.update_parallel_config(reset=True)

        n_sub = len(self.sub_reductions)
        jobs = int(np.clip(n_sub, 1, self.parallel_reductions))

        sub_reductions = multiprocessing.multitask(
            self.parallel_safe_validate_sub_reductions,
            range(n_sub), self.sub_reductions, None,
            jobs=jobs, max_nbytes=None,
            force_threading=True, logger=log)

        sub_reductions = [x for x in sub_reductions if x.is_valid()]
        if len(sub_reductions) == 0:
            log.warning("There are no valid sub-reductions available.")
            return

        self.assign_sub_reductions(sub_reductions)

        if not self.configuration.get_bool('lab'):
            if self.configuration.get_bool('commonwcs'):
                self.init_collective_source_model()
            else:
                for sub_reduction in self.sub_reductions:
                    if sub_reduction.size > 0:
                        sub_reduction.init_source_model()

        self.assign_sub_reductions(multiprocessing.multitask(
            self.parallel_safe_init_pipelines,
            range(n_sub), self.sub_reductions, None,
            jobs=jobs, max_nbytes=None,
            force_threading=True, logger=log))

        self.read_end_time = time.time()
        if None not in [self.read_start_time, self.read_end_time]:
            dt = self.read_end_time - self.read_start_time
            log.debug(f"Total read time = {dt} seconds")

    @classmethod
    def parallel_safe_validate_sub_reductions(cls, sub_reductions, index):
        """
        Multitask safe function to validate each sub-reduction.

        Parameters
        ----------
        sub_reductions : list (Reduction)
            All sub-reductions to validate.
        index : int
            The index of the sub-reduction to validate.

        Returns
        -------
        None
        """
        sub_reduction = sub_reductions[index]
        sub_reduction.validate()
        return sub_reduction

    @classmethod
    def parallel_safe_init_pipelines(cls, sub_reductions, index):
        """
        Multitask safe function to initialize sub-reduction pipelines.

        Parameters
        ----------
        sub_reductions : list (Reduction)
            All sub-reductions to initialize.
        index : int
            The index of the sub-reduction to initialize.

        Returns
        -------
        None
        """
        sub_reduction = sub_reductions[index]
        sub_reduction.init_pipelines()
        return sub_reduction

    def assign_sub_reductions(self, sub_reductions):
        """
        Reassign sub-reductions to the parent (this) reduction.

        This should be performed following any parallel process in order to
        ensure that all sub-reductions reference the parent, and the parallel
        configuration is valid.

        Parameters
        ----------
        sub_reductions : list (Reduction)

        Returns
        -------
        None
        """
        for sub_reduction in sub_reductions:
            sub_reduction.parent_reduction = self
        self.update_parallel_config(reset=True)

    def set_observing_time_options(self):
        """
        Set the configuration options for the observing time.

        Returns
        -------
        None
        """
        obstime = self.get_total_observing_time()
        for key, options in self.configuration.conditions.options.items():
            if key.startswith('obstime'):
                if len(key) < (len('obstime') + 2):
                    continue
                operator = key[len('obstime')]
                value = float(key[len('obstime') + 1:]) * units.Unit('second')
                if operator == '<':
                    if obstime < value:
                        self.apply_options_to_scans(options)
                elif operator == '>':
                    if obstime > value:
                        self.apply_options_to_scans(options)

    def apply_options_to_scans(self, options):
        """
        Apply configuration options to all scans.

        Parameters
        ----------
        options : dict

        Returns
        -------
        None
        """
        for scan in self.scans:
            scan.configuration.apply_configuration_options(options)

    def get_total_observing_time(self):
        """
        Return the total observing time for all scans in the reduction.

        Returns
        -------
        observing_time: astropy.units.Quantity
            The total observing time.
        """
        exposure = 0.0 * units.Unit('second')
        for scan in self.scans:
            exposure += scan.get_observing_time()
        return exposure

    def init_source_model(self):
        """
        Initialize the source model

        Returns
        -------
        None
        """
        self.source = self.info.get_source_model_instance(
            self.scans, reduction=self)

        if self.source is None:
            log.warning("No source model or invalid source model type.")
            return

        self.source.create_from(self.scans, assign_scans=False)
        self.source.assign_reduction(self)

    def init_collective_source_model(self):
        """
        Create a source model for each sub reduction with the same WCS.

        Returns
        -------
        None
        """
        if self.sub_reductions is None:
            raise ValueError("Reduction contains no sub-reductions.")

        log.debug("Creating shared WCS map for all sub-reductions.")

        all_scans = []
        for sub_reduction in self.sub_reductions:
            all_scans.extend(sub_reduction.scans)

        # Derive from the first reduction settings
        for reduction in self.sub_reductions:
            if reduction.size > 0:
                break
        else:
            log.warning("No data from which to create a map")
            return

        common_map = reduction.info.get_source_model_instance(
            all_scans, reduction=reduction)

        if common_map is None:
            log.warning("No source model or invalid source model type.")
            return

        common_map.create_from(all_scans, assign_scans=False)

        for sub_reduction in self.sub_reductions:
            if sub_reduction.size > 0:
                source = common_map.get_clean_local_copy(full=True)
                source.assign_reduction(sub_reduction)

    def update_runtime_config(self, reset=False):
        """
        Update important configuration settings during prior to run.

        The output path and parallel processing configuration will
        be determined during this stage.

        Parameters
        ----------
        reset : bool, optional
            If `True`, re-assign the available parallel jobs to this reduction
            and all sub-reductions if necessary.

        Returns
        -------
        None
        """
        if self.configuration.has_option('outpath'):
            self.set_outpath()
        self.update_parallel_config(reset=reset)

    def set_outpath(self):
        """
        Set the output directory based on the configuration.

        If the configuration path does not exist, it will be created if the
        'outpath.create' option is set.  Otherwise, an error will be raised.

        Returns
        -------
        None
        """
        self.configuration.set_outpath()

    def update_parallel_config(self, reset=False):
        """
        Update the maximum number of jobs to parallel process.

        Parameters
        ----------
        reset : bool, optional
            If `True`, re-assign the available parallel jobs to this reduction
            and all sub-reductions if necessary.

        Returns
        -------
        None
        """
        self.max_jobs = 1
        self.max_cores = 1

        if (self.configuration is None
                or not self.configuration.has_option('parallel')):
            return

        total_cores = multiprocessing.get_core_number()
        have_idle = self.configuration.has_option('parallel.idle')
        if have_idle:
            idle = self.configuration.get_float('parallel.idle', default=0)
            total_cores -= int(idle * total_cores)
            total_cores = int(np.clip(total_cores, 1, None))

        if self.configuration.has_option('parallel.cores'):
            cores = self.configuration.get_float('parallel.cores')
            self.max_cores = multiprocessing.valid_relative_jobs(cores)
        elif have_idle:
            self.max_cores = total_cores

        if self.configuration.has_option('parallel.jobs'):
            jobs = self.configuration.get_float('parallel.jobs')
            self.max_jobs = multiprocessing.valid_relative_jobs(jobs)
        elif have_idle:
            self.max_jobs = total_cores

        self.max_jobs = int(np.clip(self.max_jobs, 1, total_cores))
        self.max_cores = int(np.clip(self.max_cores, 1, total_cores))
        self.assign_parallel_jobs(reset=reset)

    def assign_parallel_jobs(self, reset=False):
        """
        Determine the parallel jobs for the reduction.

        Determines:
            1 - The number of sub-reductions to read in parallel
            2 - The number of scans to process in parallel
            3 - The number of tasks (processes within scans) to run in parallel

        Parameters
        ----------
        reset : bool, optional
            If `True`, allow the parallel settings for sub-reductions to be
            updated.

        Returns
        -------
        None
        """
        # Only the parent reduction should assign parallel jobs.
        if self.is_sub_reduction:
            self.parent_reduction.assign_parallel_jobs()
            return

        if not reset and self.jobs_assigned:
            return

        mode = self.configuration.get_string('parallel.mode', default='hybrid')

        # In the case that there are no sub-reductions
        if self.sub_reductions is None or len(self.sub_reductions) == 0:
            self.parallel_reductions = 1
            n_scans = self.size
            if self.reduction_files is None:
                n_read = 0
            else:
                n_read = len(self.reduction_files)

            if n_read == 0:
                # Assume no files or scans assigned yet
                self.parallel_read = self.max_cores
            else:
                self.parallel_read = int(np.clip(n_read, 1, self.max_cores))

            if mode == 'scans':
                self.parallel_tasks = 1
                self.parallel_scans = int(np.clip(n_scans, 1, self.max_jobs))
            elif mode == 'ops':
                self.parallel_scans = 1
                self.parallel_tasks = self.max_jobs
            else:
                self.parallel_scans = int(np.clip(n_scans, 1, self.max_jobs))
                self.parallel_tasks = int(np.clip(
                    self.max_jobs // self.parallel_scans, 1, self.max_jobs))

            self.available_reduction_jobs = (
                self.parallel_scans * self.parallel_tasks)
            self.jobs_assigned = True
            return

        n_scans = np.asarray(
            [sub_reduction.size for sub_reduction in self.sub_reductions])
        total_scans = np.sum(n_scans)
        n_scan_jobs = n_scans.copy()
        n_reductions = n_scan_jobs.size
        task_jobs = np.ones(n_reductions, dtype=int)

        n_read = [0 if sr.reduction_files is None else len(sr.reduction_files)
                  for sr in self.sub_reductions]
        n_read = np.asarray(n_read)
        while np.sum(n_read) > self.max_cores:
            n_read[np.argmax(n_read)] -= 1
        n_read = np.clip(n_read, 1, None)

        # The core operations
        self.parallel_read = int(np.clip(np.sum(n_read), 1, self.max_cores))
        self.parallel_reductions = int(np.clip(n_reductions, 1,
                                               self.max_cores))

        # The thread based operations
        if mode == 'ops':
            self.parallel_scans = 1
            self.parallel_tasks = self.max_jobs
            n_scan_jobs.fill(1)
            task_jobs.fill(int(
                np.clip(self.max_jobs // self.parallel_reductions, 1, None)))
        else:
            self.parallel_scans = int(np.clip(total_scans, 1, self.max_jobs))
            while n_scan_jobs.sum() > self.max_jobs:
                n_scan_jobs[np.argmax(n_scan_jobs)] -= 1
            n_scan_jobs = np.clip(n_scan_jobs, 1, None)

        if mode == 'scans':
            self.parallel_tasks = 1
        elif mode != 'ops':  # hybrid
            while np.sum(n_scan_jobs * task_jobs) < self.max_jobs:
                min_idx = np.argmin(n_scan_jobs * (task_jobs + 1))
                task_jobs[min_idx] += 1
                if np.sum(n_scan_jobs * task_jobs) > self.max_jobs:
                    task_jobs[min_idx] -= 1
                    break
            self.parallel_tasks = int(
                np.clip(self.max_jobs // self.parallel_scans,
                        1, self.max_jobs))

        self.available_reduction_jobs = (
            self.parallel_scans * self.parallel_tasks)
        available_reduction_jobs = n_scan_jobs * task_jobs
        for i, sub_reduction in enumerate(self.sub_reductions):
            sub_reduction.parallel_reductions = 1
            sub_reduction.parallel_scans = n_scan_jobs[i]
            sub_reduction.parallel_tasks = task_jobs[i]
            sub_reduction.available_reduction_jobs = \
                available_reduction_jobs[i]
            sub_reduction.parallel_read = n_read[i]
            sub_reduction.jobs_assigned = True

        self.jobs_assigned = True

    def init_pipelines(self):
        """
        Initialize the reduction pipeline.

        The parallel pipelines defines that maximum number of pipelines that
        should be created that may iterate in parallel.  Parallel tasks are
        the number of cores left available to process in parallel by the
        pipeline.

        Returns
        -------
        None
        """
        if self.is_sub_reduction:
            info_str = f'Sub-reduction {self.reduction_number}'
        else:
            info_str = 'Reduction'

        n_scans = self.size

        info_str += f' will process {n_scans} scan{"s" if n_scans > 1 else ""}'

        if self.available_reduction_jobs > 1:
            info_str += " using "
            if self.parallel_scans > 1 and self.parallel_tasks > 1:
                info_str += (f'{self.parallel_scans} parallel scans X '
                             f'{self.parallel_tasks} parallel tasks.')
            elif self.parallel_scans > 1:
                info_str += f'{self.parallel_scans} parallel scans.'
            else:
                info_str += f'{self.parallel_tasks} parallel tasks.'
        else:
            info_str += " serially."

        log.debug(info_str)

        self.pipeline = Pipeline(reduction=self)
        self.pipeline.set_source_model(self.source)
        for scan in self.scans:
            self.pipeline.add_scan(scan)
            for integration in scan.integrations:
                integration.set_thread_count(self.parallel_tasks)

    def set_object_options(self, source_name):
        """
        Set the configuration options for an observing source.

        Parameters
        ----------
        source_name : str
            The source name.

        Returns
        -------
        None
        """
        self.configuration.set_object(source_name, validate=True)

    def set_iteration(self, iteration, rounds=None, for_scans=True):
        r"""
        Set the configuration for a given iteration

        Parameters
        ----------
        iteration : float or int or str
            The iteration to set.  A positive integer defines the exact
            iteration number.  A negative integer defines the iteration
            relative to the last (e.g. -1 is the last iteration). A float
            represents a fraction (0->1) of the number of rounds, and
            a string may be parsed in many ways such as last, first, a float,
            integer, or percentage (if suffixed with a %).
        rounds : int, optional
            The maximum number of iterations.
        for_scans : bool, optional
            If `True`, set the iteration for all scans as well.

        Returns
        -------
        None
        """
        if rounds is not None:
            self.configuration.iterations.max_iteration = rounds
            self.rounds = rounds

        self.configuration.set_iteration(iteration)
        if for_scans:
            for scan in self.scans:
                scan.set_iteration(iteration, rounds=rounds)

    def reduce(self):
        """
        Perform the reduction.

        Returns
        -------
        None
        """
        self.reduce_start_time = time.time()
        if self.sub_reductions is not None:
            self.reduce_sub_reductions()
            self.reduce_end_time = time.time()
            return

        if not self.is_valid():
            log.warning("No valid scans: cannot reduce")
            self.reduce_end_time = time.time()
            return

        log.info(f"Reducing {self.size} scan(s).")
        if self.configuration.get_bool('bright'):
            log.info("Bright source reduction.")
        elif self.configuration.get_bool('faint'):
            log.info("Faint source reduction.")
        elif self.configuration.get_bool('deep'):
            log.info("Deep source reduction.")
        else:
            log.info("Default reduction.")

        if self.configuration.get_bool('extended'):
            log.info("Assuming extended source(s).")

        log.info(f"Assuming {self.info.instrument.get_source_size()} "
                 f"sized source(s).")

        if self.rounds is None or self.rounds < 0:
            raise ValueError("No rounds specified in configuration.")

        for iteration in range(1, self.rounds + 1):
            log.info(f"Round {iteration}:")
            self.set_iteration(iteration, rounds=self.rounds)
            self.iterate()

        self.write_products()
        self.reduce_end_time = time.time()
        reduce_time = self.reduce_end_time - self.reduce_start_time

        if self.is_sub_reduction:
            reduction_string = f'Sub-reduction {self.reduction_number}'
        else:
            reduction_string = 'Reduction'

        log.info(f"{reduction_string} complete! ({reduce_time:.3f} seconds)")

        if not self.is_sub_reduction:
            if None not in [self.reduce_end_time, self.read_start_time]:
                dt = self.reduce_end_time - self.read_start_time
                log.debug(f"Total reduction time = {dt:.3f} seconds.")

    def reduce_sub_reductions(self):
        """
        Reduce all sub-reductions.

        Returns
        -------
        None
        """
        if self.sub_reductions is None:
            raise RuntimeError("No sub-reductions exist.")

        n_sub = len(self.sub_reductions)
        if n_sub <= 0:
            log.warning("There are no sub-reductions to reduce.")
            return

        jobs = int(np.clip(self.parallel_reductions, 1, n_sub))
        args = (self.sub_reductions,)
        kwargs = None
        msg = f"Performing {n_sub} reductions"
        if jobs > 1:
            msg = f"{msg} (in parallel using {jobs} jobs)"
        log.debug(msg)

        use_pickle = jobs > 1
        if use_pickle:
            self.pickle_sub_reductions()

        self.sub_reductions = multiprocessing.multitask(
            self.parallel_safe_reduce_sub_reduction, range(n_sub),
            args, kwargs, jobs=jobs, max_nbytes=None,
            force_processes=True, logger=log)

        if use_pickle:
            self.unpickle_sub_reductions(delete=True)

        self.reduce_end_time = time.time()
        if None not in [self.reduce_end_time, self.reduce_start_time]:
            reduction_time = self.reduce_end_time - self.reduce_start_time
            log.info(f"Reduction complete! ({reduction_time:.3f} seconds)")

    @classmethod
    def parallel_safe_reduce_sub_reduction(cls, args, reduction_number):
        """
        Reduce a single sub-reduction.

        If the sub-reduction is a string and point to a file, it will be taken
        to be a cloudpickle file and restored.  If the reduction was
        successful, this file will be deleted.

        Parameters
        ----------
        args : 1-tuple
            A tuple containing all sub-reductions
        reduction_number : int
            The sub-reduction to reduce.

        Returns
        -------
        sub_reduction : Reduction or str
           A Reduction object if pickling is not enabled, or a path to the
           pickle file if it is.
        """
        sub_reductions = args[0]
        sub_reduction = sub_reductions[reduction_number]
        sub_reduction, reduction_file = multiprocessing.unpickle_file(
            sub_reduction)

        log.info(f"Reducing sub-reduction {sub_reduction.reduction_id}")
        sub_reduction.reduce()

        sub_reduction = multiprocessing.pickle_object(
            sub_reduction, reduction_file)
        return sub_reduction

    def iterate(self, tasks=None):
        """
        Perform a single iteration.

        Parameters
        ----------
        tasks : list (str)
            A list of tasks to perform for the iteration.

        Returns
        -------
        None
        """
        if tasks is None:
            tasks = self.configuration.get_list('ordering')
            tasks = [task.lower().strip() for task in tasks]
            if 'source' in tasks and self.solve_source():
                source_index = tasks.index('source') + 1
                self.iterate(tasks[:source_index])
                tasks = tasks[source_index:]

        if len(tasks) == 0:
            return

        self.queue = [scan for scan in self.scans]
        if self.solve_source() and 'source' in tasks:
            self.source.renew()

        self.iterate_pipeline(tasks)
        self.summarize()

        if self.solve_source() and 'source' in tasks:
            self.source.process()
            self.source.sync()
            log.info(f" [Source] {' '.join(self.source.process_brief)}")
            self.source.clear_process_brief()

        if self.configuration.is_configured('whiten'):
            if self.configuration.is_configured('whiten.once'):
                self.configuration.purge('whiten')

    def iterate_pipeline(self, tasks):
        """
        Perform a single iteration of the pipeline.

        Parameters
        ----------
        tasks : list (str)
            A list of the pipeline tasks to run.

        Returns
        -------
        None
        """
        self.pipeline.set_ordering(tasks)
        self.pipeline.iterate()

    def checkout(self, integration):
        """
        Remove an integration from the queue.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        if self.queue is not None:
            if integration in self.queue:
                self.queue.remove(integration)

    def solve_source(self):
        """
        Return whether to solve for the source.

        Returns
        -------
        bool
        """
        if self.source is None:
            return False
        return self.configuration.get_bool('source')

    def summarize(self):
        """
        Summarize (print logs) for the iteration.

        Returns
        -------
        None
        """
        for scan in self.scans:
            for integration in scan.integrations:
                if integration not in self.queue:
                    self.summarize_integration(integration)

    @staticmethod
    def summarize_integration(integration):
        """
        Summarize (print logs) for an integration.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        if integration.comments is None:
            comments = ''
        else:
            comments = ''.join(integration.comments)

        log.info(f"[{integration.get_display_id()}] {comments}")

    def write_products(self):
        """
        Write the products of the reduction to file.

        Returns
        -------
        None
        """
        if self.source is None:
            return
        self.source.suggestions()

        if self.source.is_valid():
            self.source.write(self.work_path)
        else:
            log.warning("The reduction did not result in a source model.")

        for scan in self.scans:
            scan.write_products()

    def add_user_configuration(self, **kwargs):
        """
        Add command line options to the reduction.

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        None
        """
        if len(kwargs) == 0:
            return

        # Need to convert all values to strings for the ConfigObj
        kwargs = self.configuration.normalize_options(kwargs)

        self.stored_user_configuration = deepcopy(kwargs)

        options_string = ', '.join([f'{k}={v}' for (k, v) in kwargs.items()])
        log.info(f"Applying user configuration settings: {options_string}")

        new_kwargs = {}
        for key, value in kwargs.items():
            if key != 'options':
                new_kwargs.update({key: value})
            elif isinstance(value, dict):
                for options_key, options_value in value.items():
                    options = self.configuration.aliases.unalias_branch(
                        {options_key: options_value})
                    new_kwargs.update(options)

        self.configuration.read_configuration(new_kwargs, validate=False)

        flattened = self.configuration.flatten(new_kwargs)

        # Need to lock user settings
        for key in flattened.keys():
            if key == 'rounds':
                self.configuration.lock_rounds(new_kwargs['rounds'])
            elif key in self.configuration.command_keys:
                continue
            elif key in self.configuration.section_keys:
                continue
            else:
                value_key = f'{key}.value'
                if self.configuration.exists(value_key):
                    self.configuration.lock(value_key)
                else:
                    self.configuration.lock(key)

        self.configuration.validate()
        self.info.validate_configuration_registration()

    def pickle_sub_reductions(self):
        """
        Convert all sub-reductions to pickle files.

        All sub-reduction Reduction objects will be converted to pickle files,
        and the `sub_reductions` attribute will contain on-disk file locations
        for those files.

        Returns
        -------
        None
        """
        if self.sub_reductions is None:
            return
        self.pickle_reduction_directory = multiprocessing.pickle_list(
            self.sub_reductions,
            prefix='SOFSCAN_pickle_reduction_cache',
            naming_attribute='reduction_id',
            class_type=Reduction)
        log.debug(
            f'Sub-reductions pickled to {self.pickle_reduction_directory}')

    def unpickle_sub_reductions(self, delete=True):
        """
        Retrieve all sub-reductions from pickle files.

        All sub-reduction Reduction objects will be restored from pickle files
        whose filenames are present in the `sub_reductions` attribute.

        Parameters
        ----------
        delete : bool, optional
            If `True`, delete all pickle files and the pickle directory.

        Returns
        -------
        None
        """
        if self.sub_reductions is None:
            return
        multiprocessing.unpickle_list(self.sub_reductions, delete=delete)
        if delete and os.path.isdir(self.pickle_reduction_directory):
            shutil.rmtree(self.pickle_reduction_directory)
            self.pickle_reduction_directory = None
        for sub_reduction in self.sub_reductions:
            sub_reduction.parent_reduction = self

    def edit_header(self, header):
        """
        Edit an image header with reduction information.

        Parameters
        ----------
        header : fits.Header
            The FITS header to edit.

        Returns
        -------
        None
        """
        if self.reduction_files is None:
            reduction_files = []
        elif isinstance(self.reduction_files, str):
            reduction_files = [self.reduction_files]
        else:
            reduction_files = self.reduction_files
        n_files = len(reduction_files)

        info = [
            ('COMMENT', "<------ SOFSCAN Runtime Configuration ------>"),
            ('SOFSCANV', ReductionVersion.get_full_version(),
             'SOFSCAN version information.'),
            ('ARGS', n_files, "Number of command line input files."),
        ]
        if n_files != 0:
            for i, filename in enumerate(reduction_files):
                info.append((f'ARG{i + 1}', filename, 'Input file.'))

        kwargs = self.stored_user_configuration
        if kwargs is not None and len(kwargs) > 0:
            kwargs = json.dumps(kwargs)
            info.append(('KWARGS', kwargs, 'User input options.'))

        info.extend([
            ('COMMENT', '<------ SOFSCAN Python & OS ------>'),
            ('PYTHON', platform.python_version(), 'The Python version.'),
            ('PYEXEC', sys.executable, 'Python executable.'),
            ('PYIMPL', platform.python_implementation(),
             'The Python implementation.'),
            ('OS', platform.platform(aliased=True), 'Operation System name.'),
            ('OSVER', platform.version(), 'OS Version.'),
            ('OSARCH', platform.machine(), 'OS architecture.'),
            ('CPUS', multiprocessing.get_core_number(),
             'Number of CPU cores/threads available.')
        ])

        try:
            bits = int(platform.architecture()[0].split('bit')[0])
            info.append(('DMBITS', bits, 'Bits in data model.'))
        except Exception as err:  # pragma: no cover
            log.debug(f"Could not determine number of bits for system: {err}")

        info.extend([
            ('CPENDIAN', sys.byteorder, 'CPU Endianness.'),
            ('MAXMEM', psutil.virtual_memory().available // (1024 * 1024),
             'MB of available memory.')])

        try:
            import locale
            language, country = locale.getlocale()[0].split('_')
            info.extend([
                ('COUNTRY', country, 'The user country.'),
                ('LANGUAGE', language, 'The user language.')
            ])
        except Exception as err:  # pragma: no cover
            log.debug(f"Unable to extract locale information: {err}")

        insert_info_in_header(header, info, delete_special=True)
        self.configuration.edit_header(header)
        ReductionVersion.add_history(header)

    def run(self, filenames, **kwargs):
        """
        Run the initialized reduction on a set of files.

        Note that all user configuration options must have been loaded at this
        stage.

        Parameters
        ----------
        filenames : str or list (str)
            The file or files to reduce.

        Returns
        -------
        hdul : fits.HDUList
            A list of HDU objects containing the reduced source map.
        """
        self.add_user_configuration(**kwargs)
        self.info.perform_reduction(self, filenames)

        if self.sub_reductions is not None:
            hduls = [sub_reduction.source.hdul for sub_reduction in
                     self.sub_reductions
                     if sub_reduction.source is not None]
            if self.source is not None:
                hduls = [self.source.hdul] + hduls
            return hduls

        if self.source is not None:
            return self.source.hdul
        else:
            return None
