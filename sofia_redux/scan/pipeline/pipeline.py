# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import log
import gc
from multiprocessing import Process
import numpy as np
import os
import cloudpickle
import shutil
import tempfile

from sofia_redux.toolkit.utilities import multiprocessing

__all__ = ['Pipeline']


class Pipeline(ABC):

    def __init__(self, reduction):
        """
        Initialize a reduction pipeline.

        The reduction pipeline is responsible for actually performing the
        reduction tasks at each iteration.  This generally involves
        performing the tasks on all integrations in all scans, and updating
        the iteration source model.

        Parameters
        ----------
        reduction : sofia_redux.scan.reduction.reduction.Reduction
        """
        self.reduction = reduction
        self.scans = None
        self.ordering = None
        self.scan_source = None
        self.current_task = None
        self.last_task = None
        self.pickle_directory = None
        self.add_source_queue = None

    @property
    def parallel_scans(self):
        """
        Return the maximum number of parallel scan operations.

        Returns
        -------
        jobs : int
        """
        if self.reduction is None or self.reduction.parallel_scans is None:
            return 1
        return self.reduction.parallel_scans

    @property
    def parallel_tasks(self):
        """
        Return the maximum number of parallel tasks (in-scan) operations.

        Returns
        -------
        jobs : int
        """
        if self.reduction is None or self.reduction.parallel_tasks is None:
            return 1
        return self.reduction.parallel_tasks

    @property
    def available_jobs(self):
        """
        Return the maximum number of jobs that may be performed in parallel.

        Returns
        -------
        jobs : int
        """
        return self.parallel_scans * self.parallel_tasks

    @property
    def configuration(self):
        """
        Return the reduction configuration.

        Returns
        -------
        Configuration
        """
        if self.reduction is None or self.reduction.info is None:
            return None
        return self.reduction.info.configuration

    @property
    def pipeline_id(self):
        """
        Return a unique identifier for the pipeline.

        Returns
        -------
        str
        """
        if self.reduction is None:
            return f'pipeline.{id(self)}'
        else:
            return f'{self.reduction.reduction_id}-pipeline.{id(self)}'

    def set_source_model(self, source):
        """
        Set the source model for the pipeline.

        Parameters
        ----------
        source : Source or None

        Returns
        -------
        None
        """
        if source is not None:
            self.scan_source = source.copy()
            self.scan_source.set_parallel(self.parallel_tasks)
        else:
            self.scan_source = None

    def add_scan(self, scan):
        """
        Add a scan to the pipeline for reduction.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        if self.scans is None:
            self.scans = [scan]
        else:
            self.scans.append(scan)

    def set_ordering(self, ordering):
        """
        Set the task ordering for the pipeline.

        Parameters
        ----------
        ordering : list (str)
            A list of tasks to perform.

        Returns
        -------
        None
        """
        self.ordering = ordering

    def update_source(self, scan):
        """
        Update the reduction source model with a scan.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        if self.reduction is None or self.reduction.source is None:
            return
        self.scan_source.renew()
        self.scan_source.set_info(scan.info)

        for integration in scan.integrations:

            if integration.has_option('jackknife'):
                sign = '+' if integration.gain > 0 else '-'
                integration.comments.append(sign)
            elif integration.gain < 0:
                integration.comments.append('-')

            self.scan_source.add_integration(integration)

        if scan.get_source_generation() > 0:
            self.scan_source.enable_level = False

        self.scan_source.process_scan(scan)

        self.reduction.source.add_model(self.scan_source, weight=scan.weight)

        self.scan_source.post_process_scan(scan)
        if self.configuration.get_bool('source.delete_scan'):
            scan.source_model = None

    def iterate(self):
        """
        Perform an iteration.

        Returns
        -------
        None
        """
        n_scans = len(self.scans)
        args = self.scans, self.ordering, self.parallel_tasks
        kwargs = None

        if self.configuration.get_bool('parallel.scans'):
            scan_jobs = int(np.clip(n_scans, 1, self.parallel_scans))
        else:
            scan_jobs = 1

        # max_bytes set to None in order to disable memory mapping
        # Memory mapping does not allow numba to modify arrays in-place.
        self.scans = multiprocessing.multitask(
            self.perform_tasks_for_scans, range(n_scans), args, kwargs,
            jobs=scan_jobs, max_nbytes=None, force_threading=True,
            logger=log)

        gc.collect()

        if self.configuration.get_bool('parallel.source'):
            if ('source' in self.ordering and
                    self.configuration.get_bool('source')):
                self.update_source_parallel_scans()
        else:
            self.update_source_serial_scans()

    def update_source_serial_scans(self):
        """
        Update the source using serial processing.

        Returns
        -------
        None
        """
        for i, scan in enumerate(self.scans):
            if ('source' in self.ordering and
                    scan.configuration.get_bool('source')):
                self.update_source(scan)

    def update_source_parallel_scans(self):
        """
        Update the source in parallel.

        Returns
        -------
        None
        """
        renewed_source = self.scan_source.copy()
        renewed_source.renew()
        renewed_source.reduction = None
        renewed_source.scans = None
        renewed_source.hdul = None
        renewed_source.info = None
        temp_directory = tempfile.mkdtemp('_sofscan_update_source_pipeline')

        n_scans = len(self.scans)
        scan_jobs = int(np.clip(n_scans, 1, self.parallel_scans))
        delete = self.configuration.get_bool('source.delete_scan')

        for i in range(scan_jobs):
            source_file = os.path.join(temp_directory, f'renewed_source_{i}.p')
            with open(source_file, 'wb') as f:
                cloudpickle.dump(renewed_source, f)
        del renewed_source

        update_files = multiprocessing.multitask(
            self.do_process, range(n_scans),
            (self.scans, temp_directory, scan_jobs, delete), None,
            jobs=scan_jobs, max_nbytes=None, force_threading=True, logger=log)

        for i, filename in enumerate(update_files):
            with open(filename, 'rb') as f:
                source = cloudpickle.load(f)
                self.reduction.source.add_model(
                    source, weight=self.scans[i].weight)
            del source
            os.remove(filename)

        gc.collect()

        for i in range(scan_jobs):
            source_file = os.path.join(temp_directory, f'renewed_source_{i}.p')
            if os.path.isfile(source_file):
                os.remove(source_file)

        shutil.rmtree(temp_directory)

    @classmethod
    def do_process(cls, args, block):
        """
        Multiprocessing safe implementation for source processing of scans.

        Parameters
        ----------
        args : 4-tuple
            args[0] = scans (list (Scan))
            args[1] = temporary directory name (str)
            args[2] = number of parallel jobs (int)
            args[3] = Whether to clear certain data from the scan (bool)
        block : int
            The index of the scan to process.

        Returns
        -------
        scan_pickle_file : str
            The filename pointing to the processed scan saved as a pickle file.
        """
        scans, temp_directory, scan_jobs, delete = args
        scan = scans[block]

        source_file = os.path.join(
            temp_directory, f'renewed_source_{block % scan_jobs}.p')

        with open(source_file, 'rb') as f:
            source = cloudpickle.load(f)

        source.set_info(scan.info)
        source.scans = [scan]

        for integration in scan.integrations:
            if integration.has_option('jackknife'):
                sign = '+' if integration.gain > 0 else '-'
                integration.comments.append(sign)
            elif integration.gain < 0:
                integration.comments.append('-')
            source.add_integration(integration)
            del integration

        if scan.get_source_generation() > 0:
            source.enable_level = False

        source.process_scan(scan)

        # Now need to save for later
        update_file = os.path.join(temp_directory, f'source_update_{block}.p')
        source.info = None
        source.reduction = None
        source.scans = None
        with open(update_file, 'wb') as f:
            cloudpickle.dump(source, f)
        source.set_info(scan.info)
        source.scans = [scan]

        source.process_scan(scan)
        source.post_process_scan(scan)

        if delete:
            scan.source_model = None
            for integration in scan.integrations:
                integration.frames.map_index = None

        # Remove all references
        scan.info.set_parent(scan)
        source.info = None
        source.scans = None
        del source
        del scan
        return update_file

    @classmethod
    def perform_tasks_for_scans(cls, args, block):
        """
        Perform a single iteration of all tasks for all scans for the
        pipeline.

        Returns
        -------
        None
        """
        scans, ordering, parallel_tasks = args
        scan = scans[block]
        for integration in scan.integrations:
            integration.next_iteration()
            integration.set_thread_count(parallel_tasks)

        for task in ordering:
            if scan.has_option(task):
                log.debug(f"Performing task: {task}")
                scan.perform(task)

        return scan
