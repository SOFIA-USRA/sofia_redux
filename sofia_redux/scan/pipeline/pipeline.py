# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import log

import numpy as np
from sofia_redux.toolkit.utilities import multiprocessing

__all__ = ['Pipeline']


class Pipeline(ABC):

    def __init__(self, reduction):
        """
        Initialize a reduction pipeline.

        Parameters
        ----------
        reduction : Reduction
        """
        self.reduction = reduction
        self.scans = None
        self.ordering = None
        self.scan_source = None
        self.current_task = None
        self.last_task = None

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
        scan_jobs = int(np.clip(n_scans, 1, self.parallel_scans))

        # max_bytes set to None in order to disable memory mapping
        # Memory mapping does not allow numba to modify arrays in-place.
        self.scans = multiprocessing.multitask(
            self.perform_tasks_for_scans, range(n_scans), args, kwargs,
            jobs=scan_jobs, max_nbytes=None, force_threading=True,
            logger=log)

        for scan in self.scans:
            if 'source' in self.ordering and scan.configuration.get_bool(
                    'source'):
                self.update_source(scan)

            for integration in scan.integrations:
                self.reduction.checkout(integration)

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
