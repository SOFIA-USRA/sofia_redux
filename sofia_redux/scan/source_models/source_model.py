# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC, abstractmethod
from astropy import log, units
from astropy.time import Time
from copy import deepcopy
import numpy as np
import queue
import re
import shutil

from sofia_redux.toolkit.utilities import multiprocessing
from sofia_redux.scan.utilities.class_provider import \
    info_class_for
from sofia_redux.scan.reduction.version import ReductionVersion

__all__ = ['SourceModel']


class SourceModel(ABC):

    recycler = None

    def __init__(self, info, reduction=None):
        self.info = None
        self.scans = None
        self.hdul = None
        self.id = ''
        self.generation = 0
        self.integration_time = 0.0 * units.Unit('second')
        self.enable_weighting = True
        self.enable_level = True
        self.enable_bias = True
        self.process_brief = None
        self.reduction = reduction
        self.set_info(info)

    @property
    def referenced_attributes(self):
        """
        Return attributes that should be referenced during a copy.

        Returns
        -------
        set (str)
        """
        return {'channels', 'scans', 'reduction'}

    @classmethod
    def clear_recycler(cls):
        """
        Remove all available models from the recycler.

        NOTE: This needs to be done with parallel queues rather than lists.

        Returns
        -------
        None
        """
        if cls.recycler is not None:
            if cls.recycler.empty():
                return
            cls.recycler.task_done()
            for _ in range(cls.recycler.qsize()):
                _ = cls.recycler.get()

    @classmethod
    def set_recycler_capacity(cls, size):
        """
        Set the number of available models in the recycler.

        Parameters
        ----------
        size : int

        Returns
        -------
        None
        """
        if size <= 0:
            cls.recycler = None
        cls.recycler = queue.Queue(maxsize=size)

    def get_recycled_clean_local_copy(self, full=False):
        """
        Return a clean local copy of a source model.

        Returns
        -------
        SourceModel
        """
        recycler = self.__class__.recycler
        if isinstance(recycler, queue.Queue) and not recycler.empty():
            model = recycler.get()
            model.renew()
            return model
        return self.get_clean_local_copy(full=full)

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
        model = self.copy()
        model.reset_processing()
        model.clear_content()
        return model

    def recycle(self):
        """
        Add this source model to the list of available recycled source models.

        Returns
        -------
        None
        """
        if self.__class__.recycler is None:
            return

        if self.__class__.recycler.full():
            log.warning("Source recycler overflow.")
            return

        self.__class__.recycler.put(self)

    @property
    def frame_flagspace(self):
        """
        Return the flags specific to integration frames.

        Returns
        -------
        Flags
        """
        if self.scans is None:
            return None
        return self.scans[0].frame_flagspace

    @property
    def channel_flagspace(self):
        """
        Return the flags specific to channels.

        Returns
        -------
        Flags
        """
        if self.scans is None:
            return None
        return self.scans[0].channel_flagspace

    @property
    def signal_mode(self):
        """
        Return the signal mode flag.

        Returns
        -------
        enum
        """
        return self.frame_flagspace.flags.TOTAL_POWER

    @property
    def exclude_samples(self):
        """
        Return the sample exclusion flags.

        Returns
        -------
        enum.Enum
        """
        return ~self.frame_flagspace.flags.SAMPLE_SOURCE_BLANK

    @property
    def logging_id(self):
        """
        Return the logging ID.

        Returns
        -------
        str
        """
        return 'model'

    @property
    def n_scans(self):
        """
        Return the number of scans in the source model.

        Returns
        -------
        int
        """
        return 0 if self.scans is None else len(self.scans)

    @property
    def configuration(self):
        """
        Return the configuration for the source model.

        Returns
        -------
        Configuration
        """
        if self.info is None:
            return None
        return self.info.configuration

    def has_option(self, option):
        """
        Return whether the configuration option is configured.

        In order to be considered "configured", the option must exist
        and also have a value.

        Parameters
        ----------
        option : str

        Returns
        -------
        configured : bool
        """
        return self.configuration.is_configured(option)

    def copy(self, with_contents=True):
        """
        Return a copy of the source model.

        Parameters
        ----------
        with_contents : bool, optional
            If `True`, return a true copy of the source model.  Otherwise, just
            return a basic version with the necessary meta data.

        Returns
        -------
        SourceModel
        """
        new = self.__class__(self.info, reduction=self.reduction)
        referenced = self.referenced_attributes
        for attribute, value in self.__dict__.items():
            if attribute in referenced:
                setattr(new, attribute, value)
            elif not with_contents:
                continue
            elif hasattr(value, 'copy'):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, deepcopy(value))

        new.process_brief = None
        return new

    def source_option(self, option_name):
        """
        Return the name of the option in configuration specific to the source.

        Parameters
        ----------
        option_name : str
            Name of the option.

        Returns
        -------
        str
        """
        return f'source.{option_name}'

    def get_first_scan(self):
        """
        Return the first scan in the source model scans.

        Returns
        -------
        Scan
        """
        if self.scans is None:
            return None
        return self.scans[0]

    def next_generation(self):
        """
        Initiate a new source model generation.

        Returns
        -------
        None
        """
        self.generation += 1

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
        self.integration_time += time

    def set_info(self, info):
        """
        Set the channels for the source.

        Parameters
        ----------
        info : Info

        Returns
        -------
        None
        """
        self.info = info
        self.info.set_parent(self)

    def add_process_brief(self, message):
        """
        Add a message to the process brief.

        Parameters
        ----------
        message : str

        Returns
        -------
        None
        """
        if self.process_brief is None:
            self.process_brief = []
        if isinstance(message, list):
            self.process_brief.extend(message)
        elif isinstance(message, str):
            self.process_brief.append(message)
        else:
            log.warning(f"Received bad process brief message: {message!r}")

    def clear_process_brief(self):
        """
        Remove all process brief information.

        Returns
        -------
        None
        """
        self.process_brief = []

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
        self.scans = scans

        if self.get_first_scan().is_nonsidereal:
            log.info("Forcing equatorial for moving object.")
            self.configuration.parse_key_value('system', 'equatorial')

        if assign_scans:
            self.assign_scans(scans)

    def assign_scans(self, scans):
        """
        Assign scans to the source model.

        Parameters
        ----------
        scans : list (Scan)
           A list of scans that should be assigned to *THIS* source model.

        Returns
        -------
        None
        """
        self.scans = scans
        jpb = self.scans[0].info.instrument.jansky_per_beam()
        for scan in self.scans:
            scan.set_source_model(self)
            for integration in scan.integrations:
                factor = integration.info.instrument.jansky_per_beam() / jpb
                if isinstance(factor, units.Quantity):
                    factor = factor.decompose().value
                integration.gain *= factor

    def assign_reduction(self, reduction):
        """
        Assign a reduction to the source model.

        Parameters
        ----------
        reduction : Reduction

        Returns
        -------
        None
        """
        self.reduction = reduction
        self.set_info(reduction.info)
        reduction.source = self
        self.assign_scans(reduction.scans)
        self.set_executor(reduction.source_executor)
        self.set_parallel(reduction.max_jobs)
        reduction.set_object_options(self.get_source_name())

    def get_average_resolution(self):
        """
        Return the average resolution.

        Returns
        -------
        astropy.units.Quantity
            The average resolution.
        """
        value = 0.0 * self.info.instrument.get_size_unit() ** 2
        weight = 0.0
        for scan in self.scans:
            for integration in scan.integrations:
                if integration.info is not self.info:
                    resolution = integration.info.instrument.resolution
                    wg2 = scan.weight * integration.gain ** 2
                    value += wg2 * resolution ** 2
                    weight += wg2

        if weight > 0:
            return np.sqrt(value / weight)
        else:
            return self.info.instrument.resolution

    def renew(self):
        """
        Renew the source model.

        Returns
        -------
        None
        """
        self.reset_processing()
        self.clear_content()

    def reset_processing(self):
        """
        Reset the source processing.

        Returns
        -------
        None
        """
        self.generation = 0
        self.integration_time *= 0

    def add_model(self, source_model, weight=1.0):
        """
        Add a source model increment onto this model.

        Parameters
        ----------
        source_model : SourceModel
            Another model to increment this model by.
        weight : float, optional
            The weight of the increment source model.

        Returns
        -------
        None
        """
        self.generation = max(self.generation, source_model.generation)
        self.integration_time += source_model.integration_time
        self.enable_level &= source_model.enable_level
        self.enable_weighting &= source_model.enable_weighting
        self.enable_bias &= source_model.enable_bias
        self.add_model_data(source_model, weight)

    def sync(self):
        """
        Removes the source model from the frame data and dependents.

        Returns
        -------
        None
        """
        if self.has_option(self.source_option('nosync')):
            return
        if self.has_option(self.source_option('coupling')):
            self.add_process_brief('(coupling)')
        self.add_process_brief('(sync)')

        n_parms = self.count_points()
        self.sync_all_integrations()

        for scan in self.scans:
            for integration in scan.integrations:
                # self.sync_integration(integration)
                integration.source_generation += 1
                integration.scan.source_points = n_parms

        self.next_generation()
        self.set_base()

    def sync_all_integrations(self):
        """
        Synchronize all integrations with the source model.

        Sync (subtract the source model) from all integrations in all scans
        used to generate the source model.  If parallel processing is enabled,
        this will be done in parallel using the available_reduction_jobs *
        parallel_tasks jobs.

        Returns
        -------
        None
        """
        scan_integration_mapping = []
        integrations = []
        integration_count = 0
        for scan in self.scans:
            scan_integrations = []
            for integration in scan.integrations:
                integrations.append(integration)
                scan_integrations.append(integration_count)
                integration_count += 1
            scan_integration_mapping.append(scan_integrations)

        # parallel integrations seems to cause problems, so
        # ensure they are processed serially for now
        #   max_parallel = self.reduction.available_reduction_jobs
        #   parallel_integrations = int(np.clip(max_parallel, 1,
        #                                       integration_count))
        parallel_integrations = 1

        if parallel_integrations <= 1:
            for integration in integrations:
                self.sync_integration(integration)
            return
        else:  # pragma: no cover
            log.debug(f"Syncing {parallel_integrations} integrations "
                      f"in parallel.")

            pickle_list = [self] + integrations
            pickle_directory = multiprocessing.pickle_list(pickle_list)
            source_pickle = pickle_list[0]
            integration_pickles = pickle_list[1:]
            args = source_pickle, integration_pickles
            kwargs = None

            integrations = multiprocessing.multitask(
                self.parallel_safe_sync_integration, range(integration_count),
                args, kwargs, jobs=parallel_integrations, max_nbytes=None,
                logger=log)

            multiprocessing.unpickle_list(integrations, delete=True)
            shutil.rmtree(pickle_directory)

            for scan_number, scan in enumerate(self.scans):
                for scan_integration_number, integration_number in enumerate(
                        scan_integration_mapping[scan_number]):
                    integration = integrations[integration_number]
                    integration.scan = scan  # In case object ID changed
                    scan.integrations[scan_integration_number] = integration

    @classmethod
    def parallel_safe_sync_integration(cls, args, integration_number):
        """
        Synchronize a single integration.

        This function is safe for use with :func:`multiprocessing.multitask`.

        Parameters
        ----------
        args : 2-tuple
            The source pickle filename (str), and a list of integration pickle
            filenames (list (str)).
        integration_number : int
            The index of the integration pickle file to sync.

        Returns
        -------
        integration_file : str
            The integration pickle file that was synchronized.  The original
            file is unpickled, synced, and then pickled to the same location.
        """
        source_pickle, integration_pickles = args
        source, source_file = multiprocessing.unpickle_file(source_pickle)
        integration, integration_file = multiprocessing.unpickle_file(
            integration_pickles[integration_number])
        source.sync_integration(integration)
        multiprocessing.pickle_object(integration, integration_file)
        return integration_file

    def get_blanking_level(self):
        """
        Return the blanking level from the configuration.

        Returns
        -------
        float
        """
        return self.configuration.get_float('blank', default=np.nan)

    def get_clipping_level(self):
        """
        Return the clipping level from the configuration.

        Returns
        -------
        float
        """
        return self.configuration.get_float('clip', default=np.nan)

    def get_point_size(self):
        """
        Return the point size of the source model.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.info.instrument.get_point_size()

    def get_source_size(self):
        """
        Return the source size of the source model.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.info.instrument.get_source_size()

    def get_executor(self):
        """
        ???

        Returns
        -------

        """
        pass

    def set_executor(self, executor):
        """
        ???

        Parameters
        ----------
        executor : ???

        Returns
        -------
        None
        """
        pass

    def get_parallel(self):
        """
        Get the number of parallel operations for the source model.

        Returns
        -------
        threads : int
        """
        pass

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
        pass

    def no_parallel(self):
        """
        ???

        Returns
        -------
        None
        """
        pass

    def get_native_unit(self):
        """
        Return the native unit for the source model

        Returns
        -------
        astropy.units.Quantity
        """
        return 1 * units.Unit(
            self.configuration.get('dataunit', default='count'))

    def get_kelvin_unit(self):
        """
        Return the kelvin unit.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.info.instrument.kelvin() * units.Unit('Kelvin')

    def get_canonical_source_name(self):
        """
        Return the source name with bad characters replaced with '_'.

        Returns
        -------
        str
        """
        bad_chars = r'( |\t|\r|\*|"|\?|\\|/)'
        name = re.sub(bad_chars, '_', self.get_source_name())
        name = '_'.join(name.split('_'))
        return name

    def get_default_core_name(self):
        """
        Return a descriptive name for the source model.

        Returns
        -------
        str
        """
        mjds = [scan.mjd for scan in self.scans]
        first_scan = self.scans[np.argmin(mjds)]
        last_scan = self.scans[np.argmax(mjds)]
        name = f'{self.get_canonical_source_name()}.{first_scan.get_id()}'
        if first_scan is not last_scan:
            if first_scan.get_id() != last_scan.get_id():
                name += f'-{last_scan.get_id()}'
        return name

    @staticmethod
    def check_pixel_count(integration):
        """
        Check if an integration has enough mapping pixels to generate a map.

        Parameters
        ----------
        integration : Integration
            The integration to check.

        Returns
        -------
        valid : bool
            `True` if the integration has enough pixels, and `False` otherwise.
        """
        pixels = integration.channels.get_mapping_pixels(match_flag=0).size
        n_obs = integration.channels.get_observing_channels().size

        if integration.has_option('mappingpixels'):
            # If there aren't enough good pixels in the scan, do not generate.
            min_size = integration.configuration.get_int('mappingpixels',
                                                         default=np.inf)
            if pixels < min_size:
                integration.comments.append('(!ch)')
                return False

        if integration.has_option('mappingfraction'):
            mapping_fraction = integration.configuration.get_float(
                'mappingfraction', default=np.inf)
            if pixels < (mapping_fraction * n_obs):
                integration.comments.append('(!ch%)')
                return False

        return True

    def get_ascii_header(self):
        """
        Return the header for an ASCII output file.

        Returns
        -------
        header : str
            A line space delimited header.
        """
        header = [f'# SOFSCAN version: '
                  f'{ReductionVersion().get_full_version()}',
                  f'# Instrument: {self.info.instrument.name}',
                  f'# Object: {self.get_source_name()}']
        if self.scans is not None and len(self.scans) > 0:
            header.append(f'# Equatorial: {self.scans[0].equatorial}')
            scan_list = ' '.join([scan.get_id() for scan in self.scans])
        else:
            scan_list = ''

        header.append(f'# Scans: {scan_list}')
        return '\n'.join(header)

    def get_table_entry(self, name):
        """
        Return the parameter value for a given name.

        Parameters
        ----------
        name : str

        Returns
        -------
        value
        """
        return None

    def parse_header(self, header):
        """
        Parse and apply a FITS header.

        Completely creates a new Channels instance with a blank info, but
        updated with the FITS header.

        Parameters
        ----------
        header : astropy.fits.Header

        Returns
        -------
        None
        """
        info_class = info_class_for(header['INSTRUME'])
        self.info = info_class()
        self.info.parse_image_header(header)

    def edit_header(self, header):
        """
        Update a header with model information.

        Parameters
        ----------
        header : astropy.fits.header.Header
            The FITS header to update.

        Returns
        -------
        None
        """
        header['DATE'] = Time.now().to_value('isot')
        header.comments['DATE'] = 'File creation time.'
        header['SCANS'] = self.n_scans
        header.comments['SCANS'] = (
            'The number of scans in this composite image.')
        header['INTEGRTN'] = self.integration_time.decompose().value
        header.comments['INTEGRTN'] = 'The total integration time in seconds.'
        if self.info is not None:
            self.info.edit_image_header(header, scans=self.scans)

        if self.reduction is not None:
            self.reduction.edit_header(header)

        if self.info is not None:
            self.info.add_history(header, scans=self.scans)

    def add_scan_hdus_to(self, hdul):
        """
        Add each scan HDU to an HDUL

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            The FITS HDU list.

        Returns
        -------
        None
        """
        if self.info is None or self.scans is None:
            return
        if not self.configuration.get_bool('write.scandata'):
            return
        for scan in self.scans:
            hdul.append(scan.get_summary_hdu(configuration=self.configuration))

    @abstractmethod
    def clear_content(self):
        pass

    @abstractmethod
    def is_valid(self):
        """
        Return whether the source model is valid or not.

        Returns
        -------
        bool
        """
        pass

    @abstractmethod
    def add_model_data(self, source_model, weight=1.0):
        """
        Add an increment source model data onto the current model.

        Parameters
        ----------
        source_model : SourceModel
            The source model increment.
        weight : float, optional
            The weight of the source model increment.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def add_integration(self, integration):
        """
        Add an integration to the source model.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def process(self):
        """
        Process the source model.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
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
        pass

    def post_process_scan(self, scan):
        """
        Apply post processing steps to a scan.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def sync_integration(self, integration, signal_mode=None):
        """
        Sync an integration.

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

    @abstractmethod
    def set_base(self):
        """
        ???

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def write(self, path):
        """
        Write the source to file.

        Parameters
        ----------
        path : str
            The file path to write to.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def get_reference(self):
        """
        Return the reference (x, y) coordinate.

        Returns
        -------
        2-tuple
        """
        pass

    def suggestions(self):
        """
        Write log messages with source reduction suggestions.

        Returns
        -------
        None
        """
        scanning_problem_only = False
        scans_with_few_pixels = 0
        for scan in self.scans:
            for integration in scan.integrations:
                if not self.check_pixel_count(integration):
                    scans_with_few_pixels += 1
                    break

        if scans_with_few_pixels > 0:
            scanning_problem_only = self.troubleshoot_few_pixels()
        elif not self.is_valid() * self.generation > 0:
            log.info(self.suggest_make_valid())
        else:
            return

        if not scanning_problem_only:
            log.info("SUGGEST: Please consult the README and/or GLOSSARY "
                     "for details.")

    @staticmethod
    def suggest_make_valid():
        """
        Return a message suggestion corrections to an invalid map.

        Returns
        -------
        message : str
        """
        msg = " * Check the console output for any problems " \
              "when reading scans."
        return msg

    def is_scanning_problem_only(self):
        """
        Return whether source model issues may be related to scanning problems.

        Returns
        -------
        bool
            `True` if scanning problems were detected; `False` otherwise.
        """
        speed_problem_only = True
        for scan in self.scans:
            low_speed = False
            for integration in scan.integrations:
                if not self.check_pixel_count(integration):
                    drift_n = int(
                        np.round(integration.filter_time_scale
                                 / integration.info.sampling_interval))
                    if drift_n <= 1:
                        low_speed = True
                    else:
                        speed_problem_only = False
                    if low_speed and not speed_problem_only:
                        break
            if low_speed:
                log.info(f"SUGGEST: Low scanning speed in {scan.get_id()}.")

        return speed_problem_only

    def troubleshoot_few_pixels(self):
        """
        Generate log messages related to small numbers of pixels in the map.

        Returns
        -------
        scanning_problem_only : bool
            `True` if the problem relates to scanning speeds; `False` if there
            are other issues in play.
        """
        log.warning(
            "It seems that one or more scans contain too few valid pixels for "
            "contributing to the source model. This may be just fine, and "
            "probably indicates that something was sub-optimal with the "
            "affected scan(s)")
        if self.is_scanning_problem_only():
            return True

        messages = ["You may try:"]
        if self.configuration.is_configured('deep'):
            messages.append(" * Reduce with 'faint' instead of 'deep'.")
        elif self.configuration.is_configured('faint'):
            messages.append(
                " * Reduce with default settings instead of 'faint'.")
        elif not self.configuration.is_configured('bright'):
            messages.append(" * Reduce with 'bright'.")

        messages.extend(self.reduction.channels.troubleshoot_few_pixels())

        if (self.configuration.has_option('mappingpixels')
                or self.configuration.has_option('mappingfraction')):
            messages.append(" * Adjust 'mappingpixels' or "
                            "'mappingfraction' to "
                            "allow source extraction with fewer pixels.")

        log.info('\n'.join(messages))
        return False

    @abstractmethod
    def count_points(self):
        """
        Return the number of points in the model.

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def get_source_name(self):
        """
        Return the source name for the source model.

        Returns
        -------
        str
        """
        pass

    @abstractmethod
    def get_unit(self):
        """
        Return the source model unit.

        Returns
        -------
        astropy.units.Quantity
        """
        pass
