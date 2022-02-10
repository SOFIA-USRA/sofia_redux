# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import log, units
from copy import deepcopy
import os

from sofia_redux.scan.utilities.class_provider import (
    channel_class_for,
    channel_data_class_for,
    channel_group_class_for,
    get_scan_class)
from sofia_redux.scan.info.base import InfoBase
from sofia_redux.scan.info.instrument import InstrumentInfo
from sofia_redux.scan.info.astrometry import AstrometryInfo
from sofia_redux.scan.info.origination import OriginationInfo
from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.utilities.class_provider import \
    info_class_for
from sofia_redux.scan.info.observation import ObservationInfo
from sofia_redux.scan.info.telescope import TelescopeInfo
from sofia_redux.scan.source_models.astro_intensity_map import \
    AstroIntensityMap
from sofia_redux.scan.source_models.sky_dip import SkyDip

__all__ = ['Info']


class Info(ABC):

    def __init__(self, configuration_path=None):
        """
        Initialize an Info object.

        Parameters
        ----------
        configuration_path : str, optional
            An alternate directory path to the configuration tree to be
            used during the reduction.  The default is
            <package>/data/configurations.
        """
        self.name = None
        self.scan = None
        self.parallelism = None
        self.parent = None
        self.configuration = Configuration(
            configuration_path=configuration_path)
        self.instrument = InstrumentInfo()
        self.astrometry = AstrometryInfo()
        self.observation = ObservationInfo()
        self.origin = OriginationInfo()
        self.telescope = TelescopeInfo()

    @property
    def referenced_attributes(self):
        """
        Return a set of attribute names that should be referenced during copy.

        Returns
        -------
        set (str)
        """
        return {'configuration', 'scan', 'parent'}

    def set_parent(self, owner):
        """
        Set the owner of the information.

        Parameters
        ----------
        owner : object

        Returns
        -------
        None
        """
        self.parent = owner

    def copy(self):
        """
        Create and return a copy of the Info object.

        The 'applied_scan' and 'configuration' attributes are referenced.  All
        other attributes are copied using their `copy` method, or deepcopy.

        Returns
        -------
        Info
        """
        new = self.__class__()
        referenced_only = self.referenced_attributes
        for attribute, value in self.__dict__.items():
            if attribute in referenced_only:
                setattr(new, attribute, value)
            elif hasattr(value, 'copy'):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, deepcopy(value))

        # So that info configurations reference new.configuration
        for info_value in new.available_info.values():
            info_value.configuration = new.configuration

        return new

    def unlink_configuration(self):
        """Ensure the configuration is not referenced."""
        self.configuration = self.configuration.copy()
        for info in self.available_info.values():
            info.configuration = self.configuration

    def unlink_scan(self):
        """Ensure the scan is not referenced."""
        self.scan = deepcopy(self.scan)

    @property
    def available_info(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, InfoBase):
                result[key] = value
        return result

    @property
    def config_path(self):
        """
        Return the configuration path for this information.

        Returns
        -------
        file_path : str
        """
        if self.name is None:
            return self.configuration.config_path
        else:
            return os.path.join(self.configuration.config_path, self.name)

    @classmethod
    def instance_from_instrument_name(cls, name, configuration_path=None):
        """
        Return an Info instance given an instrument name.

        Parameters
        ----------
        name : str
            The name of the instrument
        configuration_path : str, optional
            An alternate directory path to the configuration tree to be
            used during the reduction.  The default is
            <package>/data/configurations.

        Returns
        -------
        Info
        """
        return info_class_for(name)(
            configuration_path=configuration_path)

    @property
    def size_unit(self):
        """
        Return the size unit for the instrument.

        Returns
        -------
        astropy.units.Unit
        """
        return self.instrument.get_size_unit()

    @property
    def frequency(self):
        """
        Return the instrument frequency.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.instrument.frequency

    @property
    def integration_time(self):
        """
        Return the instrument integration time.

        Returns
        -------
        astropy.units.Quantity
            The instrument integration time in seconds.
        """
        return self.instrument.integration_time

    @integration_time.setter
    def integration_time(self, value):
        """
        Set the instrument integration time.

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        if not isinstance(value, units.Quantity):
            raise ValueError(f"Integration time must be {units.Quantity}.")
        self.instrument.integration_time = value.copy()

    @property
    def resolution(self):
        """
        Return the instrument resolution (spatial size)

        Returns
        -------
        astropy.units.Quantity
        """
        return self.instrument.resolution

    @resolution.setter
    def resolution(self, value):
        """
        Set the instrument resolution (spatial size).

        Parameters
        ----------
        value : astropy.units.Quantity

        Returns
        -------
        None
        """
        unit = self.size_unit
        if isinstance(value, units.Quantity):
            self.instrument.resolution = value.to(unit)
        else:
            self.instrument.resolution = value * unit

    @property
    def sampling_interval(self):
        """
        Return the instrument sampling interval (time).

        Returns
        -------
        astropy.units.Quantity
        """
        return self.instrument.sampling_interval

    @sampling_interval.setter
    def sampling_interval(self, value):
        """
        Set the instrument sampling interval (time).

        Parameters
        ----------
        value : units.Quantity

        Returns
        -------
        None
        """
        if not isinstance(value, units.Quantity):
            raise ValueError(f"Sampling interval must be {units.Quantity}.")
        self.instrument.sampling_interval = value.copy()

    @property
    def gain(self):
        """
        Return the overall instrument gain.

        Returns
        -------
        float
        """
        return self.instrument.gain

    @gain.setter
    def gain(self, value):
        """
        Set the overall instrument gain.

        Parameters
        ----------
        value : float

        Returns
        -------
        None
        """
        self.instrument.gain = value

    @property
    def telescope_name(self):
        """
        Return the name of the telescope.

        Returns
        -------
        name : str
        """
        return self.telescope.get_telescope_name()

    @property
    def jansky_per_beam(self):
        """
        Return the Jansky's per beam.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.instrument.jansky_per_beam()

    @property
    def data_unit(self):
        """
        Return the data unit of the channel data.

        Returns
        -------
        astropy.units.Unit
        """
        return self.instrument.get_data_unit()

    @property
    def kelvin(self):
        """
        Return the instrument temperature in Kelvin.

        Returns
        -------
        astropy.units.Quantity
        """
        return self.instrument.kelvin()

    @property
    def point_size(self):
        """
        Return the point size of the instrument.

        Returns
        -------
        units.Quantity
        """
        return self.instrument.get_point_size()

    @property
    def source_size(self):
        """
        Return the source size of the observation.

        Returns
        -------
        units.Quantity
        """
        return self.instrument.get_source_size()

    def get_channel_class(self):
        """
        Returns a Channels instance for a given instrument.

        Returns
        -------
        Channels
        """
        return channel_class_for(self.instrument.name)

    def get_channel_data_class(self):
        """
        Return the appropriate ChannelData class for a given instrument.

        Returns
        -------
        channel_data : class (ChannelData)
        """
        return channel_data_class_for(self.instrument.name)

    def get_channel_group_class(self):
        """
        Returns the appropriate ChannelGroup class for a given instrument.

        Returns
        -------
        class (ChannelGroup)
        """
        return channel_group_class_for(self.instrument.name)

    def get_scan_class(self):
        """
        Returns the appropriate Scan class for a given instrument.

        Returns
        -------
        class (ChannelGroup)
        """
        return get_scan_class(self.instrument.name)

    def get_channels_instance(self):
        """
        Return a Channels instance for this information.

        Returns
        -------
        Channels
        """
        channel_class = self.get_channel_class()
        return channel_class(info=self)

    def get_source_model_instance(self, scans, reduction=None):
        """
        Return the source model applicable to the channel type.

        Parameters
        ----------
        scans : list (Scan)
            A list of scans for which to create the source model.
        reduction : Reduction, optional
            The reduction to which the model will belong.

        Returns
        -------
        Map
        """
        source_type = self.configuration.get_string('source.type')
        if source_type is None:
            return None

        if source_type == 'skydip':
            return SkyDip(info=self, reduction=reduction)
        elif source_type == 'map':
            return AstroIntensityMap(info=self, reduction=reduction)
        elif source_type == 'null':
            return None
        else:
            return None

    def validate_configuration_registration(self):
        """
        Ensure that all configuration files are registered.

        Returns
        -------
        None
        """
        config_files = self.configuration.config_files
        if config_files is not None:
            for config_file in config_files:
                self.register_config_file(config_file)

    def register_config_file(self, filename):
        """
        Register that a configuration file has been read.

        Parameters
        ----------
        filename : str

        Returns
        -------
        None
        """
        pass

    def set_date_options(self, mjd):
        """
        Set the configuration options for a given date (in MJD).

        Parameters
        ----------
        mjd : float

        Returns
        -------
        None
        """
        self.configuration.set_date(float(mjd), validate=True)

    def set_mjd_options(self, mjd):
        """
        Set the configuration options for a given date (in MJD).

        Parameters
        ----------
        mjd : float

        Returns
        -------
        None
        """
        self.set_date_options(mjd)

    def set_serial_options(self, serial):
        """
        Set the configuration options for a given serial number.

        Parameters
        ----------
        serial : int or str

        Returns
        -------
        None
        """
        self.configuration.set_serial(serial, validate=True)

    def set_object_options(self, source_name):
        """
        Set the configuration object options for a given source.

        Parameters
        ----------
        source_name : str

        Returns
        -------
        None
        """
        self.configuration.set_object(source_name, validate=True)

    def parse_header(self, header):
        """
        Parse and apply a FITS header to the instrument information.

        Parameters
        ----------
        header : fits.Header

        Returns
        -------
        None
        """
        self.set_fits_header_options(header)
        self.apply_configuration()

    def set_fits_header_options(self, header, extension=0):
        """
        Set the configuration FITS options from a FITS header.

        Parameters
        ----------
        header : fits.Header
        extension : int, optional
            The HDUL extension of the header.  This is only stored for
            reference and is not used.

        Returns
        -------
        None
        """
        self.configuration.read_fits(header, extension=extension,
                                     validate=True)

    def read_configuration(self, configuration_file='default.cfg',
                           validate=True):
        """
        Read and apply a configuration file.

        Parameters
        ----------
        configuration_file : str, optional
            Path to, or name of, a configuration file.
        validate : bool, optional
            If `True` (default), validate information read from the
            configuration file.

        Returns
        -------
        None
        """
        if self.configuration.instrument_name is None:
            self.configuration.set_instrument(self.name)
        self.configuration.read_configuration(
            configuration_file, validate=validate)
        self.register_config_file(configuration_file)
        self.validate_configuration_registration()

    def apply_configuration(self):
        """
        Apply a configuration to the information.

        This should be once the FITS information from a scan file has been
        applied to the information via `parse_header`.

        Returns
        -------
        None
        """
        if self.configuration.instrument_name is None:
            self.configuration.set_instrument(self.name)
        if not isinstance(self.configuration, Configuration):
            raise ValueError(f"scan must be a {Configuration} instance.")
        for info in self.available_info.values():
            info.set_configuration(self.configuration)

    def validate(self):
        """
        Validate all information following a read of scans/integrations.

        At this point the astrometry can be verified.

        Returns
        -------
        None
        """
        for info in self.available_info.values():
            info.validate()

    def validate_scan(self, scan):
        """
        Validate a

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        for info in self.available_info.values():
            info.validate_scan(scan)

    def parse_image_header(self, header):
        """
        Parse an image header and apply a new header.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.

        Returns
        -------
        None
        """
        for info in self.available_info.values():
            info.parse_image_header(header)

    def edit_image_header(self, header, scans=None):
        """
        Add or edit image information in a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.
        scans : list (Scan), optional
            A list of scans to use during editing.

        Returns
        -------
        None
        """
        for info in self.available_info.values():
            info.edit_image_header(header, scans=scans)

    def edit_scan_header(self, header, scans=None):
        """
        Add or edit scan information in a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The FITS header to edit.
        scans : list (Scan), optional
            A list of scans to use during editing.

        Returns
        -------
        None
        """
        for info in self.available_info.values():
            info.edit_scan_header(header, scans=scans)

    def add_history(self, header, scans=None):
        """
        Add HISTORY messages to a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            The header to update with HISTORY messages.
        scans : list (Scan), optional
            A list of scans to add HISTORY messages from if necessary.

        Returns
        -------
        None
        """
        pass

    def validate_scans(self, scans):
        """
        Validate a list of scans specific to the instrument

        Parameters
        ----------
        scans : list (Scan)
            A list of scans.

        Returns
        -------
        None
        """
        if scans is None:
            return

        for i, scan in enumerate(scans):
            if scan is None:
                continue

            if not scan.is_valid():
                continue

            if self.configuration.get_bool('jackknife.alternate'):
                log.info("JACKKNIFE: Alternating scans.")
                if scan.size == 0:
                    return
                if i % 2 == 0:
                    for integration in scan.integrations:
                        integration.gain *= -1.0

    @classmethod
    def get_focus_string(cls, focus):
        """
        Return a string representing the focus.

        Parameters
        ----------
        focus : InstantFocus

        Returns
        -------
        str
        """
        if focus is None:
            return ' No instant focus.'
        msg = ''
        if focus.x is not None:
            msg += f"\n  Focus.dX --> {focus.x.to('mm')}"
        if focus.y is not None:
            msg += f"\n  Focus.dY --> {focus.y.to('mm')}"
        if focus.z is not None:
            msg += f"\n  Focus.dZ --> {focus.z.to('mm')}"
        return msg

    def get_name(self):
        """
        Return the name of the information.

        Returns
        -------
        name : str
        """
        if self.name is None:
            return ''
        return self.name

    def set_name(self, name):
        """
        Set the name for the information.

        Parameters
        ----------
        name : str

        Returns
        -------
        None
        """
        self.name = str(name)

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

    def perform_reduction(self, reduction, filenames):
        """
        Fully reduce a given reduction and set of files.

        While it is possible for the reduction object to fully reduce a set of
        files, certain special considerations may be required for certain
        instruments.  Therefore, the instrument specific Info object is given
        control of how a reduction should progress.

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
        reduction.read_scans(filenames)
        reduction.validate()
        reduction.reduce()
