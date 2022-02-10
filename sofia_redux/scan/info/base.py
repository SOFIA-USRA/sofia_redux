# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from astropy import units
from copy import deepcopy
import numpy as np

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.utilities.bracketed_values import BracketedValues

__all__ = ['InfoBase']


class InfoBase(ABC):

    UNKNOWN_STRING_VALUE = 'UNKNOWN'
    UNKNOWN_INT_VALUE = -9999
    UNKNOWN_FLOAT_VALUE = -9999.0

    def __init__(self):
        self.configuration = None
        self.scan_applied = False
        self.scan = None
        self.configuration_applied = False

    @property
    def referenced_attributes(self):
        """
        Return a set of attribute names that should be referenced during copy.

        Returns
        -------
        set (str)
        """
        return {'configuration', 'scan'}

    @property
    def log_id(self):
        """
        Return the string log ID for the info.

        Returns
        -------
        str
        """
        return 'base'

    @property
    def log_prefix(self):
        """
        Return the log prefix of the information object.

        This is used to specify table entries for extraction.

        Returns
        -------
        str
        """
        return f'{self.log_id.lower().strip()}.'

    def copy(self):
        """
        Return a full copy of the information.

        Returns
        -------
        InfoBase
        """
        referenced_only = self.referenced_attributes
        new = self.__class__()
        for attribute, value in self.__dict__.items():
            if attribute in referenced_only:
                setattr(new, attribute, value)
            elif hasattr(value, 'copy'):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, deepcopy(value))
        return new

    @property
    def options(self):
        """
        Return the FITS header options.

        Returns
        -------
        FitsOptions
        """
        if self.configuration is None or not self.configuration.fits.enabled:
            return None
        return self.configuration.fits

    def set_configuration(self, configuration):
        """
        Set the configuration for the information.

        Parameters
        ----------
        configuration : Configuration

        Returns
        -------
        None
        """
        if not isinstance(configuration, Configuration):
            raise ValueError(f"configuration must be a "
                             f"{Configuration} instance.")
        self.configuration = configuration
        self.apply_configuration()
        if self.scan_applied:
            self.validate()

    def set_scan(self, scan):
        """
        Set the scan applicable to the information.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        self.apply_scan(scan)
        if self.configuration_applied:
            self.validate()

    def __str__(self):
        """
        Return a string representation of the info base.

        Returns
        -------
        str
        """
        s = ''
        for key, value in self.__dict__.items():
            s += f'{key}: {value}\n'
        return s

    def apply_configuration(self):
        """
        Apply the configuration to the information.

        Returns
        -------
        None
        """
        self.configuration_applied = True

    def apply_scan(self, scan):
        """
        Apply scan information to the information.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        self.scan_applied = True
        self.scan = scan

    def validate(self):
        """
        Validate the data obtained from FITS header and configuration

        Returns
        -------
        None
        """
        pass

    def validate_scan(self, scan):
        """
        Validate scan information with *THIS* information.

        Parameters
        ----------
        scan : Scan

        Returns
        -------
        None
        """
        pass

    def parse_image_header(self, header):
        """
        Apply settings from a FITS image header.

        Parameters
        ----------
        header : astropy.fits.Header

        Returns
        -------
        None
        """
        pass

    def edit_image_header(self, header, scans=None):
        """
        Edit an image header with available information.

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
        pass

    def edit_scan_header(self, header, scans=None):
        """
        Edit a scan header with available information.

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
        pass

    def merge(self, last):
        """
        Set end of any range values to the last ranged end value.

        Parameters
        ----------
        last : InfoBase

        Returns
        -------
        None
        """
        for attribute, value in self.__dict__.items():
            if isinstance(value, BracketedValues):
                last_value = getattr(last, attribute, None)
                if isinstance(last_value, BracketedValues):
                    value.end = last_value.end

    @classmethod
    def valid_header_value(cls, value):
        """
        Check whether a header value is valid.

        Parameters
        ----------
        value : None or bool or int or str or float or Quantity

        Returns
        -------
        valid : bool
        """
        if isinstance(value, bool):
            return True
        elif isinstance(value, int):
            return value != cls.UNKNOWN_INT_VALUE
        elif isinstance(value, str):
            return value != cls.UNKNOWN_STRING_VALUE
        elif isinstance(value, float):
            return (value != cls.UNKNOWN_FLOAT_VALUE) and not np.isnan(value)
        elif isinstance(value, units.Quantity):
            x = value.value
            return (x != cls.UNKNOWN_FLOAT_VALUE) and not np.isnan(x)
        else:
            return False

    def get_table_entry(self, name):
        """
        Given a name, return the parameter stored in the information object.

        Note that names do not exactly match to attribute names.

        Parameters
        ----------
        name : str

        Returns
        -------
        value
        """
        for key, value in self.__dict__.items():
            if key == name:
                return value
        else:
            return None
