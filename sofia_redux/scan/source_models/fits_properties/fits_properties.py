# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABC
from copy import deepcopy

__all__ = ['FitsProperties']


class FitsProperties(ABC):

    default_creator = 'SOFSCAN'
    default_copyright = 'LOL'

    def __init__(self):
        self.creator = self.default_creator
        self.copyright = self.default_copyright
        self.filename = ''
        self.object_name = ''
        self.telescope_name = ''
        self.instrument_name = ''
        self.observer_name = ''
        self.observation_date_string = ''

    @property
    def referenced_attributes(self):
        """
        Return attribute names that should be referenced during a copy.

        Returns
        -------
        set (str)
        """
        return set([])

    def copy(self):
        """
        Return a copy of the map data.

        Returns
        -------
        FitsData
        """
        new = self.__class__()
        referenced = self.referenced_attributes
        for attribute, value in self.__dict__.items():
            if attribute in referenced:
                setattr(new, attribute, value)
            elif hasattr(value, 'copy'):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, deepcopy(value))

        return new

    def set_filename(self, filename):
        """
        Set the file name.

        Parameters
        ----------
        filename : str

        Returns
        -------
        None
        """
        self.filename = str(filename)

    def set_creator_name(self, creator_name):
        """
        Set the creator name.

        Parameters
        ----------
        creator_name : str

        Returns
        -------
        None
        """
        self.creator = str(creator_name)

    def set_copyright(self, copyright_string):
        """
        Set the copyright.

        Parameters
        ----------
        copyright_string : str

        Returns
        -------
        None
        """
        self.copyright = str(copyright_string)

    def set_object_name(self, object_name):
        """
        Set the object name.

        Parameters
        ----------
        object_name : str

        Returns
        -------
        None
        """
        self.object_name = str(object_name)

    def set_telescope_name(self, telescope_name):
        """
        Set the telescope name.

        Parameters
        ----------
        telescope_name : str

        Returns
        -------
        None
        """
        self.telescope_name = str(telescope_name)

    def set_instrument_name(self, instrument_name):
        """
        Set the instrument name.

        Parameters
        ----------
        instrument_name : str

        Returns
        -------
        None
        """
        self.instrument_name = str(instrument_name)

    def set_observer_name(self, observer_name):
        """
        Set the observer name.

        Parameters
        ----------
        observer_name : str

        Returns
        -------
        None
        """
        self.observer_name = str(observer_name)

    def set_observation_date_string(self, observation_date_string):
        """
        Set the observation date string.

        Parameters
        ----------
        observation_date_string : str

        Returns
        -------
        None
        """
        self.observation_date_string = str(observation_date_string)

    def parse_header(self, header):
        """
        Parse and apply a FITS header.

        Parameters
        ----------
        header : astropy.io.fits.Header

        Returns
        -------
        None
        """
        self.set_object_name(header.get('OBJECT'))
        self.set_telescope_name(header.get('TELESCOP'))
        self.set_instrument_name(header.get('INSTRUME'))
        self.set_observer_name(header.get('OBSERVER'))
        self.set_observation_date_string(header.get('DATE-OBS'))

    def edit_header(self, header):
        """
        Edit a FITS header with FITS property information.

        Parameters
        ----------
        header : astropy.io.fits.Header
            The header to update.

        Returns
        -------
        None
        """
        header['OBJECT'] = self.object_name, "Observed object's name."
        header['TELESCOP'] = self.telescope_name, 'Name of telescope.'
        header['INSTRUME'] = self.instrument_name, 'Name of instrument used.'
        header['OBSERVER'] = self.object_name, 'Name of observer(s).'
        header['DATE-OBS'] = (self.observation_date_string,
                              'Start of observation.')
        header['CREATOR'] = self.creator, self.copyright

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
        return None

    def info(self, header_string=None):
        """
        Return some relevant info as a string.

        Parameters
        ----------
        header_string : str, optional
            If supplied, the message will be pre-pended to this string.

        Returns
        -------
        str
        """
        buffer = ''
        if self.filename not in ['', None]:
            buffer += f' Image File: {self.filename}. -> \n\n'
        return buffer + self.brief(header_string=header_string)

    def brief(self, header_string=None):
        """
        Return a brief header summary.

        Parameters
        ----------
        header_string : str, optional
            If supplied, the message will be pre-pended to this string.

        Returns
        -------
        str
        """
        buffer = ''
        if self.object_name not in ['', None]:
            buffer += f'[{self.object_name}]\n'
        if isinstance(header_string, str):
            buffer += header_string
        return buffer

    def reset_processing(self):
        """
        Reset the processing status.

        Returns
        -------
        None
        """
        pass
