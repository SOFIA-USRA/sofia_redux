# Licensed under a 3-clause BSD style license - see LICENSE.rst

import configobj
import os
from astropy import log
from astropy.io import fits

from sofia_redux.scan.configuration.options import Options

__all__ = ['FitsOptions']


class FitsOptions(Options):

    def __init__(self, allow_error=False, verbose=True):
        """
        Initialize a FitsOptions object.

        The FitsOptions object contains configuration options specifically
        relating to a FITS header.
        """
        super().__init__(allow_error=allow_error, verbose=verbose)
        self.enabled = False
        self.header = None
        self.filename = None
        self.extension = None
        self.preserved_cards = {}

    def copy(self):
        """
        Return a copy of the FitsOptions.

        Returns
        -------
        FitsOptions
        """
        return super().copy()

    def clear(self):
        """
        Clear all options.

        Returns
        -------
        None
        """
        super().clear()
        self.header = None
        self.filename = None
        self.extension = None
        self.preserved_cards = {}

    def __getitem__(self, key):
        """
        Retrieve a value from the FITS options.

        Parameters
        ----------
        key : str
            The FITS key to retrieve.

        Raises
        ------
        KeyError
            In cases where the key does not exist in the options.

        Returns
        -------
        str or object
        """
        key = str(key).strip()
        if key not in self.options:
            raise KeyError(f"{key} is not a valid key.")
        return self.get(key)

    def __setitem__(self, key, value):
        """
        Set the value for a given key in the FITS options.

        Parameters
        ----------
        key : str
            The key to set.
        value : str or object
            The value to set for `key`.

        Returns
        -------
        None
        """
        self.set(key, value)

    def __contains__(self, key):
        """
        Return whether the FITS options contains the given key.

        Parameters
        ----------
        key : str
            The FITS key to check.

        Returns
        -------
        bool
        """
        return key in self.options

    def read(self, header, extension=0):
        """
        Read a FITS header and apply the contents to the options.

        Parameters
        ----------
        header : fits.Header or str or dict or configobj.ConfigObj
            The FITS header, options, or FITS file to read.
        extension : int, optional
            The HDU extension to read in cases where a file path is supplied in
            the `header` parameter.

        Returns
        -------
        options : configobj.ConfigObj
        """
        if not isinstance(header, (fits.header.Header, str, dict)):
            raise ValueError(f"Header must be {fits.Header}, {dict}, "
                             f"or {str} (filename).")

        if isinstance(header, str):
            if not os.path.isfile(header):
                raise ValueError(f"Not a valid file: {header}")

            try:
                header = fits.getheader(header, ext=extension)
            except Exception as err:
                log.error(f"Could not read header in extension {extension} "
                          f"of file: {header}")
                raise err
            self.filename = header
            self.extension = extension
        self.header = header  # reference (not copy)

        return configobj.ConfigObj(
            dict((k, str(v)) for k, v in header.items()))  # stringify

    def reread(self):
        """
        Re-read the current FITS header.

        Returns
        -------
        None
        """
        if self.header is None:
            return
        options = self.read(self.header)
        self.update_header(options)

    def get(self, key, default=None, unalias=True):
        """
        Return the option value for a given key.

        Parameters
        ----------
        key : str
            The FITS key value to retrieve.
        default : str or object, optional
            The value to return in cases where `key` does not exist in the
            options.
        unalias : bool, optional
            If `True`, unalias the key before attempting to retrieve
            it's value.

        Returns
        -------
        value : str or object
            The FITS header value for `key`.
        """
        return self.options.get(str(key).strip(), default=default)

    def set(self, key, value):
        """
        Set the value for a given FITS header key.

        Parameters
        ----------
        key : str
            The key for which to set a value.
        value : str or object, optional
            The value to set.

        Returns
        -------
        None
        """
        self.options[str(key).strip()] = value

    def update_header(self, header_options, extension=0):
        """
        Update the

        Parameters
        ----------
        header_options : fits.Header or str or dict or configobj.ConfigObj
            The FITS header, file path, or options with which to update the
            FitsOptions.
        extension : int, optional
            The HDU extension to read in cases where a file path was passed
            in as the argument.

        Returns
        -------
        None
        """
        self.options.merge(self.read(header_options, extension=extension))

    def keys(self):
        """
        Return a list of all keys in the FITS options.

        Returns
        -------
        list (str)
        """
        return self.options.keys()

    def set_preserved_card(self, key, header=None):
        """
        Add a key to the preserved header cards.

        Parameters
        ----------
        key : str
            The name of the FITS options key to preserve.
        header : fits.Header, optional
            The header from which to extract the key value and comments.
            If not supplied, defaults to the currently stored header.

        Returns
        -------
        None
        """
        if header is None:
            header = self.header
            if header is None:
                return

        if key not in header:
            return
        self.preserved_cards[key] = header[key], header.comments[key]

    def reset_preserved_cards(self):
        """
        Clear all preserved header cards.

        Returns
        -------
        None
        """
        self.preserved_cards = {}
