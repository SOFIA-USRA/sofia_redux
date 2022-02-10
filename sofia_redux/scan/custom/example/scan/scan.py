# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits

from sofia_redux.scan.scan.scan import Scan

__all__ = ['ExampleScan']


class ExampleScan(Scan):

    def __init__(self, channels, reduction=None):
        self.hdul = None
        super().__init__(channels, reduction=reduction)

    @property
    def referenced_attributes(self):
        """
        Return the names of attributes that are referenced during a copy.

        Returns
        -------
        attribute_names : set (str)
        """
        attributes = super().referenced_attributes
        attributes.add('hdul')
        return attributes

    def get_integration_instance(self):
        """
        Return an integration instance of the correct type for the scan.

        Returns
        -------
        integration : ExampleIntegration
        """
        return super().get_integration_instance()

    def read(self, filename, read_fully=True):
        """
        Read a filename to populate the scan.

        The read should validate the channels before instantiating integrations
        for reading.

        Parameters
        ----------
        filename : str
            The name of the file to read.
        read_fully : bool, optional
            If `True`, perform a full read (default)

        Returns
        -------
        None
        """
        self.hdul = fits.open(filename)
        self.read_hdul(self.hdul, read_fully=read_fully)
        self.close_fits()

    def close_fits(self):
        """
        Close the scan FITS file.

        Returns
        -------
        None
        """
        if self.hdul is None:
            return
        self.hdul.close()
        self.hdul = None

    def read_hdul(self, hdul, read_fully=True):
        """
        Read an open FITS HDUL.

        Parameters
        ----------
        hdul : fits.HDUList
            The FITS HDU list to read.
        read_fully : bool, optional
            If `True` (default), fully read the file.

        Returns
        -------
        None
        """
        self.info.parse_header(hdul[0].header.copy())
        self.channels.read_data(hdul)
        self.channels.validate_scan(self)
        self.read_integration(self.hdul)

    def read_integration(self, hdul):
        """
        Add integrations to this scan from a HDU list.

        Parameters
        ----------
        hdul : fits.HDUList

        Returns
        -------
        None
        """
        self.integrations = []
        integration = self.get_integration_instance()
        integration.read(hdul)
        self.integrations.append(integration)

    def get_id(self):
        """
        Return the scan ID.

        Returns
        -------
        str
        """
        return self.info.observation.obs_id
