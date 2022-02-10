# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log

from sofia_redux.scan.integration.integration import Integration

__all__ = ['ExampleIntegration']


class ExampleIntegration(Integration):

    def __init__(self, scan=None):
        """
        Initialize an example integration.

        Parameters
        ----------
        scan : ExampleScan
        """
        super().__init__(scan=scan)

    def read(self, hdul):
        """
        Read integration information from an HDU list.

        Parameters
        ----------
        hdul : fits.HDUList
            A list of data HDUs containing "timestream" data.

        Returns
        -------
        None
        """
        log.info("Processing scan data:")
        try:
            n_records = hdul[1].data.size
        except IndexError:
            log.warning('No data present.')
            return

        self.frames.initialize(self, n_records)

        self.frames.read_hdu(hdul[1])
