# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.info.observation import ObservationInfo

__all__ = ['ExampleObservationInfo']


class ExampleObservationInfo(ObservationInfo):

    def __init__(self):
        """
        Initialize the observation information for the example instrument.
        """
        super().__init__()
        self.scan_id = None
        self.obs_id = None

    def apply_configuration(self):
        """
        Read and apply the FITS options from the configuration.

        Sets the source name from 'OBJECT' in the FITS header, and the scan ID
        from 'SCANID' in the FITS header.

        Returns
        -------
        None
        """
        options = self.options
        if options is None:
            return
        self.set_source(options.get_string("OBJECT", default='UNKNOWN'))
        self.scan_id = options.get_string('SCANID', default='UNKNOWN')
        self.obs_id = f'{self.source_name}.{self.scan_id}'

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
        if name == 'obsid':
            return self.obs_id
        elif name == 'scanid':
            return self.scan_id
        return super().get_table_entry(name)
