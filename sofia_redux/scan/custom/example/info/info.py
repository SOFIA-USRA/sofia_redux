# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.example.info.simulation import SimulationInfo

__all__ = ['ExampleInfo']


class ExampleInfo(SimulationInfo):

    def __init__(self, configuration_path=None):
        """
        Initialize an ExampleInfo object.

        Parameters
        ----------
        configuration_path : str, optional
            An alternate directory path to the configuration tree to be
            used during the reduction.  The default is
            <package>/data/configurations.
        """
        super().__init__(configuration_path=configuration_path)
        self.name = 'example'

    def get_name(self):
        """
        Return the name of the information.

        Returns
        -------
        name : str
        """
        if self.instrument is None or self.instrument.name is None:
            return super().get_name()
        return self.instrument.name

    def validate_scans(self, scans):
        """
        Validate a list of scans specific to the instrument

        Parameters
        ----------
        scans : list (SofiaScan)
            A list of scans.

        Returns
        -------
        None
        """
        if scans is None or len(scans) == 0 or scans[0] is None:
            super().validate_scans(scans)
            return

    @staticmethod
    def get_file_id():
        """
        Return the file ID.

        Returns
        -------
        str
        """
        return "EXPL"
