# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.info.camera.instrument import CameraInstrumentInfo

__all__ = ['ExampleInstrumentInfo']


class ExampleInstrumentInfo(CameraInstrumentInfo):

    def __init__(self):
        """
        Initialize the instrument information for the example instrument.

        The example instrument is set to a Cassegrain mount with a resolution
        of 10 arc seconds and a sampling interval of 100 ms.
        """
        super().__init__()
        self.name = 'example'
        self.set_mount("CASSEGRAIN")
        self.resolution = 10.0 * units.Unit('arcsec')
        self.sampling_interval = 0.1 * units.Unit('s')
        self.integration_time = self.sampling_interval
