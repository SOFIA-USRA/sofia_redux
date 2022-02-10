# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units

from sofia_redux.scan.info.camera.instrument import CameraInstrumentInfo

__all__ = ['ExampleInstrumentInfo']


class ExampleInstrumentInfo(CameraInstrumentInfo):

    def __init__(self):
        super().__init__()
        self.name = 'example'
        self.set_mount("CASSEGRAIN")
        self.resolution = 10.0 * units.Unit('arcsec')
        self.sampling_interval = 0.1 * units.Unit('s')
        self.integration_time = self.sampling_interval
