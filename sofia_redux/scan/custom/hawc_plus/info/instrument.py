# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units, log
import numpy as np

from sofia_redux.scan.custom.sofia.info.instrument import SofiaInstrumentInfo
from sofia_redux.scan.utilities.utils import to_header_float

__all__ = ['HawcPlusInstrumentInfo']


class HawcPlusInstrumentInfo(SofiaInstrumentInfo):

    def __init__(self):
        super().__init__()
        self.name = 'hawc_plus'
        self.band_id = None
        self.hwp_step = 0.25 * units.Unit('deg')
        self.hwp_telescope_vertical = 0.0 * units.Unit('deg')

    def apply_configuration(self):
        super().apply_configuration()
        options = self.options
        if options is None:
            return

        sampling_freq = options.get_float("SMPLFREQ", default=np.nan
                                          ) * units.Unit('Hz')
        if np.isnan(sampling_freq) or sampling_freq < 0:
            log.warning("Missing SMPLFREQ. Will assume 203.5 Hz.")
            sampling_freq = 203.25 * units.Unit('Hz')

        self.integration_time = (1 / sampling_freq).decompose()  # to seconds
        self.sampling_interval = self.integration_time.copy()

        spectel1 = str(self.spectral_element_1).strip().upper()
        if spectel1.startswith("HAW_") and len(spectel1) > 4:
            self.band_id = spectel1[4]
        else:
            self.band_id = '-'

        if 'filter' not in self.configuration:
            filter_value = f'{self.wavelength.value}sum'
            self.configuration.put('filter', filter_value)
            log.info(f"HAWC+ Filter set to {self.configuration['filter']}.")

    def edit_header(self, header):
        """
        Edit an image header with available information.

        Parameters
        ----------
        header : astropy.fits.Header
            The FITS header to apply.

        Returns
        -------
        None
        """
        super().edit_header(header)
        header['SMPLFREQ'] = (
            to_header_float(1.0 / self.sampling_interval, 'Hz'),
            '(Hz) Detector readout rate.')
