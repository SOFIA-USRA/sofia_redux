# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.filters.fixed_filter import FixedFilter
from sofia_redux.scan.utilities.range import Range
from sofia_redux.scan.utilities import utils

__all__ = ['KillFilter']


class KillFilter(FixedFilter):

    def __init__(self, integration=None, data=None):
        """
        Initializes a kill filter.

        The kill filter fully rejects or allows a response at a given
        frequency.

        Parameters
        ----------
        integration : Integration, optional
        data : numpy.ndarray (float), optional
            An array of shape (nt, n_channels) where nt is the nearest power of
            2 integer above the number of integration frames. i.e., if
            n_frames=5, nt=8, or if n_frames=13, nt=16.  If not provided will
            be set to frame_data * frame_relative_weight.
        """
        self.reject = None
        super().__init__(integration=integration, data=data)

    def get_reject_mask(self):
        """
        Return the rejection mask for the kill filter.

        Returns
        -------
        mask : numpy.ndarray (bool)
        """
        return self.reject

    def set_integration(self, integration):
        """
        Set the filter integration.

        Parameters
        ----------
        integration : Integration

        Returns
        -------
        None
        """
        super().set_integration(integration)
        self.reject = np.full(self.nf + 1, False)

    def kill(self, frequency_range):
        """
        Sets the kill response of

        Parameters
        ----------
        frequency_range : Range
            The range of frequencies to kill.

        Returns
        -------
        None
        """
        min_f = frequency_range.min
        max_f = frequency_range.max
        if isinstance(min_f, units.Quantity):
            min_f = min_f.decompose().value
            max_f = max_f.decompose().value

        from_f = np.clip(int(min_f / self.df), 0, None)
        to_f = np.clip(int(np.ceil(max_f / self.df)), None, self.nf)
        if from_f > to_f:
            return
        self.reject[from_f:to_f + 1] = True
        self.auto_dft()

    def update_config(self):
        """
        Apply settings from the integration configuration.

        Sets the rejection response for the kill filter between two frequencies
        specified in the configuration key "filter.kill.bands".  The
        configuration value should be of the form "f1:f2" where f1 and f2 are
        the start and end frequencies (in Hz).   Multiple frequency rejection
        bands may be specified with a comma in the configuration::

            filter.kill.bands = <f1>:<f2>,<f3>:<f4>

        The above lines will reject frequencies between f1 and f2, and also
        between f3 and f4.

        Returns
        -------
        None
        """
        super().update_config()
        if not self.has_option('bands'):
            return

        ranges = utils.get_string_list(self.option('bands'))
        for band_range in ranges:
            frequency_range = Range.from_spec(band_range, is_positive=True)
            self.kill(frequency_range)

    def auto_dft(self):
        """
        Determine whether to use a full or discrete FFT.

        The DFT assumes 51 ops per datum, per rejected frequency.
        The FFT assumes 2 * FFT (back and forth) with 31 ops in each loop,
        9.5 ops per datum, 34.5 ops per datum rearrange.  The type of transform
        is determined by the fewest operations.

        This is surprisingly accurate.

        Returns
        -------
        None
        """
        dft_freq = int(51 * self.count_parms() * self.size)
        fft_freq = int(np.round(np.log10(self.nt) / np.log10(2))) * self.nt
        fft_freq = 2 * ((31 * fft_freq) + (44 * self.nt))
        self.set_dft(dft_freq < fft_freq)

    def response_at(self, fch):
        """
        Return the response at a given frequency channel(s).

        Parameters
        ----------
        fch : int or numpy.ndarray (int)
            The frequency channel or channels in question.

        Returns
        -------
        response : float or numpy.ndarray (float)
        """
        if not isinstance(fch, np.ndarray) or fch.shape == ():
            if fch == self.reject.size:
                return 0.0 if self.reject[1] else 1.0
            else:
                return self.reject[fch]

        fch_roll = fch.copy()
        fch_roll[fch == self.reject.size] = 1
        return np.logical_not(self.reject[fch_roll]).astype(float)

    def count_parms(self):
        """
        Return the total dependent count of the filter.

        Returns
        -------
        int
        """
        if self.reject is None:
            return 0
        return np.sum(self.reject[self.get_high_pass_index():])

    def get_id(self):
        """
        Return the filter ID.

        Returns
        -------
        filter_id : str
        """
        return 'K'

    def get_config_name(self):
        """
        Return the configuration name.

        Returns
        -------
        config_name : str
        """
        return 'filter.kill'
