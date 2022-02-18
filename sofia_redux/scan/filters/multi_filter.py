# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import inspect
import numpy as np
import scipy

from sofia_redux.scan.filters.varied_filter import VariedFilter
from sofia_redux.scan.filters.filter import Filter

__all__ = ['MultiFilter']


class MultiFilter(VariedFilter):

    def __init__(self, integration=None, data=None):
        """
        Initialize an integration multi-filter.

        The multi-filter contains multiple sub-filters that operate on an
        integration in sequence.

        Parameters
        ----------
        integration : Integration, optional
        data : numpy.ndarray (float), optional
            An array of shape (nt, n_channels) where nt is the nearest power of
            2 integer above the number of integration frames. i.e., if
            n_frames=5, nt=8, or if n_frames=13, nt=16.  If not provided will
            be set to frame_data * frame_relative_weight.
        """
        self.filters = []
        self.n_enabled = 0
        super().__init__(integration=integration, data=data)

    def copy(self):
        """
        Return a copy of the filter.

        All attributes are copied aside from the integration and channels which
        are referenced only.

        Returns
        -------
        Filter
        """
        new = super().copy()
        new.filters = []
        if self.filters is None:
            new.filters = None
            return new
        for sub_filter in self.filters:
            new.filters.append(sub_filter.copy())
        return new

    @property
    def size(self):
        """
        Return the number of sub-filters in the multi-filter.

        Returns
        -------
        n_filters : int
        """
        if self.filters is None:
            return 0
        return len(self.filters)

    def __contains__(self, filter_or_name):
        """
        Return whether the multi-filter contains a filter.

        Parameters
        ----------
        filter_or_name : Filter or str
            A filter object or class or name of the filter to check.

        Returns
        -------
        bool
        """
        return self[filter_or_name] is not None

    def __getitem__(self, filter_or_name):
        """
        Return a given filter from the multi-filter

        Parameters
        ----------
        filter_or_name : Filter or str or int
            A filter object, class, name, label, or filter index of the
            filter to retrieve.

        Returns
        -------
        Filter or None
        """
        if self.size == 0:
            return None
        if isinstance(filter_or_name, Filter):
            if filter_or_name in self.filters:
                return filter_or_name
        elif inspect.isclass(filter_or_name):
            for filter_object in self.filters:
                if filter_object.__class__ == filter_or_name:
                    return filter_object
        elif isinstance(filter_or_name, str):
            check_name = filter_or_name.strip().lower()
            for filter_object in self.filters:
                name = filter_object.get_config_name().split('.')[-1].lower()
                if name == check_name:
                    return filter_object
                label = filter_object.get_id().lower()
                if label == check_name:
                    return filter_object
        elif isinstance(filter_or_name, int):
            if filter_or_name < self.size:
                return self.filters[filter_or_name]
        return None

    def reindex(self):
        """
        Reindex the channel groups to be consistent with parent channels.

        In addition to the main multi-filter, all sub-filters are re-indexed
        too.

        Returns
        -------
        None
        """
        super().reindex()
        if self.filters is not None:
            for sub_filter in self.filters:
                sub_filter.reindex()

    def get_filters(self):
        """
        Return all sub-filters of the multi-filter.

        Returns
        -------
        filters : list (Filter)
        """
        return self.filters

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
        if self.data is None:
            self.make_temp_data()
        if self.filters is not None:
            for sub_filter in self.filters:
                sub_filter.set_temp_data(self.data)
                sub_filter.set_integration(integration)
        self.update_source_profile()

    def set_channels(self, channels):
        """
        Set the filter channels.

        The channels attribute will be set to a ChannelGroup type.

        Parameters
        ----------
        channels : Channels or ChannelData or ChannelGroup

        Returns
        -------
        None
        """
        super().set_channels(channels)
        if self.filters is not None:
            for sub_filter in self.filters:
                sub_filter.set_channels(channels)

    def add_filter(self, sub_filter):
        """
        Add a sub-filter to the list of filters in the multi-filter.

        The sub-filter integration is set to the multi-filter integration.  If
        an integration already exists for the sub-filter, it must be the same
        as the multi-filter integration.  The sub-filter channels are also set
        to the multi-filter channels.

        Parameters
        ----------
        sub_filter : Filter
            The filter to add.

        Returns
        -------
        None
        """
        if sub_filter.integration is None:
            sub_filter.set_integration(self.integration)
        elif sub_filter.integration is not self.integration:
            raise ValueError("Cannot compound filter from a different "
                             "integration.")
        sub_filter.set_channels(self.get_channels())
        if self.filters is None:
            self.filters = [sub_filter]
        else:
            self.filters.append(sub_filter)

    def set_filter(self, filter_index, sub_filter):
        """
        Insert a sub-filter into the list of filters in the multi-filter.

        The sub-filter integration is set to the multi-filter integration.  If
        an integration already exists for the sub-filter, it must be the same
        as the multi-filter integration.  The sub-filter channels are also set
        to the multi-filter channels.

        If the filter index is greater than the number of filters, pads the
        filters list with `None` until the length of the filters is able to
        support the requested index.

        Parameters
        ----------
        filter_index : int
            The index at which to insert the sub-filter.
        sub_filter : Filter
            The filter to insert.

        Returns
        -------
        None
        """
        if sub_filter.integration is None:
            sub_filter.set_integration(self.integration)
        elif sub_filter.integration is not self.integration:
            raise ValueError("Cannot compound filter from a different "
                             "integration.")
        sub_filter.set_channels(self.channels)
        if self.filters is None:
            self.filters = []
        add_blanks = filter_index - self.size
        if add_blanks > 0:
            self.filters.extend([None] * (add_blanks + 1))
        self.filters[filter_index] = sub_filter

    def remove_filter(self, filter_or_name):
        """
        Remove a sub-filter from the list of multi-filter sub-filters.

        Parameters
        ----------
        filter_or_name : Filter or str or int
            A filter object or class or name or filter index of the filter to
            retrieve.

        Returns
        -------
        None
        """
        if isinstance(filter_or_name, int):
            if filter_or_name < self.size:
                del self.filters[filter_or_name]
                return

        filter_object = self[filter_or_name]
        if filter_object is not None:
            for filter_index, sub_filter in enumerate(self.filters):
                if sub_filter is filter_object:
                    del self.filters[filter_index]
                    return

    def update_config(self):
        """
        Determine whether the filter is configuration and if it's pedantic.

        Will also perform the same operation for all sub-filters.

        Returns
        -------
        None
        """
        super().update_config()
        for sub_filter in self.filters:
            sub_filter.is_sub_filter = True
            sub_filter.update_config()

    def is_enabled(self):
        """
        Return whether the filter is enabled.

        In addition to the multi-filter being enabled, a `True` result requires
        that at least one sub-filter is also enabled.

        Returns
        -------
        bool
        """
        if not super().is_enabled():
            return False
        self.n_enabled = 0
        for f in self.filters:
            if f.is_enabled():
                self.n_enabled += 1
        return self.n_enabled > 0

    def pre_filter(self):
        """
        Perform the pre-filtering steps.

        The pre-filtering is also performed for all enabled sub-filters.

        Returns
        -------
        None
        """
        super().pre_filter()
        for sub_filter in self.filters:
            if sub_filter.is_enabled():
                sub_filter.pre_filter()

    def post_filter(self):
        """
        Perform the post-filtering steps.

        The post-filtering is also performed for all enabled sub-filters.

        Returns
        -------
        None
        """
        for sub_filter in self.filters:
            if sub_filter.is_enabled():
                sub_filter.post_filter()
        super().post_filter()

    def fft_filter(self, channels=None):
        """
        Apply the FFT filter to the temporary data.

        Converts data into a rejected (un-levelled) signal

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channels for which to apply the filter.  If not supplied,
            defaults to the stored filtering channels.

        Returns
        -------
        None
        """
        if channels is None:
            channels = self.get_channels()

        data = self.get_temp_data().copy()

        data = scipy.fft.rfft(data, axis=1)

        # Remove the mean
        data[:, 0].real = 0.0
        n_freq = data.shape[1]
        f_channels = np.arange(n_freq)

        filtered = np.zeros_like(data)

        # Apply the filters sequentially
        for sub_filter in self.filters:
            if not sub_filter.is_enabled():
                continue
            log.debug(f"FFT filtering {sub_filter.get_config_name()}.")

            # Make sure that the filter uses the spectrum from the master array
            if sub_filter.data is not data:
                sub_filter.set_temp_data(data)

            sub_filter.points = self.points
            sub_filter.pre_filter_channels(channels=channels)
            sub_filter.update_profile(channels=channels)

            response = sub_filter.response_at(f_channels)
            rejection = 1.0 - response
            filtered = data * rejection
            data *= response

            sub_filter.post_filter_channels(channels=channels)

        # Convert to rejected signal
        filtered = scipy.fft.irfft(filtered, axis=1)
        self.set_temp_data(filtered)

    def response_at(self, fch):
        """
        Return the response at a given frequency channel(s).

        Parameters
        ----------
        fch : int or numpy.ndarray (int or bool) or slice
            The frequency channel or channels in question.

        Returns
        -------
        response : float or numpy.ndarray (float)
        """
        if not isinstance(fch, np.ndarray) or fch.shape == ():
            full_response = 1.0
            singular = True
        else:
            full_response = np.ones(fch.size, dtype=float)
            singular = False

        if self.filters is None:
            return full_response
        for sub_filter in self.filters:
            if not sub_filter.is_enabled():
                continue

            sub_response = sub_filter.response_at(fch)
            sub_singular = (not(isinstance(sub_response, np.ndarray))
                            or sub_response.shape == ())
            if sub_singular or singular or (
                    sub_response.ndim == full_response.ndim):
                full_response = full_response * sub_response
            elif full_response.ndim == 2:  # pragma: no cover
                # In case other types of filters are added.
                full_response = full_response * sub_response[None]
            else:
                full_response = full_response[None] * sub_response

        return full_response

    def get_id(self):
        """
        Return the filter ID.

        Returns
        -------
        filter_id : str
        """
        filter_id = ''
        if self.filters is None:
            return filter_id
        for sub_filter in self.filters:
            if sub_filter.is_enabled():
                if len(filter_id) > 0:
                    filter_id += ':'
                filter_id += sub_filter.get_id()
        return filter_id

    def get_config_name(self):
        """
        Return the configuration name.

        Returns
        -------
        config_name : str
        """
        return 'filter'

    def dft_filter(self, channels=None):
        """
        Return the filter rejection using a discrete FFT.

        UNSUPPORTED FOR THE MULTI-FILTER.

        Parameters
        ----------
        channels : ChannelGroup, optional
            The channel group for which the filtering applied.  By default,
            set to the filtering channels.

        Returns
        -------
        None
        """
        super().dft_filter(channels=channels)
