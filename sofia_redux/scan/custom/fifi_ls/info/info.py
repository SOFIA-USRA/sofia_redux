# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numba as nb
import numpy as np

from sofia_redux.scan.custom.fifi_ls.info.astrometry import (
    FifiLsAstrometryInfo)
from sofia_redux.scan.custom.fifi_ls.info.detector_array import (
    FifiLsDetectorArrayInfo)
from sofia_redux.scan.custom.fifi_ls.info.instrument import (
    FifiLsInstrumentInfo)
from sofia_redux.scan.custom.sofia.info.info import SofiaInfo
from sofia_redux.scan.custom.sofia.info.extended_scanning import (
    SofiaExtendedScanningInfo)
from sofia_redux.scan.utilities.utils import (
    insert_info_in_header, to_header_float)
from sofia_redux.scan.source_models.source_numba_functions import (
    get_source_signal)
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.signal.correlated_signal import CorrelatedSignal
from sofia_redux.scan.signal import signal_numba_functions as snf


__all__ = ['FifiLsInfo', 'normalize_scan_coordinates']


@nb.njit(cache=True, fastmath=False)
def normalize_scan_coordinates(ra, dec, x, y, z, data, error, valid, flags,
                               channel_valid):  # pragma: no cover
    """
    Convert (frame, pixel) indexed data to consistent flat arrays.

    Parameters
    ----------
    ra : numpy.ndarray (float)
        The right-ascension values of shape (n_frames, n_pixels).
    dec : numpy.ndarray (float)
        The declination values of shape (n_frames, n_pixels).
    x : numpy.ndarray (float)
        The detector x-coordinates of shape (n_frames, n_pixels).
    y : numpy.ndarray (float)
        The detector y-coordinates of shape (n_frames, n_pixels).
    z : numpy.ndarray (float)
        The pixel wavelength values of shape (n_pixels,).
    data : numpy.ndarray (float)
        The detector data values of shape (n_frames, n_pixels).
    error : numpy.ndarray (float)
        The detector error values of shape (n_frames, n_pixels).
    valid : numpy.ndarray (bool)
        The valid frames (`True`) of shape (n_frames,).
    flags : numpy.ndarray (int)
        The sample flags of shape (n_frames, n_pixels).  Only zero valued flags
        will be included in the output.
    channel_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_pixels,) where `False` indicates that a
        given channel is bad and should not be included.

    Returns
    -------
    flattened : 7-tuple (numpy.ndarray)
        Seven arrays each of shape (n,) where n is the number of zero-valued
        samples that are also valid frames.  The order of the arrays are:
        (RA, DEC, WAVE, X, Y, DATA, ERROR).
    """
    n_frames, n_pixels = x.shape
    n_max = data.size
    ra_out = np.empty(n_max, dtype=nb.float64)
    dec_out = np.empty(n_max, dtype=nb.float64)
    x_out = np.empty(n_max, dtype=nb.float64)
    y_out = np.empty(n_max, dtype=nb.float64)
    z_out = np.empty(n_max, dtype=nb.float64)
    d_out = np.full(n_max, np.nan)
    e_out = np.zeros(n_max)

    i = 0
    for pixel in range(n_pixels):
        if not channel_valid[pixel]:
            continue
        z_value = z[pixel]
        for frame in range(n_frames):
            if not valid[frame]:
                continue
            if flags[frame, pixel]:
                continue
            ra_out[i] = ra[frame, pixel]
            dec_out[i] = dec[frame, pixel]
            x_out[i] = x[frame, pixel]
            y_out[i] = y[frame, pixel]
            z_out[i] = z_value
            d_out[i] = data[frame, pixel]
            e_out[i] = error[frame, pixel]
            i += 1
    return (ra_out[:i], dec_out[:i], z_out[:i], x_out[:i], y_out[:i],
            d_out[:i], e_out[:i])


@nb.njit(cache=True, fastmath=False)
def correct_for_gain(data, frame_valid, frame_gains, sync_gains,
                     channel_flags, sample_flags, source_blank_flag
                     ):  # pragma: no cover
    """
    Use the calculated gains from the scan reduction to normalize data.

    Parameters
    ----------
    data : numpy.ndarray (float)
        The frame data to be corrected of shape (n_frames, n_channels).
        Updated in-place.
    frame_valid : numpy.ndarray (bool)
        A boolean mask of shape (n_frames,) where `False` indicates that a
        particular frame is invalid and should not be included.
    frame_gains : numpy.ndarray (float)
        The frame gains as calculated by the scan reduction of shape
        (n_frames,).
    sync_gains : numpy.ndarray (float)
        The channel gains as calculated by the scan reduction of shape
        (n_channels,).
    channel_flags : numpy.ndarray (int)
        An array of channel flags of shape (n_channels,) where any nonzero
        value indicates that there is some problem with the calculated gain
        value or channel properties, and that it should not be included in the
        corrected data.
    sample_flags : numpy.ndarray (int)

    source_blank_flag

    Returns
    -------

    """
    n_frames, n_channels = data.shape
    for frame in range(n_frames):
        if not frame_valid[frame]:
            continue
        fg = frame_gains[frame]
        for channel in range(n_channels):
            if channel_flags[channel] != 0:
                data[frame, channel] = np.nan
                continue
            flag = sample_flags[frame, channel]
            if flag != 0 and flag != source_blank_flag:
                data[frame, channel] = np.nan
                continue
            gain = sync_gains[channel] * fg
            if gain == 0 or np.isnan(gain):
                data[frame, channel] = np.nan
            else:
                data[frame, channel] /= gain


class FifiLsInfo(SofiaInfo):

    def __init__(self, configuration_path=None):
        """
        Initialize the FIFI-LS information.

        The HAWC+ information contains metadata on various parts of an
        observation that are specific to observations with the instrument.

        Parameters
        ----------
        configuration_path : str, optional
            An alternate directory path to the configuration tree to be
            used during the reduction.  The default is
            <package>/data/configurations.
        """
        super().__init__(configuration_path=configuration_path)
        self.name = 'fifi_ls'
        self.astrometry = FifiLsAstrometryInfo()
        self.detector_array = FifiLsDetectorArrayInfo()
        self.instrument = FifiLsInstrumentInfo()
        self.spectroscopy = None
        self.scanning = SofiaExtendedScanningInfo()

    @classmethod
    def get_file_id(cls):
        """
        Return the file ID.

        Returns
        -------
        str
        """
        return 'FIFI'

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
        self.detector_array.edit_header(header)

        freq = (1.0 / self.sampling_interval).to('Hz').value
        info = [('COMMENT', "<------ FIFI-LS Header Keys ------>"),
                ('SMPLFREQ', to_header_float(freq),
                 "(Hz) Detector readout rate.")]

        insert_info_in_header(header, info, delete_special=True)

    def validate_scans(self, scans):
        """
        Validate a list of scans specific to the instrument.

        Parameters
        ----------
        scans : list (HawcPlusScan)
            A list of scans.  Scans are culled in-place if they do not meet
            certain criteria.

        Returns
        -------
        None
        """
        if scans is None or len(scans) < 2 or scans[0] is None:
            super().validate_scans(scans)
            return

        n_scans = len(scans)

        first_scan = scans[0]
        wavelength = first_scan.info.instrument.wavelength
        instrument_config = first_scan.info.instrument.instrument_config
        keep_scans = np.full(n_scans, True)

        for i in range(1, n_scans):
            scan = scans[i]
            if scan.info.instrument.wavelength != wavelength:
                log.warning(f"Scan {scan.get_id()} in a different band. "
                            f"Removing from set.")
                keep_scans[i] = False
            elif scan.info.instrument.instrument_config != instrument_config:
                log.warning(f"Scan {scan.get_id()} is in a different "
                            f"instrument configuration. Removing from set.")
                keep_scans[i] = False

        for i in range(n_scans - 1, 0, -1):
            if not keep_scans[i]:
                del scans[i]

        super().validate_scans(scans)

    def max_pixels(self):
        """
        Return the maximum number of pixels.

        Returns
        -------
        count : int
        """
        return self.detector_array.pixels

    def get_si_pixel_size(self):
        """
        Get the science instrument pixel size.

        Returns
        -------
        size : Coordinate2D
            The (x, y) pixel sizes, each of which is a units.Quantity.
        """
        return self.detector_array.pixel_sizes

    def perform_reduction(self, reduction, filenames):
        """
        Fully reduce a given reduction and set of files.

        While it is possible for the reduction object to fully reduce a set of
        files, certain special considerations may be required for certain
        instruments.  Therefore, the instrument specific Info object is given
        control of how a reduction should progress.

        Parameters
        ----------
        reduction : Reduction
            The reduction object.
        filenames : str or list (str)
            A single file (str) or list of files to be included in the
            reduction.

        Returns
        -------
        None
        """
        is_resample = self.configuration.get_bool('fifi_ls.resample')
        insert_source = self.configuration.get_bool(
            'fifi_ls.insert_source')
        if is_resample:
            log.info("Performing reduction for subsequent FIFI-LS resampling.")

        reduction.read_scans(filenames)
        reduction.validate()

        if is_resample and not insert_source:
            start_data = {}
            for scan in reduction.scans:
                data = []
                for integration in scan.integrations:
                    data.append((integration.frames.data.copy(),
                                 integration.channels.data.offset.copy()))
                start_data[scan.get_id()] = data
        else:
            start_data = {}

        reduction.reduce()

        if not is_resample:
            return

        reduction.source.sync()  # Ensure the source is synced to the data

        if not insert_source:
            log.info('Removing decorrelations and offsets from original data.')
            for scan in reduction.scans:
                for (integration, (frame_data, start_offset)) in zip(
                        scan.integrations, start_data[scan.get_id()]):

                    correction = np.zeros(integration.frames.data.shape,
                                          dtype=float)

                    for mode, signal in integration.signals.items():
                        if not isinstance(signal, CorrelatedSignal
                                          ):  # pragma: no cover
                            continue
                        snf.resync_gains(
                            frame_data=correction,
                            signal_values=signal.value,
                            resolution=mode.get_frame_resolution(integration),
                            delta_gains=signal.sync_gains,
                            channel_indices=mode.channel_group.indices,
                            frame_valid=integration.frames.valid)

                    # Correct for offset
                    offset = start_offset - integration.channels.data.offset
                    cleaned = frame_data + offset[None] + correction
                    integration.frames.data = cleaned

        else:
            log.info('Reinserting source back into cleaned data.')
            source = reduction.source
            signal_mode = source.signal_mode

            for scan in reduction.scans:
                for integration in scan.integrations:
                    frames = integration.frames
                    frame_gains = integration.gain * frames.get_source_gain(
                        signal_mode)
                    source_gains = integration.source_sync_gain
                    channel_data = integration.channels.data

                    source_signal, source_error = get_source_signal(
                        frame_data=frames.data,
                        frame_valid=frames.valid,
                        frame_gains=frame_gains,
                        frame_weights=frames.relative_weight,
                        channel_flags=channel_data.flag,
                        channel_variance=channel_data.variance,
                        map_values=source.map.data,
                        map_valid=source.map.valid,
                        map_indices=frames.map_index.coordinates,
                        sync_gains=source_gains)

                    integration.frames.data += source_signal

    @staticmethod
    def combine_reduction_scans_for_resampler(reduction):  # pragma: no cover
        """
        Used to combine and extract the data for subsequent reduction.

        Parameters
        ----------
        reduction : Reduction

        Returns
        -------
        combined_data : dict
        """
        combined_info = {}
        n_samples = 0
        source = reduction.source
        signal_mode = source.signal_mode
        header_list = []
        filenames = []
        all_results = []
        scan_samples = []
        corner_coordinates = []
        corner_xy_coordinates = []

        for scan in reduction.scans:

            # Need to get the corner positions for each scan
            positions = scan.channels.data.position
            min_x, min_y = positions.min.coordinates.to('arcsec').value
            max_x, max_y = positions.max.coordinates.to('arcsec').value
            corners = Coordinate2D(np.asarray([[min_x, min_x, max_x, max_x],
                                               [min_y, max_y, max_y, min_y]]),
                                   unit='arcsec')

            for integration in scan.integrations:
                frames = integration.frames
                frame_gains = integration.gain * frames.get_source_gain(
                    signal_mode)
                source_gains = integration.source_sync_gain
                channel_data = integration.channels.data

                source_signal, source_error = get_source_signal(
                    frame_data=frames.data,
                    frame_valid=frames.valid,
                    frame_gains=frame_gains,
                    frame_weights=frames.relative_weight,
                    channel_flags=channel_data.flag,
                    channel_variance=channel_data.variance,
                    map_values=source.map.data,
                    map_valid=source.map.valid,
                    map_indices=frames.map_index.coordinates,
                    sync_gains=source_gains)

                total_data = frames.data.copy()
                blank_flag = int(frames.flagspace.convert_flag(
                    'SAMPLE_SOURCE_BLANK').value)

                # Re-normalize data and error
                correct_for_gain(data=total_data,
                                 frame_valid=frames.valid,
                                 frame_gains=frame_gains,
                                 sync_gains=source_gains,
                                 channel_flags=channel_data.flag,
                                 sample_flags=frames.sample_flag,
                                 source_blank_flag=blank_flag)

                equatorial = frames.get_equatorial(channel_data.position)
                detector = frames.info.detector_array
                xy = detector.equatorial_to_detector_coordinates(equatorial)
                ra = equatorial.ra.to('hourangle').value
                dec = equatorial.dec.to('degree').value
                x = xy.x.to('arcsec').value
                y = xy.y.to('arcsec').value
                wave = channel_data.wavelength.to('um').value
                valid = frames.valid & (frames.flag == 0)
                flags = frames.sample_flag.copy()
                flags[flags == blank_flag] = 0

                corner_equatorial = frames.get_equatorial(corners)
                corner_ra = corner_equatorial.ra.to('hourangle').value
                corner_dec = corner_equatorial.dec.to('degree').value
                corner_coordinates.append((corner_ra, corner_dec))
                corner_xy = detector.equatorial_to_detector_coordinates(
                    corner_equatorial)
                corner_xy_coordinates.append(
                    [corner_xy.x.to('arcsec').value,
                     corner_xy.y.to('arcsec').value])
                channel_valid = channel_data.flag == 0

                r = normalize_scan_coordinates(
                    ra, dec, x, y, wave, total_data, source_error, valid,
                    flags, channel_valid)
                n_samples += r[0].size
                scan_samples.append(r[0].size)
                header_list.append(scan.configuration.fits.header.copy())
                filenames.append(scan.info.origin.filename)
                all_results.append(r)

        coordinates = np.empty((3, n_samples), dtype=float)
        xy_coordinates = np.empty((2, n_samples), dtype=float)
        data = np.empty(n_samples, dtype=float)
        error = np.empty(n_samples, dtype=float)
        start = 0
        i = 1
        while all_results:
            r = all_results.pop(0)
            n = r[0].size
            coordinates[:, start:start + n] = r[:3]
            xy_coordinates[:, start:start + n] = r[3:5]
            data[start:start + n] = r[5]
            error[start:start + n] = r[6]
            start += n
            i += 1

        combined_info['filenames'] = filenames
        combined_info['headers'] = header_list
        combined_info['coordinates'] = coordinates
        combined_info['xy_coordinates'] = xy_coordinates
        combined_info['flux'] = data
        combined_info['error'] = error
        combined_info['samples'] = np.asarray(scan_samples)
        combined_info['corners'] = corner_coordinates
        combined_info['xy_corners'] = corner_xy_coordinates

        return combined_info
