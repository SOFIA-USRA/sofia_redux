# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.custom.fifi_ls.integration\
    .fifi_ls_integration_numba_functions import flag_zeroed_channels


def test_flag_zeroed_channels():
    n_channels = 3
    n_frames = 5
    frame_data = np.zeros((n_frames, n_channels), dtype=float)
    discard_flag = 1
    frame_valid = np.full(n_frames, True)
    frame_valid[2] = False
    channel_indices = np.arange(n_channels)
    channel_flags = np.asarray([0, 0, 1])
    frame_data[0, 0] = 2  # no flag
    frame_data[2, 1] = 2  # invalid frame - so flag all zero
    frame_data[0, 2] = 2  # unflag
    flag_zeroed_channels(frame_data, frame_valid, channel_indices,
                         channel_flags, discard_flag)
    assert np.allclose(channel_flags, [0, 1, 0])
