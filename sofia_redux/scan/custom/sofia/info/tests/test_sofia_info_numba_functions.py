# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.custom.sofia.info.sofia_info_numba_functions import \
    get_drift_corrections


def test_get_drift_correction():
    # Test no frames
    frame_utc = np.zeros(1)
    frame_valid = np.empty(0, dtype=bool)
    drift_utc_ranges = np.zeros((0, 2))
    drift_deltas = np.zeros((0, 2))

    correction, extrapolation_frame = get_drift_corrections(
        frame_utc, frame_valid, drift_utc_ranges, drift_deltas)
    assert correction.shape == (1, 2) and np.allclose(correction, 0)
    assert extrapolation_frame == -1

    n_frames = 100
    frame_utc = np.arange(n_frames, dtype=float)
    frame_valid = np.full(n_frames, True)
    frame_valid[:10] = False

    drift_utc_ranges = np.asarray([[0, 24], [25, 49], [50, 74], [75, 90]],
                                  dtype=float)

    drift_deltas = np.asarray([[24, 48], [24, 48], [24, 48], [15, 30]])

    correction, extrapolation_frame = get_drift_corrections(
        frame_utc, frame_valid, drift_utc_ranges, drift_deltas)

    expected = np.arange(100) % 25
    expected[:10] = 0
    assert np.allclose(correction[:, 0], expected)
    assert np.allclose(correction[:, 1], expected * 2)
