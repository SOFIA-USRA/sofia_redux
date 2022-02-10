# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.chopper.chopper_numba_functions import find_transitions


def test_find_transitions():

    # Create ramped box step function, rotated at 30 degrees
    angle = np.deg2rad(30)
    amplitude = 2.0
    x_pos = amplitude * np.cos(angle)
    y_pos = amplitude * np.sin(angle)
    x_neg = -x_pos
    y_neg = -y_pos
    n_transitions = 6
    n = 100 * n_transitions
    x = np.empty(n)
    y = np.empty(n)
    length = 16
    w = length // 2
    ramp_x = np.linspace(x_neg, x_pos, length)
    ramp_y = np.linspace(y_neg, y_pos, length)

    for i in range(n_transitions):
        start = i * 100
        end = start + 100
        if (i % 2) == 0:
            x[start:end] = x_pos
            y[start:end] = y_pos
            x[start:start + w] = ramp_x[w:]
            y[start:start + w] = ramp_y[w:]
            x[end - w: end] = -ramp_x[:w]
            y[end - w: end] = -ramp_y[:w]
        else:
            x[start:end] = x_neg
            y[start:end] = y_neg
            x[start:start + w] = -ramp_x[w:]
            y[start:start + w] = -ramp_y[w:]
            x[end - w: end] = ramp_x[:w]
            y[end - w: end] = ramp_y[:w]

    threshold = 1.5

    (start, end, transitions,
     angle, distance) = find_transitions(x, y, threshold)
    assert start == 1 and end == 494 and transitions == 7
    assert np.isclose(np.rad2deg(angle), 30)
    assert np.isclose(np.median(distance), 2)

    x0 = x.copy()
    x0[300:] = 0.0
    start, end, transitions, angle, distance = find_transitions(
        x0, y, np.max(y) * 0.7)
    assert start == 1 and end == 495 and transitions == 7
    assert np.isclose(np.rad2deg(angle), 49.686120647)
    assert np.isclose(np.median(distance), 1)

    start, end, transitions, angle, distance = find_transitions(
        x * 0, y * 0, 1)
    assert start == -1 and end == -1 and transitions == 0 and angle == 0
    assert distance.size == 0
