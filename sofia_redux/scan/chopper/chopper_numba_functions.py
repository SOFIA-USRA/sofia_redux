# Licensed under a 3-clause BSD style license - see LICENSE.rst

import math
import numba as nb
import numpy as np


nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['find_transitions']


@nb.njit(cache=True, nogil=False, parallel=False)
def find_transitions(x, y, threshold):  # pragma: no cover
    """
    Given (x, y) chopper positions, find the transitions and other parameters.

    Given a chopper position in (x, y) coordinates, centered about (0, 0),
    return the index on (x, y) marking the `start` of the first transition,
    the index marking the `end` of the last transition, the number of
    `transitions`, the calculated `angle` between the two chop positions, and
    the `distance` from the center while chopping.

    This algorithm assumes that there are two chopping positions present in the
    (x, y) data (on and off) that are relative to a central (0, 0) nominal chop
    position.  A chop transition is marked when the absolute x or y position
    is greater than `tolerance` and of the opposite sign to the last detected
    chop transition.  Once a transition has been detected in both the x and
    y directions, the distance will be calculated as:

        d = sqrt(x^2 + y^2)

    for each subsequent (x, y) position.  The returned angle is weighted by
    distance (d) from the center and is given by:

        a = sum(d * arctan(s * y, s * x)) / sum(d)

    where s is given by:

        s = sign(y); x = 0
        s = sign(x); otherwise

    It is up to the user on how to best extract the chop amplitude from the
    data given the chop profile.  For a standard box-step chop profile, a
    median on the distances should suffice, while more advanced analysis
    should be applied to other patterns.

    Parameters
    ----------
    x : numpy.ndarray (float)
        The x-direction chopper position of shape (n,).
    y : numpy.ndarray (float)
        The y-direction chopper position of shape (n,).
    threshold : float
        The distance away from the center (x, y) = (0, 0) that would be
        considered representing a single chopper transition
        (chopper amplitude).

    Returns
    -------
    start, end, transitions, angle, distance : int, int, int, float, np.ndarray
    """
    n = x.size
    x_positive = False
    y_positive = False
    x_transitions = 0
    y_transitions = 0
    x_from = -1
    y_from = -1
    x_to = -1
    y_to = -1
    sum_a = 0.0
    sum_w = 0.0
    started = False
    distance = np.empty(n, dtype=np.float64)
    count = 0
    start = 0
    end = 0
    transitions = 0
    for i in range(n):

        dx = x[i]
        dy = y[i]
        if not started:
            x_positive = dx > 0.0
            y_positive = dy > 0.0
            started = True
            continue

        if ((x_positive and dx < threshold)
                or (not x_positive and dx > threshold)):
            x_positive = not x_positive
            if x_transitions == 0:
                x_from = i
            else:
                x_to = i
            x_transitions += 1

        if ((y_positive and dy < threshold)
                or (not y_positive and dy > threshold)):
            y_positive = not y_positive
            if y_transitions == 0:
                y_from = i
            else:
                y_to = i
            y_transitions += 1

        if x_transitions > 0 and y_transitions > 0:
            d = math.hypot(dx, dy)
            if d > threshold:
                if dx == 0.0:
                    sign = -1 if dy < 0.0 else 1
                else:
                    sign = -1 if dx < 0.0 else 1
                sum_a += d * math.atan2(sign * dy, sign * dx)
                sum_w += d
            distance[count] = d
            count += 1

        if x_transitions > y_transitions:
            start = x_from
            end = x_to
            transitions = x_transitions
        else:
            start = y_from
            end = y_to
            transitions = y_transitions

    if sum_w > 0:
        angle_radians = sum_a / sum_w
    else:
        angle_radians = 0.0

    return start, end, transitions, angle_radians, distance[:count]
