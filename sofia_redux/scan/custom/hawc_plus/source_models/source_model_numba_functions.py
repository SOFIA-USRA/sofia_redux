# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def combine_rt_map_data(r, r_weight, r_exposure, r_valid,
                        t, t_weight, t_exposure, t_valid,
                        bad_flag, sign=1):

    rt = np.zeros(r.shape, dtype=nb.float64)
    rt_weight = np.zeros(r.shape, dtype=nb.float64)
    rt_exposure = np.zeros(r.shape, dtype=nb.float64)
    rt_flag = np.zeros(r.shape, dtype=nb.int64)

    # Flatten everything
    r_flat = r.flat
    r_weight_flat = r_weight.flat
    r_exposure_flat = r_exposure.flat
    r_valid_flat = r_valid.flat
    t_flat = t.flat
    t_weight_flat = t_weight.flat
    t_exposure_flat = t_exposure.flat
    t_valid_flat = t_valid.flat
    rt_flat = rt.flat
    rt_weight_flat = rt_weight.flat
    rt_exposure_flat = rt_exposure.flat
    rt_flag_flat = rt_flag.flat

    for i in range(r.size):
        if not r_valid_flat[i] or not t_valid_flat[i]:
            rt_flag_flat[i] = bad_flag
            continue
        rw = r_weight_flat[i]
        tw = t_weight_flat[i]
        if rw == 0 or tw == 0:
            rt_flag_flat[i] = bad_flag
            continue

        r_value = r_flat[i]
        t_value = t_flat[i]
        if not np.isfinite(r_value) or not np.isfinite(t_value):
            rt_flag_flat[i] = bad_flag
            continue

        r_time = r_exposure_flat[i]
        t_time = t_exposure_flat[i]
        if r_time <= 0 or t_time <= 0:
            rt_flag_flat[i] = bad_flag
            continue

        rtw = rw + tw
        r1 = r_value
        t1 = t_value

        if sign > 0:
            rt_value = r1 + t1
        else:
            rt_value = r1 - t1

        rt_flat[i] = rt_value
        rt_weight_flat[i] = rtw
        rt_exposure_flat[i] = r_time + t_time

    return rt, rt_weight, rt_exposure, rt_flag
