# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for R ratio calculation."""

import numpy as np

from sofia_redux.calibration.pipecal_rratio import pipecal_rratio


class TestRRatio(object):

    def test_calc(self):
        za = 36.0
        alt = 39.0
        pwv = 6.0

        za_ref = 45.0
        alt_ref = 41.0
        pwv_ref = 7.3

        # realistic coefficients
        # (from forcast/response/rfit_*_single_20160127, FOR_F197)
        za_coeff = [0.99999990, -0.0087751624, 0.00099167718,
                    -0.00027061611, 6.0513493e-05, -3.8735213e-06]
        alt_coeff = [0.99995775, 0.0016109900, -4.3034104e-05,
                     -3.0131956e-05, -1.2056881e-06, 1.7214695e-06]
        pwv_coeff = [0.99993170, -0.010966916, -0.0028665582,
                     -0.00039106328, 6.4916506e-05, 4.1356799e-05,
                     -2.5119877e-05]

        # right answer, from IDL implementation
        ans_alt = 0.99832129
        ans_pwv = 1.0035746

        # result from Python implementation
        res_alt = pipecal_rratio(za, alt, za_ref, alt_ref,
                                 za_coeff, alt_coeff, pwv=False)
        res_pwv = pipecal_rratio(za, pwv, za_ref, pwv_ref,
                                 za_coeff, pwv_coeff, pwv=True)

        assert np.allclose(res_alt, ans_alt)
        assert np.allclose(res_pwv, ans_pwv)

        # at reference values: should be first terms, multiplied
        res_alt = pipecal_rratio(za_ref, alt_ref, za_ref, alt_ref,
                                 za_coeff, alt_coeff, pwv=False)
        res_pwv = pipecal_rratio(za_ref, pwv_ref, za_ref, pwv_ref,
                                 za_coeff, pwv_coeff, pwv=True)

        assert np.allclose(res_alt, alt_coeff[0] * za_coeff[0])
        assert np.allclose(res_pwv, pwv_coeff[0] * za_coeff[0])

        # unrealistic coefficients, for integrity check: should return 1.0
        coeff = [1.0]
        res_alt = pipecal_rratio(za, alt, za_ref, alt_ref,
                                 coeff, coeff, pwv=False), 1.0
        res_pwv = pipecal_rratio(za, alt, za_ref, alt_ref,
                                 coeff, coeff, pwv=True)
        assert np.allclose(res_alt, 1.0)
        assert np.allclose(res_pwv, 1.0)
