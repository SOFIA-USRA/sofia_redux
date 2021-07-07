# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
import pytest

import sofia_redux.calibration.pipecal_photometry as ph
from sofia_redux.calibration.pipecal_error import PipeCalError
from sofia_redux.calibration.tests import resources


class TestPhotometry(object):

    def phot_dict(self, phot):
        pkeys = [p['key'] for p in phot]
        pvals = [p['value'] for p in phot]
        phd = {k: v for k, v in zip(pkeys, pvals)}
        return phd

    def test_photometry(self):
        rtol = 0.05
        atol = 0.1

        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        phot = ph.pipecal_photometry(image, variance)

        # values from running idl pipecal on resources fake data
        idl_vals = [
            {"key": "PHOTPROF", "value": "MOFFAT",
             "comment": "Profile fit type for photometry"},
            {"key": "PHOTIMSZ", "value": "138",
             "comment": "Sub-image size for photometry"},
            {"key": "PHOTAPER", "value": "12.000000",
             "comment": "Aperture radius for photometry"},
            {"key": "PHOTSKAP", "value": "15.0000,25.0000",
             "comment": "Sky aperture radii for photometry (pix)"},
            {"key": "STCENTX", "value": "250.000 0.0162868",
             "comment": "Star centroid x-value (pix)"},
            {"key": "STCENTY", "value": "250.000 0.0162112",
             "comment": "Star centroid y-value (pix)"},
            {"key": "STPEAK", "value": "10.4327 0.0925834",
             "comment": "Star fit peak value (Me/s)"},
            {"key": "STBKG", "value": "-0.00956431 0.00148007",
             "comment": "Star fit background level (Me/s)"},
            {"key": "STFWHMX", "value": "4.25627 0.0395566",
             "comment": "Star fit FWHM in x-direction (pix)"},
            {"key": "STFWHMY", "value": "4.25632 0.0385464",
             "comment": "Star fit FWHM in y-direction (pix)"},
            {"key": "STANGLE", "value": "139.688 7968.32",
             "comment": "Star fit position angle (deg)"},
            {"key": "STPWLAW", "value": "6.00000 0.00000",
             "comment": "Star fit power law index"},
            {"key": "STPRFLX", "value": "242.425 3.81111",
             "comment": "Star flux from profile (Me/s)"},
            {"key": "STAPFLX", "value": "229.450 24.3604",
             "comment": "Star flux from aper. photometry (Me/s)"},
            {"key": "STAPSKY", "value": "9.99990e-07 0.0282095",
             "comment": "Sky flux from aper. photometry (Me/s)"}
        ]

        array_keys = ['STCENTX', 'STCENTY', 'STPEAK', 'STBKG', 'STFWHMX',
                      'STFWHMY', 'STANGLE', 'STPWLAW', 'STPRFLX',
                      'STAPFLX', 'STAPSKY', 'PHOTSKAP']
        string_keys = ['PHOTPROF']

        for key, val in enumerate(idl_vals):
            print('idl {}: {}'.format(key, val))
            print('py {}: {}'.format(key, phot[key]))
            py = phot[key]
            assert val['key'] == py['key']

            if val['key'] == 'STANGLE':
                # angle fits are unconstrained for these data
                continue

            if val['key'] in array_keys:
                if not isinstance(val['value'], list):
                    idl_arr = [float(j) for j in
                               val['value'].replace(',', ' ').split()]
                else:
                    idl_arr = val['value']
                if not isinstance(py['value'], list):
                    py_arr = [float(j) for j in
                              py['value'].replace(',', ' ').split()]
                else:
                    py_arr = py['value']

                assert np.allclose(py_arr[0], idl_arr[0],
                                   rtol=rtol, atol=atol)
                assert py_arr[1] > 0
            elif val['key'] in string_keys:
                assert val['value'] == py['value']

            else:
                assert np.allclose(float(py['value']), float(val['value']),
                                   rtol=rtol, atol=atol)

    def test_photometry_errors(self, capsys):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        # non-array image
        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(10, variance)
        capt = capsys.readouterr()
        assert 'Invalid image type' in capt.err

        # non-2D image
        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(np.array(10), variance)
        capt = capsys.readouterr()
        assert 'Image must be 2d' in capt.err

        # non-array variance
        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, 10)
        capt = capsys.readouterr()
        assert 'Invalid variance type' in capt.err

        # non-2D variance
        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, np.array(10))
        capt = capsys.readouterr()
        assert 'Variance must be 2d' in capt.err

        # mismatched variance
        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, np.zeros((10, 10)))
        capt = capsys.readouterr()
        assert 'Variance must be same shape' in capt.err

        default_phot = self.phot_dict(ph.pipecal_photometry(image, variance))

        # bad fitsize: uses default
        phot = ph.pipecal_photometry(image, variance, fitsize='a')
        assert self.phot_dict(phot)['PHOTIMSZ'] == default_phot['PHOTIMSZ']

        # bad fwhm: uses default
        phot = ph.pipecal_photometry(image, variance, fwhm='a')
        assert self.phot_dict(phot)['STFWHMX'] == default_phot['STFWHMX']

        # bad aprad: uses default
        phot = ph.pipecal_photometry(image, variance, aprad='a')
        assert self.phot_dict(phot)['PHOTAPER'] == default_phot['PHOTAPER']

        # bad profile
        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, variance, profile='unknown')
        capt = capsys.readouterr()
        assert 'Invalid profile selection' in capt.err

        # bad skyrad
        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, variance, skyrad=10)
        capt = capsys.readouterr()
        assert 'Invalid sky radius' in capt.err

    def test_photometry_profiles(self):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        # test all three models -- should at least return the same centroid.
        phot1 = ph.pipecal_photometry(image, variance, profile='gaussian')
        phot2 = ph.pipecal_photometry(image, variance, profile='lorentzian')
        phot3 = ph.pipecal_photometry(image, variance, profile='moffat')

        rtol = 0.01
        atol = 0.1
        skip_keys = ['PHOTPROF',
                     'STPWLAW', 'STANGLE',
                     'STPEAK', 'STPRFLX',
                     'STFWHMX', 'STFWHMY']
        for i, p1 in enumerate(phot1):
            p2 = phot2[i]
            p3 = phot3[i]
            print(p1['key'], p1['value'], p2['value'], p3['value'])

            if p1['key'] in skip_keys:
                continue
            elif isinstance(p1['value'], str):
                assert p1['value'] == p2['value']
                assert p1['value'] == p3['value']
            elif isinstance(p1['value'], list):
                assert np.allclose(p1['value'][0], p2['value'][0],
                                   rtol=rtol, atol=atol)
                assert np.allclose(p1['value'][1], p2['value'][1],
                                   rtol=rtol, atol=atol)
                assert np.allclose(p1['value'][0], p3['value'][0],
                                   rtol=rtol, atol=atol)
                assert np.allclose(p1['value'][1], p3['value'][1],
                                   rtol=rtol, atol=atol)
            else:
                assert np.allclose(p1['value'], p2['value'],
                                   rtol=rtol, atol=atol)
                assert np.allclose(p1['value'], p3['value'],
                                   rtol=rtol, atol=atol)

    def test_fitpeak_problem(self, mocker, capsys):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        # mock a problem in fitting the first stamp
        def mock_fit(stamp, **kwargs):
            raise RuntimeError('bad stamp')
        mocker.patch(
            'sofia_redux.calibration.pipecal_photometry.pipecal_fitpeak',
            mock_fit)

        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, variance)
        capt = capsys.readouterr()
        assert 'Unable to fit stamp' in capt.err

        # mock a problem in the second fit
        def mock_fit(stamp, **kwargs):
            par = {'baseline': 0.1,
                   'dpeak': 20.0,
                   'col_mean': 15, 'row_mean': 20,
                   'col_sigma': 4, 'row_sigma': 2,
                   'theta': 10 * np.pi / 180.}
            if stamp.size < 137**2:
                return par, par, 1
            else:
                raise RuntimeError('bad image')
        mocker.patch(
            'sofia_redux.calibration.pipecal_photometry.pipecal_fitpeak',
            mock_fit)

        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, variance, profile='gaussian')
        capt = capsys.readouterr()
        assert 'Unable to fit subimage' in capt.err

        # mock bad fit values
        def mock_fit(stamp, **kwargs):
            par = {'baseline': np.nan,
                   'dpeak': 20.0,
                   'col_mean': 15, 'row_mean': 20,
                   'col_sigma': 4, 'row_sigma': 2,
                   'theta': 10 * np.pi / 180.}
            return par, par, 1
        mocker.patch(
            'sofia_redux.calibration.pipecal_photometry.pipecal_fitpeak',
            mock_fit)

        phot = ph.pipecal_photometry(image, variance, profile='gaussian')
        dphot = self.phot_dict(phot)
        # all values from fit except centroid and peak are set to zero
        assert np.allclose([dphot['STBKG'],
                            dphot['STFWHMX'], dphot['STFWHMY'],
                            dphot['STANGLE'], dphot['STPWLAW'],
                            dphot['STPRFLX']], 0)

    def test_phot_problem(self, mocker, capsys):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        # mock a non-finite return from aperture phot
        def mock_phot(*args, **kwargs):
            return {'aperture_sum': np.nan}
        mocker.patch(
            'sofia_redux.calibration.pipecal_photometry.'
            'photutils.aperture_photometry',
            mock_phot)

        # sets fluxes, errors to zero if not finite
        phot = ph.pipecal_photometry(image, variance, profile='gaussian')
        dphot = self.phot_dict(phot)
        assert np.allclose([dphot['STAPFLX'][0], dphot['STAPFLX'][1],
                            dphot['STAPSKY'][0], dphot['STAPSKY'][1]], 0.0)

    def test_phot_angle(self, mocker):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        # mock fit values for x > y
        def mock_fit(stamp, **kwargs):
            par = {'baseline': 0.0,
                   'dpeak': 20.0,
                   'col_mean': 15, 'row_mean': 20,
                   'col_sigma': 4, 'row_sigma': 2,
                   'theta': -10 * np.pi / 180.}
            return par, par.copy(), 1
        mocker.patch(
            'sofia_redux.calibration.pipecal_photometry.pipecal_fitpeak',
            mock_fit)

        phot = ph.pipecal_photometry(image, variance, profile='gaussian')
        dphot = self.phot_dict(phot)

        # angle should be corrected to a positive value, same orientation
        assert np.allclose(dphot['STANGLE'][0], 170.)

        # mock fit values for y > x
        def mock_fit(stamp, **kwargs):
            par = {'baseline': 0.0,
                   'dpeak': 20.0,
                   'col_mean': 15, 'row_mean': 20,
                   'col_sigma': 2, 'row_sigma': 4,
                   'theta': -10 * np.pi / 180.}
            print(par)
            return par, par.copy(), 1
        mocker.patch(
            'sofia_redux.calibration.pipecal_photometry.pipecal_fitpeak',
            mock_fit)

        phot = ph.pipecal_photometry(image, variance, profile='gaussian')
        dphot = self.phot_dict(phot)

        # angle should be corrected to a positive value, 90 deg off
        assert np.allclose(dphot['STANGLE'][0], 80.)

    @pytest.mark.parametrize('profile', ['gaussian', 'lorentzian', 'moffat'])
    def test_photometry_error_values(self, profile):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        # add noise
        rand = np.random.RandomState(42)
        image += rand.normal(0, 5, image.shape)
        variance += rand.normal(0, 25, image.shape)

        # unscaled image
        phot = ph.pipecal_photometry(image, variance, profile=profile)
        u_dphot = self.phot_dict(phot)

        # scaled image
        scale = 1e6
        phot = ph.pipecal_photometry(
            image * scale, variance * scale**2, profile=profile)
        s_dphot = self.phot_dict(phot)

        # relative errors for profile values should be the same
        for key in ['STPRFLX', 'STCENTX', 'STCENTY', 'STPEAK',
                    'STBKG', 'STFWHMX', 'STFWHMY']:
            print(key, u_dphot[key], s_dphot[key],
                  u_dphot[key][1] / u_dphot[key][0],
                  s_dphot[key][1] / s_dphot[key][0])
            assert np.allclose(u_dphot[key][1] / u_dphot[key][0],
                               s_dphot[key][1] / s_dphot[key][0],
                               atol=0.05, rtol=0.05)

    def test_stamp_size(self, capsys):
        olevel = log.level
        log.setLevel('DEBUG')
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        # auto: stamp is fwhm * 5
        phot = ph.pipecal_photometry(image, variance, fitsize=200, fwhm=10,
                                     stampsize='auto')
        dphot = self.phot_dict(phot)
        capt = capsys.readouterr()
        assert 'Initial stamp size: 50' in capt.out
        cx1, cy1 = dphot['STCENTX'], dphot['STCENTY']

        # auto: stamp is fwhm * 5, minimum of 20
        phot = ph.pipecal_photometry(image, variance, fitsize=200, fwhm=1,
                                     stampsize='auto')
        dphot = self.phot_dict(phot)
        capt = capsys.readouterr()
        assert 'Initial stamp size: 20' in capt.out
        cx2, cy2 = dphot['STCENTX'], dphot['STCENTY']

        # centroid is the same
        assert np.allclose(cx2, cx1, atol=0.05, rtol=0.05)
        assert np.allclose(cy2, cy1, atol=0.05, rtol=0.05)

        # auto: stamp is fwhm * 5, maximum of fitsize
        phot = ph.pipecal_photometry(image, variance, fitsize=200, fwhm=200,
                                     stampsize='auto')
        dphot = self.phot_dict(phot)
        capt = capsys.readouterr()
        assert 'Initial stamp size: 200' in capt.out
        cx3, cy3 = dphot['STCENTX'], dphot['STCENTY']
        assert np.allclose(cx3, cx1, atol=0.05, rtol=0.05)
        assert np.allclose(cy3, cy1, atol=0.05, rtol=0.05)

        # specify stamp size instead
        phot = ph.pipecal_photometry(image, variance, fitsize=200, fwhm=1,
                                     stampsize=10)
        dphot = self.phot_dict(phot)
        capt = capsys.readouterr()
        assert 'Initial stamp size: 10' in capt.out
        cx4, cy4 = dphot['STCENTX'], dphot['STCENTY']
        assert np.allclose(cx4, cx1, atol=0.05, rtol=0.05)
        assert np.allclose(cy4, cy1, atol=0.05, rtol=0.05)

        # specify invalid stamp size
        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, variance, fitsize=200, fwhm=1,
                                  stampsize='bad')
        capt = capsys.readouterr()
        assert 'Invalid stampsize' in capt.err

        log.setLevel(olevel)

    def test_nan_data(self, capsys):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        image *= np.nan

        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, variance)
        assert 'Stamp image is empty' in capsys.readouterr().err

    def test_allow_badfit(self, capsys, mocker):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        mocker.patch('sofia_redux.calibration.'
                     'pipecal_photometry.pipecal_fitpeak',
                     side_effect=RuntimeError())

        # error if bad fit not allowed
        with pytest.raises(PipeCalError):
            ph.pipecal_photometry(image, variance, allow_badfit=False)

        # no error if badfit is allowed: still does aperture photometry
        # at source position
        phot = ph.pipecal_photometry(image, variance, allow_badfit=True)
        dphot = self.phot_dict(phot)
        capt = capsys.readouterr()
        assert 'Unable to fit stamp' in capt.err
        assert 'Unable to fit subimage' in capt.err
        assert np.allclose(dphot['STAPFLX'], [229.450, 24.3604],
                           atol=0.05, rtol=0.05)

    def test_no_recenter(self):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]

        # good data should work with or without the recenter
        phot = ph.pipecal_photometry(image, variance, stampsize=50,
                                     stamp_center=False)
        dphot = self.phot_dict(phot)
        cx1, cy1 = dphot['STCENTX'], dphot['STCENTY']

        phot = ph.pipecal_photometry(image, variance, stampsize=50,
                                     stamp_center=True)
        dphot = self.phot_dict(phot)
        cx2, cy2 = dphot['STCENTX'], dphot['STCENTY']
        assert np.allclose(cx2, cx1, atol=0.05, rtol=0.05)
        assert np.allclose(cy2, cy1, atol=0.05, rtol=0.05)

        # different input position without recenter will be different
        srcpos = (cx1[0] + 10, cx2[0] + 10)
        phot = ph.pipecal_photometry(image, variance, stampsize=50,
                                     stamp_center=False,
                                     srcpos=srcpos)
        dphot = self.phot_dict(phot)
        cx1, cy1 = dphot['STCENTX'], dphot['STCENTY']

        phot = ph.pipecal_photometry(image, variance, stampsize=50,
                                     stamp_center=True,
                                     srcpos=srcpos)
        dphot = self.phot_dict(phot)
        cx2, cy2 = dphot['STCENTX'], dphot['STCENTY']
        assert not np.allclose(cx2, cx1, atol=0.05, rtol=0.05)
        assert not np.allclose(cy2, cy1, atol=0.05, rtol=0.05)

    def test_no_background(self):
        hdul = resources.forcast_legacy_data()
        image = hdul[0].data[0]
        variance = hdul[0].data[1]
        # add a higher background level
        image += 10

        # background annulus
        phot = ph.pipecal_photometry(image, variance, skyrad=(15, 25))
        dphot = self.phot_dict(phot)
        assert np.allclose(dphot['STAPSKY'], [3.6, 0.01],
                           atol=0.05, rtol=0.05)
        assert np.allclose(dphot['STAPFLX'], [229.450, 24.3604],
                           atol=0.05, rtol=0.05)
        f1 = dphot['STAPFLX'][0]

        # no background annulus
        phot = ph.pipecal_photometry(image, variance, skyrad=(0, 0))
        dphot = self.phot_dict(phot)
        assert np.allclose(dphot['STAPSKY'], [0, 0])
        assert not np.allclose(dphot['STAPFLX'], [229.450, 24.3604],
                               atol=0.05, rtol=0.05)

        # flux is higher without background subtracted
        f2 = dphot['STAPFLX'][0]
        assert f2 > f1
