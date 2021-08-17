# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for pipecal utility/convenience functions."""

from astropy import log
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.calibration import pipecal_util as util
from sofia_redux.calibration.pipecal_config import pipecal_config
from sofia_redux.calibration.pipecal_error import PipeCalError
from sofia_redux.calibration.pipecal_photometry import pipecal_photometry
from sofia_redux.calibration.pipecal_rratio import pipecal_rratio
from sofia_redux.calibration.tests import resources


class TestUtil(object):
    def test_avg_za(self, capsys):
        # minimal header
        header = fits.header.Header()

        # start/end both good
        header['ZA_START'] = 40.0
        header['ZA_END'] = 50.0
        assert np.allclose(util.average_za(header), 45.0)

        # start good, end bad
        header['ZA_START'] = 40.0
        header['ZA_END'] = -9999
        assert np.allclose(util.average_za(header), 40.0)
        header['ZA_END'] = 'BAD'
        assert np.allclose(util.average_za(header), 40.0)

        # start bad, end good
        header['ZA_START'] = -9999
        header['ZA_END'] = 50.0
        assert np.allclose(util.average_za(header), 50.0)
        header['ZA_START'] = 'BAD'
        assert np.allclose(util.average_za(header), 50.0)

        # both bad
        del header['ZA_START']
        header['ZA_END'] = -9999
        with pytest.raises(PipeCalError):
            util.average_za(header)
        capt = capsys.readouterr()
        assert 'Bad ZA value' in capt.err

    def test_avg_alt(self, capsys):
        # minimal header
        header = fits.header.Header()

        # start/end both good
        header['ALTI_STA'] = 40000.0
        header['ALTI_END'] = 50000.0
        assert np.allclose(util.average_alt(header), 45.0)

        # start good, end bad
        header['ALTI_STA'] = 40000.0
        header['ALTI_END'] = -9999
        assert np.allclose(util.average_alt(header), 40.0)
        header['ALTI_END'] = 'BAD'
        assert np.allclose(util.average_alt(header), 40.0)

        # start bad, end good
        header['ALTI_STA'] = -9999
        header['ALTI_END'] = 50000.0
        assert np.allclose(util.average_alt(header), 50.0)
        header['ALTI_STA'] = 'BAD'
        assert np.allclose(util.average_alt(header), 50.0)

        # both bad
        del header['ALTI_STA']
        header['ALTI_END'] = -9999
        with pytest.raises(PipeCalError):
            util.average_alt(header)
        capt = capsys.readouterr()
        assert 'Bad altitude value' in capt.err

    def test_avg_pwv(self, capsys):
        # minimal header
        header = fits.header.Header()

        # start/end both good
        header['WVZ_STA'] = 4.0
        header['WVZ_END'] = 5.0
        assert np.allclose(util.average_pwv(header), 4.5)

        # start good, end bad
        header['WVZ_STA'] = 4.0
        header['WVZ_END'] = -9999
        assert np.allclose(util.average_pwv(header), 4.0)
        header['WVZ_END'] = 'BAD'
        assert np.allclose(util.average_pwv(header), 4.0)

        # start bad, end good
        header['WVZ_STA'] = -9999
        header['WVZ_END'] = 5.0
        assert np.allclose(util.average_pwv(header), 5.0)
        header['WVZ_STA'] = 'BAD'
        assert np.allclose(util.average_pwv(header), 5.0)

        # both bad
        del header['WVZ_STA']
        header['WVZ_END'] = -9999
        with pytest.raises(PipeCalError):
            util.average_pwv(header)
        capt = capsys.readouterr()
        assert 'Bad PWV value' in capt.err

    def test_guess_source(self, capsys, mocker):
        # set log to debug for this test
        olevel = log.level
        log.setLevel('DEBUG')

        # simulated data
        hdul = resources.forcast_data()
        header = hdul[0].header
        image = hdul[0].data
        srcpos = [header['SRCPOSX'], header['SRCPOSY']]
        crpix = [header['CRPIX1'], header['CRPIX2']]

        # source pos in header
        test = util.guess_source_position(header, image)
        assert test == srcpos
        capt = capsys.readouterr()
        assert 'Found SRCPOS in header' in capt.out

        # source pos not in header, find with find_peaks
        del header['SRCPOSX']
        del header['SRCPOSY']
        test = util.guess_source_position(header, image)
        assert test == srcpos
        capt = capsys.readouterr()
        assert 'SRCPOS from find_peaks' in capt.out

        # source pos in header, but set to zero -- use find_peaks
        header['SRCPOSX'] = 0.0
        header['SRCPOSY'] = 0.0
        test = util.guess_source_position(header, image)
        assert test == srcpos
        capt = capsys.readouterr()
        assert 'SRCPOS from find_peaks' in capt.out

        # find_peaks failure: use crpix
        mocker.patch(
            'sofia_redux.calibration.pipecal_util.photutils.find_peaks',
            return_value=None)
        test = util.guess_source_position(header, image)
        assert test == crpix
        capt = capsys.readouterr()
        assert 'SRCPOS from CRPIX' in capt.out

        # find_peaks failure, no crpix: return None
        del header['CRPIX1']
        del header['CRPIX2']
        test = util.guess_source_position(header, image)
        assert test is None
        capt = capsys.readouterr()
        assert 'SRCPOS not found' in capt.out

        log.setLevel(olevel)

    def test_add_calfac_keys(self):
        # simulated data
        hdul = resources.forcast_data()
        header = hdul[0].header

        # get config from data header
        config = pipecal_config(header)

        # expected header output using config values
        runits = config['runits']
        keys = {'PROCSTAT': ('LEVEL_3', 'Processing status'),
                'BUNIT': ('Jy/pixel', 'Data units'),
                'CALFCTR': (config['calfac'],
                            'Calibration factor ({}/Jy)'.format(runits)),
                'ERRCALF': (config['ecalfac'],
                            'Calibration factor uncertainty '
                            '({}/Jy)'.format(runits)),
                'LAMREF': (config['wref'], 'Reference wavelength (microns)'),
                'LAMPIVOT': (config['lpivot'], 'Pivot wavelength (microns)'),
                'COLRCORR': (config['color_corr'], 'Color correction factor'),
                'REFCALZA': (config['rfit_am']['zaref'],
                             'Reference calibration zenith angle'),
                'REFCALAW': (config['rfit_alt']['altwvref'],
                             'Reference calibration altitude'),
                'REFCALF3': (config['refcal_file'].partition(
                    config['caldata'])[-1], 'Calibration reference file')}
        expected = fits.header.Header()
        for key, val in keys.items():
            expected.set(key, val[0], val[1])

        # blank header for testing
        test = fits.header.Header()

        # update, check against expected
        util.add_calfac_keys(test, config)
        assert test == expected

        # check various procstat conditions
        # should be LEVEL_3, unless it was already LEVEL_4
        test = fits.header.Header()
        test['PROCSTAT'] = 'LEVEL_2'
        util.add_calfac_keys(test, config)
        assert test['PROCSTAT'] == 'LEVEL_3'

        test = fits.header.Header()
        test['PROCSTAT'] = 'LEVEL_3'
        util.add_calfac_keys(test, config)
        assert test['PROCSTAT'] == 'LEVEL_3'

        test = fits.header.Header()
        test['PROCSTAT'] = 'LEVEL_4'
        util.add_calfac_keys(test, config)
        assert test['PROCSTAT'] == 'LEVEL_4'

    def test_add_phot_keys(self):
        # simulated data
        hdul = resources.forcast_data()
        header = hdul[0].header
        image = hdul[0].data
        variance = hdul[1].data

        # get config from data header
        config = pipecal_config(header)

        # photometry data structure
        phot = pipecal_photometry(image, variance)

        # blank header for testing
        test = fits.header.Header()

        # update with no config or srcpos
        util.add_phot_keys(test, phot)
        for entry in phot:
            if isinstance(entry['value'], list):
                assert test[entry['key']] == entry['value'][0]
                assert test[entry['key'] + 'E'] == entry['value'][1]
            else:
                assert test[entry['key']] == entry['value']
        assert 'SRCPOSX' not in test
        assert 'LAMREF' not in test

        # update with config and srcpos
        test = fits.header.Header()
        util.add_phot_keys(test, phot, config, srcpos=(10, 10))
        assert 'STAPFLX' in test
        assert test['SRCPOSX'] == 10
        assert test['SRCPOSY'] == 10
        assert test['LAMREF'] == config['wref']
        assert test['MODLFLX'] == config['std_flux']
        assert test['REFSTD1'] == \
               config['filterdef_file'].partition(config['caldata'])[-1]

        # check for avgcalfc
        assert test['AVGCALFC'] == config['avgcalfc']
        assert test['AVGCALER'] == config['avgcaler']
        assert test['AVGCALFC'] != config['calfac']
        assert test['AVCLFILE'] == \
               config['avgcal_file'].partition(config['caldata'])[-1]

        # remove a couple keys from config
        del config['std_flux']
        del config['filterdef_file']
        test = fits.header.Header()
        util.add_phot_keys(test, phot, config)
        assert 'STAPFLX' in test
        assert test['LAMREF'] == config['wref']
        assert 'MODLFLX' not in test
        assert 'MODLFLXE' not in test
        assert 'REFSTD1' not in test
        assert 'REFSTD2' not in test
        assert 'REFSTD3' not in test

        # update without phot -- config keys only
        test = fits.header.Header()
        util.add_phot_keys(test, None, config)
        assert 'STAPFLX' not in test
        assert test['LAMREF'] == config['wref']

    def test_get_fluxcal_factor(self):
        # simulated data
        hdul = resources.forcast_data()
        header = hdul[0].header

        # get config from data header
        config = pipecal_config(header)

        # blank header for testing
        test = fits.header.Header()

        # no update or history
        cf, ecf = util.get_fluxcal_factor(test, config)
        assert len(test) == 0
        assert cf == config['calfac']
        assert ecf == config['ecalfac']

        # history without update -- no change
        test = fits.header.Header()
        util.get_fluxcal_factor(test, config, write_history=True)
        assert len(test) == 0

        # update without history -- calfac added, no history
        test = fits.header.Header()
        util.get_fluxcal_factor(test, config, update=True)
        assert test['CALFCTR'] == config['calfac']
        assert 'HISTORY' not in test

        # both update and history
        test = fits.header.Header()
        util.get_fluxcal_factor(test, config, update=True,
                                write_history=True)
        assert test['CALFCTR'] == config['calfac']
        assert 'Flux calibration' in str(test['HISTORY'])

        # missing calfac, without history
        test = fits.header.Header()
        del config['calfac']
        cf, ecf = util.get_fluxcal_factor(test, config, update=True)
        assert cf is None
        assert ecf is None
        assert 'CALFCTR' not in test
        assert 'HISTORY' not in test

        # missing calfac, with history
        test = fits.header.Header()
        cf, ecf = util.get_fluxcal_factor(test, config, update=True,
                                          write_history=True)
        assert cf is None
        assert ecf is None
        assert 'CALFCTR' not in test
        assert 'No reference flux calibration' in str(test['HISTORY'])
        assert 'for SPECTEL=' in str(test['HISTORY'])

        # missing calfac and spectel, with history
        test = fits.header.Header()
        del config['spectel']
        util.get_fluxcal_factor(test, config, update=True,
                                write_history=True)
        assert 'No reference flux calibration' in str(test['HISTORY'])
        assert 'for SPECTEL=' not in str(test['HISTORY'])

    def test_apply_fluxcal(self):
        # simulated data
        hdul = resources.forcast_data()
        header = hdul[0].header
        image = hdul[0].data
        variance = hdul[1].data
        cov = variance.copy()

        # get config from data header
        config = pipecal_config(header)

        # blank header for testing
        test = fits.header.Header()

        # with calfac, no variance or cov, write history
        c_image = util.apply_fluxcal(image, test, config, write_history=True)
        assert 'Flux calibration' in str(test['HISTORY'])
        assert test['CALFCTR'] == config['calfac']
        # one returned value
        assert not isinstance(c_image, tuple)
        assert np.allclose(c_image, image / config['calfac'])

        # with calfac, image + variance
        # no history this time
        test = fits.header.Header()
        c_image, c_var = util.apply_fluxcal(image, test, config,
                                            variance=variance,
                                            write_history=False)
        assert 'HISTORY' not in test
        assert test['CALFCTR'] == config['calfac']
        assert np.allclose(c_image, image / config['calfac'])
        assert np.allclose(c_var, variance / config['calfac']**2)

        # with calfac, image + var + cov
        c_image, c_var, c_cov = util.apply_fluxcal(image, test, config,
                                                   variance=variance,
                                                   covariance=cov)
        assert np.allclose(c_image, image / config['calfac'])
        assert np.allclose(c_var, variance / config['calfac']**2)
        assert np.allclose(c_cov, cov / config['calfac']**2)

        # with calfac, image + cov only
        # 3 returned values always if covariance is specified
        c_image, c_var, c_cov = util.apply_fluxcal(image, test, config,
                                                   covariance=cov)
        assert np.allclose(c_image, image / config['calfac'])
        assert c_var is None
        assert np.allclose(c_cov, cov / config['calfac']**2)

        # no calfac
        test = fits.header.Header()
        del config['calfac']
        c_image, c_var, c_cov = util.apply_fluxcal(
            image, test, config, variance=variance,
            covariance=cov,)
        assert np.allclose(c_image, image)
        assert np.allclose(c_var, variance)
        assert np.allclose(c_cov, cov)
        assert 'CALFCTR' not in test

    def test_get_tellcor_factor(self):
        # simulated data
        hdul = resources.forcast_data()
        header = hdul[0].header

        # get config from data header
        config = pipecal_config(header)

        # header for testing (needs alt/za, so make copy)
        test = header.copy()

        # expected value
        ref_r = pipecal_rratio(45, 41, 45, 41,
                               config['rfit_am']['coeff'],
                               config['rfit_alt']['coeff'],
                               pwv=False)
        obs_r = pipecal_rratio(util.average_za(header),
                               util.average_alt(header),
                               45, 41,
                               config['rfit_am']['coeff'],
                               config['rfit_alt']['coeff'],
                               pwv=False)
        expected = ref_r / obs_r

        # no update
        cf = util.get_tellcor_factor(test, config)
        assert 'TELCORR' not in test
        assert 'HISTORY' not in test
        assert cf == expected

        # with update
        test = header.copy()
        util.get_tellcor_factor(test, config, update=True)
        assert test['TELCORR'] == expected
        assert 'Telluric correction factor' in str(test['HISTORY'])
        assert 'altitude=' in str(test['HISTORY'])

        # use pwv
        ref_r = pipecal_rratio(45., 7.3, 45., 7.3,
                               config['rfit_am']['coeff'],
                               config['rfit_pwv']['coeff'],
                               pwv=True)
        obs_r = pipecal_rratio(util.average_za(header),
                               util.average_pwv(header),
                               45., 7.3,
                               config['rfit_am']['coeff'],
                               config['rfit_pwv']['coeff'],
                               pwv=True)
        expected = ref_r / obs_r

        test = header.copy()
        util.get_tellcor_factor(test, config, update=True, use_wv=True)
        assert test['TELCORR'] == expected
        assert 'Telluric correction factor' in str(test['HISTORY'])
        assert 'PWV=' in str(test['HISTORY'])

    def test_apply_tellcor(self):
        # simulated data
        hdul = resources.forcast_data()
        header = hdul[0].header
        image = hdul[0].data
        variance = hdul[1].data
        cov = variance.copy()

        # get config from data header
        config = pipecal_config(header)

        # header for testing (needs alt/za, so make copy)
        test = header.copy()

        # expected value
        expected = util.get_tellcor_factor(header.copy(), config)

        # no variance or cov
        c_image = util.apply_tellcor(image, test, config)
        assert 'Telluric correction' in str(test['HISTORY'])
        assert test['TELCORR'] == expected
        # one returned value
        assert not isinstance(c_image, tuple)
        assert np.allclose(c_image, image * expected)

        # with image + variance
        c_image, c_var = util.apply_tellcor(image, test, config,
                                            variance=variance)
        assert np.allclose(c_image, image * expected)
        assert np.allclose(c_var, variance * expected**2)

        # with image + var + cov
        c_image, c_var, c_cov = util.apply_tellcor(image, test, config,
                                                   variance=variance,
                                                   covariance=cov)
        assert np.allclose(c_image, image * expected)
        assert np.allclose(c_var, variance * expected**2)
        assert np.allclose(c_cov, cov * expected**2)

        # with image + cov only
        # 3 returned values always if covariance is specified
        c_image, c_var, c_cov = util.apply_tellcor(image, test, config,
                                                   covariance=cov)
        assert np.allclose(c_image, image * expected)
        assert c_var is None
        assert np.allclose(c_cov, cov * expected**2)

        # raises error if alt/za not available
        del test['ZA_START']
        del test['ZA_END']
        with pytest.raises(PipeCalError) as err:
            util.apply_tellcor(image, test, config)
        assert 'Response data not found' in str(err)

    def test_run_phot(self, capsys, mocker):
        hdul = resources.forcast_data()
        header = hdul[0].header
        image = hdul[0].data
        variance = hdul[1].data
        config = pipecal_config(header)

        # default run, with source position and model flux
        test = header.copy()
        util.run_photometry(image, test, variance, config)
        capt = capsys.readouterr()
        assert 'Source Flux' in capt.out
        assert 'Model Flux' in capt.out

        # run with minimal config: should work for basic photometry
        test = header.copy()
        util.run_photometry(image, test, variance, {})
        capt = capsys.readouterr()
        assert 'Source Flux' in capt.out
        assert 'No model flux' in capt.err

        # run on calibrated file: reports a percent difference
        test = header.copy()
        cal = util.apply_fluxcal(image, test, config)
        util.run_photometry(cal, test, variance, config)
        capt = capsys.readouterr()
        assert 'Source Flux' in capt.out
        assert 'Model Flux' in capt.out
        assert 'Percent difference' in capt.out

        # mock an error in photometry
        mocker.patch('sofia_redux.calibration.pipecal_util.add_phot_keys',
                     return_value=None)
        test = header.copy()
        util.run_photometry(image, test, variance, config)
        capt = capsys.readouterr()
        assert 'Source Flux' not in capt.out
        assert 'Photometry failed' in capt.err

        # mock an error in modlflux key, and a bad flux value
        # (add flux keys but not std. flux)
        def mock_keys(hdr, *args, **kwargs):
            hdr['STAPFLX'] = -10.0
            hdr['STAPFLXE'] = 1.0
            hdr['STCENTX'] = 10.0
            hdr['STCENTY'] = 10.0
        mocker.patch('sofia_redux.calibration.pipecal_util.add_phot_keys',
                     mock_keys)
        test = header.copy()
        util.run_photometry(image, test, variance, config)
        capt = capsys.readouterr()
        assert 'Source Flux' in capt.out
        assert 'Model Flux' not in capt.out
        assert 'Bad flux; not adding REFCALFC' in capt.err
