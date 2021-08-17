# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import shutil

import numpy as np
import pytest

import sofia_redux.calibration
from sofia_redux.calibration.pipecal_config \
    import pipecal_config, read_respfile
from sofia_redux.calibration.pipecal_error import PipeCalError
from sofia_redux.calibration.tests import resources


class TestConfig(object):

    def test_respfile(self, capsys):
        """Test read_respfile"""

        cwd = os.path.dirname(os.path.realpath(__file__))
        fname = f'{cwd}/../data/forcast/response/rfit_alt_single_20130702.txt'
        spectel = 'FOR_F054'

        # correct values
        altmin = 35.0
        altmax = 45.0
        altref = 41.0
        zamin = 30.0
        zamax = 70.0
        zaref = 45.0
        respref = 214.36301

        fields = ['altwvref', 'altwvrange', 'zaref', 'zarange', 'respref']
        values = [altref, [altmin, altmax], zaref, [zamin, zamax], respref]

        # check correct values
        response = read_respfile(fname, spectel)
        rtol = 1e-2
        for field, value in zip(fields, values):
            assert np.allclose(response[field], value, rtol=rtol)

        correct_coeffs = [0.99990758, 0.0033000098, -5.0777290e-05,
                          -6.7140983e-05, -3.2985944e-06, 3.8826696e-06]
        assert np.allclose(response['coeff'], correct_coeffs, rtol=rtol)

    def test_respfile_errors(self, capsys, tmpdir):
        """Test various error conditions in read_respfile"""
        spectel = 'FOR_F054'

        # check missing file
        with pytest.raises(PipeCalError):
            read_respfile('does_not_exist.txt', spectel)
        capt = capsys.readouterr()
        assert 'Cannot open' in capt.err

        # should look like:
        # ALTMIN=35.0 ALTMAX=45.0 ALTREF=41.0
        # ZAMIN=30.0 ZAMAX=70.0 ZAREF=45.0
        # 5.35200  FOR_F054 214.36301 0.99990758 0.0033000098 ...

        # check entirely bad response file
        badfile = tmpdir.join('badfile.txt')
        badfile.write('BADVAL')
        with pytest.raises(PipeCalError):
            read_respfile(str(badfile), spectel)
        capt = capsys.readouterr()
        assert 'Could not read reference values' in capt.err

        # check missing alt/za values
        badfile = tmpdir.join('badfile2.txt')
        badfile.write('# ALTMIN=35.0 ALTMAX=45.0')
        with pytest.raises(PipeCalError):
            read_respfile(str(badfile), spectel)
        capt = capsys.readouterr()
        assert 'Could not read reference values' in capt.err

        # check missing alt/za values
        badfile = tmpdir.join('badfile3.txt')
        badfile.write('# ALTMIN=35.0 ALTMAX=45.0 ALTREF=41.0\n'
                      '# ZAMIN=30.0 ZAREF=45.0')
        with pytest.raises(PipeCalError):
            read_respfile(str(badfile), spectel)
        capt = capsys.readouterr()
        assert 'Could not read reference values' in capt.err

        # check missing column
        badfile = tmpdir.join('badfile4.txt')
        badfile.write('# ALTMIN=35.0 ALTMAX=45.0 ALTREF=41.0\n'
                      '# ZAMIN=30.0 ZAMAX=70.0 ZAREF=45.0\n'
                      '1 2 3')
        with pytest.raises(PipeCalError):
            read_respfile(str(badfile), spectel)

        # check no matching spectel: returns empty, no error
        badfile = tmpdir.join('badfile5.txt')
        badfile.write('# ALTMIN=35.0 ALTMAX=45.0 ALTREF=41.0\n'
                      '# ZAMIN=30.0 ZAMAX=70.0 ZAREF=45.0\n'
                      '1 2 3 4 5 6')
        result = read_respfile(str(badfile), spectel)
        assert len(result) == 0

        # check bad response reference
        badfile = tmpdir.join('badfile6.txt')
        badfile.write('# ALTMIN=35.0 ALTMAX=45.0 ALTREF=41.0\n'
                      '# ZAMIN=30.0 ZAMAX=70.0 ZAREF=45.0\n'
                      '1 {} BAD 1 2 3'.format(spectel))
        with pytest.raises(PipeCalError):
            read_respfile(str(badfile), spectel)

        # check bad coefficients
        badfile = tmpdir.join('badfile7.txt')
        badfile.write('# ALTMIN=35.0 ALTMAX=45.0 ALTREF=41.0\n'
                      '# ZAMIN=30.0 ZAMAX=70.0 ZAREF=45.0\n'
                      '1 {} 3 a b c'.format(spectel))
        with pytest.raises(PipeCalError):
            read_respfile(str(badfile), spectel)

    def test_config(self):
        """Test pipecal_config"""

        # data path
        cwd = os.path.dirname(
            os.path.realpath(sofia_redux.calibration.__file__))
        caldata = os.path.join(cwd, 'data', '')

        # test data
        hdul = resources.forcast_data()
        header = hdul[0].header

        # assemble the expected descriptive keys for the config
        correct = dict()
        correct['caldata'] = caldata
        correct['date'] = 20181231
        correct['instrument'] = 'FORCAST'
        correct['spectel'] = 'FOR_F197'
        correct['altcfg1'] = 0
        correct['object'] = 'ALPHABOO'

        correct['filterdef_file'] = \
            os.path.join(caldata, 'forcast', 'filter_def',
                         'filter_def_20160125.txt')
        correct['refcal_file'] = \
            os.path.join(caldata, 'forcast', 'ref_calfctr',
                         'refcalfac_20190710.txt')
        correct['avgcal_file'] = \
            os.path.join(caldata, 'forcast', 'ref_calfctr',
                         'refcalfac_master.txt')
        correct['rfitam_file'] = \
            os.path.join(caldata, 'forcast', 'response',
                         'rfit_am_single_20160127.txt')
        correct['rfitalt_file'] = \
            os.path.join(caldata, 'forcast', 'response',
                         'rfit_alt_single_20160127.txt')
        correct['rfitpwv_file'] = \
            os.path.join(caldata, 'forcast', 'response',
                         'rfit_pwv_single_20160127.txt')
        correct['stdeflux_file'] = \
            os.path.join(caldata, 'forcast', 'standard_flux',
                         'model_err_20150515.txt')

        correct['aprad'] = 12.0
        correct['bgin'] = 15.0
        correct['bgout'] = 25.0
        correct['fitsize'] = 138
        correct['fwhm'] = 5.0
        correct['runits'] = 'Me/s'

        # generate actual config
        test_config = pipecal_config(header)

        for key in correct:
            assert correct[key] == test_config[key]

        # also check that other keys are present, without checking values
        # that will depend on current config
        extra_keys = ['rfit_am', 'rfit_alt', 'rfit_pwv',
                      'wref', 'lpivot', 'color_corr', 'calfac', 'ecalfac',
                      'avgcalfc', 'avgcaler', 'std_eflux', 'std_scale']
        for key in extra_keys:
            assert key in test_config

    def test_config_forcast(self):
        # test forcast data configuration
        hdul = resources.forcast_data()
        header = hdul[0].header
        header['SPECTEL1'] = 'FOR_F197'
        header['SPECTEL2'] = 'FOR_F371'

        # altcfg 0 (single mode), SW camera
        header['INSTCFG'] = 'IMAGING_SWC'
        header['DICHROIC'] = 'Mirror (swc)'
        header['DETCHAN'] = 'SW'
        result = pipecal_config(header)
        assert result['instrument'] == 'FORCAST'
        assert result['spectel'] == 'FOR_F197'
        assert result['altcfg1'] == 0

        # altcfg 1 (dual mode, Barr 2), SW camera
        header['INSTCFG'] = 'IMAGING_DUAL'
        header['DICHROIC'] = 'Dichroic'
        result = pipecal_config(header)
        assert result['spectel'] == 'FOR_F197'
        assert result['altcfg1'] == 1

        # altcfg 2 (dual mode, Barr 3), SW camera
        header['INSTCFG'] = 'IMAGING_DUAL'
        header['DICHROIC'] = 'Barr #3'
        result = pipecal_config(header)
        assert result['spectel'] == 'FOR_F197'
        assert result['altcfg1'] == 2

        # altcfg 0 (single mode), LW camera
        header['INSTCFG'] = 'IMAGING_LWC'
        header['DICHROIC'] = 'Open(lwc)'
        header['DETCHAN'] = 'LW'
        result = pipecal_config(header)
        assert result['instrument'] == 'FORCAST'
        assert result['spectel'] == 'FOR_F371'
        assert result['altcfg1'] == 0

        # altcfg 1 (dual mode, Barr 2), LW camera
        header['INSTCFG'] = 'IMAGING_DUAL'
        header['DICHROIC'] = 'Dichroic'
        result = pipecal_config(header)
        assert result['spectel'] == 'FOR_F371'
        assert result['altcfg1'] == 1

        # altcfg 2 (dual mode, Barr 3), LW camera
        header['INSTCFG'] = 'IMAGING_DUAL'
        header['DICHROIC'] = 'Barr #3'
        result = pipecal_config(header)
        assert result['spectel'] == 'FOR_F371'
        assert result['altcfg1'] == 2

    def test_config_hawc(self):
        # test hawc data configuration
        hdul = resources.hawc_pol_data()
        header = hdul[0].header
        header['SPECTEL1'] = 'HAW_D'

        # altcfg 0: chop/nod, with hwp
        header['INSTMODE'] = 'C2N (NMC)'
        header['SPECTEL2'] = 'HAW_HWP_D'
        result = pipecal_config(header)
        assert result['instrument'] == 'HAWC_PLUS'
        assert result['spectel'] == 'HAW_D'
        assert result['altcfg1'] == 0

        # altcfg 1: chop/nod, without hwp
        header['SPECTEL2'] = 'HAW_HWP_Open'
        result = pipecal_config(header)
        assert result['altcfg1'] == 1

        # altcfg 2: scan, with hwp
        hdul = resources.hawc_im_data()
        header = hdul[0].header
        header['INSTMODE'] = 'OTFMAP'
        header['SPECTEL1'] = 'HAW_D'
        header['SPECTEL2'] = 'HAW_HWP_D'
        result = pipecal_config(header)
        assert result['instrument'] == 'HAWC_PLUS'
        assert result['spectel'] == 'HAW_D'
        assert result['altcfg1'] == 2

        # altcfg 1: chop/nod, without hwp
        header['SPECTEL2'] = 'HAW_HWP_Open'
        result = pipecal_config(header)
        assert result['altcfg1'] == 3

    def test_config_flitecam(self):
        # test flitecam data configuration

        # altcfg 0: flitecam
        hdul = resources.flitecam_data()
        header = hdul[0].header
        result = pipecal_config(header)
        assert result['instrument'] == 'FLITECAM'
        assert result['spectel'] == header['SPECTEL1']
        assert result['altcfg1'] == 0

        # altcfg 1: flipo
        hdul = resources.flipo_data()
        header = hdul[0].header
        result = pipecal_config(header)
        assert result['instrument'] == 'FLITECAM'
        assert result['spectel'] == header['SPECTEL1']
        assert result['altcfg1'] == 1

    def test_unsupported(self, capsys):
        # unknown instrument
        hdul = resources.basic_data()
        header = hdul[0].header
        result = pipecal_config(header)
        capt = capsys.readouterr()
        assert 'Unsupported instrument' in capt.err
        assert 'spectel' not in result

    def test_config_errors(self, capsys, tmpdir, mocker):
        """Test various error conditions in pipecal_config"""
        # data path
        cwd = os.path.dirname(os.path.realpath(__file__))
        caldata = os.path.join(os.path.dirname(cwd), 'data', '')

        # test data
        hdul = resources.forcast_data()
        header = hdul[0].header

        # test missing keywords
        bad = header.copy()
        del bad['SPECTEL1']
        test_config = pipecal_config(bad)
        assert len(test_config) == 0
        capt = capsys.readouterr()
        assert 'Missing required keywords' in capt.err

        # test bad date (defaults to 99999999)
        bad = header.copy()
        bad['DATE-OBS'] = 'BADVAL-BADVAL-BADVAL'
        test_config = pipecal_config(bad)
        assert test_config['date'] == 99999999

        # test missing calibration default file
        # mock the cal path with the tmpdir
        mocker.patch('sofia_redux.calibration.pipecal_config._get_cal_path',
                     return_value=str(tmpdir))
        pipecal_config(header)
        capt = capsys.readouterr()
        assert 'Calibration default file' in capt.err
        assert 'does not exist' in capt.err

        # make a bad default file
        caldefault = tmpdir.join('caldefault.txt')
        caldefault.write('# test\n1 2 3 4 5\n')
        assert os.path.isfile(caldefault)
        pipecal_config(header)
        capt = capsys.readouterr()
        assert 'Calibration default file' in capt.err
        assert 'is poorly formatted' in capt.err

        # make a good default file, no match for data
        caldefault.write('19990101 0 fdef.txt  merr.txt  rfcal.txt  '
                         'avcal.txt  rf1.txt  rf2.txt  rf3.txt\n'
                         '19990101 1 fdef.txt  merr.txt  rfcal.txt  '
                         'avcal.txt  rf1.txt  rf2.txt  rf3.txt\n')
        assert os.path.isfile(caldefault)
        pipecal_config(header)
        capt = capsys.readouterr()
        assert 'No pipecal data found for date' in capt.err

        # make a good default file, still missing standards file
        caldefault.write('99999999 0 fdef.txt  merr.txt  rfcal.txt  '
                         'avcal.txt  rf1.txt  rf2.txt  rf3.txt\n'
                         '99999999 1 fdef.txt  merr.txt  rfcal.txt  '
                         'avcal.txt  rf1.txt  rf2.txt  rf3.txt\n')
        pipecal_config(header)
        capt = capsys.readouterr()
        assert 'Standards default file' in capt.err
        assert 'does not exist' in capt.err

        # make a bad standards file
        stddefault = tmpdir.join('stddefault.txt')
        stddefault.write('# test\n1 2 3 4 5\n')
        assert os.path.isfile(stddefault)
        pipecal_config(header)
        capt = capsys.readouterr()
        assert 'Standards default file' in capt.err
        assert 'is incorrectly formatted' in capt.err

        # copy in a filter definition
        fdef = os.path.join(caldata, 'forcast', 'filter_def',
                            'filter_def_20160125.txt')
        shutil.copyfile(fdef, str(tmpdir.join('fdef.txt')))

        # make a good standards file, no stdflux file matching
        # (no error)
        stddefault.write('# test\n'
                         '19990101 0 ALPHABOO .\n'
                         '19990101 1 ALPHABOO .\n')
        result = pipecal_config(header)
        assert 'stdflux' not in result
        capt = capsys.readouterr()
        assert capt.err == ''

        # add a stdflux file that doesn't exist
        # (same effect)
        stddefault.write('99999999 0 ALPHABOO stdflux.txt\n'
                         '99999999 1 ALPHABOO stdflux.txt\n')
        result = pipecal_config(header)
        assert 'std_flux' not in result
        capt = capsys.readouterr()
        assert capt.err == ''

        # make a bad file
        stdflux = tmpdir.join('stdflux.txt')
        stdflux.write('# 1\n# 2\n# 3\n# 4\n# 5\n# 6\nBADVAL BADVAL\n')
        assert os.path.isfile(str(stdflux))
        pipecal_config(header)
        capt = capsys.readouterr()
        assert 'Standard flux file' in capt.err
        assert 'poorly formatted' in capt.err

        # make a good file, no match
        stdflux.write('# 1\n# 2\n# 3\n# 4\n# 5\n# 6\n'
                      '0 34.6769 2 3 4 5 6 7 8 9 55.0 11\n'
                      '0 37.1236 2 3 4 5 6 7 8 9 65.0 11\n'
                      )
        pipecal_config(header)
        capt = capsys.readouterr()
        assert 'Standard flux not found for wavelength' in capt.err

        # make a match, but bad flux value
        stdflux.write('# 1\n# 2\n# 3\n# 4\n# 5\n# 6\n'
                      '0 19.6993 2 3 4 5 6 7 8 9 A 11\n'
                      '0 37.1236 2 3 4 5 6 7 8 9 65.0 11\n'
                      )
        pipecal_config(header)
        capt = capsys.readouterr()
        assert 'Bad standard flux value' in capt.err

        # good flux value
        stdflux.write('# 1\n# 2\n# 3\n# 4\n# 5\n# 6\n'
                      '0 19.6993 2 3 4 5 6 7 8 9 55.0 11\n'
                      '0 37.1236 2 3 4 5 6 7 8 9 65.0 11\n'
                      )
        result = pipecal_config(header)
        capt = capsys.readouterr()
        assert capt.err == ''
        assert result['std_flux'] == 55.0

        # test both kinds of refcal files
        for key, fname in [('calfac', 'rfcal.txt'), ('avgcalfc', 'avcal.txt')]:
            # make a bad refcalfac file
            refcal = tmpdir.join(fname)
            refcal.write('BADVAL')
            result = pipecal_config(header)
            capt = capsys.readouterr()
            assert 'Reference calibration factor file' in capt.err
            assert 'poorly formatted' in capt.err
            assert key not in result

            # good format, bad value
            refcal.write('FOR_F197 0 A 0.1\n'
                         'FOR_F197 1 B 0.1\n')
            result = pipecal_config(header)
            capt = capsys.readouterr()
            assert capt.err == ''
            assert key not in result

            # good format, good value
            refcal.write('FOR_F197 0 1.1 0.1\n'
                         'FOR_F197 1 2.1 0.1\n')
            result = pipecal_config(header)
            capt = capsys.readouterr()
            assert capt.err == ''
            assert result[key] == 1.1

        # make a bad model error file
        merr = tmpdir.join('merr.txt')
        merr.write('BADVAL')
        result = pipecal_config(header)
        capt = capsys.readouterr()
        assert 'Standard error file' in capt.err
        assert 'poorly formatted' in capt.err
        assert 'std_eflux' not in result

        # good format, bad value
        merr.write('#   stars\n'
                   'ALPHABOO   A 1.0\n'
                   'ALPHACET   B 1.0\n')
        result = pipecal_config(header)
        capt = capsys.readouterr()
        assert capt.err == ''
        assert 'std_eflux' not in result

        # good format, good value
        merr.write('#   stars\n'
                   'ALPHABOO   5.0 1.0\n'
                   'ALPHACET   5.0 1.0\n')
        result = pipecal_config(header)
        capt = capsys.readouterr()
        assert capt.err == ''
        assert result['std_eflux'] == 5.0

    @pytest.mark.parametrize('name',
                             ['ALPHABOO', 'ALPHA BOO', 'alpha Boo',
                              'alpha_boo', 'AL-phaboo', '  Al pha_Boo'])
    def test_object_name(self, name):
        hdul = resources.forcast_data()
        header = hdul[0].header

        # test variations on standard name
        header['OBJECT'] = name
        result = pipecal_config(header)
        assert 'std_flux' in result
        assert result['object'] == 'ALPHABOO'
