# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.flitecam.getcalpath import getcalpath

IMG_CALDEFAULT = [
    # imaging: all dates, all filters same values
    ('FLT_H', '2013-04-01T00:00:00', 'header_req_ima',
     'lc_coeffs_20140325'),
    ('FLT_Pa_cont', '2021-06-01T00:00:00', 'header_req_ima',
     'lc_coeffs_20140325'),
]

GRI_CALDEFAULT = [
    # early
    ('FLT_A1_LM', '2015-09-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'FLT_A1_LM_flat', 'flt_a1_lm_map2pos_ngc7027',
     '', 'lma'),
    ('FLT_A2_KL', '2015-09-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_kla_multinight', 'kla_map_arcs',
     'rspA2KL_flipo', 'kla'),
    ('FLT_A3_HW', '2015-09-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'mask_hwa', 'hwa_map_rou',
     'rspA3Hw_flipo', 'hwa'),
    ('FLT_B1_LM', '2015-09-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_lmb_multinight', 'lmb_map_arcs',
     'rspB1LM_flipo', 'lmb'),
    ('FLT_B2_HW', '2015-09-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'mask_hwb', 'hwb_map_rou',
     'rspB2Hw_flipo', 'hwb'),
    ('FLT_B3_J', '2015-09-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'mask_jb', 'jb_map_rou',
     'rspB3J_flipo', 'jb'),
    ('FLT_C2_LM', '2015-09-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_lmc_multinight', 'lmc_map_arcs',
     'rspC2LM_flipo', 'lmc'),
    ('FLT_C3_KW', '2015-09-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_kwc_multinight', 'kwc_map_rou',
     'rspC3Kw_flipo', 'kwc'),
    ('FLT_C4_H', '2015-09-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'mask_hc', 'hc_map_rou',
     'rspC4H_flipo', 'hc'),
    # OC3J
    ('FLT_A2_KL', '2015-11-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_kla_multinight', 'kla_map_arcs',
     'rspA2KL_fcam', 'kla'),
    ('FLT_B1_LM', '2015-11-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_lmb_multinight', 'lmb_map_arcs',
     'rspB1LM_fcam', 'lmb'),
    ('FLT_C2_LM', '2015-11-01T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_lmc_multinight', 'lmc_map_arcs',
     'rspC2LM_fcam', 'lmc'),
    # OC4J
    ('FLT_A1_LM', '2016-10-20T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'FLT_A1_LM_flat', 'flt_a1_lm_map2pos_ngc7027',
     'FC_GRI_A1LM_SS20_RSP', 'lma'),
    ('FLT_A2_KL', '2016-10-20T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_kla_multinight', 'kla_map_arcs',
     'FC_GRI_A2KL_SS20_RSP', 'kla'),
    ('FLT_A3_HW', '2016-10-20T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'mask_hwa', 'hwa_map_rou',
     'FC_GRI_A3Hw_SS20_RSP', 'hwa'),
    ('FLT_B1_LM', '2016-10-20T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_lmb_multinight', 'lmb_map_arcs',
     'FC_GRI_B1LM_SS20_RSP', 'lmb'),
    ('FLT_B2_HW', '2016-10-20T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'mask_hwb', 'hwb_map_rou',
     'FC_GRI_B2Hw_SS20_RSP', 'hwb'),
    ('FLT_B3_J', '2016-10-20T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'mask_jb', 'jb_map_rou',
     'FC_GRI_B3J_SS20_RSP', 'jb'),
    ('FLT_C2_LM', '2016-10-20T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_lmc_multinight', 'lmc_map_arcs',
     'FC_GRI_C2LM_SS20_RSP', 'lmc'),
    ('FLT_C3_KW', '2016-10-20T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_kwc_multinight', 'kwc_map_rou',
     'FC_GRI_C3Kw_SS20_RSP', 'kwc'),
    ('FLT_C4_H', '2016-10-20T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'mask_hc', 'hc_map_rou',
     'FC_GRI_C4H_SS20_RSP', 'hc'),
    # OC5L
    ('FLT_A2_KL', '2017-10-07T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_kla_multinight', 'kla_map_arcs',
     'FP_GRI_A2KL_SS20_RSP', 'kla'),
    ('FLT_B1_LM', '2017-10-07T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_lmb_multinight', 'lmb_map_arcs',
     'rspB1LM_flipo_v2', 'lmb'),
    ('FLT_C2_LM', '2017-10-07T00:00:00', 'header_req_gri',
     'lc_coeffs_20140325', 'flat_lmc_multinight', 'lmc_map_arcs',
     'rspC2LM_flipo_v2', 'lmc')
]


class TestGetcalpath(object):

    def make_header(self):
        # basic header
        header = fits.Header()
        header['SPECTEL1'] = 'test1'
        header['SPECTEL2'] = 'test2'
        return header

    def test_bad_header(self, capsys):
        # empty header - gets defaults for missing keys
        header = fits.Header()
        result = getcalpath(header)
        assert not result['error']
        assert result['spectel'] == ''
        assert result['slit'] == ''
        assert result['dateobs'] == 99999999

        header = self.make_header()
        result = getcalpath(header)
        assert not result['error']
        assert result['spectel'] == 'TEST1'
        assert result['slit'] == 'TEST2'
        assert result['dateobs'] == 99999999

    def test_missing_caldefault(self, capsys, mocker, tmpdir):
        mocker.patch('os.path.dirname', return_value=str(tmpdir))
        header = self.make_header()
        result = getcalpath(header)
        assert result['error']
        assert 'Problem reading default file' in capsys.readouterr().err

    def test_modes(self):
        header = self.make_header()

        header['INSTCFG'] = 'IMAGING'
        result = getcalpath(header)
        assert result['gmode'] == -1
        assert result['name'] == 'IMA'

        header['INSTCFG'] = 'SPECTROSCOPY'
        result = getcalpath(header)
        assert result['gmode'] == 1
        assert result['name'] == 'GRI'

        header['INSTCFG'] = 'GRISM'
        result = getcalpath(header)
        assert result['gmode'] == 1
        assert result['name'] == 'GRI'

    def test_date(self, capsys):
        header = self.make_header()
        header['DATE-OBS'] = '2014-02-21T10:55:26.809'
        result = getcalpath(header)
        assert result['dateobs'] == 20140221

        # test bad date -- defaults to latest (99999999)
        header['DATE-OBS'] = 'BADVAL-BADVAL-BADVAL'
        result = getcalpath(header)
        assert result['dateobs'] == 99999999

        # test really bad date -- beyond 99999999
        capsys.readouterr()
        header['DATE-OBS'] = '100000-02-04T06:18:22.791'
        result = getcalpath(header)
        capt = capsys.readouterr()
        assert 'Problem reading defaults' in capt.err
        assert result['error'] is True

    @pytest.mark.parametrize('spectel,date,kw,lin', IMG_CALDEFAULT)
    def test_img_modes(self, spectel, date, kw, lin):
        header = self.make_header()
        header['INSTCFG'] = 'IMAGING'
        header['SPECTEL1'] = spectel
        header['DATE-OBS'] = date

        result = getcalpath(header)
        assert kw in result['kwfile']
        assert lin in result['linfile']

        # all files should exist
        assert os.path.exists(result['kwfile'])
        assert os.path.exists(result['linfile'])

    @pytest.mark.parametrize('spectel,date,kw,lin,mask,wave,resp,line',
                             GRI_CALDEFAULT)
    def test_grism_modes(self, spectel, date, kw, lin, mask, wave, resp, line):
        header = self.make_header()
        header['INSTCFG'] = 'SPECTROSCOPY'
        header['SPECTEL1'] = spectel
        header['DATE-OBS'] = date

        # only one slit used for flitecam
        header['SPECTEL2'] = 'FLT_SS20'

        # expected values for resolution -- should be closeish
        resolution = {
            'FLT_A1_LM': 1075,
            'FLT_A2_KL': 1140,
            'FLT_A3_HW': 1290,
            'FLT_B1_LM': 1200,
            'FLT_B2_HW': 1320,
            'FLT_B3_J': 1425,
            'FLT_C2_LM': 1300,
            'FLT_C3_KW': 1390,
            'FLT_C4_H': 1400
        }

        result = getcalpath(header)

        # expected file names
        assert kw in result['kwfile']
        assert lin in result['linfile']
        assert mask in result['maskfile']
        assert wave in result['wavefile']
        assert resp in result['respfile']
        assert line in result['linefile']

        # all cal files should exist
        assert os.path.exists(result['kwfile'])
        assert os.path.exists(result['linfile'])
        assert os.path.exists(result['maskfile'])
        assert os.path.exists(result['wavefile'])
        assert os.path.exists(result['linefile'])

        # except possibly response
        if result['respfile'] != '':
            assert os.path.exists(result['respfile'])

        # if resolution is present, it should be near the
        # expected values
        if result['resolution'] > 0:
            assert np.allclose(result['resolution'],
                               resolution[spectel],
                               rtol=0.2)

        # all waveshifts should be pretty near zero if present
        if np.abs(result['waveshift']) > 0:
            assert np.allclose(result['waveshift'],
                               0, atol=0.001)

    def test_get_grism_cal(self, capsys, tmpdir):
        import sofia_redux.instruments.flitecam.getcalpath as gcp

        # missing caldefault file
        result = {'spectel': 'FLT_B3_J', 'slit': 'FLT_SS20',
                  'dateobs': 99999999}
        gcp._get_grism_cal('BADVAL', result)

        assert result['error']
        assert 'Problem reading default file' in capsys.readouterr().err

        # write a caldefault file with a good resolution number
        pathcal = str(tmpdir.join('grism'))
        os.makedirs(pathcal)
        cfile = tmpdir.join('grism', 'caldefault.txt')
        cfile.write('99999999 FLT_B3_J FLT_SS20 . . . . . 124\n')
        gcp._get_grism_cal(tmpdir, result)
        assert result['resolution'] == 124

        # now a bad resolution
        result = {'spectel': 'FLT_B3_J', 'slit': 'FLT_SS20',
                  'dateobs': 99999999}
        os.remove(cfile)
        cfile.write('99999999 FLT_B3_J FLT_SS20 . . . . . BAD\n')
        gcp._get_grism_cal(tmpdir, result)
        assert 'resolution' not in result
        assert result['error']
        assert 'Problem reading resolution' in capsys.readouterr().err
