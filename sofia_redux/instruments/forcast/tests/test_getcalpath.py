# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np
import pytest

import sofia_redux.instruments.forcast as drip
from sofia_redux.instruments.forcast.getcalpath import getcalpath

IMG_CALDEFAULT = [
    # img swc
    (0, 'FOR_F077', '2013-04-01T00:00:00',
     'early', 'OC1', 'OC1', 'pinhole'),
    (0, 'FOR_F077', '2013-06-01T00:00:00',
     'OC1B', 'OC1', 'OC1', 'pinhole'),
    (0, 'FOR_F077', '2013-11-01T00:00:00',
     'OC1DF', 'OC1', 'OC1', 'pinhole'),
    (0, 'FOR_F077', '2014-06-01T00:00:00',
     'OC2', 'OC2', 'OC2', 'pinhole'),
    (0, 'FOR_F077', '2015-06-01T00:00:00',
     'OC3', 'OC3', 'OC2', 'pinhole'),
    (0, 'FOR_F077', '2015-11-01T00:00:00',
     'OC3', 'OC3_2', 'OC3', 'pinhole'),
    (0, 'FOR_F077', '2016-02-01T00:00:00',
     'OC3', 'OC4', 'OC4A', 'pinhole'),
    (0, 'FOR_F077', '2016-07-01T00:00:00',
     'OC3', 'OC4', 'OC4G', 'pinhole'),
    (0, 'FOR_F077', '2016-09-01T00:00:00',
     'OC3', 'OC4', 'OC4G', 'pinhole'),
    (0, 'FOR_F077', '2017-08-02T00:00:00',
     'OC3', 'OC4', 'OC5J_FT425', 'pinhole'),
    (0, 'FOR_F077', '2017-08-03T00:00:00',
     'OC3', 'OC4', 'OC5J_FT426', 'pinhole'),
    (0, 'FOR_F077', '2017-08-06T00:00:00',
     'OC3', 'OC4', 'OC5J_FT427', 'pinhole'),
    (0, 'FOR_F077', '2017-08-07T00:00:00',
     'OC3', 'OC4', 'OC5J_FT428', 'pinhole'),
    (0, 'FOR_F077', '2017-09-01T00:00:00',
     'OC3', 'OC4', 'OC5K', 'pinhole'),
    (0, 'FOR_F077', '2018-09-01T00:00:00',
     'OC3', 'OC4', 'OC6J', 'pinhole'),
    (0, 'FOR_F077', '2019-07-01T00:00:00',
     'OC7', 'OC4', 'OC7D', 'SWC_2019'),
    # img lwc
    (1, 'FOR_F371', '2013-04-01T00:00:00',
     'early', 'OC1', 'OC1', 'pinhole'),
    (1, 'FOR_F371', '2013-06-01T00:00:00',
     'OC1B', 'OC1', 'OC1', 'pinhole'),
    (1, 'FOR_F371', '2013-11-01T00:00:00',
     'OC1DF', 'OC1', 'OC1', 'pinhole'),
    (1, 'FOR_F371', '2014-06-01T00:00:00',
     'OC2', 'OC2', 'OC2', 'pinhole'),
    (1, 'FOR_F371', '2015-06-01T00:00:00',
     'OC3', 'OC3', 'OC2', 'pinhole'),
    (1, 'FOR_F371', '2015-11-01T00:00:00',
     'OC3', 'OC3_2', 'OC3', 'pinhole'),
    (1, 'FOR_F371', '2016-02-01T00:00:00',
     'OC3', 'OC4', 'OC4A', 'pinhole'),
    (1, 'FOR_F371', '2016-07-01T00:00:00',
     'OC3', 'OC4', 'OC4A', 'pinhole'),
    (1, 'FOR_F371', '2016-09-01T00:00:00',
     'OC3', 'OC4', 'OC4A', 'pinhole'),
    (1, 'FOR_F371', '2017-08-02T00:00:00',
     'OC3', 'OC4', 'OC5', 'pinhole'),
    (1, 'FOR_F371', '2017-08-03T00:00:00',
     'OC3', 'OC4', 'OC5', 'pinhole'),
    (1, 'FOR_F371', '2017-08-06T00:00:00',
     'OC3', 'OC4', 'OC5', 'pinhole'),
    (1, 'FOR_F371', '2017-08-07T00:00:00',
     'OC3', 'OC4', 'OC5', 'pinhole'),
    (1, 'FOR_F371', '2017-09-01T00:00:00',
     'OC3', 'OC4', 'OC5', 'pinhole'),
    (1, 'FOR_F371', '2018-09-01T00:00:00',
     'OC3', 'OC4', 'OC6J', 'pinhole'),
    (1, 'FOR_F371', '2019-07-01T00:00:00',
     'OC7', 'OC4', 'OC7D', 'LWC_2019'),
]

GRI_CALDEFAULT = [
    # g063
    (0, 'FOR_G063', '2013-04-01T00:00:00', 'early', 'OC1', 'OC1', 'OC1B'),
    (0, 'FOR_G063', '2013-06-01T00:00:00', 'OC1B', 'OC1', 'OC1', 'OC1B'),
    (0, 'FOR_G063', '2013-11-01T00:00:00', 'OC1DF', 'OC1', 'OC1', 'OC1DF'),
    (0, 'FOR_G063', '2014-06-01T00:00:00', 'OC2', 'OC2', 'OC2', 'OC2'),
    (0, 'FOR_G063', '2014-06-04T00:00:00', 'OC2', 'OC2', 'OC2', 'OC2'),
    (0, 'FOR_G063', '2015-06-01T00:00:00', 'OC3', 'OC3', 'OC2', 'OC2'),
    (0, 'FOR_G063', '2015-11-01T00:00:00', 'OC3', 'OC3_2', 'OC3', 'OC2'),
    (0, 'FOR_G063', '2016-02-01T00:00:00', 'OC3', 'OC4', 'OC4A', 'OC2'),
    (0, 'FOR_G063', '2016-07-01T00:00:00', 'OC3', 'OC4', 'OC4G', 'OC2'),
    (0, 'FOR_G063', '2016-09-01T00:00:00', 'OC3', 'OC4', 'OC4G', 'OC2'),
    (0, 'FOR_G063', '2017-08-02T00:00:00', 'OC3', 'OC4', 'OC5J_FT425', 'OC2'),
    (0, 'FOR_G063', '2017-08-03T00:00:00', 'OC3', 'OC4', 'OC5J_FT426', 'OC2'),
    (0, 'FOR_G063', '2017-08-06T00:00:00', 'OC3', 'OC4', 'OC5J_FT427', 'OC2'),
    (0, 'FOR_G063', '2017-08-07T00:00:00', 'OC3', 'OC4', 'OC5J_FT428', 'OC2'),
    (0, 'FOR_G063', '2017-09-01T00:00:00', 'OC3', 'OC4', 'OC5K', 'OC2'),
    (0, 'FOR_G063', '2018-09-01T00:00:00', 'OC3', 'OC4', 'OC6J', 'OC2'),
    (0, 'FOR_G063', '2019-07-01T00:00:00', 'OC7', 'OC4', 'OC7D', 'OC7'),
    # g111
    (0, 'FOR_G111', '2013-04-01T00:00:00', 'early', 'OC1', 'OC1', 'OC1B'),
    (0, 'FOR_G111', '2013-06-01T00:00:00', 'OC1B', 'OC1', 'OC1', 'OC1B'),
    (0, 'FOR_G111', '2013-11-01T00:00:00', 'OC1DF', 'OC1', 'OC1', 'OC1DF'),
    (0, 'FOR_G111', '2014-06-01T00:00:00', 'OC2', 'OC2', 'OC2', 'OC2'),
    (0, 'FOR_G111', '2014-06-04T00:00:00', 'OC2', 'OC2', 'OC2', 'OC2'),
    (0, 'FOR_G111', '2015-06-01T00:00:00', 'OC3', 'OC3', 'OC2', 'OC2'),
    (0, 'FOR_G111', '2015-11-01T00:00:00', 'OC3', 'OC3_2', 'OC3', 'OC2'),
    (0, 'FOR_G111', '2016-02-01T00:00:00', 'OC3', 'OC4', 'OC4A', 'OC2'),
    (0, 'FOR_G111', '2016-07-01T00:00:00', 'OC3', 'OC4', 'OC4G', 'OC2'),
    (0, 'FOR_G111', '2016-09-01T00:00:00', 'OC3', 'OC4', 'OC4G', 'OC2'),
    (0, 'FOR_G111', '2017-08-02T00:00:00', 'OC3', 'OC4', 'OC5J_FT425', 'OC2'),
    (0, 'FOR_G111', '2017-08-03T00:00:00', 'OC3', 'OC4', 'OC5J_FT426', 'OC2'),
    (0, 'FOR_G111', '2017-08-06T00:00:00', 'OC3', 'OC4', 'OC5J_FT427', 'OC2'),
    (0, 'FOR_G111', '2017-08-07T00:00:00', 'OC3', 'OC4', 'OC5J_FT428', 'OC2'),
    (0, 'FOR_G111', '2017-09-01T00:00:00', 'OC3', 'OC4', 'OC5K', 'OC2'),
    (0, 'FOR_G111', '2018-09-01T00:00:00', 'OC3', 'OC4', 'OC6J', 'OC2'),
    (0, 'FOR_G111', '2019-07-01T00:00:00', 'OC7', 'OC4', 'OC7D', 'OC7'),
    # g227
    (1, 'FOR_G227', '2013-04-01T00:00:00', 'early', 'OC1', 'OC1', 'OC1B'),
    (1, 'FOR_G227', '2013-06-01T00:00:00', 'OC1B', 'OC1', 'OC1', 'OC1B'),
    (1, 'FOR_G227', '2013-11-01T00:00:00', 'OC1DF', 'OC1', 'OC1', 'OC1DF'),
    (1, 'FOR_G227', '2014-06-01T00:00:00', 'OC2', 'OC2', 'OC2', 'OC2'),
    (1, 'FOR_G227', '2015-06-01T00:00:00', 'OC3', 'OC3', 'OC2', 'OC2'),
    (1, 'FOR_G227', '2015-11-01T00:00:00', 'OC3', 'OC3_2', 'OC3', 'OC2'),
    (1, 'FOR_G227', '2016-02-01T00:00:00', 'OC3', 'OC4', 'OC4A', 'OC2'),
    (1, 'FOR_G227', '2016-07-01T00:00:00', 'OC3', 'OC4', 'OC4A', 'OC2'),
    (1, 'FOR_G227', '2016-09-01T00:00:00', 'OC3', 'OC4', 'OC4A', 'OC2'),
    (1, 'FOR_G227', '2017-08-02T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G227', '2017-08-03T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G227', '2017-08-06T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G227', '2017-08-07T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G227', '2017-09-01T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G227', '2018-09-01T00:00:00', 'OC3', 'OC4', 'OC6J', 'OC5'),
    (1, 'FOR_G227', '2019-07-01T00:00:00', 'OC7', 'OC4', 'OC7D', 'OC7'),
    # g329
    (1, 'FOR_G329', '2013-04-01T00:00:00', 'early', 'OC1', 'OC1', 'OC1B'),
    (1, 'FOR_G329', '2013-06-01T00:00:00', 'OC1B', 'OC1', 'OC1', 'OC1B'),
    (1, 'FOR_G329', '2013-11-01T00:00:00', 'OC1DF', 'OC1', 'OC1', 'OC1DF'),
    (1, 'FOR_G329', '2014-06-01T00:00:00', 'OC2', 'OC2', 'OC2', 'OC2'),
    (1, 'FOR_G329', '2015-06-01T00:00:00', 'OC3', 'OC3', 'OC2', 'OC2'),
    (1, 'FOR_G329', '2015-11-01T00:00:00', 'OC3', 'OC3_2', 'OC3', 'OC2'),
    (1, 'FOR_G329', '2016-02-01T00:00:00', 'OC3', 'OC4', 'OC4A', 'OC2'),
    (1, 'FOR_G329', '2016-07-01T00:00:00', 'OC3', 'OC4', 'OC4A', 'OC2'),
    (1, 'FOR_G329', '2016-09-01T00:00:00', 'OC3', 'OC4', 'OC4A', 'OC2'),
    (1, 'FOR_G329', '2017-08-02T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G329', '2017-08-03T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G329', '2017-08-06T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G329', '2017-08-07T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G329', '2017-09-01T00:00:00', 'OC3', 'OC4', 'OC5', 'OC5'),
    (1, 'FOR_G329', '2018-09-01T00:00:00', 'OC3', 'OC4', 'OC6J', 'OC5'),
    (1, 'FOR_G329', '2019-07-01T00:00:00', 'OC7', 'OC4', 'OC7D', 'OC7'),
]


class TestGetcalpath(object):

    def make_header(self):
        # basic header
        header = fits.Header()
        header['SPECTEL1'] = 'test1'
        header['SPECTEL2'] = 'test2'
        header['DETCHAN'] = 0
        header['SLIT'] = 'test3'
        return header

    def test_errors(self, capsys):
        header = fits.Header()
        result = getcalpath(header)
        assert result['error']
        assert 'Problem reading SPECTEL1' in capsys.readouterr().err

        header = self.make_header()
        result = getcalpath(header)
        assert not result['error']

    def test_spectel(self):
        header = self.make_header()
        header['DETCHAN'] = 0
        assert getcalpath(header)['spectel'] == header['SPECTEL1']
        header['DETCHAN'] = 1
        assert getcalpath(header)['spectel'] == header['SPECTEL2']
        header['DETCHAN'] = 'SW'
        assert getcalpath(header)['spectel'] == header['SPECTEL1']
        header['DETCHAN'] = 'LW'
        assert getcalpath(header)['spectel'] == header['SPECTEL2']

    def test_slit(self):
        header = self.make_header()
        slit1 = header.get('slit')
        header['SLIT'] = 'foo'
        result = getcalpath(header)
        assert result['slit'] == 'FOO' and result['slit'] != slit1

    def test_modes(self):
        header = self.make_header()
        header['DETCHAN'] = 0
        header['SLIT'] = 'FOR_LS24'
        spectels = ['FOR_G063', 'FOR_G111',
                    'FOR_G227', 'FOR_G329', 'NONE']
        modes = [2, 3, 4, 5, -1]

        names = ['G063', 'G111', 'G227', 'G329', 'IMG_SWC']
        for spectel, name, mode in zip(spectels, names, modes):
            header['SPECTEL1'] = spectel
            result = getcalpath(header)
            assert mode == result['gmode']
            assert name == result['name']

        header['DETCHAN'] = 1
        header['SLIT'] = 'NONE'
        names[-1] = 'IMG_LWC'
        for spectel, name, mode in zip(spectels, names, modes):
            header['SPECTEL2'] = spectel
            result = getcalpath(header)
            assert mode == result['gmode']
            assert name == result['name']

    def test_files(self, capsys):
        header = self.make_header()
        header['DETCHAN'] = 0
        header['SPECTEL1'] = 'FOR_F197'
        header['SPECTEL2'] = 'FOR_F371'
        header['DATE-OBS'] = '2000-02-04T06:18:22.791'
        old_file = getcalpath(header)['conffile']
        header['DATE-OBS'] = '2050-02-04T06:18:22.791'
        new_file = getcalpath(header)['conffile']
        assert len(new_file) != 0
        assert len(old_file) != 0
        assert old_file != new_file

        # test explicit pathcal
        capsys.readouterr()
        # bad
        pathcal = 'BADVAL'
        result = getcalpath(header, pathcal=pathcal)
        capt = capsys.readouterr()
        assert 'Problem reading default file' in capt.err
        assert result['conffile'] == ''
        # good
        pathcal = os.path.join(os.path.dirname(drip.__file__), 'data')
        pathcal_file = getcalpath(header, pathcal=pathcal)['conffile']
        assert pathcal_file == new_file

        # test bad date -- defaults to latest (99999999)
        header['DATE-OBS'] = 'BADVAL-BADVAL-BADVAL'
        baddate_file = getcalpath(header)['conffile']
        assert baddate_file == new_file

        # test really bad date -- beyond 99999999
        capsys.readouterr()
        header['DATE-OBS'] = '100000-02-04T06:18:22.791'
        result = getcalpath(header)
        capt = capsys.readouterr()
        assert 'Problem reading defaults' in capt.err
        assert result['conffile'] == ''

    @pytest.mark.parametrize('detchan,spectel,date,conf,kw,bad,pin',
                             IMG_CALDEFAULT)
    def test_img_modes(self, detchan, spectel, date, conf, kw, bad, pin):
        header = self.make_header()
        header['DETCHAN'] = detchan
        if detchan == 0:
            header['SPECTEL1'] = spectel
            camera = 'swc'
        else:
            header['SPECTEL2'] = spectel
            camera = 'lwc'
        header['DATE-OBS'] = date

        result = getcalpath(header)
        assert conf in result['conffile']
        assert kw in result['kwfile']
        assert pin in result['pinfile']

        # bad pix should say swc or lwc as appropriate
        assert camera in result['badfile']
        assert bad in result['badfile']

        # all files should exist
        assert os.path.exists(result['conffile'])
        assert os.path.exists(result['kwfile'])
        assert os.path.exists(result['badfile'])
        assert os.path.exists(result['pinfile'])

    @pytest.mark.parametrize('detchan,spectel,date,conf,kw,bad,wave',
                             GRI_CALDEFAULT)
    def test_grism_modes(self, detchan, spectel, date, conf, kw, bad, wave):
        header = self.make_header()
        header['DETCHAN'] = detchan
        if detchan == 0:
            header['SPECTEL1'] = spectel
            camera = 'swc'
        else:
            header['SPECTEL2'] = spectel
            camera = 'lwc'
        header['DATE-OBS'] = date

        # expected values for resolution -- should be closeish
        resolution = {
            'G063': {'LS24': 180, 'LS47': 130},
            'G111': {'LS24': 256, 'LS47': 151},
            'G227': {'LS24': 130, 'LS47': 124},
            'G329': {'LS24': 187, 'LS47': 152},
        }

        for slit in ['FOR_LS24', 'FOR_LS47']:
            # grism files usually contain these
            grism = spectel[4:]
            slitname = slit[4:]

            header['SLIT'] = slit

            result = getcalpath(header)
            assert conf in result['conffile']
            assert kw in result['kwfile']

            # bad pix should say swc or lwc as appropriate
            assert camera in result['badfile']
            assert bad in result['badfile']

            # no pinhole for grism modes
            assert result['pinfile'] == ''

            # order mask should contain grism and slit
            assert grism in result['maskfile']
            assert slitname in result['maskfile']

            # wavecal file should contain grism
            assert wave in result['wavefile']
            assert grism in result['wavefile']

            # response file should contain grism and slit if present
            if result['respfile'] != '':
                assert grism in result['respfile']
                assert slitname in result['respfile']
                assert os.path.exists(result['respfile'])

            # slit file should contain grism and slit if present
            if result['slitfile'] != '':
                assert grism in result['slitfile']
                assert slitname in result['slitfile']
                assert os.path.exists(result['slitfile'])

            # all other cal files should exist
            assert os.path.exists(result['conffile'])
            assert os.path.exists(result['kwfile'])
            assert os.path.exists(result['badfile'])
            assert os.path.exists(result['maskfile'])
            assert os.path.exists(result['wavefile'])

            # if resolution is present, it should be near the
            # expected values
            if result['resolution'] > 0:
                assert np.allclose(result['resolution'],
                                   resolution[grism][slitname],
                                   rtol=0.2)

    def test_get_grism_cal(self, capsys, tmpdir):
        import sofia_redux.instruments.forcast.getcalpath as gcp

        # missing caldefault file
        result = {'spectel': 'FOR_G063', 'slit': 'FOR_LS24',
                  'dateobs': 99999999}
        gcp._get_grism_cal('BADVAL', result)

        assert result['error']
        assert 'Problem reading default file' in capsys.readouterr().err

        # write a caldefault file with a good resolution number
        pathcal = str(tmpdir.join('grism'))
        os.makedirs(pathcal)
        cfile = tmpdir.join('grism', 'caldefault.txt')
        cfile.write('99999999 FOR_G063 FOR_LS24 . . 124 . . .\n')
        gcp._get_grism_cal(tmpdir, result)
        assert result['resolution'] == 124

        # now a bad resolution
        result = {'spectel': 'FOR_G063', 'slit': 'FOR_LS24',
                  'dateobs': 99999999}
        os.remove(cfile)
        cfile.write('99999999 FOR_G063 FOR_LS24 . . BAD . . .\n')
        gcp._get_grism_cal(tmpdir, result)
        assert 'resolution' not in result
        assert result['error']
        assert 'Problem reading resolution' in capsys.readouterr().err
