# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from astropy.io import fits

from sofia_redux.instruments.exes import makeflat as mf
from sofia_redux.instruments.exes.readhdr import readhdr


@pytest.fixture
def basic_params():
    nz, ny, nx = (3, 10, 15)
    cards = np.ones((nz, ny, nx))
    variance = np.ones((nz, ny, nx))
    header = readhdr(fits.Header({'NSPAT': nx, 'NSPEC': ny,
                                  'WAVENO0': 1210.0,
                                  'FLATTAMB': 290.0, 'FLATEMIS': 0.1}),
                     check_header=False)
    params = mf._check_inputs(cards, header, variance)
    return params


@pytest.fixture
def process_params():
    ny, nx = 20, 10
    black = np.full((ny, nx), 10.0)
    sky = np.full((ny, nx), 5.0)
    shiny = np.full((ny, nx), 6.0)
    dark = np.full((ny, nx), 3.0)
    cards = np.array([black, sky, shiny])
    var = np.full_like(cards, 2.0)
    params = {'nx': nx, 'ny': ny, 'cards': cards, 'variance': var,
              'black_frame': 0, 'sky_frame': 1, 'shiny_frame': 2,
              'dark': dark}
    return params


class TestMakeFlat(object):

    @pytest.mark.parametrize('cardmode', ['CARD', 'NONE'])
    def test_makeflat(self, mocker, basic_params, cardmode):
        basic_params['cardmode'] = cardmode
        m1 = mocker.patch.object(mf, '_check_inputs',
                                 return_value=basic_params)
        m2 = mocker.patch.object(mf, '_set_black_frame')
        m3 = mocker.patch.object(mf, '_set_shiny_and_sky_frames')
        m4 = mocker.patch.object(mf, '_set_process_type')
        m5 = mocker.patch.object(mf, '_check_saturation')
        m6 = mocker.patch.object(mf, '_process_cards')
        m7 = mocker.patch.object(
            mf, '_calculate_responsive_quantum_efficiency')
        m8 = mocker.patch.object(mf, '_undistort_flat')
        m9 = mocker.patch.object(mf, '_create_flat')

        mf.makeflat(1, 2, 3)

        for mock in [m1, m2, m3, m4, m5, m6, m7, m8]:
            assert mock.call_count == 1

        if cardmode == 'NONE':
            assert m9.call_count == 0
            assert np.allclose(basic_params['flat'], 1.0)
        else:
            assert m9.call_count == 1

    def test_makeflat_single(self, rdc_low_flat_hdul):
        # regression test for single-order synthetic data
        cards = rdc_low_flat_hdul[0].data
        header = rdc_low_flat_hdul[0].header
        variance = rdc_low_flat_hdul[1].data ** 2

        params = mf.makeflat(cards, header, variance)
        flat = params['flat']
        var = params['flat_variance']
        illum = params['illum']
        assert flat.shape == cards[0].shape
        assert np.allclose(np.mean(flat), 0.1619346)
        assert np.allclose(np.mean(var), 3.481595e-8)
        assert np.allclose(np.sum(illum), 321780)
        assert np.allclose(params['header']['BNU_T'], 118.05324)

    def test_makeflat_multi(self, rdc_high_low_flat_hdul):
        # regression test for multi-order synthetic data
        cards = rdc_high_low_flat_hdul[0].data
        header = rdc_high_low_flat_hdul[0].header
        variance = rdc_high_low_flat_hdul[1].data ** 2

        params = mf.makeflat(cards, header, variance)
        flat = params['flat']
        var = params['flat_variance']
        illum = params['illum']
        assert flat.shape == cards[0].shape
        assert np.allclose(np.mean(flat), 0.01745146)
        assert np.allclose(np.mean(var), 3.815235e-10)
        assert np.allclose(np.sum(illum), 686960)
        assert np.allclose(params['header']['BNU_T'], 112.930494)

    def test_blackbody_pnu(self):
        # values from IDL implementation:
        # c, h, kB are estimated so will be less accurate than this version
        expected = [1.4949697e14, 1.4895577e14, 1.4841625e14,
                    1.4787849e14, 1.4734258e14, 1.4680832e14,
                    1.4627583e14, 1.4574512e14, 1.4521610e14,
                    1.4468881e14]
        result = mf.blackbody_pnu(1210. + np.arange(10), 273.)
        assert np.allclose(result, expected, rtol=1e-4)

        expected = [1.0431983e15, 1.0413343e15, 1.0394676e15,
                    1.0375979e15, 1.0357247e15, 1.0338491e15,
                    1.0319706e15, 1.0300887e15, 1.0282046e15,
                    1.0263177e15]
        result = mf.blackbody_pnu(480. + np.arange(10), 260.)
        assert np.allclose(result, expected, rtol=1e-4)

    def test_bnu(self):
        # values from IDL implementation:
        # c, h, kB are estimated so will be less accurate than this version
        expected = [35.930885, 35.830395, 35.730103, 35.630009, 35.530151,
                    35.430481, 35.331024, 35.231792, 35.132755, 35.033928]
        result = mf.bnu(1210. + np.arange(10), 273.)
        assert np.allclose(result, expected, rtol=1e-3)

        expected = [99.462242, 99.491348, 99.519478, 99.546570, 99.572594,
                    99.597633, 99.621643, 99.644585, 99.666565, 99.687523]
        result = mf.bnu(480. + np.arange(10), 260.)
        assert np.allclose(result, expected, rtol=1e-3)

    def test_check_inputs(self, capsys):
        nz, ny, nx = (3, 10, 15)
        cards = np.ones((nz, ny, nx))
        variance = np.ones((nz, ny, nx))
        dark = np.zeros((ny, nx))
        header = readhdr(fits.Header({'NSPAT': nx, 'NSPEC': ny}),
                         check_header=False)

        # data/var mismatch
        with pytest.raises(ValueError) as err:
            mf._check_inputs(cards, header, variance[0])
        assert 'Card shape does not match variance' in str(err)

        # dark mismatch
        with pytest.raises(ValueError) as err:
            mf._check_inputs(cards, header, variance, dark=dark[0])
        assert 'Dark does not match' in str(err)

        # bad cardmode
        header['CARDMODE'] = 'NONE'
        with pytest.raises(ValueError) as err:
            mf._check_inputs(cards, header, variance, dark=dark)
        assert 'CARDMODE is unspecified' in str(err)

        # flatmode for camera: shiny
        header['CARDMODE'] = 'OTHER'
        header['INSTCFG'] = 'CAMERA'
        header['FRAMETIM'] = 0
        result = mf._check_inputs(cards, header, variance, dark=dark)
        assert result['cardmode'] == 'SHINY'
        assert 'Setting flatmode = shiny for camera' in capsys.readouterr().out

        # unless frames=2 => BLKSKY
        result = mf._check_inputs(cards[:2], header, variance[:2], dark=dark)
        assert result['cardmode'] == 'BLKSKY'

        # or frames=2 => BLK
        result = mf._check_inputs(cards[0], header, variance[0], dark=dark)
        assert result['cardmode'] == 'BLK'

        # for non-cross-dispersed: focal length is xdfl0, R is xdr
        assert result['focal_length'] == header['XDFL0']
        assert result['r_number'] == header['XDR']

        # maxval for empty gain/time is raw satval
        assert result['maxval'] == header['SATVAL']

        capsys.readouterr()

        # cross-dispersed
        header['INSTCFG'] = 'HIGH_MED'
        header['PAGAIN'] = 10.0
        header['FRAMETIM'] = 20.0
        result = mf._check_inputs(cards, header, variance, dark=dark)
        assert 'Setting flatmode = shiny' not in capsys.readouterr().out

        # for cross-dispersed: focal length is hrfl0, R is hrr
        assert result['focal_length'] == header['HRFL0']
        assert result['r_number'] == header['HRR']

        # maxval corrected by gain/time
        assert result['maxval'] == header['SATVAL'] / 200

        # bad frames
        bad = cards.copy()
        bad[0] = np.nan
        with pytest.raises(ValueError) as err:
            mf._check_inputs(bad, header, variance, dark=dark)
        assert 'Cannot proceed: Bad data' in str(err)

        bad = cards.copy()
        bad[1] = np.nan
        with pytest.raises(ValueError) as err:
            mf._check_inputs(bad, header, variance, dark=dark)
        assert 'Cannot proceed: Bad data' in str(err)

        capsys.readouterr()

        bad = cards.copy()
        bad[2] = np.nan
        result = mf._check_inputs(bad, header, variance, dark=dark)
        capt = capsys.readouterr()
        assert 'Bad data' in capt.out
        assert 'This is allowable' in capt.out
        assert result['cardmode'] == 'OTHER'

        header['CARDMODE'] = 'SHINY'
        result = mf._check_inputs(bad, header, variance, dark=dark)
        capt = capsys.readouterr()
        assert 'Bad data' in capt.err
        assert 'Changing mode to BLKSKY' in capt.out
        assert result['cardmode'] == 'BLKSKY'

    def test_set_black_frame(self, capsys):
        params = {'cardmode': 'BLK', 'card_means': [1, 2, 3], 'ncards': 3}

        # provided black frame

        # good
        mf._set_black_frame(params, 1)
        assert params['black_frame'] == 1

        # bad type
        with pytest.raises(ValueError) as err:
            mf._set_black_frame(params, 'bad')
        assert 'Cannot use black_frame=-1' in str(err)

        # bad number
        with pytest.raises(ValueError) as err:
            mf._set_black_frame(params, 3)
        assert 'Cannot use black_frame=3' in str(err)

        capsys.readouterr()

        # no frame provided

        # highest mean
        mf._set_black_frame(params, None)
        assert params['black_frame'] == 2
        assert 'frame 2 is brightest' in capsys.readouterr().out

        # sky: always first frame
        params['cardmode'] = 'SKY'
        mf._set_black_frame(params, None)
        assert params['black_frame'] == 0

    def test_set_shiny_sky_4c(self):
        # 4 cards
        params = {'card_means': [1, 2, 3, 4],
                  'ncards': 4, 'black_frame': 0}

        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 1
        assert params['sky_frame2'] == 3
        assert params['shiny_frame'] == 2

        params['black_frame'] = 1
        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 0
        assert params['sky_frame2'] == 2
        assert params['shiny_frame'] == 3

        params['black_frame'] = 2
        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 1
        assert params['sky_frame2'] == 3
        assert params['shiny_frame'] == 0

        params['black_frame'] = 3
        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 2
        assert params['sky_frame2'] == 0
        assert params['shiny_frame'] == 1

    def test_set_shiny_sky_3c(self):
        # 3 cards
        params = {'card_means': [1, 2, 3],
                  'ncards': 3, 'black_frame': 0}

        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 1
        assert params['sky_frame2'] == 1
        assert params['shiny_frame'] == 2

        params['black_frame'] = 1
        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 0
        assert params['sky_frame2'] == 2
        assert params['shiny_frame'] == 1

        params['black_frame'] = 2
        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 1
        assert params['sky_frame2'] == 1
        assert params['shiny_frame'] == 0

    def test_set_shiny_sky_2c(self):
        # 2 cards
        params = {'card_means': [1, 2],
                  'ncards': 2, 'black_frame': 0}

        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 1
        assert params['sky_frame2'] == 1
        assert params['shiny_frame'] == 0

        params['black_frame'] = 1
        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 0
        assert params['sky_frame2'] == 0
        assert params['shiny_frame'] == 1

    def test_set_shiny_sky_1c(self):
        # 1 card
        params = {'card_means': [1],
                  'ncards': 1, 'black_frame': 0}

        mf._set_shiny_and_sky_frames(params)
        assert params['sky_frame'] == 0
        assert params['sky_frame2'] == 0
        assert params['shiny_frame'] == 0

    @pytest.mark.parametrize('cardmode,ptype',
                             [('BLK', 'BLK'),
                              ('NONE', 'BLK'),
                              ('UNKNOWN', 'BLK'),
                              ('SKY', 'SKY'),
                              ('SHINY', 'SHINY'),
                              ('BLKSKY', 'BLKSKY'),
                              ('OBJ', 'BLKSKY'),
                              ('BLKOBJ', 'BLKSKY'),
                              ('BLKSHINY', 'BLKSHINY'),
                              ])
    def test_set_process_type(self, cardmode, ptype):
        params = {'cardmode': cardmode,
                  'black_frame': 0, 'shiny_frame': 1}
        mf._set_process_type(params)
        assert params['process_type'] == ptype

        params['shiny_frame'] = 0
        if 'SHINY' not in ptype:
            mf._set_process_type(params)
            assert params['process_type'] == ptype
        else:
            with pytest.raises(ValueError) as err:
                mf._set_process_type(params)
            assert 'unusable without shiny' in str(err)

        params['shiny_frame'] = 1
        params['cardmode'] = 'BAD'
        with pytest.raises(ValueError) as err:
            mf._set_process_type(params)
        assert 'Unrecognizable' in str(err)

    @pytest.mark.parametrize('ptype,nbad',
                             [('BLK', 4), ('SKY', 5), ('SHINY', 6)])
    def test_check_saturation(self, ptype, nbad):
        ny, nx = 20, 10
        black = np.ones((ny, nx))
        black[:4, 0] = 1000
        sky = np.ones((ny, nx))
        sky[:5, 0] = 1000
        shiny = np.ones((ny, nx))
        shiny[:6, 0] = 1000
        cards = np.array([black, sky, shiny])
        params = {'nx': nx, 'ny': ny,
                  'process_type': ptype, 'maxval': 100, 'cards': cards,
                  'black_frame': 0, 'sky_frame': 1, 'shiny_frame': 2}

        # nbad found depends on process type
        mf._check_saturation(params, max_saturation=1)
        assert params['mask'].shape == (ny, nx)
        assert (~params['mask']).sum() == nbad

        # too many bad
        with pytest.raises(ValueError) as err:
            mf._check_saturation(params)
        assert f'{nbad} pixels saturated' in str(err)

        # maxval too low: all good
        params['maxval'] = -1
        mf._check_saturation(params, max_saturation=1)
        assert params['mask'].shape == (ny, nx)
        assert (~params['mask']).sum() == 0

        # none bad
        params['maxval'] = 100
        cards *= 0
        mf._check_saturation(params, max_saturation=1)
        assert params['mask'].shape == (ny, nx)
        assert (~params['mask']).sum() == 0

    @pytest.mark.parametrize('ptype,pfunc',
                             [('BLK', '_process_blk'),
                              ('SKY', '_process_sky'),
                              ('SHINY', '_process_shiny'),
                              ('BLKSKY', '_process_blksky'),
                              ('BLKSHINY', '_process_blkshiny')])
    def test_process_cards(self, mocker, ptype, pfunc):
        m1 = mocker.patch.object(mf, pfunc)
        params = {'process_type': ptype, 'card1': 10, 'card2': 20}
        mf._process_cards(params)
        assert m1.call_count == 1
        # cards replaced with [card1, card2]
        assert np.all(params['cards'] == [10, 20])

    def test_process_cards_bad_ptype(self):
        with pytest.raises(ValueError) as err:
            mf._process_cards({'process_type': 'bad'})
        assert 'Unknown process type' in str(err)

    def test_process_blk(self, capsys, process_params):
        params = process_params
        dark = params['dark']
        cards = params['cards']
        var = params['variance']
        black = cards[params['black_frame']]
        sky = cards[params['sky_frame']]

        # 2 frames: black and sky
        params['dark'] = None
        mf._process_blk(params)
        assert np.all(params['card1'] == black)
        assert np.all(params['card2'] == (black - sky) / black)
        assert np.all(params['diff'] == black)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))

        # 1 frame: black only
        params['sky_frame'] = 0
        params['cards'] = np.expand_dims(cards[0], axis=0)
        params['variance'] = np.expand_dims(var[0], axis=0)
        mf._process_blk(params)
        assert np.all(params['card1'] == black)
        assert np.all(params['card2'] == 0)
        assert np.all(params['diff'] == black)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))
        assert 'No slit dark available' in capsys.readouterr().out

        # dark provided
        params['dark'] = dark
        mf._process_blk(params)
        assert np.all(params['card1'] == black)
        assert np.all(params['card2'] == black - dark)
        assert np.all(params['diff'] == black - dark)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))
        assert 'Subtracting slit dark' in capsys.readouterr().out

    def test_process_sky(self, capsys, process_params):
        params = process_params
        cards = params['cards']
        sky = cards[params['sky_frame']]

        mf._process_sky(params)
        assert np.all(params['card1'] == sky)
        assert np.all(params['card2'] == sky)
        assert np.all(params['diff'] == sky)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))

    def test_process_shiny(self, capsys, process_params):
        params = process_params
        cards = params['cards']
        shiny = cards[params['shiny_frame']]
        sky = cards[params['sky_frame']]

        # with sky frame
        mf._process_shiny(params)
        assert np.all(params['card1'] == shiny)
        assert np.all(params['card2'] == sky / shiny)
        assert np.all(params['diff'] == shiny)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))

        # sky = shiny
        params['sky_frame'] = params['shiny_frame']
        mf._process_shiny(params)
        assert np.all(params['card1'] == shiny)
        assert np.all(params['card2'] == 0)
        assert np.all(params['diff'] == shiny)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))

    def test_process_blksky(self, capsys, process_params):
        params = process_params
        dark = params['dark']
        cards = params['cards']
        black = cards[params['black_frame']]
        sky = cards[params['sky_frame']]
        shiny = cards[params['shiny_frame']]

        # 3 frames: black, sky, shiny
        params['dark'] = None
        mf._process_blksky(params)
        assert np.all(params['card1'] == black - shiny)
        assert np.all(params['card2'] == (black - sky) / black)
        assert np.all(params['diff'] == black - sky)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))

        # 2 frames: black, sky
        params['shiny_frame'] = params['black_frame']
        mf._process_blksky(params)
        assert np.all(params['card1'] == black)
        assert np.all(params['card2'] == (black - sky) / black)
        assert np.all(params['diff'] == black - sky)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))

        # dark provided
        params['dark'] = dark
        mf._process_blksky(params)
        assert np.all(params['card1'] == black)
        assert np.all(params['card2'] == black - dark)
        assert np.all(params['diff'] == black - dark)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))
        assert 'Subtracting slit dark' in capsys.readouterr().out

    def test_process_blkshiny(self, capsys, process_params):
        params = process_params
        cards = params['cards']
        black = cards[params['black_frame']]
        shiny = cards[params['shiny_frame']]
        sky = cards[params['sky_frame']]

        # with sky frame
        mf._process_blkshiny(params)
        assert np.all(params['card1'] == black - shiny)
        assert np.all(params['card2'] == (black - sky) / black)
        assert np.all(params['diff'] == black - shiny)
        assert np.all(params['card_variance'] == 2)
        assert np.all(params['stddev'] == np.sqrt(2))

    def test_rqe(self, capsys, basic_params):
        params = basic_params
        params['black_frame'] = 0

        mf._calculate_responsive_quantum_efficiency(params)
        assert np.allclose(params['rqe'], 1.4267995e-08)
        assert 'Mean RQE over black' in capsys.readouterr().out

    def test_undistort_flat(self, capsys, basic_params):
        params = basic_params
        ny, nx = params['ny'], params['nx']
        params['stddev'] = np.sqrt(params['variance'][0])
        params['mask'] = np.full((ny, nx), True)

        mf._undistort_flat(params)

        # data is cleaned
        assert 'exes.clean' in capsys.readouterr().err

        # data matches single frame shape
        data = params['tortdata']
        illum = params['tortillum']
        assert data.shape == (ny, nx)
        assert illum.shape == (ny, nx)

        # data is all 1, due to output clipping to input range
        np.allclose(data, 1.0)
        assert np.all(illum[1:-2, 1:-2] == 1)

    @pytest.mark.parametrize('threshold,effective_threshold',
                             [(0, 0.05), (2, 0.25), (0.5, 0.125)])
    def test_create_flat(self, capsys, basic_params, threshold,
                         effective_threshold):
        params = basic_params
        ny, nx = params['ny'], params['nx']
        diff = np.arange(ny * nx, dtype=float).reshape((ny, nx))
        params['diff'] = diff
        params['stddev'] = 0.05 * diff
        params['tortillum'] = np.ones((ny, nx), dtype=int)
        params['mask'] = np.full((ny, nx), True)
        params['header']['THRFAC'] = threshold

        mf._create_flat(params)

        # calibration stored in header and applied to flat,
        # minimum values set to zero
        bnut = params['header']['BNU_T']
        assert np.allclose(bnut, 37.6900538)

        zidx = params['diff'] <= diff.mean() * effective_threshold
        assert np.allclose(params['flat'][zidx], 0)
        assert np.allclose(params['flat_variance'][zidx], 0)
        assert np.allclose(params['flat'][~zidx],
                           bnut / diff[~zidx])
        assert np.allclose(params['flat_variance'][~zidx],
                           (bnut * params['stddev'][~zidx])**2
                           / diff[~zidx]**4)

        # torted illumination flat except for borders and bad pix
        illum = params['illum']
        if threshold < 2:
            assert np.all(illum[1:-2, 1:-2] == 1)
            assert np.all(illum[0] == [-1, -1, -1, -1, -1, -1, -1,
                                       1, 1, 1, 1, 1, 1, 1, -1])
            assert np.all(illum[:, 0] == [-1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
            if threshold > 0:
                assert np.all(illum[-1] == [-1, -1, 1, 1, 1, 1, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0])
                assert np.all(illum[:, -1] == [-1, 1, 1, 1, 1, 1, 1, 1, 0, 0])

    def test_create_bad_flat(self, capsys, basic_params):
        params = basic_params
        ny, nx = params['ny'], params['nx']
        diff = np.arange(ny * nx, dtype=float).reshape((ny, nx))
        params['diff'] = diff.copy()
        params['stddev'] = 0.05 * diff.copy()
        params['tortillum'] = np.ones((ny, nx), dtype=int)
        params['mask'] = np.full((ny, nx), True)

        params['diff'] = -1 * diff.copy()
        mf._create_flat(params)
        assert 'Mean flat diff <= 0' in capsys.readouterr().err
        assert np.allclose(params['flat'], 1.0)
        assert np.allclose(params['flat_variance'], 0.0)
        assert np.allclose(params['illum'], 1.0)

        params['diff'] = np.nan * diff.copy()
        mf._create_flat(params)
        assert 'No pixels found above threshold' in capsys.readouterr().err
        assert np.allclose(params['flat'], 1.0)
        assert np.allclose(params['flat_variance'], 0.0)
        assert np.allclose(params['illum'], 1.0)
