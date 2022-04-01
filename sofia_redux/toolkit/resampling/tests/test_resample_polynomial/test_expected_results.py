# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial
from sofia_redux.toolkit.utilities.func import julia_fractal

import numpy as np


# The following tests are designed to ensure that the expected results of
# complex reductions remain unaltered.

get_all = {'get_error': True, 'get_counts': True, 'get_weights': True,
           'get_distance_weights': True, 'get_rchi2': True,
           'get_cross_derivatives': True, 'get_offset_variance': True}


def dict_all_results(results):
    out = {}
    for i, name in enumerate(['fit', 'error', 'counts', 'weights', 'distance',
                              'rchi2', 'deriv', 'offset']):
        out[name] = results[i]
    return out


def test_1d_example():
    rand = np.random.RandomState(1)
    noise = rand.rand(100) - 0.5
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.sin(x) + noise
    error = np.full(x.size, np.std(noise))
    x_out = np.linspace(-np.pi, np.pi, 1000)

    resampler = ResamplePolynomial(x, y, window=np.pi / 2, order=2)
    results = resampler(x_out, smoothing=0.4, relative_smooth=True,
                        order_algorithm='extrapolate', **get_all)
    vals = dict_all_results(results)

    # Sample some random indices
    indices = rand.choice(np.arange(x_out.size), 10)
    assert np.allclose(
        vals['fit'][indices],
        [-0.30864694, 0.66400339, 0.3812583, -0.93341651, -0.17525373,
         0.62353959, 0.64398911, 0.54194401, 0.95147111, 0.9167358],
        rtol=1e-2
    )
    assert np.allclose(
        vals['error'][indices],
        [0.10422094, 0.07142018, 0.07939968, 0.08160594, 0.07227806,
         0.07381277, 0.07195642, 0.07219886, 0.06902329, 0.07248957],
        rtol=1e-2
    )
    assert np.allclose(
        vals['counts'][indices],
        [27, 50, 31, 50, 49, 34, 49, 50, 50, 40],
        rtol=1e-2
    )
    assert np.allclose(
        vals['weights'][indices],
        [15.96411071, 27.08272602, 19.22282607, 27.0827596, 27.00072882,
         21.90525394, 27.00077502, 27.08244709, 27.08230762, 25.00474049],
        rtol=1e-2
    )
    assert np.allclose(vals['distance'], vals['weights'])
    assert np.allclose(vals['rchi2'][indices], 1)
    assert np.allclose(
        vals['deriv'][indices].ravel(),
        [1.70535923, 1.30217186, 3.07511476, 0.37463185, 1.46783429,
         2.38893013, 1.33583128, 1.3941511, 0.9029972, 1.46716173],
        rtol=1e-2
    )
    assert np.allclose(
        vals['offset'][indices],
        [1.92693208e+00, 2.14841232e-05, 1.10315197e+00, 1.15536840e-05,
         1.94789384e-05, 5.79557550e-01, 6.36046968e-06, 1.03983156e-04,
         1.45232673e-04, 1.78016266e-01],
        rtol=1e-2
    )

    # Test the adaptive case (scaled algorithm since 1-D)

    resampler = ResamplePolynomial(x, y, error=error,
                                   window=np.pi / 2, order=2)
    results = resampler(x_out, smoothing=0.4, relative_smooth=True,
                        adaptive_threshold=1.0, **get_all)
    vals = dict_all_results(results)
    assert np.allclose(
        vals['fit'][indices],
        [np.nan, 0.64972326, 0.41087771, -0.9036128, -0.15569441,
         0.63025555, 0.6364205, 0.53562748, 0.93864567, 0.92114922],
        equal_nan=True, rtol=1e-2)
    assert np.allclose(
        vals['error'][indices],
        [np.nan, 0.06637172, 0.10075495, 0.06643629, 0.06706474,
         0.07619934, 0.06701519, 0.0663814, 0.06636787, 0.0708717],
        equal_nan=True, rtol=1e-2)
    assert np.allclose(
        vals['counts'][indices],
        [25, 50, 29, 50, 49, 32, 49, 50, 50, 38],
        rtol=1e-2
    )
    assert np.allclose(
        vals['weights'][indices],
        [0., 504.14675961, 299.65264418, 502.49627497, 494.72285728,
         334.83515644, 496.57451494, 503.96184244, 504.08019556, 396.86125807],
        rtol=1e-2
    )
    assert np.allclose(
        vals['distance'][indices],
        [0., 43.69554791, 25.97157717, 43.55249665, 42.87875683,
         29.02092564, 43.03924422, 43.67952072, 43.68977865, 34.39686914],
        rtol=1e-2
    )
    assert np.allclose(
        vals['rchi2'][indices],
        [np.nan, 0.98916963, 0.82945118, 1.03822911, 1.04092724,
         0.82255194, 1.00284766, 1.02060422, 0.84521437, 0.81706457],
        equal_nan=True, rtol=1e-2)
    assert np.allclose(
        vals['deriv'][indices].ravel(),
        [-1.36311572e+57, 1.24403687e+00, 1.54083430e+00, 6.30830047e-01,
         1.44176684e+00, 1.20949072e+00, 1.27794454e+00, 1.18835437e+00,
         1.51336449e+00, 1.43060335e+00],
        rtol=1e-2
    )
    assert np.allclose(
        vals['offset'][indices],
        [np.nan, 2.14841232e-05, 1.53507580e+00, 1.15536840e-05,
         1.94789384e-05, 8.36760824e-01, 6.36046968e-06, 1.03983156e-04,
         1.45232673e-04, 2.84969673e-01], equal_nan=True, rtol=1e-2
    )


def test_2d_example():
    image = julia_fractal(64, 64)
    s2n = 50.0
    rand = np.random.RandomState(1)
    noise = rand.randn(64, 64)
    noise -= np.min(noise)
    noise /= s2n
    image += noise
    error = np.full(image.shape, 1 / s2n)

    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    x_out = np.linspace(20, 40, 200)
    y_out = np.linspace(5, 25, 200)
    coordinates = np.vstack([c.ravel() for c in [x, y]])

    # standard results
    resampler = ResamplePolynomial(coordinates, image.ravel(),
                                   error=error.ravel(), order=3)
    results = resampler(x_out, y_out, smoothing=0.05, jobs=-1,
                        relative_smooth=True, order_algorithm='extrapolate',
                        **get_all)
    vals = dict_all_results(results)

    rand = np.random.RandomState(2)
    x_inds = rand.choice(results[0].shape[0], 10)
    y_inds = rand.choice(results[0].shape[1], 10)
    indices = y_inds, x_inds

    assert np.allclose(
        vals['fit'][indices],
        [0.24736943, 0.07289773, 0.08230824, 0.05716328, 0.07439903,
         0.38469025, 0.25575886, 0.06743079, 0.240303, 0.48381683],
        rtol=1e-2
    )
    assert np.allclose(
        vals['error'][indices],
        [0.02636175, 0.03087019, 0.02821441, 0.02947445, 0.02821061,
         0.02734793, 0.03016577, 0.03024338, 0.02767716, 0.02641496],
        rtol=1e-2
    )
    assert np.allclose(
        vals['counts'][indices],
        [36, 32, 36, 35, 34, 36, 32, 34, 35, 37],
        rtol=1e-2
    )
    assert np.allclose(
        vals['weights'][indices],
        [4326.71996344, 4261.39753329, 4331.77722815, 4325.3820202,
         4346.28544853, 4324.4832625, 4263.00978972, 4303.64983975,
         4320.07180358, 4351.67222626],
        rtol=1e-2
    )
    assert np.allclose(
        vals['distance'][indices],
        [1.73068799, 1.70455901, 1.73271089, 1.73015281, 1.73851418,
         1.7297933, 1.70520392, 1.72145994, 1.72802872, 1.74066889],
        rtol=1e-2
    )
    assert np.allclose(
        vals['rchi2'][indices],
        [2.27196135, 0.11711652, 0.11440767, 0.05282344, 0.14306539,
         11.84362885, 24.18734016, 0.05686596, 0.94745175, 1.84671635],
        rtol=1e-2
    )
    assert np.allclose(
        vals['deriv'][indices],
        [[[1.23323990e-01, -8.30580642e-02],
          [-8.30580642e-02, 1.47640842e-01]],

         [[6.50544354e-03, -4.30101548e-03],
          [-4.30101548e-03, 9.20006317e-03]],

         [[7.81194681e-03, -8.60946542e-03],
          [-8.60946542e-03, 1.05919253e-02]],

         [[2.10466857e-03, -1.68676368e-03],
          [-1.68676368e-03, 9.96062586e-03]],

         [[1.82769320e-03, 1.34894620e-03],
          [1.34894620e-03, 3.59555155e-03]],

         [[1.76209837e-01, -4.14669197e-01],
          [-4.14669197e-01, 1.48903267e+00]],

         [[6.07946686e-01, 3.99764124e-01],
          [3.99764124e-01, 7.77754739e-01]],

         [[6.62309695e-03, 8.70273814e-03],
          [8.70273814e-03, 1.81563992e-02]],

         [[4.53006858e-02, 1.33044741e-02],
          [1.33044741e-02, 6.22389602e-02]],

         [[7.30295888e-02, -2.03527611e-02],
          [-2.03527611e-02, 2.75613864e-02]]],
        rtol=1e-1
    )
    assert np.allclose(
        vals['offset'][indices],
        [0.00166611, 0.00057243, 0.00125939, 0.00372879, 0.00175924,
         0.00105621, 0.00143352, 0.00202815, 0.00069292, 0.00056732],
        rtol=1e-2
    )

    # Test scaled adaptive algorithm
    results = resampler(x_out, y_out, smoothing=0.05,
                        relative_smooth=True, order_algorithm='extrapolate',
                        adaptive_algorithm='scaled', adaptive_threshold=1.0,
                        jobs=-1, **get_all)
    vals = dict_all_results(results)

    assert np.allclose(
        vals['fit'][indices],
        [0.23375892, 0.07242548, 0.08175571, 0.06281888, 0.07275225,
         0.33776382, 0.1869372, 0.06508216, 0.2491362, 0.50916927],
        rtol=1e-2
    )
    assert np.allclose(
        vals['error'][indices],
        [0.03178707, 0.01870266, 0.02141927, 0.01892529, 0.01829044,
         0.11199227, 0.048033, 0.01739582, 0.02537561, 0.04589881],
        rtol=1e-2
    )
    assert np.allclose(
        vals['counts'][indices],
        [36, 32, 36, 35, 34, 36, 32, 34, 35, 37]
    )
    assert np.allclose(
        vals['weights'][indices],
        [3205.9744538, 10232.81486406, 8309.31653702, 8708.77276968,
         8131.92672052, 2323.48762026, 2429.66660047, 8423.68489276,
         4766.70977743, 4024.10003897],
        rtol=1e-2
    )
    assert np.allclose(
        vals['distance'][indices],
        [1.28238978, 4.09312595, 3.32372661, 3.48350911, 3.25277069,
         0.92939505, 0.97186664, 3.36947396, 1.90668391, 1.60964002],
        rtol=1e-2
    )
    assert np.allclose(
        vals['rchi2'][indices],
        [1.43241248, 0.56322213, 0.30626535, 0.58469197, 0.33874503,
         0.17383593, 2.50591551, 0.45422697, 0.19432942, 0.26716398],
        rtol=1e-2
    )
    assert np.allclose(
        vals['deriv'][indices],
        [[[9.97194254e-02, -1.08954998e-01],
          [-1.08954998e-01, 1.45597386e-01]],

         [[2.70973196e-03, -1.23581493e-03],
          [-1.23581493e-03, 3.15046934e-03]],

         [[6.05050554e-03, -4.25487083e-04],
          [-4.25487083e-04, 5.48338990e-03]],

         [[7.72192450e-04, -7.37644808e-05],
          [-7.37644808e-05, 8.58142009e-04]],

         [[9.97401291e-04, 3.10651848e-05],
          [3.10651848e-05, 1.10731677e-03]],

         [[1.52734186e-01, 6.33076322e-02],
          [6.33076322e-02, 1.23995890e-01]],

         [[1.66500116e-01, 1.51219715e-01],
          [1.51219715e-01, 2.45139095e-01]],

         [[2.94056508e-03, 4.33756140e-03],
          [4.33756140e-03, 7.24257901e-03]],

         [[5.38330655e-02, 3.60339129e-02],
          [3.60339129e-02, 2.74118728e-02]],

         [[8.69500760e-02, -7.69468190e-02],
          [-7.69468190e-02, 2.56175897e-01]]],
        rtol=1e-1
    )
    assert np.allclose(
        vals['offset'][indices],
        [0.00166611, 0.00057243, 0.00125939, 0.00372879, 0.00175924,
         0.00105621, 0.00143352, 0.00202815, 0.00069292, 0.00056732],
        rtol=1e-2
    )

    # Test shaped adaptive algorithm
    results = resampler(x_out, y_out, smoothing=0.05,
                        relative_smooth=True, order_algorithm='extrapolate',
                        adaptive_algorithm='shaped', adaptive_threshold=1.0,
                        **get_all)
    vals = dict_all_results(results)

    assert np.allclose(
        vals['fit'][indices],
        [0.24638073, 0.07019944, 0.08696563, 0.0571854, 0.07380982,
         0.34665808, 0.19300889, 0.06691261, 0.25068628, 0.51701834],
        rtol=1e-2
    )
    assert np.allclose(
        vals['error'][indices],
        [0.02823859, 0.01830435, 0.02296557, 0.01949401, 0.01961805,
         0.04166166, 0.07218005, 0.02050251, 0.02276781, 0.03791761],
        rtol=1e-2
    )
    assert np.allclose(
        vals['counts'][indices],
        [36, 32, 36, 35, 34, 36, 32, 34, 35, 37]
    )
    assert np.allclose(
        vals['weights'][indices],
        [3590.25498936, 11740.8629434, 7125.55195515, 8745.089046,
         7891.47006395, 3585.70933458, 2553.37253768, 8852.6356721,
         5366.46221703, 4524.50133832],
        rtol=1e-2
    )
    assert np.allclose(
        vals['distance'][indices],
        [1.436102, 4.69634518, 2.85022078, 3.49803562, 3.15658803,
         1.43428373, 1.02134902, 3.54105427, 2.14658489, 1.80980054],
        rtol=1e-2
    )
    assert np.allclose(
        vals['rchi2'][indices],
        [1.66144867, 0.51322155, 0.25529615, 0.38708086, 0.35009122,
         0.91542981, 2.09961444, 0.3593031, 0.21380265, 0.62488994],
        rtol=1e-2
    )
    assert np.allclose(
        vals['deriv'][indices],
        [[[1.29328277e-01, -3.20560019e-03],
          [-3.20560019e-03, 1.85416185e-01]],

         [[2.45956361e-03, -8.04401915e-04],
          [-8.04401915e-04, 4.67562587e-03]],

         [[1.68169865e-03, -1.86718493e-03],
          [-1.86718493e-03, 8.05088336e-03]],

         [[9.93682872e-03, 3.26568914e-03],
          [3.26568914e-03, 6.36160055e-03]],

         [[7.14280838e-04, 1.00059174e-03],
          [1.00059174e-03, 3.20872172e-03]],

         [[2.65449561e-01, -5.33105578e-01],
          [-5.33105578e-01, 1.10997977e+00]],

         [[2.60126778e-01, 5.72268084e-02],
          [5.72268084e-02, 6.91935904e-02]],

         [[5.71827703e-03, 5.44675069e-03],
          [5.44675069e-03, 6.32822044e-03]],

         [[5.91001435e-02, 4.04182816e-02],
          [4.04182816e-02, 3.05502178e-02]],

         [[1.19545116e-01, -3.82214355e-02],
          [-3.82214355e-02, 5.87952344e-01]]],
        rtol=1e-1
    )
    assert np.allclose(
        vals['offset'][indices],
        [0.00166611, 0.00057243, 0.00125939, 0.00372879, 0.00175924,
         0.00105621, 0.00143352, 0.00202815, 0.00069292, 0.00056732],
        rtol=1e-2
    )
