# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial

import numpy as np
from skimage.data import coins


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
    image = coins()
    blank = image[:25, :300]
    image = coins()[12:76, 303:367]  # 64x64 pixels
    blank = blank / image.max()
    image = image / image.max()

    error = np.full(image.shape, np.std(blank))
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
        [0.68628546, 0.79656139, 0.55561359, 0.74560031, 0.79303773,
         0.4350173, 0.73267058, 0.48039404, 0.75398415, 0.66701528],
        rtol=1e-2
    )
    assert np.allclose(
        vals['error'][indices],
        [0.06378189, 0.07468996, 0.06826447, 0.07131074, 0.06825551,
         0.06616791, 0.0729861, 0.07317236, 0.06696454, 0.06391068],
        rtol=1e-2
    )
    assert np.allclose(
        vals['counts'][indices],
        [36, 32, 36, 35, 34, 36, 32, 34, 35, 37],
        rtol=1e-2
    )
    assert np.allclose(
        vals['weights'][indices],
        [739.11525709, 727.95650284, 739.97916821, 738.88670191,
         742.45754609, 738.73317093, 728.23191778, 735.17428551,
         737.97958008, 743.37774652],
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
        [0.01379423, 1.48113587, 0.35192532, 0.46076381, 0.04814567,
         0.18977021, 0.16100593, 0.07120683, 0.02412596, 0.01970182],
        rtol=1e-2
    )
    assert np.allclose(
        vals['deriv'][indices],
        [[[7.73904862e-03, -1.93634763e-03],
          [-1.93634763e-03, 2.98021349e-03]],

         [[6.94634571e-01, 5.43928917e-01],
          [5.43928917e-01, 5.97785735e-01]],

         [[1.73491932e-02, 7.70325100e-03],
          [7.70325100e-03, 3.88665090e-02]],

         [[2.90161858e-01, 2.98267037e-01],
          [2.98267037e-01, 4.33464825e-01]],

         [[2.21517834e-02, -1.56457177e-02],
          [-1.56457177e-02, 1.90337833e-02]],

         [[3.88825567e-01, -6.42997044e-01],
          [-6.42997044e-01, 1.12130680e+00]],

         [[8.08015751e-02, 3.04441452e-02],
          [3.04441452e-02, 6.78821731e-02]],

         [[1.42781766e-01, 9.71004452e-02],
          [9.71004452e-02, 9.23220596e-02]],

         [[5.44234379e-03, 4.23758430e-03],
          [4.23758430e-03, 4.27035873e-03]],

         [[9.59949785e-03, -7.25118518e-04],
          [-7.25118518e-04, 1.44783814e-02]]],
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
        [0.68025418, 0.77808745, 0.52701211, 0.73387656, 0.77815641,
         0.4469629, 0.72547085, 0.48498082, 0.75452326, 0.66246624],
        rtol=1e-2
    )
    assert np.allclose(
        vals['error'][indices],
        [0.03827069, 0.05805472, 0.0450944, 0.05015295, 0.04098312,
         0.04998848, 0.04821716, 0.04733662, 0.03922033, 0.04383654],
        rtol=1e-2
    )
    assert np.allclose(
        vals['counts'][indices],
        [36, 32, 36, 35, 34, 36, 32, 34, 35, 37]
    )
    assert np.allclose(
        vals['weights'][indices],
        [1880.97626261, 927.42368574, 1324.11368922, 1119.74410252,
         1617.80941123, 1071.59961698, 1183.84989807, 1306.12220665,
         1821.32075627, 1509.7608489],
        rtol=1e-2
    )
    assert np.allclose(
        vals['distance'][indices],
        [4.40443217, 2.17162481, 3.10050108, 2.62195598, 3.78820933,
         2.50922243, 2.772064, 3.05837282, 4.2647448, 3.53520637],
        rtol=1e-2
    )
    assert np.allclose(
        vals['rchi2'][indices],
        [0.08527751, 3.01355787, 1.16110893, 1.73417909, 0.40187175,
         1.25224871, 0.44806641, 0.50673207, 0.14954966, 0.22609291],
        rtol=1e-2
    )
    assert np.allclose(
        vals['deriv'][indices],
        [[[0.00486515, 0.00305658],
          [0.00305658, 0.00379233]],

         [[0.16718942, 0.18274602],
          [0.18274602, 0.24974218]],

         [[0.01556815, 0.01084947],
          [0.01084947, 0.01409344]],

         [[0.10735855, 0.09185641],
          [0.09185641, 0.11469069]],

         [[0.03153735, -0.01201194],
          [-0.01201194, 0.0142071]],

         [[0.29928189, -0.4149837],
          [-0.4149837, 0.64357539]],

         [[0.03592587, 0.01157044],
          [0.01157044, 0.03201321]],

         [[0.0827349, 0.03362447],
          [0.03362447, 0.03824523]],

         [[0.00334145, 0.00385317],
          [0.00385317, 0.00874397]],

         [[0.00608948, -0.00153193],
          [-0.00153193, 0.00446786]]],
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
        [0.68296064, 0.73710596, 0.52085781, 0.72153002, 0.78965367,
         0.46366592, 0.73117522, 0.52314707, 0.75634777, 0.65593053],
        rtol=1e-2
    )
    assert np.allclose(
        vals['error'][indices],
        [0.0411598, 0.06504268, 0.0465513, 0.04978293, 0.04818291,
         0.05505095, 0.0489499, 0.05201444, 0.04475005, 0.0393651],
        rtol=1e-2
    )
    assert np.allclose(
        vals['counts'][indices],
        [36, 32, 36, 35, 34, 36, 32, 34, 35, 37]
    )
    assert np.allclose(
        vals['weights'][indices],
        [1746.17299067, 1277.72968976, 2198.36310078, 1117.62644742,
         1759.03261984, 991.49053059, 1121.21411078, 1308.72504377,
         1850.93124567, 1771.63757718],
        rtol=1e-2
    )
    assert np.allclose(
        vals['distance'][indices],
        [4.08878126, 2.99188983, 5.14761476, 2.61699734, 4.11889295,
         2.32164163, 2.6253981, 3.06446753, 4.33407975, 4.1484083],
        rtol=1e-2
    )
    assert np.allclose(
        vals['rchi2'][indices],
        [0.08484625, 1.90425448, 0.85342021, 1.80968629, 0.22912724,
         5.205958, 0.23500118, 0.47533556, 0.04258422, 0.5647755],
        rtol=1e-2
    )
    assert np.allclose(
        vals['deriv'][indices],
        [[[0.00752256, 0.00137517],
          [0.00137517, 0.00131708]],

         [[0.5537669, 0.33305741],
          [0.33305741, 0.44566386]],

         [[0.01968198, 0.00826993],
          [0.00826993, 0.07253312]],

         [[0.13436461, 0.08221129],
          [0.08221129, 0.08824205]],

         [[0.07953451, -0.02396745],
          [-0.02396745, 0.03882596]],

         [[0.1397034, -0.18425574],
          [-0.18425574, 0.57979638]],

         [[0.03969487, 0.03657964],
          [0.03657964, 0.07348977]],

         [[0.15959266, -0.06155905],
          [-0.06155905, 0.10343057]],

         [[0.01467744, 0.0042856],
          [0.0042856, 0.01324727]],

         [[0.00690038, 0.00221446],
          [0.00221446, 0.01368642]]],
        rtol=1e-1
    )
    assert np.allclose(
        vals['offset'][indices],
        [0.00166611, 0.00057243, 0.00125939, 0.00372879, 0.00175924,
         0.00105621, 0.00143352, 0.00202815, 0.00069292, 0.00056732],
        rtol=1e-2
    )
