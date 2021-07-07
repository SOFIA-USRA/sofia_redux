# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.modeling import models, fitting
import numpy as np
import pytest

from sofia_redux.toolkit.fitting.fitpeaks1d import (
    fitpeaks1d, robust_masking, guess_xy_mad, medabs_baseline,
    get_model_name, get_x_parname, get_width_parname, get_amplitude_parname,
    get_n_submodels, parse_width_arg, get_min_width, get_fitter, dofit,
    box_convolve, get_search_model, initial_search, get_final_model,
    get_background_fit, update_model)


@pytest.fixture
def separated_peaks_offset():
    x = np.linspace(0, 250, 2501)
    y = np.zeros(2501)
    npeaks = 5
    coeffs = np.zeros((npeaks, 3))
    coeffs[::-1, 0] = (np.arange(5) + 2) ** 2  # amplitude
    coeffs[1::2, 0] *= -1  # positive and negative peaks
    coeffs[:, 1] = np.arange(5) * 50 + 25  # x centroid
    coeffs[::-1, 2] = np.arange(5) * 2 + 0.5  # fwhm
    for c in coeffs:
        y += models.Gaussian1D(*c)(x)
    y += 10  # baseline
    return x, y, coeffs


@pytest.fixture
def compound_peak():
    x = np.linspace(0, 10, 1001)
    model1 = models.Gaussian1D(amplitude=3, mean=5, stddev=0.5)
    model2 = models.Gaussian1D(amplitude=4, mean=6, stddev=0.2)
    y = model1(x) + model2(x)
    noise = np.random.normal(0, 0.1, y.shape)
    return x, y, noise


@pytest.fixture
def single_peak_sloped_bg():
    x = np.linspace(0, 10, 1001)
    peak = models.Gaussian1D(amplitude=3, mean=5, stddev=0.5)
    y = peak(x) + x * 0.25 + 1
    return x, y


def test_robust_masking():
    data = np.random.normal(0, 1, 1000)
    data[500] += 1000
    result = robust_masking(data, threshold=10)
    assert isinstance(result[500], np.ma.core.MaskedConstant)
    assert np.sum(result.mask) == 1


def test_medabs_baseline():
    y = np.random.normal(100, 0.01, 1000)
    x = np.arange(1000)
    result = medabs_baseline(x, y)
    assert np.allclose(result[0], 0, atol=0.1)
    assert np.allclose(result[1], 100, atol=0.1)


def test_guess_xy_mad():
    x = np.arange(1000) + 0.5
    y = np.random.normal(100, 0.01, 1000)
    y[np.arange(10) * 100] += np.arange(10) + 100
    result = guess_xy_mad(x, y)
    assert np.allclose(result, [900.5, 209], atol=0.1)


def test_get_model_name():
    model = models.Gaussian1D
    assert get_model_name(model) == 'Gaussian1D'
    model = model()
    assert get_model_name(model) == 'Gaussian1D'
    assert get_model_name('Gaussian1D') == 'Gaussian1D'
    assert get_model_name(None) is None

    compound_model = model + model
    assert get_model_name(compound_model, base=True) == 'Gaussian1D'
    assert get_model_name(model) == 'Gaussian1D'
    assert get_model_name(compound_model).startswith('CompoundModel')
    assert get_model_name(None) is None


def test_get_x_parname():
    assert get_x_parname(models.Gaussian1D()) == 'mean'
    assert get_x_parname(models.Voigt1D()) == 'x_0'
    assert get_x_parname(models.Sine1D()) == 'phase'


def test_get_width_parname():
    names = {'Moffat1D': 'gamma',
             'Box1D': 'width',
             'Gaussian1D': 'stddev',
             'Lorentz1D': 'fwhm',
             'MexicanHat1D': 'sigma',
             'Sersic1D': 'r_eff',
             'Trapezoid1D': 'width',
             'Voigt1D': 'fwhm_G'}
    for k, v in names.items():
        assert get_width_parname(k) == v
    assert get_width_parname('foobar') is None


def test_get_amplitude_parname():
    assert get_amplitude_parname(models.Gaussian1D()) == 'amplitude'
    assert get_amplitude_parname(models.Voigt1D()) == 'amplitude_L'


def test_get_n_submodels():
    # mock some old and new style astropy models
    class TestModel1(object):
        def __init__(self):
            pass

    class TestModel2(object):
        def __init__(self):
            self.n_submodels = 2

    class TestModel3(object):
        def n_submodels(self):
            return 3

    model1 = TestModel1()
    assert get_n_submodels(model1) == 1
    model2 = TestModel2()
    assert get_n_submodels(model2) == 2
    model3 = TestModel3()
    assert get_n_submodels(model3) == 3

    # if not an int for model type 2, returns 1
    model2.n_submodels = None
    assert get_n_submodels(model2) == 1


def test_parse_width_arg():
    assert parse_width_arg(models.Gaussian1D(), (None, 1)) == ('stddev', 1)
    assert parse_width_arg(models.Gaussian1D(), None) is None
    assert parse_width_arg(models.Gaussian1D(), 5) == 5
    with pytest.raises(ValueError):
        parse_width_arg(models.Gaussian1D(), ('stddev', 1, 2))

    with pytest.raises(ValueError):
        parse_width_arg(models.Gaussian1D(), ('stddev', 'a'))

    with pytest.raises(ValueError):
        parse_width_arg(models.Gaussian1D(), ('foobar', 1))

    with pytest.raises(ValueError):
        parse_width_arg(models.Gaussian1D(), (1, 1))

    with pytest.raises(ValueError):
        parse_width_arg(models.Gaussian1D((1, 'foobar')))

    with pytest.raises(ValueError):
        parse_width_arg(models.Linear1D(), (None, 1))

    # noinspection PyCallByClass
    badclass = models.Gaussian1D.rename('foobar')
    with pytest.raises(TypeError):
        parse_width_arg(object, ('width', 1))

    with pytest.raises(ValueError):
        parse_width_arg(badclass(), (None, 1))


def test_update_model():
    model = models.Gaussian1D()
    model.foo = 'not_bar'
    model.d2 = {'another': 1}
    kwargs = {'foo': 'bar',  # weird parameter
              'mean': 1.5,  # existing attribute
              'd2': {'another': 2}}
    update_model(model, kwargs)
    assert model.mean == 1.5
    assert model.foo == 'bar'
    assert model.d2['another'] == 2


def test_get_min_width():
    model = models.Gaussian1D()
    x = np.arange(100) * 0.5  # 0.5 separation
    x = np.hstack((x, x))  # duplicate values
    np.random.shuffle(x)  # not ordered
    assert get_min_width(model, None, x) is None
    assert get_min_width(model, 5, x) == ('stddev', 5)
    assert get_min_width(model, ('stddev', 3), x) == ('stddev', 3)
    assert get_min_width(model, ('stddev', None), x) == ('stddev', 0.25)
    assert get_min_width(model, (None, None), x) == ('stddev', 0.25)
    with pytest.raises(ValueError) as err:
        get_min_width(model, [1, None], x=None)
    assert "require x coordinates" in str(err.value).lower()


def test_get_fitter():
    class DummyFitter(object):
        def __init__(self, **kw):
            for k, x in kw.items():
                setattr(self, k, x)
    fitter_class = fitting.LevMarLSQFitter

    assert isinstance(get_fitter(fitter_class), fitter_class)
    kwargs = {'foo': 1}
    init_fitter = get_fitter(DummyFitter, **kwargs)
    assert init_fitter.foo == 1
    fitter = get_fitter(fitter_class, outlier_func=robust_masking,
                        robust={'threshold': 10101}, outlier_iter=6)
    assert isinstance(fitter, fitting.FittingWithOutlierRemoval)
    assert fitter.outlier_func is robust_masking
    assert fitter.niter == 6
    assert fitter.outlier_kwargs['threshold'] == 10101


def test_dofit():
    fitter = fitting.LevMarLSQFitter()
    model = models.Gaussian1D(1, 5, 1)
    x = np.linspace(0, 10, 100)
    y = model(x) + np.random.normal(0, 0.01, 100)
    result = dofit(fitter, model, x, y[0])
    assert np.isnan(result.parameters).all()
    assert result.fit_info['y'] == y[0]
    assert np.allclose(result.fit_info['x'], x)
    fitter = get_fitter(fitting.LevMarLSQFitter, robust=True)
    kwargs = {'maxiter': 1}
    result = dofit(fitter, models.Gaussian1D(), x, y, **kwargs)
    assert not np.allclose(result.parameters, model.parameters, atol=0.1)
    assert result.fit_info['nfev'] == 2
    result = dofit(fitter, models.Gaussian1D(), x, y)
    assert np.allclose(result.parameters, model.parameters, atol=0.1)


def test_apply_box():
    model = models.Voigt1D(x_0=5, amplitude_L=1, fwhm_G=1.5)
    box_model = models.Box1D

    with pytest.raises(ValueError):
        box_convolve(model, box_model, box_x='foo')
    with pytest.raises(ValueError):
        box_convolve(model, box_model, model_x='foo')
    with pytest.raises(ValueError):
        box_convolve(model, box_model, box_width=('foo', 1))
    with pytest.raises(ValueError):
        box_convolve(model, box_model, box_width=('fwhm_G', 'a'))

    # fixed width
    result = box_convolve(model, box_model, 3)
    assert result[1].width == 3
    assert not result[1].tied['amplitude']
    assert not result[1].tied['width']
    tied_x = result[1].tied['x_0'](result)
    assert tied_x == 5
    assert result[1].fixed['amplitude']
    assert not result[1].fixed['x_0']
    assert result[1].fixed['width']

    # variable width
    result = box_convolve(model, box_model, ('fwhm_G', 4))
    assert not result[1].tied['amplitude']
    tied_x = result[1].tied['x_0'](result)
    assert tied_x == 5
    tied_w = result[1].tied['width'](result)
    assert tied_w == 6
    assert result[1].fixed['amplitude']
    assert not result[1].fixed['width']
    assert not result[1].fixed['x_0']

    # box_params
    result = box_convolve(model, models.Trapezoid1D,
                          box_params={
                              'amplitude': 1,
                              'width': 3,
                              'slope': 0.5,
                              'fixed': {'slope': False}})
    assert result[1].amplitude == 1
    assert result[1].slope == 0.5
    assert result[1].width == 3
    assert result[1].tied['x_0'](result) == 5

    # kwargs - OOOOoooo linking phase to position - cool!
    result = box_convolve(
        model, models.Sine1D, box_x='phase',
        box_params={'amplitude': 1, 'frequency': 2 / np.pi},
        phase_1=-2 / np.pi * 5,
        tied={'phase_1': lambda m: -2 / np.pi * getattr(m[0], 'x_0')})
    assert result[1].tied['phase'](result) == -2 / np.pi * result.x_0_0

    # compound models
    result = box_convolve(model + model, box_model, box_width=('x_0', 1.0))
    assert result.tied['x_0_2'] is not False


def test_get_search_model():

    peak_model = models.Gaussian1D

    # basic model and kwargs test
    result = get_search_model(peak_model, amplitude=2)
    assert result.amplitude == 2
    assert result.bounds['mean'] == (None, None)
    assert result.bounds['stddev'] != (1.234, None)

    # min_width test
    result = get_search_model(peak_model, min_width=1.234)
    assert result.bounds['stddev'] == (1.234, None)
    # noinspection PyCallByClass
    bad_model = peak_model.rename('foobie')
    with pytest.raises(ValueError):
        get_search_model(bad_model, min_width=1.234)
    result = get_search_model(
        bad_model, min_width=('stddev', 1.234))
    assert result.bounds['stddev'] == (1.234, None)

    # xbounds test
    result = get_search_model(peak_model, xrange=(1, 2))
    assert result.bounds['mean'] == (1, 2)

    # test box
    box = models.Box1D
    result = get_search_model(peak_model, box, 3, amplitude_0=-2)
    assert result[0].amplitude == -2
    assert result[1].amplitude == 1
    assert result[1].width == 3
    assert result[1].tied['x_0'](result) == result[0].mean

    # check the box_convolve arguments get through ok
    result = get_search_model(peak_model, box, 3, box_x='amplitude')
    assert result[1].tied['amplitude'](result) == result[0].mean


def test_initial_search(separated_peaks_offset):
    x, y, coeffs = separated_peaks_offset

    peak_model = models.Gaussian1D
    box = models.Box1D
    fitter = get_fitter(fitting.LevMarLSQFitter)
    model = get_search_model(
        peak_model, box, box_width=('stddev', 4), min_width=0.1)

    with pytest.raises(ValueError):
        initial_search(fitter, model, x, y, 5, xpeak_parname='foo')
    with pytest.raises(ValueError):
        initial_search(fitter, model, x, y, 5, ypeak_parname='foo')

    # test default options work ok
    s = initial_search(fitter, model, x, y, 5)
    assert np.allclose(s[:, :2], coeffs[:, :2], rtol=0.01)

    # test fitopts works
    s = initial_search(fitter, model, x, y, 5, fitopts={'maxiter': 1})
    assert not np.allclose(s[:, :2], coeffs[:, :2], rtol=0.01)

    # test baseline_func works
    def kill_them_all(_, iny):
        return iny * 0, iny * 0

    s = initial_search(fitter, model, x, y, 5, baseline_func=kill_them_all)
    assert np.allclose(s[:, 1], 0, rtol=0.01)
    assert np.allclose(s[:, 0], y[0], rtol=0.01)

    # test guess_func works
    def min_peak(inx, iny):
        xind = np.argmin(iny)
        return inx[xind], iny[xind]

    def just_baseline(_, iny):
        medy = np.full(y.shape, float(np.median(iny)))
        return iny - medy, medy

    s = initial_search(fitter, model, x, y, 1,
                       guess_func=min_peak, baseline_func=just_baseline)
    assert np.allclose(s[0][:2], coeffs[1][:2], rtol=0.01)  # largest neg peak

    # test guess  # just x and y positions really
    guess = coeffs[3, 1]
    s = initial_search(fitter, model, x, y, 1, guess=guess)
    assert np.allclose(s[0][:2], coeffs[3][:2], rtol=0.01)


def test_get_background_fit(separated_peaks_offset):
    x, y, coeffs = separated_peaks_offset
    ynoise = y + np.random.normal(0, 0.1, y.size) + x * 0.25
    fitter = get_fitter(fitting.LevMarLSQFitter)
    peak_model = models.Gaussian1D()
    background_class = models.Linear1D
    bfit = get_background_fit(fitter, peak_model, background_class,
                              x, ynoise, coeffs)
    assert np.allclose(bfit, [0.25, 10], atol=0.1)

    bfit = get_background_fit(fitter, peak_model, background_class(),
                              x, ynoise, coeffs)
    assert np.allclose(bfit, [0.25, 10], atol=0.1)

    # test kwargs
    bfit = get_background_fit(fitter, peak_model, background_class,
                              x, ynoise, coeffs, intercept=np.nan)
    assert np.isnan(bfit[1])


def test_bgargs(separated_peaks_offset, capsys):
    x, y, coeffs = separated_peaks_offset
    ynoise = y + np.random.normal(0, 0.1, y.size) + x * 0.25
    fitter = get_fitter(fitting.LevMarLSQFitter)
    peak_model = models.Gaussian1D

    # test bgargs: one argument
    class TestClass(models.Linear1D):
        def __init__(self, arg1, **kwargs):
            print(arg1)
            super().__init__(**kwargs)

    # test in get_backround_fit
    bfit = get_background_fit(fitter, peak_model(), TestClass,
                              x, ynoise, coeffs, bg_args=123)
    assert np.allclose(bfit, [0.25, 10], atol=0.1)
    assert '123' in capsys.readouterr().out

    # now test in get_final_model
    pinit = [[1, 2, 3]]
    model = get_final_model(peak_model, pinit,
                            background_class=TestClass, bg_args=123)
    assert np.allclose(model[0].parameters, pinit[0])
    assert '123' in capsys.readouterr().out

    # test bgargs: multiple argument
    class TestClass2(models.Linear1D):
        def __init__(self, arg1, arg2, **kwargs):
            print(arg1)
            print(arg2)
            super().__init__(**kwargs)

    bfit = get_background_fit(fitter, peak_model(), TestClass2,
                              x, ynoise, coeffs, bg_args=(456, 789))
    assert np.allclose(bfit, [0.25, 10], atol=0.1)
    assert '456\n789' in capsys.readouterr().out

    model = get_final_model(peak_model, pinit,
                            background_class=TestClass2, bg_args=(456, 789))
    assert np.allclose(model[0].parameters, pinit)
    assert '456\n789' in capsys.readouterr().out


def test_get_final_model():
    peak_class = models.Lorentz1D
    pinit = [[1, 2, 3]]
    background_class = models.PowerLaw1D
    binit = [0.1, 0.2, 0.3]
    min_width = 4
    xbounds = 0, 5
    kwargs = {'bounds': {'alpha_1': (0, 1)}}
    model = get_final_model(peak_class, pinit,
                            background_class=background_class, binit=binit,
                            min_width=min_width, xbounds=xbounds, **kwargs)
    assert np.allclose(model[0].parameters, pinit)
    assert np.allclose(model[1].parameters, binit)
    assert model[1].bounds['alpha'] == (0, 1)
    assert model[0].bounds['fwhm'] == (4, None)
    assert model[0].bounds['x_0'] == (0, 5)
    assert model.name == '1_peak_with_background'

    # multiple peaks
    pinit = [[1, 2, 3], [4, 5, 6]]
    model = get_final_model(peak_class, pinit,
                            background_class=background_class, binit=binit,
                            min_width=min_width, xbounds=xbounds)
    assert model.name == '2_peaks_with_background'
    assert np.allclose(model[0].parameters, pinit[0])
    assert np.allclose(model[1].parameters, pinit[1])
    assert np.allclose(model[2].parameters, binit)

    # single peak no background
    model = get_final_model(peak_class, [pinit[0]], min_width=min_width,
                            xbounds=xbounds)
    assert model.name == '1_peak'
    assert np.allclose(model.parameters, pinit[0])
    assert model.bounds['x_0'] == (0, 5)

    # single peak, default background params
    model = get_final_model(peak_class, [pinit[0]],
                            background_class=background_class(), binit=None,
                            min_width=min_width, xbounds=xbounds)
    assert np.allclose(model[1].parameters, 1)


def test_fitpeak1d(separated_peaks_offset):
    x, y, coeffs = separated_peaks_offset
    with pytest.raises(TypeError):
        fitpeaks1d(x, y, peak_class=None)

    with pytest.raises(ValueError):
        # shape mismatch
        fitpeaks1d(x, y[:-1])

    with pytest.raises(ValueError):
        # all NaN
        fitpeaks1d(np.arange(100), np.full(100, np.nan))

    with pytest.raises(ValueError):
        # Not enough points
        fitpeaks1d(np.arange(4), np.arange(4))

    with pytest.raises(ValueError):
        # invalid xrange
        fitpeaks1d(x, y, xrange=(1, 2, 3))

    noise = np.random.normal(0, 0.01, y.size)
    y += noise

    # Test optional function
    def opt_func(_, iny):
        return iny / 2

    fit = fitpeaks1d(x, y, npeaks=5, optional_func=opt_func)

    expected_coeffs = coeffs.copy()
    expected_coeffs[:, 0] /= 2
    for peak, c in zip(fit[:5], expected_coeffs):
        assert np.allclose(peak.parameters, c, atol=0.01)

    fit = fitpeaks1d(x, y, npeaks=5, optional_func=opt_func, maxiter=2000)
    expected_coeffs = coeffs.copy()
    expected_coeffs[:, 0] /= 2
    for peak, c in zip(fit[:5], expected_coeffs):
        assert np.allclose(peak.parameters, c, atol=0.01)

    # Test with no box: still same result
    fit = fitpeaks1d(x, y, npeaks=5, optional_func=opt_func, box_class=None)
    expected_coeffs = coeffs.copy()
    expected_coeffs[:, 0] /= 2
    for peak, c in zip(fit[:5], expected_coeffs):
        assert np.allclose(peak.parameters, c, atol=0.01)
