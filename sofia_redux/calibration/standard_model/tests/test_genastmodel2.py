# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import glob
import pytest
import numpy as np
import numpy.testing as npt

from sofia_redux.calibration.standard_model import genastmodel2
from sofia_redux.calibration.standard_model.tests import resources
from sofia_redux.calibration.pipecal_error import PipeCalError


@pytest.fixture
def params():
    return resources.horizon()


@pytest.fixture
def model():
    wave = np.linspace(5, 300, 100)
    flux = np.log10(wave) + np.random.normal(0, 0.25, len(wave))
    return np.array([wave, flux])


@pytest.fixture
def fit(model):
    return genastmodel2.interpolate_model(model[0, :], model[1, :])


@pytest.fixture
def timepoint():
    return resources.timepoint()


def test_interpolate_model(model):
    fit = genastmodel2.interpolate_model(model[0, :], model[1, :])
    assert fit.shape == (2, 20)
    npt.assert_allclose(fit[0, 0], model[0, 0], rtol=0.2)


def test_interpolate_model_request(model):
    requested = np.linspace(10, 100, 10)
    fit = genastmodel2.interpolate_model(model[0, :], model[1, :],
                                         requested_wave=requested)
    assert fit.shape == (2, len(requested))


def test_write_models(params, tmpdir, model, fit):
    with tmpdir.as_cwd():
        outfile = tmpdir / 'test.out'
        genastmodel2.write_models(model, fit, params, outfile)
        filenames = [outfile, 'bb_fit.out']
        for filename in filenames:
            assert os.path.isfile(filename)


def test_asteroid_model(params, timepoint):
    date = timepoint.date()
    time = timepoint.time()
    outprefix = 'test_out'
    outfile = f'{outprefix}{date}_{time}_model.out'
    outfile = outfile.replace(':', '')
    model = genastmodel2.asteroid_model(params=params, date=date, time=time,
                                        return_model=True,
                                        save_model=True)
    assert os.path.isfile(outfile)
    read_model = np.loadtxt(outfile, skiprows=9)
    npt.assert_allclose(model, read_model.T, atol=1e-6)
    os.remove(outfile)
    images = glob.glob('*png')
    for image in images:
        os.remove(image)
    os.remove('bb_fit.out')


def test_parse_args():
    horizons = 'hor'
    obs_file = 'obs'
    prefix = 'out'
    args = [horizons, obs_file, prefix]
    args = genastmodel2.parse_args(args)

    assert args.params == [horizons]
    assert args.obs_file == [obs_file]
    assert args.outprefix == prefix


@pytest.mark.parametrize('prefix,expected', [('test', 'test'),
                                             ('', 'test_out')])
def test_check_args(prefix, expected):
    horizons_file = 'horizons.out'
    obs_file = 'obs_file.txt'

    for filename in [horizons_file, obs_file]:
        with open(filename, 'w') as f:
            f.write('test')

    args = [horizons_file, obs_file, prefix]
    args = genastmodel2.parse_args(args)
    args = genastmodel2.check_args(args)

    assert args.params == horizons_file
    assert args.obs_file == obs_file
    assert args.outprefix == expected

    for filename in [horizons_file, obs_file]:
        os.remove(filename)


def test_check_args_lists():
    horizons_file = 'horizons.out'
    obs_file = 'obs_file.txt'
    prefix = 'test'

    for filename in [horizons_file, obs_file]:
        with open(filename, 'w') as f:
            f.write('test')

    args = [horizons_file, obs_file, prefix]
    args = genastmodel2.parse_args(args)
    args.params = args.params
    args.obs_file = args.obs_file
    args = genastmodel2.check_args(args)

    assert args.params == horizons_file
    assert args.obs_file == obs_file
    assert args.outprefix == prefix

    for filename in [horizons_file, obs_file]:
        os.remove(filename)


def test_check_args_exception():
    horizons_file = 'horizons.out'
    obs_file = 'obs_file.txt'
    prefix = 'test'

    args = [horizons_file, obs_file, prefix]
    args = genastmodel2.parse_args(args)
    with pytest.raises(PipeCalError):
        _ = genastmodel2.check_args(args)

    with open(horizons_file, 'w') as f:
        f.write('test')

    args = [horizons_file, obs_file, prefix]
    args = genastmodel2.parse_args(args)
    with pytest.raises(PipeCalError):
        _ = genastmodel2.check_args(args)

    os.remove(horizons_file)
