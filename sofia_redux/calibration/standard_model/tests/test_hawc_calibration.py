# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import glob
import datetime
import pandas as pd
import pytest
import numpy.testing as npt

import sofia_redux.calibration.standard_model.hawc_calibration as hc
from sofia_redux.calibration.standard_model.tests import resources
from sofia_redux.calibration.pipecal_error import PipeCalError


@pytest.fixture
def herschel_path():
    import sofia_redux.calibration
    module_path = os.path.dirname(sofia_redux.calibration.__file__)
    herschel_path = os.path.join(module_path, 'data', 'models',
                                 'call_esa2_2_i.dat')
    return herschel_path


@pytest.fixture
def caldata(herschel_path):
    return os.path.dirname(herschel_path)


@pytest.fixture(params=['Ceres', 'Neptune'])
def obs_file(request, tmpdir):
    target = request.param
    filename = tmpdir.join(f'{target}_obstimes.txt')
    with open(filename, 'w') as f:
        f.write(f'2016-12-01 05:00:00    {target}\n')
        f.write(f'2016-12-16 05:00:00    {target}\n')
        f.write(f'2017-10-17 11:00:00    {target}\n')
        f.write(f'2017-11-16 07:00:00    {target}\n')
    return os.path.abspath(filename)


@pytest.fixture(params=['Ceres', 'Neptune'])
def bad_obs_file(request, tmpdir):
    target = request.param
    filename = tmpdir.join(f'{target}_obstimes.txt')
    with open(filename, 'w') as f:
        f.write('2016-12-01 05:00:00\n')
        f.write('2016-12-16 05:00:00\n')
        f.write('2017-10-17 11:00:00\n')
        f.write('2017-11-16 07:00:00\n')
    return os.path.abspath(filename)


@pytest.fixture(params=['Ceres', 'Neptune'])
def obs_file_bad_date(request, tmpdir):
    target = request.param
    filename = tmpdir.join(f'{target}_obstimes.txt')
    with open(filename, 'w') as f:
        f.write(f'2016-12-01 05:00:00    {target}\n')
        f.write(f'2016-15-16 05:00:00    {target}\n')
        f.write(f'2017-10-17 11:00:00    {target}\n')
        f.write(f'2017-11-16 07:00:00    {target}\n')
    return os.path.abspath(filename)


@pytest.fixture
def atran(tmpdir):
    return resources.atran(str(tmpdir), num_points=100)


@pytest.fixture
def date():
    return resources.timepoint().date()


@pytest.fixture
def time():
    return resources.timepoint().time()


@pytest.fixture
def minor_row(date, time):
    r = pd.Series({'target': 'Ceres', 'date': date, 'time': time})
    return r


@pytest.fixture
def major_row(date, time):
    r = pd.Series({'target': 'Callisto', 'date': date, 'time': time})
    return r


@pytest.mark.parametrize('target,expected',
                         [('Mars', 1.16715), ('Jupiter', 4.76609),
                          ('Saturn', 8.76912), ('Uranus', 19.90532),
                          ('Neptune', 29.49400), ('Callisto', 4.76609),
                          ('Ganymede', 4.76609), ('Europa', 4.93100),
                          ('Io', 4.93100), ('Titan', 8.76913)])
def test_model_dist(target, expected, caldata):
    model = hc.model_dist(target, caldata)
    npt.assert_almost_equal(model, expected, decimal=5)


def test_bad_target(caldata):
    with pytest.raises(PipeCalError):
        _ = hc.model_dist('Sun', caldata)


def test_generate_major_outfile(herschel_path, date):
    target = 'Callisto'
    expected = 'Callisto_2017Sep28_ESA2.txt'
    model = hc.generate_major_outfile(target, herschel_path, date.__str__())
    assert model == expected


def test_generate_minor_outfile(minor_row):
    outfile = hc.generate_minor_outfile(minor_row)
    assert outfile == 'Ceres_2017-09-28_110820_model.out'


def test_parse_atran_filename(atran):
    altitude, zenith = hc.parse_atran_filename(atran)
    assert altitude == '41K'
    assert zenith == '45deg'


def test_generate_major_cal_outfile(major_row, atran, herschel_path):
    outfile = hc.generate_major_cal_outfile(major_row, atran, herschel_path)
    assert outfile == 'HAWC_Callisto_ESA2_2017Sep28_41K_45deg.out'

    major_row['date'] = major_row['date'].__str__()
    outfile = hc.generate_major_cal_outfile(major_row, atran, herschel_path)
    assert outfile == 'HAWC_Callisto_ESA2_2017Sep28_41K_45deg.out'


def test_generate_minor_cal_outfile(minor_row, atran):
    outfile = hc.generate_minor_cal_outfile(minor_row, atran)
    assert outfile == 'HAWC_Ceres_2017Sep28_41K_45deg.out'

    minor_row['date'] = minor_row['date'].__str__()
    outfile = hc.generate_minor_cal_outfile(minor_row, atran)
    assert outfile == 'HAWC_Ceres_2017Sep28_41K_45deg.out'


def test_parse_args(obs_file, atran):
    args = [obs_file, '-a', atran]
    args = hc.parse_args(args)
    assert args.obs_file == [obs_file]
    assert args.atran == atran


def test_check_args(obs_file, atran):
    args = [obs_file, '-a', atran]
    args = hc.parse_args(args)
    args = hc.check_args(args)
    assert args.obs_file == obs_file
    assert args.atran == atran

    args = [obs_file]
    args = hc.parse_args(args)
    args = hc.check_args(args)
    assert args.obs_file == obs_file
    assert args.atran == 'atran_41K_45deg_40-300mum.fits'


@pytest.mark.parametrize('target,expected',
                         [('Uranus', 'major'), ('neptune', 'major'),
                          ('ganymede', 'major'), ('Callisto', 'major'),
                          ('ceres', 'minor'), ('Vesta', 'minor')])
def test_classify_target(target, expected):
    classification = hc.classify_target(target)
    assert classification == expected


def test_read_obstimes(obs_file):
    obs_times = hc.read_obstimes(obs_file)
    assert len(obs_times['date']) == 4
    assert len(set(obs_times['target'])) == 1
    assert all(obs_times['datetime'].apply(lambda x: x.time() < datetime.time(
        hour=12)))


def test_read_bad_obstimes(bad_obs_file, obs_file_bad_date):
    with pytest.raises(PipeCalError):
        hc.read_obstimes('dummy.file')

    with pytest.raises(PipeCalError):
        hc.read_obstimes(bad_obs_file)

    with pytest.raises(PipeCalError):
        hc.read_obstimes(obs_file_bad_date)


@pytest.mark.parametrize('target,expected',
                         [('uranus', 'ura_esa2_2_i.dat'),
                          ('neptune', 'nep_esa5_2_i.dat'),
                          ('ganymede', 'gany_esa2_2_i.dat'),
                          ('callisto', 'call_esa2_2_i.dat')])
def test_select_herschel_file(target, expected):
    herschel_model = hc.select_herschel_file(target)
    assert os.path.basename(herschel_model) == expected


def test_calibration_data_path():
    caldata = hc.calibration_data_path()
    assert os.path.isdir(caldata)
    for filt in list('ABCDE'):
        assert os.path.isfile(os.path.join(caldata, f'HAWC_band{filt}.txt'))


@pytest.mark.parametrize('target,expected',
                         [('Neptune', 29.493995),
                          ('uranus', 19.905319),
                          ('io', 4.931002)])
def test_model_dist_calpath(target, expected):
    caldata = hc.calibration_data_path()
    distance = hc.model_dist(target, caldata)
    npt.assert_almost_equal(distance, expected, decimal=6)


@pytest.mark.parametrize('distance,params,expected',
                         [(2, {'delta': 4}, 0.25),
                          (10, {'delta': 30}, 0.111111),
                          (10, {'delta': 5}, 4.0)])
def test_scale_factor(distance, params, expected):
    fscale = hc.scale_factor(distance, params)
    npt.assert_almost_equal(fscale, expected, decimal=5)


def test_apply_scale_factor(tmpdir):
    with tmpdir.as_cwd():
        herschel_file = hc.select_herschel_file('uranus')
        outfile = tmpdir / 'test.out'
        fscale = 0.25
        model = hc.apply_scale_factor(herschel_file, outfile, fscale)
        assert os.path.isfile(outfile)
        assert isinstance(model, pd.DataFrame)
        plots = glob.glob('scaled_flux*eps')
        for plot in plots:
            assert os.path.isfile(plot)
        assert os.path.isfile('scaled_flux_ura_esa2_2_i.png')


def test_model_minor_body(obs_file, atran, tmpdir):
    with tmpdir.as_cwd():
        obs_times = hc.read_obstimes(obs_file)
        if obs_times['target'][0] == 'Ceres':
            obs = hc.model_minor_body(obs_times, atran)
            assert obs.shape == (len(obs_times), 6)
        outputs = glob.glob('Ceres_*[.txt,model.out]')
        for output in outputs:
            assert os.path.isfile(output)


def test_model_major_body(obs_file, caldata, atran, tmpdir):
    requests = pytest.importorskip('requests')
    with tmpdir.as_cwd():
        obs_times = hc.read_obstimes(obs_file)
        if obs_times['target'][0] == 'Neptune':
            try:
                obs = hc.model_major_body(obs_times, caldata, atran)
            except requests.exceptions.ConnectionError:
                # sometimes internet hiccups happen; don't fail
                # the test in this case
                return
            assert obs.shape == (len(obs_times), 6)
        outputs = glob.glob('Neptune_*txt')
        model_outputs = glob.glob('Neptune_*model.out')
        outputs += model_outputs
        for output in outputs:
            assert os.path.isfile(output)


def test_main(obs_file, atran, tmpdir):
    with tmpdir.as_cwd():
        hc.main([obs_file, '-a', atran])
        target = obs_file.split('_')[0]
        model_files = glob.glob(f'{target}*_model.out')
        param_files = glob.glob(f'{target}*_params.out')
        outer_files = glob.glob(f'{target}*.txt')
        images = glob.glob(f'HAWC_{target}*png')
        others = glob.glob(f'HAWC_{target}*deg.out')
        filenames = model_files + param_files + outer_files + images + others
        for filename in filenames:
            assert os.path.isfile(filename)


def test_calibration(obs_file, atran, capsys, tmpdir):
    with tmpdir.as_cwd():
        hc.calibration(obs_file, atran)
        captured = capsys.readouterr()
        assert 'Using ATRAN' in captured.out
        assert 'Plotting' in captured.out
        assert 'Lambda_iso' in captured.out

        target = os.path.basename(obs_file).split('_')[0]
        model_files = glob.glob(f'{target}*_model.out')
        param_files = glob.glob(f'{target}*_params.out')
        outer_files = glob.glob(f'{target}*.txt')
        images = glob.glob(f'HAWC_{target}*png')
        others = glob.glob(f'HAWC_{target}*deg.out')
        filenames = model_files + param_files + outer_files + images + others
        for filename in filenames:
            assert os.path.isfile(filename)
