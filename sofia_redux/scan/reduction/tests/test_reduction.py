# Licensed under a 3-clause BSD style license - see LICENSE.rst

import json
import os
import shutil
import time

from astropy.io import fits
from astropy import units as u
from configobj import ConfigObj
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.example.channels.channels import ExampleChannels
from sofia_redux.scan.custom.example.info.info import ExampleInfo
from sofia_redux.scan.custom.example.integration.integration \
    import ExampleIntegration
from sofia_redux.scan.custom.example.scan.scan import ExampleScan
from sofia_redux.scan.pipeline.pipeline import Pipeline
from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.scan.source_models.astro_intensity_map \
    import AstroIntensityMap
from sofia_redux.toolkit.utilities.fits import set_log_level


class MockScan(object):
    """Mock scan behavior."""
    def __init__(self, valid=True):
        self.valid = valid
        self.size = 1
        self.configuration = Configuration()
        self.id = 0

    def is_valid(self):
        return self.valid

    def get_observing_time(self):
        return 1.0 * u.s

    def validate(self):
        if not self.is_valid():
            self.size = 0

    def get_id(self):
        return self.id

    def write_products(self):
        print(f'write scan {self.id}')


class TestReduction(object):
    @pytest.fixture(autouse=True)
    def cd_tmpdir(self, tmpdir):
        with tmpdir.as_cwd():
            yield

    def test_instrument(self):
        # nonexistent instrument
        with pytest.raises(ModuleNotFoundError):
            Reduction('test')

        # implemented instrument
        reduction = Reduction('example')
        assert reduction.instrument == 'example'
        assert reduction.name == 'example'
        assert isinstance(reduction.channels, ExampleChannels)
        assert isinstance(reduction.info, ExampleInfo)
        assert isinstance(reduction.configuration, Configuration)

        # None instrument: succeeds, no info or channels set
        reduction = Reduction(None)
        assert reduction.instrument is None
        assert reduction.name is None
        assert reduction.channels is None
        assert reduction.info is None
        assert reduction.configuration is None

    def test_blank_copy(self):
        # nothing returned if no instrument
        reduction = Reduction(None)
        blank = reduction.blank_copy()
        assert blank is None

        # test reduction: only config copied, not scans, etc.
        reduction = Reduction('example')
        reduction.scans = ['test', 'value']

        blank = reduction.blank_copy()
        assert isinstance(blank, Reduction)
        assert blank.instrument == 'example'
        assert isinstance(blank.channels, ExampleChannels)
        assert len(blank.scans) == 0

    def test_size(self):
        reduction = Reduction('example')
        assert reduction.size == 0

        reduction.scans = None
        assert reduction.size == 0

        reduction.scans = []
        assert reduction.size == 0

        reduction.scans = ['a', 'b', 'c']
        assert reduction.size == 3

    def test_rounds(self):
        reduction = Reduction('example')
        reduction.rounds = 4
        assert reduction.rounds == 4
        assert reduction.configuration.iterations.max_iteration == 4
        reduction.rounds = 2
        assert reduction.rounds == 2
        assert reduction.configuration.iterations.max_iteration == 2

        reduction = Reduction(None)
        assert reduction.rounds == 0
        with pytest.raises(ValueError) as err:
            reduction.rounds = 4
        assert 'Cannot set rounds' in str(err)

    def test_info(self):
        reduction = Reduction('example')
        reduction.info = 'test'
        assert reduction.info == 'test'
        assert reduction.channels.info == 'test'

        reduction = Reduction(None)
        assert reduction.info is None
        with pytest.raises(ValueError) as err:
            reduction.info = 'test'
        assert 'Cannot set info' in str(err)

    def test_sub_reduction_info(self):
        reduction = Reduction(None)
        assert reduction.total_reductions == 1

        reduction.sub_reductions = ['test', 'values']
        assert reduction.total_reductions == 2
        assert not reduction.is_sub_reduction
        assert reduction.reduction_id == f'{id(reduction)}.0'

        reduction2 = Reduction(None)
        reduction2.parent_reduction = reduction
        assert reduction2.total_reductions == 2
        assert reduction2.is_sub_reduction
        assert reduction2.is_sub_reduction
        assert reduction2.reduction_id == f'{id(reduction)}.0-' \
                                          f'{id(reduction2)}.0'

    def test_iteration(self):
        reduction = Reduction(None)
        assert reduction.iteration() == 0

        reduction = Reduction('example')
        reduction.configuration.iterations.current_iteration = 1
        assert reduction.iteration() == 1
        reduction.configuration.iterations = None
        assert reduction.iteration() == 0

    def test_read_scan(self, mocker, capsys):
        reduction = Reduction('example')
        mocker.patch.object(reduction.channels, 'read_scan',
                            return_value='test')
        assert reduction.read_scan('test.fits') == 'test'
        assert 'Reading scan: test.fits' in capsys.readouterr().out

    def test_read_scans(self, mocker, capsys, tmpdir, scan_file, bad_file):
        reduction = Reduction('example')
        mocker.patch('psutil.cpu_count', return_value=1)

        # no files
        reduction.read_scans()
        assert 'No files' in capsys.readouterr().err

        # read some files
        f1 = scan_file
        f2 = str(tmpdir.join('test2.fits'))
        f3 = str(tmpdir.join('test3.fits'))
        shutil.copyfile(scan_file, f2)
        shutil.copyfile(scan_file, f3)
        with set_log_level('DEBUG'):
            reduction.read_scans(filenames=[f1, f2, f3])
        assert len(reduction.scans) == 3
        for scan in reduction.scans:
            assert isinstance(scan, ExampleScan)
        assert 'Reading 3 files' in capsys.readouterr().out

        # try to read an invalid file: ignored, good files kept
        f4 = bad_file
        with set_log_level('DEBUG'):
            reduction.read_scans(filenames=[f1, f4])
        assert len(reduction.scans) == 1
        assert 'Reading 2 files' in capsys.readouterr().out

        # read 1 file, but split it
        reduction.configuration.apply_configuration_options(
            {'segment': 100, 'subscan': {'split': True, 'minlength': 1.0}})
        with set_log_level('DEBUG'):
            reduction.read_scans(filenames=[f1])
        assert len(reduction.scans) == 2
        assert 'Reading 1 files' in capsys.readouterr().out

        # check call to read when sub_reductions is not None:
        # bypasses regular read
        m1 = mocker.patch.object(reduction, 'read_sub_reduction_scans')
        reduction.sub_reductions = [Reduction(None), Reduction(None)]
        with set_log_level('DEBUG'):
            reduction.read_scans([f1, f2])
        m1.assert_called_once()
        assert 'Reading 2 files' not in capsys.readouterr().out

    def test_return_scan_from_read_arguments(self, mocker, capsys,
                                             scan_file, bad_file):
        # mock unpickling to return example class
        reduction = Reduction('example')
        mocker.patch('sofia_redux.toolkit.utilities.'
                     'multiprocessing.unpickle_file',
                     return_value=(reduction.channels, None))

        # read a good file
        scan = Reduction.return_scan_from_read_arguments(scan_file, 'test')
        assert isinstance(scan, ExampleScan)
        assert 'Successfully read scan' in capsys.readouterr().out

        # read a bad file
        scan = Reduction.return_scan_from_read_arguments(bad_file, 'test')
        assert scan is None
        capt = capsys.readouterr()
        assert 'contains no valid data' in capt.err
        assert 'Successfully read scan' not in capt.out

        # mock a file that is a scan but does not validate
        mocker.patch.object(reduction.channels, 'read_scan',
                            return_value=MockScan(False))
        scan = Reduction.return_scan_from_read_arguments(scan_file, 'test')
        assert scan is None
        capt = capsys.readouterr()
        assert 'contains no valid data' in capt.err
        assert 'Successfully read scan' not in capt.out

    def test_read_sub_reduction_scans(self, capsys, scan_file, bad_file):
        # raises error for no reductions
        reduction = Reduction(None)
        with pytest.raises(ValueError) as err:
            reduction.read_sub_reduction_scans()
        assert 'No sub-reductions' in str(err)
        reduction.sub_reductions = []
        with pytest.raises(ValueError):
            reduction.read_sub_reduction_scans()
        assert 'No sub-reductions' in str(err)

        # set one sub-reduction, no files assigned
        subred = Reduction('example')
        reduction.sub_reductions = [subred]
        with set_log_level('DEBUG'):
            reduction.read_sub_reduction_scans()
        assert 'Reading 0 files' in capsys.readouterr().out
        assert len(subred.scans) == 0

        # assign a good and bad file and read them
        subred.reduction_files = [scan_file, bad_file]
        with set_log_level('DEBUG'):
            reduction.read_sub_reduction_scans()
        assert 'Reading 2 files' in capsys.readouterr().out
        assert len(subred.scans) == 1
        assert isinstance(subred.scans[0], ExampleScan)

        # read 1 file, but split it
        subred.reduction_files = [scan_file]
        subred.configuration.apply_configuration_options(
            {'segment': 100, 'subscan': {'split': True, 'minlength': 1.0}})
        with set_log_level('DEBUG'):
            reduction.read_sub_reduction_scans()
        assert 'Reading 1 files' in capsys.readouterr().out
        assert len(subred.scans) == 2

    def test_parallel_safe_read_all_files(self, mocker):
        # mock scan read, pickling
        mocker.patch.object(Reduction, 'return_scan_from_read_arguments',
                            return_value='test1')
        mocker.patch('sofia_redux.toolkit.utilities.'
                     'multiprocessing.pickle_object',
                     return_value='test2')

        # none pickle directory: skip pickling, return scan
        args = ([[1, 2, 3]], None)
        out = Reduction.parallel_safe_read_all_files(args, 0)
        assert out == 'test1'

        # non-none pickle directory: return pickle file
        args = ([[1, 2, 3]], 'pickle')
        out = Reduction.parallel_safe_read_all_files(args, 0)
        assert out == 'test2'

    def test_parallel_safe_read_sub_reduction_scans(self, mocker, capsys,
                                                    scan_file):
        # return none if no files assigned
        subs = [Reduction(None)]
        out = Reduction.parallel_safe_read_sub_reduction_scans((subs,), 0)
        assert 'No reduction files' in capsys.readouterr().err
        assert out is None

        # read a scan if specified
        subs = [Reduction('example')]
        subs[0].reduction_files = [scan_file]
        out = Reduction.parallel_safe_read_sub_reduction_scans((subs,), 0)
        assert 'No reduction files' not in capsys.readouterr().err
        assert isinstance(out, Reduction)
        assert len(out.scans) == 1
        assert isinstance(out.scans[0], ExampleScan)

    def test_assign_reduction_files(self):
        reduction = Reduction(None)

        # string specification: allows split on ; or tab
        reduction.assign_reduction_files('')
        assert reduction.reduction_files == []
        reduction.assign_reduction_files('1; 2; 3')
        assert reduction.reduction_files == ['1', '2', '3']
        reduction.assign_reduction_files('1\t2\t3')
        assert reduction.reduction_files == ['1', '2', '3']
        reduction.assign_reduction_files('test.fits')
        assert reduction.reduction_files == ['test.fits']

        # list specification
        reduction.assign_reduction_files(['test.fits'])
        assert reduction.reduction_files == ['test.fits']

        # does grouping for sub-reductions
        # raises error if files/reductions are mismatched
        reduction.sub_reductions = [Reduction(None), Reduction(None)]
        with pytest.raises(ValueError) as err:
            reduction.assign_reduction_files(['test.fits'])
        assert 'does not match' in str(err)

        # otherwise distributes 1 for 1
        reduction.assign_reduction_files(['test1.fits', 'test2.fits'])
        assert reduction.sub_reductions[0].reduction_files == ['test1.fits']
        assert reduction.sub_reductions[1].reduction_files == ['test2.fits']

    def test_validate(self, mocker, scan_file, bad_file, capsys):
        reduction = Reduction('example')

        # error for no scans
        with pytest.raises(ValueError) as err:
            reduction.validate()
        assert 'No scans to reduce' in str(err)

        # read scans
        reduction.read_scans([scan_file, bad_file])
        reduction.validate()
        assert len(reduction.scans) == 1

        # no valid scans in reduction
        mocker.patch.object(reduction, 'is_valid', return_value=False)
        reduction.validate()
        assert 'Reduction contains no valid scans' in capsys.readouterr().err

        # none in subreduction
        reduction.parent_reduction = Reduction(None)
        reduction.validate()
        assert 'Sub-reduction 0 contains ' \
               'no valid scans' in capsys.readouterr().err

    def test_validate_scans(self):
        reduction = Reduction(None)
        assert len(reduction.scans) == 0

        # no-op without scans
        reduction.validate_scans()
        assert len(reduction.scans) == 0

        # invalid scans are dropped
        reduction.scans = [MockScan(True), MockScan(False), MockScan(True)]
        reduction.validate_scans()
        assert len(reduction.scans) == 2

    def test_is_valid(self):
        # not valid without valid scans
        reduction = Reduction(None)
        assert not reduction.is_valid()

        reduction.scans = [MockScan(False), MockScan(False)]
        assert not reduction.is_valid()

        # valid with at least one good scan
        reduction.scans = [MockScan(True), MockScan(False)]
        assert reduction.is_valid()

    def test_validate_sub_reductions(self, mocker, capsys,
                                     scan_file, bad_file):
        reduction = Reduction('example')

        # error for no subs
        with pytest.raises(ValueError) as err:
            reduction.validate_sub_reductions()
        assert 'does not contain sub-reductions' in str(err)

        subred = Reduction('example')
        reduction.sub_reductions = [subred]

        # works from top level without scans
        reduction.validate()
        assert len(subred.scans) == 0
        assert 'no valid sub-reductions' in capsys.readouterr().err

        # read scans
        reduction.sub_reductions = [subred]
        subred.read_scans([scan_file, bad_file])
        reduction.validate_sub_reductions()
        assert len(subred.scans) == 1

        # read 2 good files, common wcs: makes common source
        subred2 = Reduction('example')
        subred2.reduction_number = 1
        reduction.sub_reductions = [subred, subred2]
        reduction.configuration.apply_configuration_options(
            {'commonwcs': True, 'parallel': False})
        subred2.read_scans([scan_file])

        # mock source model inits
        m1 = mocker.patch.object(reduction, 'init_collective_source_model')
        m2 = mocker.patch.object(subred, 'init_source_model')
        m3 = mocker.patch.object(subred2, 'init_source_model')
        reduction.validate_sub_reductions()
        m1.assert_called_once()

        # add a start time to check log message
        reduction.read_start_time = time.time()

        # without common wcs: calls separate init in subreductions
        reduction.configuration.apply_configuration_options(
            {'blacklist': 'commonwcs'})
        with set_log_level('DEBUG'):
            reduction.validate_sub_reductions()
        m1.assert_called_once()
        m2.assert_called()
        m3.assert_called()
        assert 'Total read time' in capsys.readouterr().out

    def test_parallel_safe_validate_sub_reductions(self, mocker):
        # mock some subreductions to validate
        subs = []
        mocks = []
        for i in range(4):
            sub = Reduction(None)
            mocks.append(mocker.patch.object(sub, 'validate'))
            sub.reduction_number = i
            subs.append(sub)

        # check that correct sub is returned
        s0 = Reduction.parallel_safe_validate_sub_reductions(subs, 0)
        s1 = Reduction.parallel_safe_validate_sub_reductions(subs, 1)
        s2 = Reduction.parallel_safe_validate_sub_reductions(subs, 2)
        assert s0.reduction_number == 0
        assert s1.reduction_number == 1
        assert s2.reduction_number == 2

        # check that validate was called for first three, not the last
        for m in mocks[:-1]:
            m.assert_called()
        mocks[-1].assert_not_called()

    def test_parallel_safe_init_pipelines(self, mocker):
        # mock some subreductions to init
        subs = []
        mocks = []
        for i in range(4):
            sub = Reduction(None)
            mocks.append(mocker.patch.object(sub, 'init_pipelines'))
            sub.reduction_number = i
            subs.append(sub)

        # check that correct sub is returned
        s0 = Reduction.parallel_safe_init_pipelines(subs, 0)
        s1 = Reduction.parallel_safe_init_pipelines(subs, 1)
        s2 = Reduction.parallel_safe_init_pipelines(subs, 2)
        assert s0.reduction_number == 0
        assert s1.reduction_number == 1
        assert s2.reduction_number == 2

        # check that init was called for first three, not the last
        for m in mocks[:-1]:
            m.assert_called()
        mocks[-1].assert_not_called()

    def test_assign_sub_reductions(self):
        subs = []
        for i in range(4):
            sub = Reduction(None)
            subs.append(sub)

        parent = Reduction(None)
        parent.assign_sub_reductions(subs)

        # parent is set in each sub
        for sub in subs:
            assert sub.parent_reduction is parent

    def test_set_observing_time_options(self, mocker):
        reduction = Reduction('example')
        reduction.scans = [MockScan(), MockScan()]
        mock1 = mocker.patch.object(reduction, 'apply_options_to_scans')

        # set obstime<45 {'stability': '2.5'}: applies to data
        expected = ConfigObj({'stability': '2.5'})
        reduction.configuration.conditions.options['obstime<45'] = expected
        reduction.set_observing_time_options()
        mock1.assert_called_with(expected)
        del reduction.configuration.conditions.options['obstime<45']

        # set option obstime>3 {'stability': '3.0'}: does not apply to data
        mock2 = mocker.patch.object(reduction, 'apply_options_to_scans')
        not_expected = ConfigObj({'stability': '3.0'})
        reduction.configuration.conditions.options['obstime>3'] = not_expected
        reduction.set_observing_time_options()
        mock2.assert_not_called()

        # add scans: should now apply to data
        reduction.scans.extend([MockScan(), MockScan()])
        reduction.set_observing_time_options()
        mock2.assert_called_with(not_expected)
        del reduction.configuration.conditions.options['obstime>3']

        # set a bad option: should be skipped
        mock3 = mocker.patch.object(reduction, 'apply_options_to_scans')
        reduction.configuration.conditions.options['obstime'] = not_expected
        reduction.set_observing_time_options()
        mock3.assert_not_called()

    def test_apply_options_to_scans(self):
        reduction = Reduction(None)
        reduction.scans = [MockScan(), MockScan()]

        # new options are added to each scan
        reduction.apply_options_to_scans({'test': 'option'})
        for scan in reduction.scans:
            assert scan.configuration['test'] == 'option'

    def test_get_observing_time(self):
        # empty reduction gives 0 obs time
        reduction = Reduction(None)
        assert reduction.get_total_observing_time() == 0 * u.s

        # mock scans return 1s obs time each
        reduction.scans = [MockScan(), MockScan()]
        assert reduction.get_total_observing_time() == 2 * u.s

    def test_init_source_model(self, mocker, capsys):
        reduction = Reduction('example')
        reduction.scans = [MockScan(), MockScan()]

        # return None source
        mocker.patch.object(reduction.info, 'get_source_model_instance',
                            return_value=None)
        reduction.init_source_model()
        assert 'No source model' in capsys.readouterr().err

        # return some source
        source = AstroIntensityMap(reduction.info)
        mocker.patch.object(reduction.info, 'get_source_model_instance',
                            return_value=source)
        m1 = mocker.patch.object(source, 'create_from')
        m2 = mocker.patch.object(source, 'assign_reduction')

        # create and assign reduction to source
        reduction.init_source_model()
        m1.assert_called_once()
        m2.assert_called_once()

    def test_init_collective_source_model(self, mocker, capsys):
        reduction = Reduction('example')

        # no subreductions: raise error
        with pytest.raises(ValueError) as err:
            reduction.init_collective_source_model()
        assert 'no sub-reductions' in str(err)

        # assign subreductions
        subs = []
        for i in range(2):
            sub = Reduction('example')
            subs.append(sub)
        reduction.sub_reductions = subs

        # warns without scans
        reduction.init_collective_source_model()
        assert 'No data' in capsys.readouterr().err

        # add scans
        for sub in subs:
            sub.scans = [MockScan()]

        # invalid model: None source
        mocker.patch.object(subs[0].info, 'get_source_model_instance',
                            return_value=None)
        reduction.init_collective_source_model()
        assert 'No source model' in capsys.readouterr().err

        # return some source
        source = AstroIntensityMap(reduction.info)
        m1 = mocker.patch.object(source, 'create_from')
        m2 = mocker.patch.object(source, 'get_clean_local_copy',
                                 return_value=source)
        m3 = mocker.patch.object(source, 'assign_reduction')
        mocker.patch.object(subs[0].info, 'get_source_model_instance',
                            return_value=source)
        reduction.init_collective_source_model()
        # source is created once; copied, assigned twice
        m1.assert_called_once()
        m2.assert_called()
        m3.assert_called()

    def test_update_runtime_config(self, tmpdir):
        reduction = Reduction('example')
        outpath = str(tmpdir.join('test'))
        reduction.configuration.apply_configuration_options(
            {'outpath': outpath})

        # error if it doesn't exist
        with pytest.raises(ValueError):
            reduction.update_runtime_config()

        # unless create is set
        reduction.configuration.apply_configuration_options(
            {'outpath': outpath, 'outpath.create': True})
        reduction.update_runtime_config()
        assert os.path.isdir(outpath)

    def test_update_parallel_config(self, mocker):
        # works without config
        reduction = Reduction(None)
        reduction.update_parallel_config()
        assert reduction.max_jobs == 1
        assert reduction.max_cores == 1

        # reduction with blank config
        reduction = Reduction('example')
        conf = Configuration()
        reduction.info.configuration = conf

        # set hybrid mode, all jobs all cores
        conf.apply_configuration_options(
            {'parallel': {'mode': 'hybrid', 'jobs': -1, 'cores': -1}})

        # mock 1 core
        mocker.patch('psutil.cpu_count', return_value=1)
        reduction.update_parallel_config()
        assert reduction.max_jobs == 1
        assert reduction.max_cores == 1

        # mock 8 cores
        mocker.patch('psutil.cpu_count', return_value=8)
        reduction.update_parallel_config()
        assert reduction.max_cores == 8
        assert reduction.max_jobs == 8

        # specify 0.5 cores instead, but still full jobs
        conf.apply_configuration_options(
            {'parallel': {'cores': 0.5}})
        reduction.update_parallel_config()
        assert reduction.max_cores == 4
        assert reduction.max_jobs == 8

        # specify half jobs
        reduction.configuration.apply_configuration_options(
            {'parallel': {'jobs': 0.5}})
        reduction.update_parallel_config()
        assert reduction.max_cores == 4
        assert reduction.max_jobs == 4

        # idle 3/4 of cores instead
        reduction.configuration.apply_configuration_options(
            {'parallel': {'idle': 0.75}})
        reduction.update_parallel_config()
        assert reduction.max_cores == 2
        assert reduction.max_jobs == 2

        # same if cores/jobs not specified
        reduction.configuration.apply_configuration_options(
            {'blacklist': 'parallel.jobs,parallel.cores'})
        reduction.update_parallel_config()
        assert reduction.max_cores == 2
        assert reduction.max_jobs == 2

    def test_assign_parallel_jobs(self):
        # config for parallel with 4 cores/jobs available
        reduction = Reduction('example')
        conf = reduction.configuration
        conf.apply_configuration_options(
            {'parallel': {'mode': 'hybrid', 'jobs': -1, 'cores': -1}})
        reduction.max_cores = 4
        reduction.max_jobs = 4

        # no sub-reductions, no files loaded
        reduction.assign_parallel_jobs(reset=True)
        assert reduction.jobs_assigned
        assert reduction.parallel_read == 4
        assert reduction.parallel_scans == 1
        assert reduction.parallel_tasks == 4
        assert reduction.available_reduction_jobs == 4

        # assign two files but don't read them
        reduction.reduction_files = ['test1.fits', 'test2.fits']
        reduction.assign_parallel_jobs(reset=True)
        assert reduction.parallel_read == 2
        assert reduction.parallel_scans == 1
        assert reduction.parallel_tasks == 4
        assert reduction.available_reduction_jobs == 4

        # load 2 files
        reduction.scans = [MockScan(), MockScan()]
        reduction.assign_parallel_jobs(reset=True)
        assert reduction.parallel_read == 2
        assert reduction.parallel_scans == 2
        assert reduction.parallel_tasks == 2
        assert reduction.available_reduction_jobs == 4

        # change config and rerun: no change without reset
        conf.apply_configuration_options({'parallel.mode': 'scans'})
        reduction.assign_parallel_jobs()
        assert reduction.parallel_scans == 2
        assert reduction.parallel_tasks == 2
        assert reduction.available_reduction_jobs == 4

        # updates with reset flag
        reduction.assign_parallel_jobs(reset=True)
        assert reduction.parallel_scans == 2
        assert reduction.parallel_tasks == 1
        assert reduction.available_reduction_jobs == 2

        # change mode: ops parallel
        conf.apply_configuration_options({'parallel.mode': 'ops'})
        reduction.assign_parallel_jobs(reset=True)
        assert reduction.parallel_scans == 1
        assert reduction.parallel_tasks == 4
        assert reduction.available_reduction_jobs == 4

    def test_assign_parallel_jobs_subreductions(self, mocker):
        # config for parallel with 4 cores/jobs available
        reduction = Reduction('example')
        conf = reduction.configuration
        conf.apply_configuration_options(
            {'parallel': {'mode': 'hybrid', 'jobs': -1, 'cores': -1}})
        reduction.max_cores = 4
        reduction.max_jobs = 4

        # add subs, no scans
        subs = []
        for i in range(3):
            sub = Reduction('example')
            sub.parent_reduction = reduction
            subs.append(sub)
        reduction.sub_reductions = subs

        reduction.assign_parallel_jobs()
        assert reduction.jobs_assigned
        assert reduction.parallel_read == 3
        assert reduction.parallel_scans == 1
        assert reduction.parallel_tasks == 4
        assert reduction.available_reduction_jobs == 4
        for sub in subs:
            assert sub.jobs_assigned
            assert sub.parallel_read == 1
            assert sub.parallel_scans == 1
            # one sub gets the spare thread
            assert sub.parallel_tasks in [1, 2]
            assert sub.available_reduction_jobs in [1, 2]

        # add more scans than cores
        for sub in subs:
            sub.reduction_files = []
            for i in range(5):
                sub.reduction_files.append(f'test{i}.fits')
                sub.scans.append(MockScan())

        reduction.assign_parallel_jobs(reset=True)
        assert reduction.parallel_read == 4
        assert reduction.parallel_scans == 4
        assert reduction.parallel_tasks == 1
        assert reduction.available_reduction_jobs == 4
        for sub in subs:
            assert sub.parallel_read in [1, 2]
            assert sub.parallel_scans in [1, 2]
            assert sub.parallel_tasks in [1, 2]
            assert sub.available_reduction_jobs in [1, 2]

        # change mode: scans only
        conf.apply_configuration_options({'parallel.mode': 'scans'})
        reduction.assign_parallel_jobs(reset=True)
        assert reduction.parallel_scans == 4
        assert reduction.parallel_tasks == 1
        assert reduction.available_reduction_jobs == 4
        for sub in subs:
            assert sub.parallel_scans in [1, 2]
            assert sub.parallel_tasks == 1
            assert sub.available_reduction_jobs in [1, 2]

        # change mode: ops only
        conf.apply_configuration_options({'parallel.mode': 'ops'})
        reduction.assign_parallel_jobs(reset=True)
        assert reduction.parallel_scans == 1
        assert reduction.parallel_tasks == 4
        assert reduction.available_reduction_jobs == 4
        for sub in subs:
            assert sub.parallel_scans == 1
            assert sub.parallel_tasks in [1, 2]
            assert sub.available_reduction_jobs in [1, 2]

        # go back to hybrid, make more cores than tasks
        conf.apply_configuration_options({'parallel.mode': 'hybrid'})
        sub = reduction.sub_reductions[0]
        reduction.sub_reductions = [sub]
        reduction.max_jobs = 12
        reduction.assign_parallel_jobs(reset=True)
        assert reduction.parallel_scans == 5
        assert reduction.parallel_tasks == 2
        assert reduction.available_reduction_jobs == 10
        assert sub.parallel_scans == 5
        assert sub.parallel_tasks == 2
        assert sub.available_reduction_jobs == 10

        # try to call a subreduction assignment directly:
        # should call the parent's instead
        m1 = mocker.patch.object(reduction, 'assign_parallel_jobs')
        sub.assign_parallel_jobs()
        m1.assert_called_once()

    def test_init_pipelines(self, capsys, scan_file, mocker):
        # no subreductions, no scans: no errors
        reduction = Reduction('example')
        with set_log_level('DEBUG'):
            reduction.init_pipelines()
        assert isinstance(reduction.pipeline, Pipeline)
        assert reduction.pipeline.scans is None
        assert 'Reduction will process 0 ' \
               'scan serially' in capsys.readouterr().out

        # add a scan, read serially
        mocker.patch('psutil.cpu_count', return_value=1)
        reduction.read_scans(scan_file)
        with set_log_level('DEBUG'):
            reduction.init_pipelines()
        assert len(reduction.pipeline.scans) == 1
        capt = capsys.readouterr().out
        assert 'Reduction will process 1 scan serially' in capt

        # mark as subreduction
        reduction.parent_reduction = 'test'
        with set_log_level('DEBUG'):
            reduction.init_pipelines()
        capt = capsys.readouterr().out
        assert 'Sub-reduction 0 will process 1 scan serially' in capt

        # set parallel config
        reduction.parent_reduction = None
        mocker.patch('psutil.cpu_count', return_value=4)
        reduction.configuration.apply_configuration_options(
            {'parallel': {'mode': 'hybrid', 'jobs': -1, 'cores': -1}})
        reduction.update_parallel_config(reset=True)
        with set_log_level('DEBUG'):
            reduction.init_pipelines()
        capt = capsys.readouterr().out
        assert 'Reduction will process 1 scan using 4 parallel tasks' in capt

        # add a scan
        reduction.scans.append(reduction.scans[0])
        reduction.update_parallel_config(reset=True)
        with set_log_level('DEBUG'):
            reduction.init_pipelines()
        capt = capsys.readouterr().out
        assert 'Reduction will process 2 scans using 2 parallel ' \
               'scans X 2 parallel tasks' in capt

        # skip parallel tasks
        reduction.parallel_tasks = 1
        with set_log_level('DEBUG'):
            reduction.init_pipelines()
        capt = capsys.readouterr().out
        assert 'Reduction will process 2 scans using 2 parallel scans' in capt

    def test_reduce(self, capsys, scan_file):
        reduction = Reduction('example')

        # no scans
        reduction.reduce()
        assert 'No valid scans' in capsys.readouterr().err

        # one scan file
        reduction = Reduction('example')
        reduction.read_scans(scan_file)
        reduction.validate()

        # default reduction: 5 rounds in example config
        reduction.reduce()
        capt = capsys.readouterr()
        assert 'Reducing 1 scan' in capt.out
        assert 'Default reduction' in capt.out
        assert 'Reduction complete' in capt.out
        for i in range(5):
            assert f'Round {i + 1}' in capt.out

        # top level source options
        conf = reduction.configuration
        for arg in ['bright', 'faint', 'deep']:
            conf.apply_configuration_options({arg: True})
            reduction.reduce()
            capt = capsys.readouterr()
            assert 'Default reduction' not in capt.out
            assert f'{arg.title()} source reduction' in capt.out
            conf.apply_configuration_options(
                {'bright': False, 'faint': False, 'deep': False})

        # extended can be used with any source option
        conf.apply_configuration_options({'extended': True})
        reduction.reduce()
        capt = capsys.readouterr()
        assert 'Default reduction' in capt.out
        assert 'Assuming extended' in capt.out
        conf.apply_configuration_options({'extended': False})

        # set alternate rounds
        reduction.rounds = 2
        reduction.reduce()
        capt = capsys.readouterr()
        for i in range(5):
            if i < 2:
                assert f'Round {i + 1}' in capt.out
            else:
                assert f'Round {i + 1}' not in capt.out

        # if rounds is unset, raises error
        reduction.rounds = None
        with pytest.raises(ValueError) as err:
            reduction.reduce()
        assert 'No rounds specified' in str(err)

        # if marked as sub-reduction, message changes
        reduction.parent_reduction = 'test'
        reduction.rounds = 1
        reduction.reduce()
        assert 'Sub-reduction 0 complete' in capsys.readouterr().out

    def test_reduce_subreductions(self, capsys, scan_file, mocker):
        reduction = Reduction('example')

        # error if subs is None
        with pytest.raises(RuntimeError) as err:
            reduction.reduce_sub_reductions()
        assert 'No sub-reductions' in str(err)

        # warns if empty list
        reduction.sub_reductions = []
        reduction.reduce_sub_reductions()
        assert 'no sub-reductions' in capsys.readouterr().err

        # add subs, no scans
        subs = []
        for i in range(3):
            sub = Reduction('example')
            sub.parent_reduction = reduction
            subs.append(sub)
        reduction.sub_reductions = subs

        # reduce calls reduce_sub_reductions
        reduction.reduce()

        # messages for all subreductions
        capt = capsys.readouterr()
        assert capt.out.count('Reducing sub-reduction') == 3
        assert capt.err.count('No valid scans') == 3

        # add scans, but mock multitask reduce and test parallel config
        mocker.patch('psutil.cpu_count', return_value=4)
        reduction.configuration.apply_configuration_options(
            {'parallel': {'mode': 'hybrid', 'jobs': -1, 'cores': -1}})
        m1 = mocker.patch.object(reduction, 'pickle_sub_reductions')
        m2 = mocker.patch.object(reduction, 'unpickle_sub_reductions')
        m3 = mocker.patch(
            'sofia_redux.toolkit.utilities.multiprocessing.multitask',
            return_value=subs)
        reduction.update_parallel_config(reset=True)

        with set_log_level('DEBUG'):
            reduction.reduce()
        capt = capsys.readouterr()
        assert 'Performing 3 reductions (in parallel using 3 jobs)' in capt.out
        assert 'Reduction complete' in capt.out

        # check mock calls
        m1.assert_called_once()
        m2.assert_called_once()
        m3.assert_called_once()

    def test_iterate(self, scan_file):
        reduction = Reduction('example')

        # test an uncommon branch in iteration
        reduction.configuration.apply_configuration_options(
            {'whiten': True, 'whiten.once': True})

        reduction.read_scans(scan_file)
        reduction.init_pipelines()

        # whiten.once turns off whiten after 1 iteration
        assert reduction.configuration.is_configured('whiten')
        reduction.iterate()
        assert not reduction.configuration.is_configured('whiten')

    def test_checkout(self):
        # mock integration removal from queue
        reduction = Reduction(None)
        reduction.queue = [1, 3, 4]
        reduction.checkout(3)
        assert reduction.queue == [1, 4]
        reduction.checkout(2)
        assert reduction.queue == [1, 4]

    def test_summarize_integration(self, capsys):
        integration = ExampleIntegration()
        integration.comments = None

        # no comments
        Reduction.summarize_integration(integration)
        assert '[1]' in capsys.readouterr().out

        # with comments
        integration.comments = ['test', 'set', 'of', 'comments']
        Reduction.summarize_integration(integration)
        assert '[1] testsetofcomments' in capsys.readouterr().out

    def test_write_products(self, capsys, mocker):
        reduction = Reduction('example')

        # no op if no source
        reduction.write_products()

        # mock a source
        source = AstroIntensityMap(reduction.info)
        m1 = mocker.patch.object(source, 'suggestions')
        m2 = mocker.patch.object(source, 'is_valid', return_value=False)
        m3 = mocker.patch.object(source, 'write')
        reduction.source = source

        # no write if not valid
        reduction.write_products()
        m1.assert_called_once()
        m2.assert_called_once()
        m3.assert_not_called()
        assert 'did not result in a source model' in capsys.readouterr().err

        # write if valid
        m4 = mocker.patch.object(source, 'is_valid', return_value=True)
        reduction.write_products()
        m4.assert_called_once()
        m3.assert_called_once()

        # if scans are present, their write is called too
        reduction.scans = [MockScan(), MockScan()]
        reduction.write_products()
        assert capsys.readouterr().out.count('write scan') == 2

    def test_add_user_configuration(self, capsys):
        reduction = Reduction('example')
        assert not reduction.configuration.is_configured('test')
        conf = reduction.configuration.copy()

        # no-op without kwargs
        reduction.add_user_configuration()
        assert reduction.configuration.options == conf.options

        # set a value
        kwargs = {'test': 'value'}
        reduction.add_user_configuration(**kwargs)
        assert 'test=value' in capsys.readouterr().out
        assert reduction.configuration.is_configured('test')
        assert reduction.configuration.get('test') == 'value'

        # user values are locked after setting
        assert 'test' in reduction.configuration.locked

        reduction.info.configuration = conf

        # set subsection values
        kwargs = {'test': {'a': 1, 'b': 2}}
        reduction.add_user_configuration(**kwargs)
        assert "test={'a': '1', 'b': '2'}" in capsys.readouterr().out
        assert reduction.configuration.get('test.a') == '1'
        assert reduction.configuration.get('test.b') == '2'
        reduction.info.configuration = conf

        # alternate specification
        kwargs = {'options': {'test.a': 1, 'test.b': 2}}
        reduction.add_user_configuration(**kwargs)
        assert "options={'test.a': '1', " \
               "'test.b': '2'}" in capsys.readouterr().out
        assert reduction.configuration.get('test.a') == '1'
        assert reduction.configuration.get('test.b') == '2'
        reduction.info.configuration = conf

        # special keyword: different lock mechanism
        reduction.add_user_configuration(rounds=10)
        assert reduction.configuration.iterations.rounds_locked

        # section and command keys do not get locked
        reduction.add_user_configuration(forget='whiten')
        assert 'forget' not in reduction.configuration.locked
        kwargs = {'object': 'test'}
        reduction.add_user_configuration(**kwargs)
        assert 'object' not in reduction.configuration.locked

    def test_pickle_sub_reductions(self, capsys):
        reduction = Reduction('example')

        # no op without subs
        with set_log_level('DEBUG'):
            reduction.pickle_sub_reductions()
            reduction.unpickle_sub_reductions()
        assert 'Sub-reductions pickled' not in capsys.readouterr().out

        subs = [Reduction('example'), Reduction('example')]
        reduction.sub_reductions = subs

        with set_log_level('DEBUG'):
            reduction.pickle_sub_reductions()
        assert 'Sub-reductions pickled' in capsys.readouterr().out
        pdir = reduction.pickle_reduction_directory
        assert os.path.isdir(pdir)

        # unpickle: sets reconstructed objects in sub_reductions list
        reduction.unpickle_sub_reductions()
        assert reduction.sub_reductions == subs
        assert len(reduction.sub_reductions) == 2
        for sub in reduction.sub_reductions:
            assert isinstance(sub, Reduction)
            assert sub.parent_reduction is reduction
        assert not os.path.isdir(pdir)
        assert reduction.pickle_reduction_directory is None

    def test_edit_header(self):
        reduction = Reduction('example')
        kwargs = {'test': 'value'}
        reduction.add_user_configuration(**kwargs)

        header = fits.Header({'TEST1': 1,
                              'TEST2': 2})
        header['HISTORY'] = 'Previous'
        header['TEST3'] = 3
        header['HISTORY'] = 'history'
        hcopy = header.copy()

        # edit header
        reduction.edit_header(header)

        # previous values should survive
        history = str(header['HISTORY'])
        assert 'Previous\nhistory' in history
        assert header['TEST1'] == 1
        assert header['TEST2'] == 2
        assert header['TEST3'] == 3

        # new values added
        assert 'Reduced: SOFSCAN' in history
        assert header['ARGS'] == 0
        assert header['KWARGS'] == '{"test": "value"}'

        # full config in json format
        conf = json.loads(header['CNFGVALS'])
        assert conf['test'] == 'value'

        # if files were loaded, they should be logged
        header = hcopy.copy()
        reduction.reduction_files = 'test.fits'
        reduction.edit_header(header)
        assert header['ARGS'] == 1
        assert header['ARG1'] == 'test.fits'

        header = hcopy.copy()
        reduction.reduction_files = ['t1.fits', 't2.fits']
        reduction.edit_header(header)
        assert header['ARGS'] == 2
        assert header['ARG1'] == 't1.fits'
        assert header['ARG2'] == 't2.fits'

    def test_run(self, mocker, scan_file):
        reduction = Reduction('example')
        output = reduction.run(scan_file)
        assert isinstance(output, fits.HDUList)

        # mock failed source creation
        reduction = Reduction('example')
        mocker.patch.object(reduction.info, 'perform_reduction')
        output = reduction.run(scan_file)
        assert output is None

    def test_run_subreduction(self, scan_file):
        reduction = Reduction('example')

        # add subs
        subs = []
        for i in range(3):
            sub = Reduction('example')
            sub.parent_reduction = reduction
            sub.reduction_number = i + 1
            subs.append(sub)
        reduction.sub_reductions = subs

        # run scans: will assign 1 each to subs
        filenames = [scan_file, scan_file, scan_file]
        output = reduction.run(filenames)
        assert isinstance(output, list)
        assert len(output) == 3
        for hdul in output:
            assert isinstance(hdul, fits.HDUList)

        # if there's a source hdul, it's returned as well
        reduction.source = AstroIntensityMap(reduction.info)
        reduction.source.hdul = 'test'

        output = reduction.run(filenames)
        assert isinstance(output, list)
        assert len(output) == 4
        assert output[0] == 'test'
        for hdul in output[1:]:
            assert isinstance(hdul, fits.HDUList)
