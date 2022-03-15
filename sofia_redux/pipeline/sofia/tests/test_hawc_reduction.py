# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the HAWC Reduction class."""

import os
import pytest

from astropy.io import fits
from astropy.io.fits.tests import FitsTestCase

from sofia_redux.pipeline.parameters import ParameterSet
from sofia_redux.pipeline.gui.qad_viewer import QADViewer

try:
    from sofia_redux.pipeline.sofia.hawc_reduction \
        import HAWCReduction
    from sofia_redux.pipeline.sofia.parameters.hawc_parameters \
        import HAWCParameters
    HAS_DRP = True
except ImportError:
    HAS_DRP = False


class MockDataFits(object):
    """Mock sufficient DataFits functionality to test Redux interface."""

    # class-level attributes to change as needed
    headval = None
    mode = 'nodpol'

    def __init__(self, config=None, *args, **kwargs):
        self.log = None
        if config is None:
            self.config = {'mode_intcal': {'stepslist': ['StepCheckhead']},
                           'mode_nodpol': {'stepslist': ['StepCheckhead',
                                                         'StepDemodulate']},
                           'mode_skycal': {'stepslist': ['StepCheckhead',
                                                         'StepDemodulate',
                                                         'StepMkflat']}}
        else:
            self.config = config
        self.config_files = ['file1', 'file2']
        self.loaded = False
        self.filename = 'test'
        self.rawname = 'test'

    def load(self, *args, **kwargs):
        self.loaded = True

    def loadhead(self, fname, **kwargs):
        self.filename = fname

    def mergeconfig(self, *args, **kwargs):
        pass

    def getheadval(self, *args, **kwargs):
        if self.headval is None:
            raise KeyError('not found')
        else:
            return self.headval

    def save(self):
        pass

    def to_hdulist(self, **kwargs):
        # return empty HDUList
        return fits.HDUList()

    def to_header_list(self):
        # return list of empty header
        return [fits.Header()]

    @staticmethod
    def reset():
        MockDataFits.headval = None
        MockDataFits.mode = 'nodpol'


class MockStep(object):
    """No-op pipeline step for testing."""
    iomode = 'SISO'
    auxout = []
    raise_error = False

    def __call__(self, *args, **kwargs):
        if self.raise_error:
            raise ValueError('test error')
        if self.iomode.endswith('SO'):
            return MockDataFits()
        else:
            return [MockDataFits()]

    @staticmethod
    def reset():
        MockStep.iomode = 'SISO'
        MockStep.auxout = []
        MockStep.raise_error = False


@pytest.mark.skipif('not HAS_DRP')
class TestHAWCReduction(object):
    def make_file(self):
        """Retrieve a test FITS file for HAWC mode."""
        fitstest = FitsTestCase()
        fitstest.setup()
        fitstest.copy_file('test0.fits')
        ffile = fitstest.temp('test0.fits')
        fits.setval(ffile, 'INSTRUME', value='HAWC_PLUS')
        return ffile

    def mock_drp(self, mocker):
        # mock datafits in reduction and parameters
        mocker.patch(
            'sofia_redux.pipeline.sofia.hawc_reduction.DataFits',
            MockDataFits)
        mocker.patch(
            'sofia_redux.pipeline.sofia.parameters.hawc_parameters.DataFits',
            MockDataFits)

    def test_startup(self):
        red = HAWCReduction()
        # check that steps were loaded
        assert len(red.step_class) > 0
        assert len(red.step_class) == len(red.step_name)
        assert len(red.step_class) == len(red.processing_steps)
        assert len(red.step_class) == len(red.param_lists)

    def test_register_viewers(self, mocker):
        mocker.patch.object(QADViewer, '__init__', return_value=None)

        red = HAWCReduction()
        vz = red.register_viewers()
        assert len(vz) == 1
        assert isinstance(vz[0], QADViewer)

    def test_load(self, mocker):
        self.mock_drp(mocker)

        red = HAWCReduction()
        ffile = self.make_file()

        red.load([ffile])

        # "DataFits" should be loaded
        assert len(red.input) == 1
        assert isinstance(red.input[0], MockDataFits)

        # parameters should be loaded
        assert isinstance(red.parameters, HAWCParameters)

        # recipe is set by override for 'nodpol' data
        # (names differ)
        assert len(red.recipe) == len(red.override_steplist['nodpol'])

        # try an alternate mode: should also be set by override
        red.override_mode = 'skycal'
        red.load([ffile])
        assert len(red.recipe) == len(red.override_steplist['skycal'])

    def test_load_intermediate(self, mocker):
        self.mock_drp(mocker)
        red = HAWCReduction()
        ffile = self.make_file()

        # set a different override recipe
        red.override_steplist = {'nodpol': ['unknown']}

        # raises value error for unknown step
        with pytest.raises(ValueError):
            red.load([ffile])

        # set another recipe: headval(prodtype) == step name, so
        # should assume is intermediate, and exclude first
        # step from recipe
        MockDataFits.mode = 'test'
        MockDataFits.headval = 'make_flats'
        red.override_steplist = {'test': ['make_flats', 'demodulate']}
        red.load([ffile])
        assert red.intermediate
        assert red.recipe == ['demodulate']
        assert red.processing_steps['demodulate'] == \
            red.extra_steps['demodulate']

        # empty extra_steps and test name comes from step instead
        red.extra_steps = {}
        red.load([ffile])
        assert red.processing_steps['demodulate'] == 'demodulate'

        # now set recipe to just the first step -- should
        # raise error for no steps to run
        red.override_steplist = {'test': ['make_flats']}
        with pytest.raises(ValueError):
            red.load([ffile])

    def test_load_mode(self, mocker):
        self.mock_drp(mocker)
        red = HAWCReduction()
        ffile = self.make_file()

        # set mode to intcal
        MockDataFits.mode = 'intcal'
        red.load([ffile])

        # recipe should be set by config
        assert red.recipe == ['checkhead']

        # set mode to None -- should raise error
        # for missing mode
        MockDataFits.mode = None
        with pytest.raises(ValueError):
            red.load([ffile])

        # set override mode -- should select that instead and
        # not raise an error
        red.override_mode = 'intcal'
        red.load([ffile])
        assert red.recipe == ['checkhead']
        assert red.parameters.override_mode == 'intcal'

        # set to an unknown mode - should raise error
        red.override_mode = 'bad_mode'
        with pytest.raises(ValueError) as err:
            red.load([ffile])
        assert 'override mode bad_mode not found' in str(err)

    def test_step(self, mocker, capsys):
        self.mock_drp(mocker)

        # also mock the run_drp_step so actual steps aren't called
        mocker.patch.object(HAWCReduction, 'run_drp_step',
                            return_value=None)

        red = HAWCReduction()
        ffile = self.make_file()

        red.load([ffile])

        # run a step
        red.step()
        assert red.step_index == 1

        # run another
        red.step()
        assert red.step_index == 2

        # run with reduce
        red = HAWCReduction()
        red.load([ffile])
        red.reduce()
        assert red.step_index == len(red.recipe)

        capt = capsys.readouterr()
        assert 'Reduction complete' in capt.out

        # try to run another step, after complete --
        # sets an error message instead
        status = red.step()
        assert status == 'No steps to run.'

    def test_display_data(self, mocker):
        self.mock_drp(mocker)
        red = HAWCReduction()

        # just set some input and auxout and verify
        # display_data is set
        init_aux = ['test1', 'test2']
        red.input = [MockDataFits(), MockDataFits(), MockDataFits()]
        red.auxout = init_aux.copy()

        # sets headers and initial auxout only
        red.set_display_data(display=False)
        assert len(red.display_data) == 1
        disp = red.display_data['QADViewer']
        assert len(disp) == len(red.input) + len(init_aux)
        assert len(red.auxout) == len(init_aux)

        for i in range(len(disp)):
            item = disp[i]
            if i < len(red.input):
                assert isinstance(item, list)
                assert isinstance(item[0], fits.Header)
            else:
                assert item == init_aux[i - len(red.input)]

        # set data
        red.set_display_data(display=True)
        assert len(red.display_data) == 1
        disp = red.display_data['QADViewer']
        assert len(disp) == len(red.input) + len(init_aux)
        assert len(red.auxout) == len(init_aux)
        for i in range(len(disp)):
            item = disp[i]
            if i < len(red.input):
                assert isinstance(item, fits.HDUList)
            else:
                assert item == init_aux[i - len(red.input)]

        # set additional auxout
        addl = ['test3', 'test4']
        red.set_display_data(display=True, auxout=addl)
        assert len(red.display_data) == 1
        disp = red.display_data['QADViewer']
        assert len(disp) == len(red.input) + len(init_aux) + len(addl)
        assert len(red.auxout) == len(init_aux) + len(addl)
        for i in range(len(disp)):
            item = disp[i]
            if i < len(red.input):
                assert isinstance(item, fits.HDUList)
            elif i < len(red.input) + len(addl):
                assert item == addl[i - len(red.input)]
            else:
                assert item == init_aux[i - len(red.input) - len(addl)]

    def test_drp_step_load(self, mocker, capsys):
        self.mock_drp(mocker)
        red = HAWCReduction()
        ffile = self.make_file()

        # set mode to intcal for simple recipe
        MockDataFits.mode = 'intcal'
        red.load([ffile])

        # mock the step
        mocker.patch(
            'sofia_redux.instruments.hawc.steps.stepcheckhead.StepCheckhead',
            MockStep)

        # run it (calls run_drp_step)
        red.step()
        capt = capsys.readouterr()
        assert 'Check Headers' in capt.out

        # now run drp_step directly

        # step_index not in recipe (no steps to run) -- raises error
        with pytest.raises(ValueError) as err:
            red.run_drp_step()
        assert 'not found' in str(err.value)

        # no input -- raises error
        red.input = []
        with pytest.raises(ValueError) as err:
            red.run_drp_step()
        assert 'No input' in str(err.value)

        # import error (step class not found)
        red.step_index = 0
        red.input = [MockDataFits()]
        red.step_class['unknown'] = 'StepUnknown'
        with pytest.raises(ValueError) as err:
            red.run_drp_step(step_name='unknown')
        assert 'unknown not found' in str(err.value)

        # reload data properly
        red.load([ffile])

        # empty parameter set for step -- should not raise error,
        # will try to load from df.filename
        red.parameters.current[0] = ParameterSet()
        red.run_drp_step()
        capt = capsys.readouterr()
        assert 'Input: {}'.format(os.path.basename(ffile)) in capt.out

        # load from rawname instead
        red.input[0].filename = 'test value'
        red.input[0].rawname = ffile
        red.run_drp_step()
        capt = capsys.readouterr()
        # rawname is loaded, fname is still printed
        assert 'Input: test value' in capt.out

        # raises error if neither is a file
        red.input[0].filename = 'test value 1'
        red.input[0].rawname = 'test value 2'
        with pytest.raises(IOError) as err:
            red.run_drp_step()
        assert 'Could not load' in str(err.value)

    def test_drp_step_iomode(self, mocker, capsys):
        self.mock_drp(mocker)
        red = HAWCReduction()
        ffile = self.make_file()

        # set mode to intcal for simple recipe
        MockDataFits.mode = 'intcal'

        # mock the step
        mocker.patch(
            'sofia_redux.instruments.hawc.steps.stepcheckhead.StepCheckhead',
            MockStep)

        # make it MI (multi-in), set an auxout, save the output
        MockStep.iomode = 'MISO'
        MockStep.auxout = ['test aux']

        red.load([ffile])
        red.parameters.current[0]['save']['value'] = True
        red.run_drp_step()
        capt = capsys.readouterr()
        assert 'Input: All files' in capt.out
        assert 'Wrote' in capt.out
        assert 'test aux' in red.display_data['QADViewer']

        # raise error in step -- in MI mode, raises error
        MockStep.raise_error = True
        with pytest.raises(ValueError):
            red.run_drp_step()
        MockStep.raise_error = False

        # reload, make it SI and run again
        MockStep.iomode = 'SISO'
        red.load([ffile])
        red.parameters.current[0]['save']['value'] = True
        red.run_drp_step()
        capt = capsys.readouterr()
        assert 'Input: {}'.format(os.path.basename(ffile)) in capt.out
        assert 'Wrote' in capt.out
        assert 'test aux' in red.display_data['QADViewer']

        # raise error in step -- in SI mode, just skips file
        MockStep.raise_error = True
        red.run_drp_step()
        capt = capsys.readouterr()
        assert 'skipping file' in capt.err
        MockStep.raise_error = False

    def test_make_flats(self, mocker, capsys):
        self.mock_drp(mocker)
        red = HAWCReduction()
        ffile = self.make_file()

        # mock run_drp_step
        mocker.patch.object(HAWCReduction, 'run_drp_step',
                            return_value=None)

        # set mode to intcal
        MockDataFits.mode = 'intcal'

        # nothing loaded
        red.make_flats()
        capt = capsys.readouterr()
        assert 'No flats' in capt.err

        # load an intcal
        red.load([ffile])
        orig_input = red.input[0]

        # no science, checkhead only
        red.make_flats()
        capt = capsys.readouterr()
        assert capt.out.count('Sub-step') == 1
        assert len(red.input) == 0

        # set a longer recipe
        recipe = ['StepCheckhead', 'StepDemodulate', 'StepDmdPlot']
        orig_input.config['mode_intcal']['stepslist'] = recipe
        red.input = [orig_input]

        red.make_flats()
        capt = capsys.readouterr()
        assert capt.out.count('Sub-step') == 3
        assert len(red.input) == 0

        # start from intermediate: calls DmdPlot only
        red.input = [orig_input]
        MockDataFits.headval = 'demodulate'
        red.make_flats()
        capt = capsys.readouterr()
        assert capt.out.count('Sub-step') == 1

        # if no steps after demodulate, raise error
        recipe = ['StepCheckhead', 'StepDemodulate']
        orig_input.config['mode_intcal']['stepslist'] = recipe
        red.input = [orig_input]
        with pytest.raises(ValueError):
            red.make_flats()

        # unset intermediate; make step unfindable -- raises error
        MockDataFits.headval = None
        del red.step_name['StepDemodulate']
        red.input = [orig_input]
        with pytest.raises(ValueError):
            red.make_flats()

    def test_process_intcal(self, mocker, capsys):
        self.mock_drp(mocker)
        red = HAWCReduction()
        ffile = self.make_file()

        # mock run_drp_step
        mocker.patch.object(HAWCReduction, 'run_drp_step',
                            return_value=None)

        # set mode to intcal
        MockDataFits.mode = 'intcal'

        # load an intcal
        red.load([ffile])
        orig_input = red.input[0]

        # set a longer recipe
        recipe = ['StepCheckhead', 'StepDemodulate', 'StepMkflat']
        orig_input.config['mode_intcal']['stepslist'] = recipe
        red.input = [orig_input]

        # same as calling the make_flats step
        red.process_intcal()
        capt = capsys.readouterr()
        assert capt.out.count('Sub-step') == 3
        assert len(red.input) == 0

    def test_demodulate(self, mocker, capsys):
        self.mock_drp(mocker)
        red = HAWCReduction()
        ffile = self.make_file()

        # mock run_drp_step
        mocker.patch.object(HAWCReduction, 'run_drp_step',
                            return_value=None)

        # nothing loaded: raises error
        with pytest.raises(ValueError):
            red.demodulate()

        # load 'nodpol' data
        red.load([ffile])
        orig_input = red.input[0]

        # run 2 demod steps (checkhead, demodulate; from config)
        red.demodulate()
        capt = capsys.readouterr()
        assert capt.out.count('Sub-step') == 2
        assert len(red.input) == 1

        # set intermediate: calls run_drp_step for demodulate only
        red.intermediate = True
        red.demodulate()
        capt = capsys.readouterr()
        assert capt.out.count('Sub-step') == 0
        assert capt.err == ''
        red.intermediate = False

        # mode not found: also just runs demodulate directly
        orig_input.mode = 'unknown'
        red.input = [orig_input]
        red.demodulate()
        capt = capsys.readouterr()
        assert capt.out.count('Sub-step') == 0
        assert capt.err == ''

        # StepDemodulate not found in recipe: issues warning only
        orig_input.mode = 'intcal'
        red.input = [orig_input]
        red.demodulate()
        capt = capsys.readouterr()
        assert capt.out.count('Sub-step') == 0
        assert 'No demodulate steps' in capt.err

        # clear step_name so that step is not found: raises error
        orig_input.mode = 'nodpol'
        red.input = [orig_input]
        del red.step_name['StepDemodulate']
        with pytest.raises(ValueError) as err:
            red.demodulate()
        assert 'not found' in str(err.value)
        red.step_name['StepDemodulate'] = 'demodulate'
        capsys.readouterr()

        # add a step with no custom parameter function --
        # should not raise error
        red.step_name['StepUnknown'] = 'unknown'
        red.processing_steps['unknown'] = 'Unknown'
        orig_input.config['mode_nodpol']['stepslist'] = \
            ['StepUnknown', 'StepDemodulate']
        red.input = [orig_input]
        red.demodulate()
        capt = capsys.readouterr()
        assert capt.out.count('Sub-step') == 2
        assert 'Parameters for step unknown not found' in capt.err

    def test_parameters(self, mocker):
        self.mock_drp(mocker)

        # start with no param lists, no config
        par = HAWCParameters()
        assert len(par.default) == 0

        # mock a config -- still not added
        config = {'test_step': {'test_key': 'test_value'}}
        par = HAWCParameters(config=config)
        assert len(par.default) == 0

        # mock a paramlist
        paramlist = {'test_step': [['test_key', 'test_default',
                                    'test_description']]}
        par = HAWCParameters(param_lists=paramlist)
        assert len(par.default) == 1
        assert isinstance(par.default['test_step'], ParameterSet)
        assert par.default['test_step'].get_value('test_key') == 'test_default'

        # override paramlist with config
        par = HAWCParameters(param_lists=paramlist, config=config)
        assert len(par.default) == 1
        assert isinstance(par.default['test_step'], ParameterSet)
        assert par.default['test_step'].get_value('test_key') == 'test_value'
        conf = par.to_config()
        assert 'mode' not in conf

        # add an override mode: should be propagated to saved config
        par = HAWCParameters(param_lists=paramlist, config=config,
                             mode='skycal')
        conf = par.to_config()
        assert conf['mode'] == 'skycal'

    def test_parameter_steps(self, mocker):
        """Check some specific parameters that affect Redux control."""
        self.mock_drp(mocker)

        # set an unrealistic recipe that exercises all
        # steps with non-default parameters
        recipe = ['checkhead', 'fluxjump', 'prepare', 'dmdcut',
                  'dmdplot', 'opacity', 'calibrate', 'merge',
                  'stdphotcal', 'polmap', 'scanmap', 'scanmappol',
                  'skydip', 'poldip', 'scanmapfocus', 'focus',
                  'make_flats', 'demodulate',
                  'scanmapflat', 'skycal', 'process_intcal']
        red = HAWCReduction()
        red.recipe = recipe
        red.parameters = HAWCParameters()

        red.load_parameters()
        for i, step in enumerate(recipe):
            parset = red.get_parameter_set(i)

            # all should at least have save, load, display, or keep_auxout
            assert len(parset) > 0
            assert 'save' in parset or 'load' in parset \
                or 'display' in parset or 'keep_auxout' in parset

        # for scanmap, test that no-out sets values correctly
        recipe = ['scanmap']
        red.recipe = recipe
        paramlist = {'scanmap': [['noout', True,
                                  'test_description']]}
        red.parameters = HAWCParameters(param_lists=paramlist)
        red.load_parameters()
        parset = red.get_parameter_set(0)
        assert parset.get_value('save') is False
        assert parset.get_value('display') is False

    def test_drp_arglist(self):
        """Check that override parameters are passed to a real DRP step."""
        recipe = ['checkhead']
        red = HAWCReduction()
        red.recipe = recipe
        red.parameters = HAWCParameters()
        red.load_parameters()

        # make some datafits objects that will fail header checks
        from sofia_redux.instruments.hawc.datafits import DataFits
        inp = []
        for i in range(3):
            df = DataFits()
            df.setheadval('TESTKEY', 'TESTVAL')
            inp.append(df)

        # default parameter: fails header check, aborts
        red.input = inp
        with pytest.raises(RuntimeError):
            red.step()

        # override parameter: should not abort
        parset = red.get_parameter_set(0)
        parset.set_value('abort', False)
        red.step()

    def teardown_method(self):
        MockDataFits.reset()
        MockStep.reset()
