# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux Reduction class."""

import os
from collections import OrderedDict

from astropy import log
from astropy.io import fits
from astropy.io.fits.tests import FitsTestCase
import pytest

from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.parameters import Parameters, ParameterSet


class TestReduction(object):
    def make_file(self, fname=None):
        """Retrieve a test FITS file."""
        if fname is None:
            fname = 'test0.fits'
        fitstest = FitsTestCase()
        fitstest.setup()
        ffile = fitstest.data(fname)
        return ffile

    def test_description(self):
        red = Reduction()
        desc = red.description
        assert red.name in desc
        assert red.pipe_version in desc
        assert red.instrument in desc
        assert red.mode in desc
        if red.pipe_version != '':
            assert desc == f'{red.name} v{red.pipe_version} ' \
                           f'for {red.instrument} in {red.mode} mode'

        # missing pipe version
        red.pipe_version = ''
        assert red.description == f'{red.name} for {red.instrument} ' \
                                  f'in {red.mode} mode'
        red.pipe_version = None
        assert red.description == f'{red.name} for {red.instrument} ' \
                                  f'in {red.mode} mode'

        # non-string pipe version
        red.pipe_version = 42
        assert red.description == f'{red.name} v42 for {red.instrument} ' \
                                  f'in {red.mode} mode'

    def test_load(self):
        reduction = Reduction()
        ffile = 'test data'
        reduction.load(ffile)
        assert reduction.step_index == 0
        assert reduction.raw_files == [ffile]

    def test_load_fits(self, tmpdir):
        reduction = Reduction()
        ffile = self.make_file()

        badfile = tmpdir.join('bad.fits')
        badfile.write('test data\n')

        inlist = [ffile, 'test_data', str(badfile)]

        reduction.load(inlist)
        assert reduction.raw_files == inlist

        reduction.load_fits_files(inlist)
        assert reduction.step_index == 0
        assert reduction.raw_files == [ffile]
        assert len(reduction.input) == 1
        assert isinstance(reduction.input[0], fits.HDUList)

    def test_load_data_id(self):
        reduction = Reduction()
        ffile = self.make_file()

        # add a new data_key
        reduction.data_keys.append('NAXIS')

        # load a fits file into an HDUList
        reduction.load(ffile)
        reduction.load_fits_files(ffile)

        # also load an HDU directly
        fdata = 'test HDU'
        hdu = reduction.input[0][0]
        reduction.raw_files.append(fdata)
        reduction.input.append(hdu)

        # also load a string
        tdata = 'test string'
        reduction.raw_files.append(tdata)
        reduction.input.append(tdata)

        # load the data description
        reduction.load_data_id()
        data_id = reduction.data_id
        assert isinstance(data_id, OrderedDict)

        # check for filenames
        assert os.path.basename(ffile) in data_id['File Name']
        assert fdata in data_id['File Name']
        assert tdata in data_id['File Name']

        # no file size for non-file data
        assert data_id['File Size'][0] != 'UNKNOWN'
        assert data_id['File Size'][1] == 'UNKNOWN'
        assert data_id['File Size'][2] == 'UNKNOWN'

        # NAXIS key only for FITS data
        assert data_id['NAXIS'][0] == '0'
        assert data_id['NAXIS'][1] == '0'
        assert data_id['NAXIS'][2] == 'UNKNOWN'

    def test_parameters(self, capsys):
        reduction = Reduction()
        ffile = 'test data'
        reduction.load(ffile)

        # get empty parameters before load
        pset = reduction.get_parameter_set()
        assert isinstance(pset, ParameterSet)
        assert len(pset.keys()) == 0

        # load parameters; still empty set because none defined
        reduction.load_parameters()
        pset = reduction.get_parameter_set()
        assert isinstance(pset, ParameterSet)
        assert len(pset.keys()) == 0

        # add a parameter to the current step
        reduction.edit_parameters('test_key', value='test_value')
        pset = reduction.get_parameter_set()
        assert isinstance(pset, ParameterSet)
        assert len(pset.keys()) == 1
        assert pset.get_value('test_key') == 'test_value'

        # add a hidden parameter to verify that it is not logged
        reduction.edit_parameters('test_key_2', value='test_value_2',
                                  hidden=True)

        # add parameters the step knows about: they will be logged
        # to INFO, WARNING, and ERROR respectively
        reduction.edit_parameters('message', value='test_message')
        reduction.edit_parameters('warning', value='test_warning')
        reduction.edit_parameters('error', value='test_error')
        reduction.step()

        capt = capsys.readouterr()

        # check for logged messages
        assert 'test_message' in capt.out
        assert 'test_warning' in capt.err
        assert 'test_error' in capt.err

        # check for logged parameters
        assert 'test_key' in capt.out
        assert 'test_key_2' not in capt.out

    def test_parameter_method(self):
        reduction = Reduction()
        ffile = 'test data'
        reduction.load(ffile)

        # make a parameter class with an override method for log_input
        # parameters
        class TestParameters(Parameters):
            def log_input(self, idx):
                self.current[idx].set_value('test_key', 'test_value')
        reduction.parameters = TestParameters()

        reduction.load_parameters()
        pset = reduction.get_parameter_set()
        assert isinstance(pset, ParameterSet)
        assert len(pset.keys()) == 1
        assert pset.get_value('test_key') == 'test_value'

    def test_parameter_set(self):
        reduction = Reduction()
        reduction.load_parameters()
        pset = reduction.get_parameter_set()
        assert len(pset.keys()) == 0

        # set a new parameter set
        new_pset = ParameterSet()
        new_pset.set_param(key='test_key', value='test_value')
        reduction.set_parameter_set(new_pset)

        # retrieve and test it
        pset = reduction.get_parameter_set()
        assert isinstance(pset, ParameterSet)
        assert len(pset.keys()) == 1
        assert pset.get_value('test_key') == 'test_value'

    def test_record_outfiles(self):
        reduction = Reduction()

        # add a new file
        reduction.record_outfile('test1')
        assert reduction.out_files == ['test1']

        # add another
        reduction.record_outfile('test2')
        assert reduction.out_files == ['test1', 'test2']

        # attempt to add one already there
        reduction.record_outfile('test1')
        assert reduction.out_files == ['test1', 'test2']

    def test_step(self, capsys):
        reduction = Reduction()
        ffile = 'test data'
        reduction.load(ffile)

        # run a step
        reduction.step()
        capt = capsys.readouterr()
        assert 'test data' in capt.out

        # attempt to run again; verify nothing happens
        reduction.step()
        capt = capsys.readouterr()
        assert 'test data' not in capt.out

        # run an alternate method that sets an error message
        test_str = 'test method'
        test_err = 'test error'

        class TestReductionClass(Reduction):
            def test_method(self):
                log.info(test_str)
                self.error = test_err

        reduction = TestReductionClass()
        status = reduction.step(alt_method='test_method')

        capt = capsys.readouterr()
        assert test_str in capt.out
        assert status == test_err

        # verify that an error is raised by reduce
        # when an error message is set
        reduction = TestReductionClass()
        reduction.recipe = ['test_method']
        reduction.processing_steps['test_method'] = 'Test method'
        with pytest.raises(RuntimeError):
            reduction.reduce()
        capt = capsys.readouterr()
        assert test_err in capt.err
