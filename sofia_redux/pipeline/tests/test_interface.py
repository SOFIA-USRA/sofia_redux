# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux Interface class."""

import logging
import os

from astropy import log
from astropy.io.fits.tests import FitsTestCase

from sofia_redux import pipeline
from sofia_redux.pipeline.interface import Interface
from sofia_redux.pipeline.chooser import Chooser
from sofia_redux.pipeline.configuration import Configuration
from sofia_redux.pipeline.reduction import Reduction
from sofia_redux.pipeline.viewer import Viewer


class TestInterface(object):
    def make_file(self):
        """Retrieve a test FITS file."""
        fitstest = FitsTestCase()
        fitstest.setup()
        ffile = fitstest.data('test0.fits')
        return ffile

    def test_no_op(self):
        # test all the functions that should just
        # return if no reduction/configuration is loaded --
        # make sure they don't throw errors
        interface = Interface()

        # these functions return nothing
        assert not interface.has_embedded_viewers()
        assert interface.load_data_id() is None
        assert interface.load_parameters() is None
        assert interface.reduce() is None
        assert interface.register_viewers() is None
        assert interface.reset_viewers() is None
        assert interface.save_input_manifest() is None
        assert interface.save_parameters() is None
        assert interface.save_output_manifest() is None
        assert interface.set_output_directory() is None
        assert interface.set_log_file() is None
        assert interface.set_recipe() is None
        assert interface.update_viewers() is None
        assert interface.update_configuration(None) is None

        # step returns empty string if no errors
        assert interface.step() == ''

        # test load without chooser
        data = 'test data'
        assert interface.load_files(data) is None
        assert interface.reduction is None

        # test reset without reduction
        assert interface.reset_reduction(data) is None
        assert interface.reduction is None

        # test load when chooser returns None
        def null_reduction(*args, **kwargs):
            return None
        chooser = Chooser()
        chooser.choose_reduction = null_reduction
        interface.chooser = chooser
        assert interface.load_files(data) is None
        assert interface.reduction is None

    def test_config(self):
        config = Configuration({'test_key': 'test_value'})
        interface = Interface(config)
        assert interface.configuration.test_key == 'test_value'

    def test_manifest(self, tmpdir):
        interface = Interface()
        ffile = self.make_file()
        manifest = tmpdir.join('infiles.txt')
        manifest.write("{}\n".format(ffile))
        fname = str(manifest)

        # tests for input manifest
        assert interface.is_input_manifest([fname])
        assert interface.read_input_manifest([fname])[0] == ffile

        # load from manifest
        interface.start(fname)
        assert interface.reduction.raw_files[0] == ffile

    def test_clear_reduction(self):
        # load data
        interface = Interface()
        ffile = self.make_file()
        interface.start(ffile)
        assert interface.reduction is not None

        # clear data
        interface.clear_reduction()
        assert interface.reduction is None

    def test_viewer(self, mocker):
        interface = Interface()

        # mock an embedded viewer for a reduction
        viewer = Viewer()
        viewer.embedded = True
        mocker.patch.object(Reduction, 'register_viewers',
                            return_value=[viewer])

        # load data
        ffile = self.make_file()
        interface.start(ffile)
        interface.register_viewers()

        # test that viewer exists and is embedded
        assert interface.has_embedded_viewers()

        # test load data
        interface.reduction.display_data = {'Viewer': [ffile]}
        interface.update_viewers()
        assert viewer.display_data == [ffile]

        # test reset
        interface.reset_viewers()
        assert viewer.display_data == []

    def test_reset_reduction(self):
        interface = Interface()
        ffile = self.make_file()
        interface.start(ffile)

        # step once
        interface.step()
        assert interface.reduction.step_index == 1

        # reset
        interface.reset_reduction(ffile)
        assert interface.reduction.step_index == 0

    def test_save_input_manifest(self, tmpdir):
        # start a reduction
        interface = Interface()
        ffile = self.make_file()
        interface.start(ffile)

        # temp directory
        tmpdir_name = str(tmpdir)

        # test output directory
        fname = 'test_input.txt'
        interface.reduction.output_directory = tmpdir_name
        interface.save_input_manifest(filename=fname)
        assert os.path.isfile(str(tmpdir.join(fname)))

        # test full filename
        fname = tmpdir.join('test_input_2.txt')
        interface.save_input_manifest(filename=str(fname))
        assert os.path.isfile(str(fname))

        # test contents
        assert fname.read() == ffile + '\n'

        # test relative paths
        fname = tmpdir.join('test_input_3.txt')
        interface.save_input_manifest(filename=str(fname),
                                      absolute_paths=False)
        assert os.path.isfile(str(fname))
        relpath = os.path.relpath(ffile, tmpdir_name)
        assert fname.read() == relpath + '\n'

        # test single infile, no infiles
        interface.reduction.raw_files = ffile
        fname = tmpdir.join('test_input_4.txt')
        interface.save_input_manifest(filename=str(fname))
        assert os.path.isfile(str(fname))
        assert fname.read() == ffile + '\n'

        interface.reduction.raw_files = []
        fname = tmpdir.join('test_input_5.txt')
        interface.save_input_manifest(filename=str(fname))
        assert not os.path.isfile(str(fname))

    def test_save_parameters(self, tmpdir):
        # start a reduction
        interface = Interface()
        ffile = self.make_file()
        interface.start(ffile)

        # test string output
        par_str = '\n'.join(interface.save_parameters()) + '\n'
        assert par_str.startswith('# Redux parameters')
        assert 'log_input' in par_str

        # test empty filename
        par_str_2 = '\n'.join(interface.save_parameters(filename='')) + '\n'
        assert par_str == par_str_2

        # temp directory
        tmpdir_name = str(tmpdir)

        # test output directory
        fname = 'test_par.txt'
        interface.reduction.output_directory = tmpdir_name
        interface.save_parameters(filename=fname)
        assert os.path.isfile(str(tmpdir.join(fname)))

        # test full filename
        fname = tmpdir.join('test_par_2.txt')
        interface.save_parameters(filename=str(fname))
        assert os.path.isfile(str(fname))

        # test contents of file
        assert fname.read() == par_str

    def test_save_output_manifest(self, tmpdir):
        # start a reduction
        interface = Interface()
        ffile = self.make_file()
        interface.start(ffile)

        # temp directory
        tmpdir_name = str(tmpdir)

        # mock an output file
        outfile = tmpdir.join('test_file.txt')
        outfile.write('test data\n')
        ffile = str(outfile)
        interface.reduction.record_outfile(ffile)

        # test output directory
        fname = 'test_output.txt'
        interface.reduction.output_directory = tmpdir_name
        interface.save_output_manifest(filename=fname)
        assert os.path.isfile(str(tmpdir.join(fname)))

        # test full filename
        fname = tmpdir.join('test_output_2.txt')
        interface.save_output_manifest(filename=str(fname))
        assert os.path.isfile(str(fname))

        # test contents
        assert fname.read() == ffile + '\n'

        # test relative paths
        fname = tmpdir.join('test_output_3.txt')
        interface.save_output_manifest(
            filename=str(fname), absolute_paths=False)
        assert os.path.isfile(str(fname))
        relpath = os.path.relpath(ffile, tmpdir_name)
        assert fname.read() == relpath + '\n'

        # test single outfile, no outfiles
        interface.reduction.out_files = ffile
        fname = tmpdir.join('test_output_4.txt')
        interface.save_output_manifest(filename=str(fname))
        assert os.path.isfile(str(fname))
        assert fname.read() == ffile + '\n'

        interface.reduction.out_files = []
        fname = tmpdir.join('test_output_5.txt')
        interface.save_output_manifest(filename=str(fname))
        assert not os.path.isfile(str(fname))

    def test_set_output_directory(self, tmpdir):
        # start a reduction with temporary output directory
        tmpdir_name = str(tmpdir)
        interface = Interface(Configuration({'output_directory': tmpdir_name}))
        ffile = self.make_file()
        interface.start(ffile)

        # test that the directory was set from config
        assert interface.reduction.output_directory == tmpdir_name

        # change the directory
        newdir_name = str(tmpdir.join('test_dir'))
        interface.set_output_directory(newdir_name)
        assert interface.reduction.output_directory == newdir_name
        assert os.path.isdir(newdir_name)

    def test_set_recipe(self):
        # set a non-default recipe (log_input twice)
        recipe = ['log_input', 'log_input']
        interface = Interface(Configuration({'recipe': recipe}))
        interface.start('test_data')

        assert interface.reduction.recipe == recipe

        interface.reduce()
        assert interface.reduction.step_index == 2

    def test_set_log_file(self, tmpdir):
        # start a reduction with temporary output directory
        tmpdir_name = str(tmpdir)
        interface = Interface(
            Configuration({'output_directory': tmpdir_name,
                           'log_file': 'test_log.txt',
                           'log_level': 'WARNING',
                           'log_format': 'test - %(message)s'}))
        ffile = self.make_file()
        interface.start(ffile)

        # test that the log file was set from config
        log_file_name = os.path.join(tmpdir_name, 'test_log.txt')
        assert interface.reduction.output_directory == tmpdir_name
        found = False
        for hand in log.handlers:
            if isinstance(hand, logging.FileHandler):
                assert hand.baseFilename == log_file_name
                assert hand.level == logging.WARNING
                found = True
                del hand
        assert found

        # change the log file
        new_file = tmpdir.join('test_log_2.txt')
        new_name = str(new_file)
        interface.set_log_file(new_name)
        found = False
        for hand in log.handlers:
            if isinstance(hand, logging.FileHandler):
                assert hand.baseFilename == new_name
                assert hand.level == logging.WARNING
                found = True
        assert found
        assert not os.path.isfile(log_file_name)

        # test a message
        log.warning('Test')
        assert 'test - Test\n' == new_file.read()

        # test unset log
        interface.unset_log_file()
        found = False
        for hand in log.handlers:
            if isinstance(hand, logging.FileHandler):
                found = True
        assert not found

    def test_tidy_log(self, capsys):
        # make a tidy log and test with messages
        Interface.tidy_log(loglevel='DEBUG')
        origin = 'pipeline.tests.test_interface'

        msg = 'Test info'
        log.info(msg)
        capt = capsys.readouterr()
        assert capt.out == '{}\n'.format(msg)
        assert origin not in capt.out

        msg = 'Test warning'
        log.warning(msg)
        capt = capsys.readouterr()
        assert capt.err.startswith('WARNING: {}'.format(msg))
        assert origin in capt.err

        msg = 'Test error'
        log.error(msg)
        capt = capsys.readouterr()
        assert capt.err.startswith('ERROR: {}'.format(msg))
        assert origin in capt.err

        msg = 'Test debug'
        log.debug(msg)
        capt = capsys.readouterr()
        assert capt.out.startswith('DEBUG: {}'.format(msg))
        assert origin in capt.out

        # call tidy log again, verify message is not doubled, but
        # log level changes
        msg = 'Another message'
        Interface.tidy_log(loglevel='INFO')
        log.debug(msg)
        capt = capsys.readouterr()
        assert capt.out == ''
        log.info(msg)
        capt = capsys.readouterr()
        assert capt.out == '{}\n'.format(msg)

        # reset log to defaults
        Interface.reset_log('DEBUG')
        log.debug(msg)
        capt = capsys.readouterr()
        assert capt.out.startswith('DEBUG: {}'.format(msg))
        assert origin in capt.out
        log.info(msg)
        capt = capsys.readouterr()
        assert capt.out.startswith('INFO: {}'.format(msg))
        assert origin in capt.out

    def test_save_configuration(self, tmpdir):
        # load a configuration
        config = Configuration({'test_key': 'test_value'})
        interface = Interface(config)
        assert interface.configuration.test_key == 'test_value'

        # add a temp directory
        tmpdir_name = str(tmpdir)
        interface.configuration.output_directory = tmpdir_name

        # test string output
        par_str = '\n'.join(interface.save_configuration()) + '\n'
        assert par_str.startswith(f'# Redux v{pipeline.__version__} '
                                  f'Configuration')
        assert 'test_key = test_value' in par_str

        # test empty filename
        par_str_2 = '\n'.join(interface.save_configuration(filename='')) + '\n'
        assert par_str == par_str_2

        # test output directory
        fname = 'test_conf.txt'
        interface.save_configuration(filename=fname)
        assert os.path.isfile(str(tmpdir.join(fname)))

        # test full filename
        fname = tmpdir.join('test_par_2.txt')
        interface.save_configuration(filename=str(fname))
        assert os.path.isfile(str(fname))

        # test contents of file
        assert fname.read() == par_str

    def test_update_configuration(self):
        # load a configuration
        config = Configuration({'test_key': 'test_value'})
        interface = Interface(config)
        assert interface.configuration.test_key == 'test_value'

        # update with another configuration
        interface.update_configuration({'test1': 1})
        assert interface.configuration.test_key == 'test_value'
        assert interface.configuration.test1 == 1
