# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys
import pytest

from sofia_redux.visualization import controller, eye

PyQt5 = pytest.importorskip('PyQt5')


class TestController(object):

    def test_parse_args(self):
        infile = 'test.fits'
        args = [infile, '--log']
        args = controller.parse_args(args)

        assert len(args.filenames) == 1
        assert os.path.basename(args.filenames[0]) == infile
        assert args.system_logs
        assert args.log_level == 'CRITICAL'

    def test_check_args(self):
        infile = 'test.fits'
        args = [infile, '--log']
        args = controller.parse_args(args)
        args = controller.check_args(args)

        assert len(args.filenames) == 1
        assert os.path.basename(args.filenames[0]) == infile
        assert args.system_logs
        assert args.log_level == 'CRITICAL'

    def test_main(self, mocker, qtbot, spectral_filenames, qapp):
        mocker.patch.object(sys, 'argv', spectral_filenames)
        mocker.patch.object(sys, 'exit')
        mocker.patch.object(PyQt5.QtWidgets.QApplication, 'exec_',
                            return_value=0)
        mocker.patch.object(PyQt5.QtWidgets, 'QApplication',
                            return_value=qapp)
        open_mock = mocker.patch.object(eye.Eye, 'open_eye')

        controller.main()

        assert open_mock.called_once()
