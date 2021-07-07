# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.visualization import log
from sofia_redux.visualization.eye import Eye
from sofia_redux.visualization import redux_viewer as rv
try:
    from sofia_redux.pipeline.viewer import Viewer
except ImportError:
    Viewer = object
    HAS_PIPELINE = False
else:
    HAS_PIPELINE = True

try:
    from PyQt5 import QtWidgets
except ImportError:
    HAS_PYQT5 = False
    QtWidgets = None
else:
    HAS_PYQT5 = True


@pytest.mark.skipif(not HAS_PIPELINE or not HAS_PYQT5,
                    reason='Missing dependencies')
class TestReduxViewer(object):
    @pytest.fixture(autouse=True, scope='function')
    def set_debug_level(self):
        # set log level to debug
        orig_level = log.level
        log.setLevel('DEBUG')
        # let tests run
        yield
        # reset log level
        log.setLevel(orig_level)

    def test_no_pipeline(self, mocker, capsys):
        mocker.patch.object(rv, 'HAS_PIPELINE', False)
        with pytest.raises(ImportError) as err:
            rv.EyeViewer()
        assert "Unable to import Viewer" in str(err)

    def test_startup(self, qtbot, qapp, open_mock):
        ev = rv.EyeViewer()
        assert isinstance(ev, Viewer)

        # no parent passed
        ev.start()
        assert ev.parent is None

        # with parent
        ev.start(parent=qtbot)
        assert ev.parent is qtbot

        # creates an unopened Eye instance: no timer loop
        assert isinstance(ev.eye, Eye)
        assert ev.eye.view.parent is qtbot
        assert not open_mock.called

    def test_close(self, caplog, open_mock):
        ev = rv.EyeViewer()
        # no op if not started
        ev.close()
        assert ev.eye is None
        assert 'Closing' not in caplog.text

        # start and close: closes window but does not clear out eye
        ev.start()
        ev.close()
        assert isinstance(ev.eye, Eye)
        assert 'Closing' in caplog.text
        assert not open_mock.called

    def test_reset(self, caplog, open_mock):
        # same as close
        ev = rv.EyeViewer()
        ev.start()
        ev.reset()
        assert isinstance(ev.eye, Eye)
        assert 'Closing' in caplog.text
        assert not open_mock.called

    def test_display_no_data(self, caplog, open_mock):
        ev = rv.EyeViewer()

        # unstarted: no op
        ev.display()
        assert 'Updating' not in caplog.text
        assert 'Closing' not in caplog.text

        # eye started, but no data: closes eye instead
        ev.start()
        ev.display()
        assert 'Updating' not in caplog.text
        assert 'Closing' in caplog.text
        assert not open_mock.called

        # same if explicitly set to empty
        ev.display_data = []
        assert not open_mock.called

    def test_display_bad_data(self, caplog, open_mock):
        ev = rv.EyeViewer()
        # set bad data
        ev.display_data = [1, 2, 3]
        # explicitly set first display flag to verify it is reset
        # if display is attempted without start
        ev.first_display = False

        # unstarted: no op
        ev.display()
        assert not open_mock.called
        assert ev.first_display

        # start, then display bad data: will abort and reset
        ev.start()
        ev.display()
        assert not open_mock.called
        assert 'Invalid data' in caplog.text
        assert ev.first_display

    def test_display_good_data(self, caplog, open_mock, grism_hdul):
        ev = rv.EyeViewer()
        # set good data
        ev.display_data = [grism_hdul]

        # start, then display data
        ev.start()
        assert ev.first_display
        ev.display()
        assert open_mock.called
        assert 'Added 1 models to panes' in caplog.text
        assert not ev.first_display

        # display again
        ev.display()
        assert open_mock.called_twice()
