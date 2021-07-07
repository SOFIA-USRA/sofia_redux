# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux Viewer class."""

from sofia_redux.pipeline.viewer import Viewer


class TestViewer(object):
    def test_display(self):
        """Test run method."""
        viewer = Viewer()
        viewer.display()

    def test_update(self):
        """Test run method."""
        data = 'test data'
        viewer = Viewer()
        viewer.update(data)

    def test_reset(self):
        """Test run method."""
        viewer = Viewer()
        viewer.reset()
