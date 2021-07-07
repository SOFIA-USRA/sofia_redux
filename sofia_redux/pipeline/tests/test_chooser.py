# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Redux Chooser class."""

from sofia_redux.pipeline.chooser import Chooser
from sofia_redux.pipeline.reduction import Reduction


class TestChooser(object):
    def test_choose_reduction(self):
        """Test run method."""
        chooser = Chooser()
        ro = chooser.choose_reduction()
        assert(type(ro) == Reduction)
