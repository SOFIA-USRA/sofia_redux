# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the Matplotlib Viewer class."""

from matplotlib import colors as mc
from matplotlib.figure import Figure
import numpy as np
import pytest

from sofia_redux.pipeline.gui.matplotlib_viewer import MatplotlibViewer

try:
    from PyQt5 import QtWidgets
except ImportError:
    QtWidgets = None
    HAS_PYQT5 = False
else:
    HAS_PYQT5 = True


@pytest.fixture()
def mpview(qtbot, mocker):
    """Make and register a Matplotlib window."""
    mocker.patch.object(QtWidgets.QDialog, 'show',
                        return_value=None)
    view = MatplotlibViewer(title='test')
    view.start(parent=qtbot)
    return view


@pytest.mark.skipif("not HAS_PYQT5")
class TestMatplotlibViewer(object):
    """Test the MatplotlibViewer class"""

    @pytest.fixture(autouse=True, scope='function')
    def mock_app(self, qapp, mocker):
        mocker.patch.object(QtWidgets, 'QApplication',
                            return_value=qapp)

    def make_plot(self, nplot=1, overplot=False):
        plots = []
        for i in range(nplot):
            disp = {'args': [np.arange(10), np.arange(10)],
                    'kwargs': {'title': f'test {i}',
                               'xlabel': 'x',
                               'ylabel': 'y'},
                    'plot_kwargs': {}}
            if overplot:
                disp['overplot'] = [{'args': [np.arange(10), np.arange(10)]},
                                    {'args': [5], 'plot_type': 'vline'},
                                    {'args': [5], 'plot_type': 'hline'}]
            plots.append(disp)
        return plots

    def test_display(self, mpview):
        # make a test plot
        plots = self.make_plot()

        # update viewer
        mpview.update(plots)

        # test image was displayed
        assert isinstance(mpview.plotter.figure, Figure)
        assert 'test' in mpview.plotter.windowTitle()

    def test_reset(self, mpview):
        # display a plot
        plots = self.make_plot()
        mpview.update(plots)
        assert isinstance(mpview.plotter.figure, Figure)

        # reset
        mpview.reset()

        # check that data is empty
        assert len(mpview.display_data) == 0

        # check that plotter is gone
        assert mpview.plotter is None

    def test_number_plots(self, mpview, capsys):
        # make some plots
        plots = self.make_plot(nplot=4)

        # test max_plot configuration
        mpview.max_plot = 2
        mpview.update(plots)
        assert "showing first 2" in capsys.readouterr().err
        assert len(mpview.plotter.figure.get_axes()) == 2
        mpview.reset()

        # test layouts -- grid, rows, columns
        mpview.max_plot = None
        mpview.plot_layout = 'grid'
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        assert len(axes) == 4
        geom = axes[-1].get_geometry()
        assert geom == (2, 2, 4)
        mpview.reset()

        mpview.layout = 'rows'
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        assert len(axes) == 4
        geom = axes[-1].get_geometry()
        print(axes, mpview.plotter.plot_layout, geom)
        assert geom == (4, 1, 4)
        mpview.reset()

        mpview.layout = 'columns'
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        assert len(axes) == 4
        geom = axes[-1].get_geometry()
        assert geom == (1, 4, 4)

        # test null plot -- should just clear axes
        mpview.plotter.plot([])
        axes = mpview.plotter.figure.get_axes()
        assert len(axes) == 0

    def test_args(self, mpview, capsys):
        # try a bad kwarg
        plots = self.make_plot()
        plots[0]['kwargs']['bad_arg'] = 'test'
        mpview.update(plots)
        assert 'Bad argument' in capsys.readouterr().err
        # line is still plotted
        axes = mpview.plotter.figure.get_axes()
        assert len(axes[0].lines) == 1

        # add an overplot
        plots = self.make_plot(overplot=True)
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        assert len(axes[0].lines) == 4

        # add an overplot with a bad kwarg
        plots[0]['overplot'][0]['kwargs'] = {'badval': True}
        mpview.update(plots)
        assert 'Bad argument' in capsys.readouterr().err
        # overplot is not added
        axes = mpview.plotter.figure.get_axes()
        assert len(axes[0].lines) == 3

    def test_share_axes(self, mpview):
        # make some plots
        plots = self.make_plot(nplot=4)

        # test sharing on
        mpview.share_axes = 'both'
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        ax1 = axes[0]
        new_limit = (0.1, 0.8)
        ax1.set_xlim(new_limit)
        ax1.set_ylim(new_limit)
        for i, ax in enumerate(axes):
            assert ax.get_xlim() == new_limit
            assert ax.get_ylim() == new_limit

        # test sharing off
        mpview.share_axes = 'none'
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        ax1 = axes[0]
        new_limit = (0.1, 0.8)
        ax1.set_xlim(new_limit)
        ax1.set_ylim(new_limit)
        for i, ax in enumerate(axes):
            if i == 0:
                assert ax.get_xlim() == new_limit
                assert ax.get_ylim() == new_limit
            else:
                assert ax.get_xlim() != new_limit
                assert ax.get_ylim() != new_limit

        # test share x only
        mpview.share_axes = 'x'
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        ax1 = axes[0]
        new_limit = (0.1, 0.8)
        ax1.set_xlim(new_limit)
        ax1.set_ylim(new_limit)
        for i, ax in enumerate(axes):
            if i == 0:
                assert ax.get_xlim() == new_limit
                assert ax.get_ylim() == new_limit
            else:
                assert ax.get_xlim() == new_limit
                assert ax.get_ylim() != new_limit

        # test share y only
        mpview.share_axes = 'y'
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        ax1 = axes[0]
        new_limit = (0.1, 0.8)
        ax1.set_xlim(new_limit)
        ax1.set_ylim(new_limit)
        for i, ax in enumerate(axes):
            if i == 0:
                assert ax.get_xlim() == new_limit
                assert ax.get_ylim() == new_limit
            else:
                assert ax.get_xlim() != new_limit
                assert ax.get_ylim() == new_limit

    def test_no_kwargs(self, mpview):
        plots = self.make_plot(nplot=2)

        # take out kwargs - should work without errors
        del plots[0]['kwargs']

        # update viewer
        mpview.update(plots)

        # all plots are shown
        axes = mpview.plotter.figure.get_axes()
        assert len(axes) == 2

    def test_colormap(self, mpview):
        plots = self.make_plot()

        # update with plots, get the next color from the default cycle
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        ax1_cyc = next(axes[0]._get_lines.prop_cycler)['color']

        # set a continuous colormap kwarg and update viewer again
        plots[0]['kwargs']['colormap'] = 'plasma'
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        ax2_cyc = next(axes[0]._get_lines.prop_cycler)['color']

        # next color is different, after changing colormap
        assert mc.to_hex(ax1_cyc) != mc.to_hex(ax2_cyc)
        assert mc.to_hex(ax2_cyc) == '#2c0594'

        # set a discrete map - should also work
        plots[0]['kwargs']['colormap'] = 'tab20b'
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        ax3_cyc = next(axes[0]._get_lines.prop_cycler)['color']

        # next color is different, after changing colormap
        assert mc.to_hex(ax2_cyc) != mc.to_hex(ax3_cyc)
        assert mc.to_hex(ax3_cyc) == '#5254a3'

    @pytest.mark.parametrize('ptype,oplot_args,nline',
                             [('vline', [2.0], 2), ('hline', [2.0], 2),
                              ('line', [(2, 2), (4, 4)], 2),
                              ('scatter', [np.arange(10), np.arange(10)], 1),
                              ('histogram', [np.arange(10)], 1),
                              ('text', [2, 4, 'sample text'], 1),
                              ('legend', [], 1),
                              ('unknown', [np.arange(10), np.arange(10)], 2)])
    def test_plot_types(self, mpview, ptype, oplot_args, nline):
        plots = self.make_plot()
        plots[0]['overplot'] = [{'plot_type': ptype, 'args': oplot_args}]
        mpview.update(plots)
        axes = mpview.plotter.figure.get_axes()
        assert len(axes[0].lines) == nline

    def test_set_scroll(self, mpview):
        plots = self.make_plot(nplot=4)
        mpview.update(plots)

        mpview.plotter.set_scroll('top')
        bar = mpview.plotter.scroll.verticalScrollBar()
        assert bar.value() == bar.minimum()

        mpview.plotter.set_scroll('bottom')
        bar = mpview.plotter.scroll.verticalScrollBar()
        assert bar.value() == bar.maximum()

        mpview.plotter.set_scroll('right')
        bar = mpview.plotter.scroll.horizontalScrollBar()
        assert bar.value() == bar.maximum()

        mpview.plotter.set_scroll('left')
        bar = mpview.plotter.scroll.horizontalScrollBar()
        assert bar.value() == bar.minimum()

    def test_clear(self, mpview):
        plots = self.make_plot()
        mpview.update(plots)
        assert len(mpview.plotter.figure.get_axes()) == 1

        mpview.plotter.clear()
        assert len(mpview.plotter.figure.get_axes()) == 0
