# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Basic Matplotlib viewer for image/plot display."""

import warnings

from astropy import log
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
import numpy as np

from sofia_redux.pipeline.viewer import Viewer

try:
    from PyQt5 import QtWidgets, QtGui, QtCore
    from matplotlib.backends.backend_qt5agg import \
        FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import \
        NavigationToolbar2QT as NavigationToolbar
except ImportError:
    HAS_PYQT5 = False
    QtGui, QtCore, FigureCanvas, NavigationToolbar = None, None, None, None

    # duck type parents to allow class definition
    class QtWidgets:
        class QDialog:
            pass
else:
    HAS_PYQT5 = True


class MatplotlibPlot(QtWidgets.QDialog):
    """Show plot data in a separate window."""
    def __init__(self, parent=None, title=None):
        """
        Build the matplotlib widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for Redux GUI.')

        super().__init__(parent)

        # plot configuration
        self.plot_layout = 'grid'
        self.max_plot = None
        self.min_width = 5.0
        self.min_height = 3.0
        self.share_axes = 'both'

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Matplotlib figure
        self.figure = Figure((self.min_width, self.min_height),
                             dpi=100, tight_layout=True)

        # Qt Canvas Widget
        csize = (int(100 * self.min_width),
                 int(100 * self.min_height))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(*csize)

        # navigation widget
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # add a scroll window
        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setGeometry(0, 0, *csize)

        self.scroll.setWidget(self.canvas)
        layout.addWidget(self.scroll)

        # make sure there's a close button
        self.setWindowFlags(QtCore.Qt.Window
                            | QtCore.Qt.WindowMaximizeButtonHint
                            | QtCore.Qt.WindowMinimizeButtonHint
                            | QtCore.Qt.WindowCloseButtonHint)

        # window title
        if title is not None:
            self.setWindowTitle(title)

    def plot(self, data):
        """
        Plot the data in the widget.

        Each primary list element in `data` will appear in
        a separate subplot. The dictionary values should contain 'args'
        and 'kwargs' to pass to the plot function.

        If the dictionary contains an 'overplot' key with a list of
        dicts containing their own 'args' values, the nested plots will be
        plotted on top of the primary plot.  Overplots may specify a
        'plot_type' which translates to matplotlib functions as follows:

            - vline : axes.axvline
            - plot : axes.plot

        Primary plot kwargs are set in the axes (via axes.set); overplot
        kwargs are passed to the overplot function directly.

        Parameters
        ----------
        data : `list` of dict
            Data sets to plot.
        """
        # clear the figure
        self.figure.clear()

        # determine layout for subplots
        n = len(data)
        if self.max_plot is not None and n > self.max_plot:
            n = self.max_plot
            log.warning("Too many plots to display; "
                        "showing first {}.".format(n))
        if n > 0:
            if self.plot_layout == 'grid':
                ncol = int(np.ceil(np.sqrt(n)))
                nrow = int(np.ceil(float(n) / ncol))
            elif self.plot_layout == 'columns':
                ncol = n
                nrow = 1
            else:
                # rows
                ncol = 1
                nrow = n
            width = self.min_width * ncol
            height = self.min_height * nrow
        else:
            ncol = 0
            nrow = 0
            width = self.min_width
            height = self.min_height

        # put each data set in a new subplot
        ax0 = None
        for i in range(n):
            dataset = data[i]

            # create an axis
            if i == 0:
                ax = self.figure.add_subplot(nrow, ncol, i + 1)
                ax.autoscale(enable=True, tight=True)
                ax0 = ax
            else:
                if self.share_axes == 'both':
                    ax = self.figure.add_subplot(nrow, ncol, i + 1,
                                                 sharex=ax0, sharey=ax0)
                elif self.share_axes == 'x':
                    ax = self.figure.add_subplot(nrow, ncol, i + 1, sharex=ax0)
                elif self.share_axes == 'y':
                    ax = self.figure.add_subplot(nrow, ncol, i + 1, sharey=ax0)
                else:
                    ax = self.figure.add_subplot(nrow, ncol, i + 1)

            if 'kwargs' in dataset:
                kwargs = dataset['kwargs'].copy()
            else:
                kwargs = {}
            if 'colormap' in kwargs:
                colormap = kwargs.pop('colormap')
                cmap = get_cmap(colormap)
                if cmap.N < 100:
                    n = cmap.N
                else:
                    n = 20
                color = cmap(np.linspace(0, 1, n))
                ax.set_prop_cycle('color', color)

            # plot data as specified
            ax.plot(*dataset['args'], **dataset['plot_kwargs'])
            if kwargs:
                try:
                    ax.set(**kwargs)
                except AttributeError:
                    log.warning('Bad argument to matplotlib plot.')
                    log.debug(('Provided kwargs for ax.set:', kwargs))

            if 'overplot' in dataset:
                for oplot in dataset['overplot']:
                    ptype = oplot.get('plot_type', 'plot')
                    kwargs = oplot.get('kwargs', {})
                    if ptype == 'vline':
                        func = ax.axvline
                    elif ptype == 'hline':
                        func = ax.axhline
                    elif ptype == 'line':
                        func = ax.axline
                    elif ptype == 'scatter':
                        func = ax.scatter
                    elif ptype == 'histogram':
                        func = ax.hist
                    elif ptype == 'text':
                        func = ax.text
                    elif ptype == 'legend':
                        func = ax.legend
                    else:
                        func = ax.plot
                    try:
                        func(*oplot['args'], **kwargs)
                    except (AttributeError, ValueError):
                        log.warning(f'Bad argument to matplotlib {ptype}.')
                        log.debug(('Provided kwargs for overplot:', kwargs))

        # resize figure and canvas
        current_size = self.scroll.size()
        new_size = (max(int(width * 100), current_size.width()),
                    max(int(height * 100), current_size.height()))
        self.figure.set_size_inches(((new_size[0]) / 100,
                                     (new_size[1]) / 100),
                                    forward=True)
        self.canvas.resize(*new_size)
        self.canvas.setMinimumSize(int(width * 100),
                                   int(height * 100))

        # redraw
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self.canvas.draw_idle()

    def set_scroll(self, location='bottom'):
        if location == 'top':
            top = self.scroll.verticalScrollBar().minimum()
            self.scroll.verticalScrollBar().setValue(top)
        elif location == 'left':
            left = self.scroll.horizontalScrollBar().minimum()
            self.scroll.horizontalScrollBar().setValue(left)
        elif location == 'right':
            right = self.scroll.horizontalScrollBar().maximum()
            self.scroll.horizontalScrollBar().setValue(right)
        else:
            bottom = self.scroll.verticalScrollBar().maximum()
            self.scroll.verticalScrollBar().setValue(bottom)

    def clear(self):
        self.figure.clear()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self.canvas.draw_idle()


class MatplotlibViewer(Viewer):
    def __init__(self, title=None):
        """Basic Matplotlib viewer."""
        super().__init__()
        self.name = "MatplotlibViewer"
        self.embedded = False

        self.plotter = None
        self.layout = 'grid'
        self.max_plot = None
        self.title = title
        self.share_axes = 'both'

    def display(self):
        """Display a plot in a Matplotlib canvas."""
        if not self.display_data:
            self.close()
            return

        if self.plotter is None or not self.plotter.isVisible():
            self.plotter = MatplotlibPlot(title=self.title)
            self.plotter.plot_layout = self.layout
            self.plotter.max_plot = self.max_plot
            self.plotter.share_axes = self.share_axes

        self.plotter.plot(self.display_data)
        self.plotter.show()

    def close(self):
        """Close the viewer."""
        if self.plotter is not None:
            log.debug("Closing Matplotlib plot.")
            self.plotter.close()
            self.plotter = None
