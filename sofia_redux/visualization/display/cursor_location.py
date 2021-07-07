#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from typing import Any, List, Dict, Tuple, Union

try:
    from PyQt5 import QtWidgets, QtGui, QtCore
    from sofia_redux.visualization.display.ui import cursor_location as cl
except ImportError:
    HAS_PYQT5 = False
    QtWidgets, QtGui, QtCore = None, None, None

    class QtWidgets:
        class QDialog:
            pass

    class cl:
        class Ui_Dialog:
            pass
else:
    from PyQt5.QtWidgets import QTableWidgetItem
    from PyQt5.QtGui import QColor
    HAS_PYQT5 = True

__all__ = ['CursorLocation']


class CursorLocation(QtWidgets.QDialog, cl.Ui_Dialog):
    """
    Cursor location display widget.

    Parameters
    ----------
    parent : View
        Parent view widget for the dialog window.

    Attributes
    ----------
    points : list
        Cursor data points to display in the widget.
    """

    def __init__(self, parent: Any) -> None:
        if not HAS_PYQT5:  # pragma: no cover
            raise ImportError('PyQt5 package is required for the Eye.')
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)
        self.setModal(0)

        self.points = list()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        """
        Close the window.

        Calls the parent `closed_cursor_popout` method.

        Parameters
        ----------
        a0 : QtGui.QCloseEvent
            The close event.
        """
        self.parentWidget().closed_cursor_popout()

    def update_points(self, data_coords: Dict,
                      cursor_coords: Union[List, Tuple]) -> None:
        """
        Update displayed data points.

        Parameters
        ----------
        data_coords : dict
            Keys are filenames; values are lists of dicts
            containing 'order', 'bin', 'bin_x', 'bin_y',
            'x_field', 'y_field', 'color', and 'visible'
            values to display.
        cursor_coords : tuple or list
            Current cursor (x, y) coordinates.
        """
        new_points = self._flatten_combine(data_coords,
                                           cursor_coords)
        if len(new_points) == 0:
            return
        self._define_table(row_count=len(new_points),
                           col_count=len(new_points[0]) - 1)
        for row_index, new_point in enumerate(new_points):
            for col_index, value in enumerate(new_point[1:]):
                if isinstance(value, float):
                    item = QTableWidgetItem(f'{value:.3g}')
                elif isinstance(value, int):
                    item = QTableWidgetItem(f'{value:d}')
                elif isinstance(value, str):
                    if value.startswith('#'):
                        # special handling for color values
                        item = QTableWidgetItem('')
                        item.setBackground(QColor(value))
                    else:
                        item = QTableWidgetItem(value)
                else:
                    item = QTableWidgetItem('######')
                item.setTextAlignment(QtCore.Qt.AlignHCenter)
                self.table_widget.setItem(row_index, col_index, item)
            # set filename as vertical header
            vhead = QTableWidgetItem(os.path.basename(str(new_point[0])))
            vhead.setTextAlignment(QtCore.Qt.AlignLeft)
            vhead.setToolTip(new_point[0])
            self.table_widget.setVerticalHeaderItem(row_index, vhead)
        self._resize()

    @staticmethod
    def _flatten_combine(data_coords: Dict,
                         cursor_coords: Union[List, Tuple]) -> List:
        """
        Flatten input coordinates to a single table.

        Parameters
        ----------
        data_coords : dict
            Keys are filenames; values are lists of dicts
            containing 'order', 'bin', 'bin_x', 'bin_y',
            'x_field', 'y_field', 'color', and 'visible'
            values to display,
        cursor_coords : tuple or list
            Current cursor (x, y) coordinates.

        Returns
        -------
        points : list
            Each element of the list is a list with values
            filename, order, color, x_field, y_field,
            x_cursor, y_cursor, x_value, y_value, column.
        """
        points = list()
        for model_id, model_data_coords in data_coords.items():
            point = None
            for values in model_data_coords:
                if not values['visible']:
                    continue
                point = [model_id, values['order'],
                         values['color'],
                         values['x_field'], values['y_field'],
                         cursor_coords[0], cursor_coords[1],
                         values['bin_x'], values['bin_y'],
                         values['bin']]
                points.append(point)
            if point is None:
                # no points found, append a blank entry for the model
                point = [model_id] + ['-'] * 9
                points.append(point)

        return points

    def _define_table(self, row_count: int, col_count: int) -> None:
        """
        Define the table widget.

        Parameters
        ----------
        row_count : int
            Number of data rows in the table, not including the
            horizontal header row.
        col_count : int
            Number of data columns in the table, not including the
            vertical header column.
        """
        self.table_widget.setRowCount(row_count)
        self.table_widget.setColumnCount(col_count)

    def _resize(self) -> None:
        """Resize the table widget to its contents."""
        self.table_widget.resizeRowsToContents()
