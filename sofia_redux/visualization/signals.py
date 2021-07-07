# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    from PyQt5 import QtCore
except ImportError:
    HAS_PYQT5 = False

    # duck type parents to allow class definition
    class QtCore:
        class QObject:
            pass

        @staticmethod
        def pyqtSignal():
            return
else:
    HAS_PYQT5 = True

__all__ = ['Signals']


class Signals(QtCore.QObject):
    """Custom signals used in the Eye GUI."""
    atrophy = QtCore.pyqtSignal()
    atrophy_controls = QtCore.pyqtSignal()
    atrophy_bg_full = QtCore.pyqtSignal()
    atrophy_bg_partial = QtCore.pyqtSignal()
    refresh_file_table = QtCore.pyqtSignal()
    refresh_order_list = QtCore.pyqtSignal()
    current_pane_changed = QtCore.pyqtSignal()
    axis_limits_changed = QtCore.pyqtSignal()
    axis_scale_changed = QtCore.pyqtSignal()
    axis_unit_changed = QtCore.pyqtSignal()
    axis_field_changed = QtCore.pyqtSignal()
    cursor_loc_changed = QtCore.pyqtSignal()
    panes_changed = QtCore.pyqtSignal()
    model_selected = QtCore.pyqtSignal()
    model_removed = QtCore.pyqtSignal()
    end_zoom_mode = QtCore.pyqtSignal()
    end_cursor_recording = QtCore.pyqtSignal()
    clear_fit = QtCore.pyqtSignal()
    toggle_fit_visibility = QtCore.pyqtSignal()
