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
    """Atrophy the view."""

    atrophy_controls = QtCore.pyqtSignal()
    """Atrophy the controls."""

    atrophy_bg_full = QtCore.pyqtSignal()
    """Atrophy the full background."""

    atrophy_bg_partial = QtCore.pyqtSignal()
    """Partially atrophy the background."""

    refresh_file_table = QtCore.pyqtSignal()
    """Refresh the file table."""

    refresh_order_list = QtCore.pyqtSignal()
    """Refresh the order list."""

    current_pane_changed = QtCore.pyqtSignal()
    """Indicate the current pane has changed."""

    axis_limits_changed = QtCore.pyqtSignal()
    """Indicate the axis limits have changed."""

    axis_scale_changed = QtCore.pyqtSignal()
    """Indicate the axis scale has changed."""

    axis_unit_changed = QtCore.pyqtSignal()
    """Indicate the axis unit has changed."""

    axis_field_changed = QtCore.pyqtSignal()
    """Indicate the axis field has changed."""

    cursor_loc_changed = QtCore.pyqtSignal()
    """Indicate the cursor location has changed."""

    panes_changed = QtCore.pyqtSignal()
    """Indicate the panes have been changed."""

    model_selected = QtCore.pyqtSignal()
    """Select a model."""

    model_removed = QtCore.pyqtSignal()
    """Remove a model."""

    end_zoom_mode = QtCore.pyqtSignal()
    """End the current zoom mode."""

    end_cursor_recording = QtCore.pyqtSignal()
    """End cursor recording."""

    clear_fit = QtCore.pyqtSignal()
    """Clear fit values."""

    toggle_fit_visibility = QtCore.pyqtSignal()
    """Toggle the fit visibility."""

    update_reference_lines = QtCore.pyqtSignal()
    """Update displayed reference lines."""

    unload_reference_model = QtCore.pyqtSignal()
    """Unload the current reference model."""

    obtain_raw_model = QtCore.pyqtSignal()
    """Get a copy of the raw model."""

    on_orders_changed = QtCore.pyqtSignal()
    """Indicate enabled orders have changed."""

    off_orders_changed = QtCore.pyqtSignal()
    """Indicate hidden orders have changed."""

    controls_updated = QtCore.pyqtSignal()
    """Indicate the controls have been updated."""
